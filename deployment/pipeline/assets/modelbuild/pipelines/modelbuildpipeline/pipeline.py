"""Example workflow pipeline script for RUL pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import json
import logging
import os

import boto3
import sagemaker
from botocore.exceptions import ClientError
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.lambda_helper import Lambda
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import (ProcessingInput, ProcessingOutput,
                                  ScriptProcessor)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.lambda_step import (LambdaOutput, LambdaOutputTypeEnum,
                                            LambdaStep)
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)


def create_lambda_role(role_name):
    """Create a role for your lambda function.

    Args:
        role_name: the name of the newly created role

    Returns:
        role arn
    """
    iam = boto3.client("iam")
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "lambda.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
            Description="Role for Lambda to call SageMaker functions",
        )

        role_arn = response["Role"]["Arn"]

        response = iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )

        response = iam.create_policy(
            PolicyName='LambdaS3Access',
            PolicyDocument=json.dumps({
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Action": [
                            "s3:DeleteObject*",
                            "s3:PutObject",
                            "s3:Abort*"
                        ],
                        "Resource": [
                            "arn:aws:s3:::sagemaker-*",
                        ],
                        "Effect": "Allow"
                    }
                ]
            }),
            Description='Enable write access to S3',
        )

        arn = response["Policy"]["Arn"]

        response = iam.attach_role_policy(
            PolicyArn=arn,
            RoleName=role_name,
        )

        return role_arn

    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Using ARN from existing role: {role_name}")
        response = iam.get_role(RoleName=role_name)
        return response["Role"]["Arn"]


def get_pipeline_session(region, default_bucket):

    """Gets the sagemaker pipeline session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.workflow.pipeline_context.PipelineSession instance
    """
    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker.workflow.pipeline_context.PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def resolve_ecr_uri_from_image_versions(sagemaker_session, image_versions, image_name):
    """Gets ECR URI from image versions
    Args:
        sagemaker_session: boto3 session for sagemaker client
        image_versions: list of the image versions
        image_name: Name of the image

    Returns:
        ECR URI of the image version
    """

    # Fetch image details to get the Base Image URI
    for image_version in image_versions:
        if image_version["ImageVersionStatus"] == "CREATED":
            image_arn = image_version["ImageVersionArn"]
            version = image_version["Version"]
            logger.info(f"Identified the latest image version: {image_arn}")
            response = sagemaker_session.sagemaker_client.describe_image_version(
                ImageName=image_name, Version=version
            )
            return response["ContainerImage"]
    return None


def resolve_ecr_uri(sagemaker_session, image_arn):
    """Gets the ECR URI from the image name

    Args:
        sagemaker_session: boto3 session for sagemaker client
        image_name: name of the image

    Returns:
        ECR URI of the latest image version
    """

    # Fetching image name from image_arn (^arn:aws(-[\w]+)*:sagemaker:.+:[0-9]{12}:image/[a-z0-9]([-.]?[a-z0-9])*$)
    image_name = image_arn.partition("image/")[2]
    try:
        # Fetch the image versions
        next_token = ""
        while True:
            response = sagemaker_session.sagemaker_client.list_image_versions(
                ImageName=image_name,
                MaxResults=100,
                SortBy="VERSION",
                SortOrder="DESCENDING",
                NextToken=next_token,
            )
            ecr_uri = resolve_ecr_uri_from_image_versions(
                sagemaker_session, response["ImageVersions"], image_name
            )
            if "NextToken" in response:
                next_token = response["NextToken"]

            if ecr_uri is not None:
                return ecr_uri

        # Return error if no versions of the image found
        error_message = f"No image version found for image name: {image_name}"
        logger.error(error_message)
        raise Exception(error_message)

    except (
        ClientError,
        sagemaker_session.sagemaker_client.exceptions.ResourceNotFound,
    ) as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)


def get_pipeline(
    region,
    role=None,
    default_bucket="sagemaker-p-project-123456789042",
    model_package_group_name="EnergyMgtPackageGroup",
    pipeline_name="EnergyMgtPipeline",
    base_job_prefix="EnergyMgt",
    project_id="p-abcdefghijkl",
):
    """Gets a SageMaker ML Pipeline instance working with on RUL data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    pipeline_session = get_pipeline_session(region, default_bucket)
    lambda_role = create_lambda_role("lambda-update-manifest-role")
    if role is None:
        role = sagemaker.session.get_execution_role(pipeline_session)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.4xlarge"
    )
    inference_instance_type = ParameterString(
        name="InferenceInstanceType", default_value="ml.m5.4xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    processing_image_name = "sagemaker-{0}-processing-imagebuild".format(project_id)
    training_image_name = "sagemaker-{0}-training-imagebuild".format(project_id)
    inference_image_name = "sagemaker-{0}-training-imagebuild".format(project_id)

    # processing step for feature engineering
    try:
        processing_image_uri = (
            pipeline_session.sagemaker_client.describe_image_version(
                ImageName=processing_image_name
            )["ContainerImage"]
        )
    except (pipeline_session.sagemaker_client.exceptions.ResourceNotFound):
        processing_image_uri = sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=region,
            version="1.3-1",
            py_version="py3",
            instance_type=processing_instance_type,
        )
    script_processor = ScriptProcessor(
        image_uri=processing_image_uri,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/EnergyMgt-preprocess",
        command=["python3"],
        sagemaker_session=pipeline_session,
        role=role,
        volume_size_in_gb=100,
        network_config=sagemaker.network.NetworkConfig(enable_network_isolation=True)
    )
    processing_args = script_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--bucket-name", default_bucket],
    )
    step_process = ProcessingStep(
        name="PreprocessEnergyMgtData",
        step_args=processing_args
    )

    # training step for generating model artifacts
    model_path = (
        f"s3://{pipeline_session.default_bucket()}/{base_job_prefix}/EnergyMgtTrain"
    )

    try:
        training_image_uri = pipeline_session.sagemaker_client.describe_image_version(
            ImageName=training_image_name
        )["ContainerImage"]
    except (pipeline_session.sagemaker_client.exceptions.ResourceNotFound):
        training_image_uri = sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=region,
            version="1.3-1",
            py_version="py3",
            instance_type=training_instance_type,
        )

    xgb_train = Estimator(
        image_uri=training_image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        base_job_name=f"{base_job_prefix}/EnergyMgt-train",
        sagemaker_session=pipeline_session,
        role=role,
        enable_sagemaker_metrics=True,
        metric_definitions=[
            {'Name': 'keras_validation_c1c2:mae', 'Regex': 'Validation_c1c2_keras_mae=(.*?);'},
            {'Name': 'keras_validation_c1c2c4:mae', 'Regex': 'Validation_c1c2c4_keras_mae=(.*?);'},
            {'Name': 'keras_validation_c1c4:mae', 'Regex': 'Validation_c1c4_keras_mae=(.*?);'},
            {'Name': 'keras_validation_c1c3c4:mae', 'Regex': 'Validation_c1c3c4_keras_mae=(.*?);'},
            {'Name': 'keras_validation_c1c2c3c4:mae', 'Regex': 'Validation_c1c2c3c4_keras_mae=(.*?);'},
            {'Name': 'keras_validation_c1c3:mae', 'Regex': 'Validation_c1c3_keras_mae=(.*?);'},
            {'Name': 'keras_validation_c1c2c3:mae', 'Regex': 'Validation_c1c2c3_keras_mae=(.*?);'},
            {'Name': 'keras_validation_c3c4:mae', 'Regex': 'Validation_c3c4_keras_mae=(.*?);'},
            {'Name': 'keras_validation_c2c3c4:mae', 'Regex': 'Validation_c2c3c4_keras_mae=(.*?);'},
            {'Name': 'keras_validation_c2c4:mae', 'Regex': 'Validation_c2c4_keras_mae=(.*?);'},
            {'Name': 'keras_validation_c2c3:mae', 'Regex': 'Validation_c2c3_keras_mae=(.*?);'},
            {'Name': 'keras_train_c1c2:mae', 'Regex': 'Train_c1c2_keras_mae=(.*?);'},
            {'Name': 'keras_train_c1c2c4:mae', 'Regex': 'Train_c1c2c4_keras_mae=(.*?);'},
            {'Name': 'keras_train_c1c4:mae', 'Regex': 'Train_c1c4_keras_mae=(.*?);'},
            {'Name': 'keras_train_c1c3c4:mae', 'Regex': 'Train_c1c3c4_keras_mae=(.*?);'},
            {'Name': 'keras_train_c1c2c3c4:mae', 'Regex': 'Train_c1c2c3c4_keras_mae=(.*?);'},
            {'Name': 'keras_train_c1c3:mae', 'Regex': 'Train_c1c3_keras_mae=(.*?);'},
            {'Name': 'keras_train_c1c2c3:mae', 'Regex': 'Train_c1c2c3_keras_mae=(.*?);'},
            {'Name': 'keras_train_c3c4:mae', 'Regex': 'Train_c3c4_keras_mae=(.*?);'},
            {'Name': 'keras_train_c2c3c4:mae', 'Regex': 'Train_c2c3c4_keras_mae=(.*?);'},
            {'Name': 'keras_train_c2c4:mae', 'Regex': 'Train_c2c4_keras_mae=(.*?);'},
            {'Name': 'keras_train_c2c3:mae', 'Regex': 'Train_c2c3_keras_mae=(.*?);'},
        ],
        enable_network_isolation=True,
        disable_profiler=True

    )

    xgb_train.set_hyperparameters(
        quantile=0.5, test_size=0.2, random_state=42, degree=4, remove_outliers=False,
        verbose=1, dual_model=False, optimizer="adam", optimizer__learning_rate=0.001,
        epochs=2000
    )
    train_args = xgb_train.fit({
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
     })
    step_train = TrainingStep(
        name="TrainEnergyMgtModel",
        step_args=train_args
    )

    try:
        inference_image_uri = pipeline_session.sagemaker_client.describe_image_version(
            ImageName=inference_image_name
        )["ContainerImage"]
    except (pipeline_session.sagemaker_client.exceptions.ResourceNotFound):
        inference_image_uri = sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=region,
            version="1.3-1",
            py_version="py3",
            instance_type=inference_instance_type,
        )

    simulate_processor = ScriptProcessor(
        image_uri=processing_image_uri,
        instance_type="ml.c5.9xlarge",
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/EnergyMgt-simulate",
        command=["python3"],
        sagemaker_session=pipeline_session,
        role=role,
        volume_size_in_gb=10,
    )
    inference_args = simulate_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ],
        code=os.path.join(BASE_DIR, "simulate.py"),
        arguments=["--model-artefact", step_train.properties.ModelArtifacts.S3ModelArtifacts]
    )
    simulate_step = ProcessingStep(
        name="SimulateEnergyMgtData",
        step_args=inference_args,
        depends_on=[step_train],
    )

    # Lambda helper class can be used to create the Lambda function
    func = Lambda(
        function_name="lambda-update-manifest",
        execution_role_arn=lambda_role,
        script=os.path.join(BASE_DIR, "lambda_handler.py"),
        handler="lambda_handler.lambda_handler",
    )

    output_param_1 = LambdaOutput(output_name="statusCode", output_type=LambdaOutputTypeEnum.String)
    output_param_2 = LambdaOutput(output_name="body", output_type=LambdaOutputTypeEnum.String)

    step_put_manifest = LambdaStep(
        name="UpdateManifestLambda",
        lambda_func=func,
        inputs={
            "TRAINING": step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            "SIMULATION": simulate_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            "BUCKET": default_bucket,
        },
        outputs=[output_param_1, output_param_2],
        depends_on=[simulate_step],
    )

    step_register = RegisterModel(
        name="RegisterEnergyMgtModel",
        estimator=xgb_train,
        image_uri=inference_image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        depends_on=[step_put_manifest],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
        ],
        steps=[step_process, step_train, simulate_step, step_put_manifest, step_register],
        sagemaker_session=pipeline_session,
    )
    return pipeline
