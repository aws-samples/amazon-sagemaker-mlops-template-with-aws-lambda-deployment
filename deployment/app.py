#!/usr/bin/env python3

import os
import random
import string
import aws_cdk as _cdk
import cdk_nag as cdknag
import boto3
from SageMakerPipelineSourceCode.SageMakerPipelineSourceCode_stack import \
    SageMakerPipelineSourceCodeStack


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str

app = _cdk.App()

aws_region = boto3.session.Session().region_name
aws_account_id = boto3.client('sts').get_caller_identity().get('Account')
project_id = get_random_string(12)

stack = SageMakerPipelineSourceCodeStack(
    app,
    "EnergyOptimization",
    sagemaker_project_name="enopt-project-cdk",
    sagemaker_project_id=f"p-{project_id}",
    enable_processing_image_build_pipeline=True,
    enable_training_image_build_pipeline=True,
    enable_inference_image_build_pipeline=False,
    aws_account_id=aws_account_id,
    aws_region=aws_region,
    container_image_tag="latest")
_cdk.Aspects.of(app).add(cdknag.AwsSolutionsChecks())


cdknag.NagSuppressions.add_stack_suppressions(
    stack,
    [
        cdknag.NagPackSuppression(
            id="AwsSolutions-IAM4",
            reason="Use AWS managed poclicies AWSLambdaBasicExecutionRole",
        )
    ],
)

cdknag.NagSuppressions.add_stack_suppressions(
    stack,
    [
        cdknag.NagPackSuppression(
            id="AwsSolutions-IAM5",
            reason="Use AWS managed poclicies CodeBuild Project with defaults from cdk",
        )
    ],
)

cdknag.NagSuppressions.add_stack_suppressions(
    stack,
    [
        cdknag.NagPackSuppression(
            id="AwsSolutions-CB4",
            reason="S3 and Sagemaker Jobs are encrpyted",
        )
    ],
)
app.synth()
