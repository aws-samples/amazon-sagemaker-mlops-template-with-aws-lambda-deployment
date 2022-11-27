from aws_cdk import Duration, Stack
from aws_cdk import aws_codebuild as _codebuild
from aws_cdk import aws_codecommit as _codecommit
from aws_cdk import aws_codepipeline as _codepipeline
from aws_cdk import aws_codepipeline_actions as _actions
from aws_cdk import aws_ec2 as _ec2
from aws_cdk import aws_ecr as _ecr
from aws_cdk import aws_events as _events
from aws_cdk import aws_events_targets as _targets
from aws_cdk import aws_iam as _iam
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_s3 as _s3
from aws_cdk import aws_s3_deployment as _s3_deploy
from aws_cdk import aws_sagemaker as _sagemaker
from constructs import Construct

from .role_policy import role_policy_ecr_image_build
from .role_policy import role_policy_model_build
from .role_policy import role_policy_model_deploy
from .role_policy import role_policy_sagemaker_pipeline_execution 

class SageMakerPipelineSourceCodeStack(Stack):
    """SageMakerPipelineSourceCodeStack class to deploy the AWS CDK stack.

    Attributes:
    :mlops_*_policy:     The IAM inline policy that gets attached
                                            to the `mlops_*_role`
    :mlops_*_role:       The IAM role that will be created
    :mlops_artifacts_bucket:                The Amazon S3 artifact bucket
    :sagemaker_project_name:                The name of the Amazon SageMaker project
    :sagemaker_project_id:                  The unique Amazon SageMaker project ID
    :aws_account_id:                        The AWS account the solution gets deployed in
    :aws_region:                            The region this stack will be deployed to
    """

    def create_iam_role(self, **kwargs) -> _iam.Role:
        """Create the IAM role

        Args:
            No arguments

        Returns:
            No return value
        """
        # Create the policy document
        self.mlops_training_image_build_policy = _iam.PolicyDocument(
            statements=role_policy_ecr_image_build)
        self.mlops_processing_image_build_policy = _iam.PolicyDocument(
            statements=role_policy_ecr_image_build)
        self.mlops_model_build_policy = _iam.PolicyDocument(
            statements=role_policy_model_build)
        self.mlops_model_deploy_policy = _iam.PolicyDocument(
            statements=role_policy_model_deploy)
        self.mlops_sagemaker_pipeline_policy = _iam.PolicyDocument(
            statements=role_policy_sagemaker_pipeline_execution)
        # Define the IAM role
        self.mlops_training_image_build_role = _iam.Role(
            self,
            "SageMakerMLOpsTrainingImageBuildRole",
            #role_name="SageMakerMLOpsEcrImageBuildRole",
            assumed_by=_iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="The SageMakerMLOpsTrainingImageBuildRole for trainign Image build.",
            inline_policies={
                "SageMakerMLOpsTrainingImageBuildPolicy": self.mlops_training_image_build_policy,
            },
        )
        self.mlops_processing_image_build_role = _iam.Role(
            self,
            "SageMakerMLOpsProcessingImageBuildRole",
            #role_name="SageMakerMLOpsEcrImageBuildRole",
            assumed_by=_iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="The SageMakerMLOpsProcessingImageBuildRole for processingImage build.",
            inline_policies={
                "SageMakerMLOpsProcessingImageBuildPolicy": self.mlops_processing_image_build_policy,
            },
        )
        self.mlops_model_build_role = _iam.Role(
            self,
            "SageMakerMLOpsModelBuildRole",
            #role_name="SageMakerMLOpsProductUseRole",
            assumed_by=_iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="The SageMakerMLOpsModelBuildRole for service interactions.",
            inline_policies={
                "SageMakerMLOpsModelBuildPolicy": self.mlops_model_build_policy,
            },
        )
        self.mlops_model_deploy_role = _iam.Role(
            self,
            "SageMakerMLOpsModelDeployRole",
            #role_name="SageMakerMLOpsProductUseRole",
            assumed_by=_iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="The SageMakerMLOpsModelDeployRole for service interactions.",
            inline_policies={
                "SageMakerMLOpsModelDeployPolicy": self.mlops_model_deploy_policy,
            },
        )
        self.mlops_sagemaker_pipeline_role = _iam.Role(
            self,
            "SageMakerMLOpsSagemakerPipelineRole",
            #role_name="SageMakerMLOpsEcrImageBuildRole",
            assumed_by=_iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="The SageMakerPipelineRole for executing pipeline .",
            managed_policies=[
                _iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerPipelinesIntegrations")
            ],
            inline_policies={
                "SageMakerMLOpsSagemkerPipelinePolicy": self.mlops_sagemaker_pipeline_policy,
            },
        )
        # Add more service principals the IAM role can assume
        self.mlops_training_image_build_role.assume_role_policy.add_statements(
            _iam.PolicyStatement(
                actions=["sts:AssumeRole"],
                effect=_iam.Effect.ALLOW,
                principals=[
                    _iam.ServicePrincipal("codebuild.amazonaws.com"),
                ]))
        self.mlops_processing_image_build_role.assume_role_policy.add_statements(
            _iam.PolicyStatement(
                actions=["sts:AssumeRole"],
                effect=_iam.Effect.ALLOW,
                principals=[
                    _iam.ServicePrincipal("codebuild.amazonaws.com"),
                ]))        
        self.mlops_model_build_role.assume_role_policy.add_statements(
            _iam.PolicyStatement(
                actions=["sts:AssumeRole"],
                effect=_iam.Effect.ALLOW,
                principals=[
                    _iam.ServicePrincipal("codebuild.amazonaws.com"),
                ]))
        self.mlops_model_deploy_role.assume_role_policy.add_statements(
            _iam.PolicyStatement(
                actions=["sts:AssumeRole"],
                effect=_iam.Effect.ALLOW,
                principals=[
                    _iam.ServicePrincipal("codebuild.amazonaws.com"),
                ]))
        self.mlops_sagemaker_pipeline_role.assume_role_policy.add_statements(
            _iam.PolicyStatement(
                actions=["sts:AssumeRole"],
                effect=_iam.Effect.ALLOW,
                principals=[
                    _iam.ServicePrincipal("sagemaker.amazonaws.com"),
                ]))

    def create_s3_artifact_bucket(self, **kwargs) -> _s3.Bucket:
        """Create the Amazon S3 bucket to store all ML artifacts in

        Args:
            No arguments

        Returns:
            No return value
        """
        # Create the Amazon S3 bucket.
        self.mlops_artifacts_bucket = _s3.Bucket(
            self,
            "MlOpsArtifactsBucket",
            bucket_name=f"sagemaker-project-{self.sagemaker_project_id}",
            server_access_logs_prefix="access-logging",
            encryption=_s3.BucketEncryption.S3_MANAGED,
            block_public_access=_s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True)

    def create_codecommit_repository(
        self,
        construct_id: str,
        repository_tag: str,
        **kwargs) -> _codecommit.Repository:
        """Create an AWS CodeCommit repository

        Args:
            construct_id:       The construct ID visible on the CloudFormation console for this resource
            repository_tag:     Indicating what repository type, values `modelbuild` or `modeldeploy`

        Returns:
            repository:         The AWS CodeCommit repository
        """
        # Create AWS CodeCommit repository
        repository = _codecommit.Repository(
            self,
            construct_id,
            repository_name=f"sagemaker-{self.sagemaker_project_name}-{self.sagemaker_project_id}-{repository_tag}",
            description=f"{repository_tag} workflow infrastructure as code for the Project {self.sagemaker_project_name}",
            code=_codecommit.Code.from_zip_file(
                file_path=f"pipeline/assets/{repository_tag}.zip",
                branch="main"))

        return repository

    def _event_rule_description_mapping(
        self,
        rule_tag: str,
        **kwargs) -> str:
        """Helper function to map event rule type to a description.

        Args:
            rule_tag:           Indicating what rule type, values `build` or `code`

        Returns:
            output:             The description to output
        """
        output = "No description available"
        mapping = {
            "build": "Rule to trigger a deployment when ModelBuild CodeCommit repository is updated",
            "code": "Rule to trigger a deployment when CodeCommit is updated with a commit",
        }
        if rule_tag in mapping:
            output = mapping[rule_tag]
        return output

    def create_codecommit_event_rules(
        self,
        construct_id: str,
        rule_tag: str,
        resource: _codepipeline.Pipeline,
        **kwargs) -> _events.Rule:
        """Create specific Event rules to trigger AWS CodePipeline based on push to
            `main` branch in the corresponding AWS CodeCommit repository.

        Args:
            construct_id:       The construct ID visible on the CloudFormation console for this resource
            rule_tag:           Indicating what rule type, values `build` or `code`
            resource:           The AWS CodePipeline to trigger

        Returns:
            event_rule:         The event rule object that was created
        """
        # Define Event rule
        event_rule = _events.Rule(
            self,
            construct_id,
            rule_name=f"sagemaker-{self.sagemaker_project_name}-{self.sagemaker_project_id}-{rule_tag}",
            description=self._event_rule_description_mapping(rule_tag=rule_tag),
            event_pattern=_events.EventPattern(
                source=["aws.codecommit"],
                resources=[resource.pipeline_arn],
                detail={
                    "referenceType": ["branch"],
                    "referenceName": ["main"]
                },
                detail_type=["CodeCommit Repository State Change"]),
            enabled=True,)

        # Add target: here the AWS CodePipeline
        event_rule.add_target(
            target=_targets.CodePipeline(
                pipeline=resource,))

        return event_rule

    def create_sagemaker_event_rule(
        self,
        resource: _codepipeline.Pipeline,
        **kwargs):
        """Create specific Event rules to trigger AWS CodePipeline based updated
            SageMaker Model registry model package.

        Args:
            resource:           The AWS CodePipeline to trigger

        Returns:
            event_rule:         The event rule object that was created
        """
        # Define Event rule
        model_deploy_sagemaker_event_rule = _events.Rule(
            self,
            "ModelDeploySageMakerEventRule",
            description="Rule to trigger a deployment when SageMaker Model registry is updated with a new model package. For example, a new model package is registered with Registry",
            rule_name=f"sagemaker-{self.sagemaker_project_name}-{self.sagemaker_project_id}-model",
            event_pattern=_events.EventPattern(
                source=["aws.sagemaker"],
                detail={
                    "ModelPackageGroupName": [f"{self.sagemaker_project_name}-{self.sagemaker_project_id}"],
                },
                detail_type=["SageMaker Model Package State Change"]),
            enabled=True,)

        # Add target: here the AWS CodePipeline
        model_deploy_sagemaker_event_rule.add_target(
            target=_targets.CodePipeline(
                pipeline=resource,))

        return model_deploy_sagemaker_event_rule

    def create_modelbuild_pipeline(
        self,
        repository: _codecommit.Repository,
        **kwargs) -> _codepipeline.Pipeline:
        """Create an entire AWS CodePipeline with an incorporated AWS CodeBuild
            step. This pipeline will use `repository` as a source and execute this
            code in the AWS CodeBuild step. This pipeline represents the model building
            step.

        Args:
            repository:             The AWS CodeCommit repository that will be leveraged
                                    in the pipeline

        Returns:
            model_build_pipeline:   The AWS CDK CodePipeline object
        """
        # Set a generic source artifact
        source_artifact = _codepipeline.Artifact()

        # Define the AWS CodeBuild project
        sagemaker_modelbuild_pipeline = _codebuild.Project(
            self,
            "SageMakerModelPipelineBuildProject",
            project_name=f"sagemaker-{self.sagemaker_project_name}-{self.sagemaker_project_id}-modelbuild",
            description="Builds the model building workflow code repository, creates the SageMaker Pipeline and executes it",
            role=self.mlops_model_build_role,
            environment=_codebuild.BuildEnvironment(
                build_image=_codebuild.LinuxBuildImage.from_code_build_image_id(id="aws/codebuild/amazonlinux2-x86_64-standard:4.0"),
                compute_type=_codebuild.ComputeType.SMALL,
                privileged=True
            ),
            environment_variables={
                "SAGEMAKER_PROJECT_NAME": _codebuild.BuildEnvironmentVariable(value=self.sagemaker_project_name),
                "SAGEMAKER_PROJECT_ID": _codebuild.BuildEnvironmentVariable(value=self.sagemaker_project_id),
                "ARTIFACT_BUCKET": _codebuild.BuildEnvironmentVariable(value=self.mlops_artifacts_bucket.bucket_name),
                "SAGEMAKER_PIPELINE_NAME": _codebuild.BuildEnvironmentVariable(value=f"sagemaker-{self.sagemaker_project_name}"),
                "SAGEMAKER_PIPELINE_ROLE_ARN": _codebuild.BuildEnvironmentVariable(value=self.mlops_sagemaker_pipeline_role.role_arn),
                "AWS_REGION": _codebuild.BuildEnvironmentVariable(value=self.aws_region),
            },
            source=_codebuild.Source.code_commit(repository=repository),
            timeout=Duration.hours(2))

        # Defines the AWS CodePipeline
        model_build_pipeline = _codepipeline.Pipeline(
            self,
            "ModelBuildPipeline",
            pipeline_name=f"sagemaker-{self.sagemaker_project_name}-{self.sagemaker_project_id}-modelbuild",
            artifact_bucket=self.mlops_artifacts_bucket,)

        # Define actions that are executed in CodePipeline:
        # - `Source` leverages the AWS CodeCommit `repository` and copies that into AWS CodeBuild
        # - `Deploy` will run AWS CodeBuild leveragin the pulled `repository`
        source_action = _actions.CodeCommitSourceAction(
            action_name="Source",
            output=source_artifact,
            branch="main",
            repository=repository,
            code_build_clone_output=True)

        build_action = _actions.CodeBuildAction(
            action_name="Deploy",
            project=sagemaker_modelbuild_pipeline,
            input=source_artifact,
            outputs=[_codepipeline.Artifact()])

        # Add the stages defined above to the pipeline
        model_build_pipeline.add_stage(
            stage_name="Source",
            actions=[source_action])
        model_build_pipeline.add_stage(
            stage_name="Deploy",
            actions=[build_action])

        return model_build_pipeline

    def create_modeldeploy_pipeline(
        self,
        repository: _codecommit.Repository,
        **kwargs) -> _codepipeline.Pipeline:
        """Create an entire AWS CodePipeline with an incorporated AWS CodeBuild
            step. This pipeline will use `repository` as a source and execute this
            code in the AWS CodeBuild step. This pipeline represents the model deployment
            step.

        Args:
            repository:             The AWS CodeCommit repository that will be leveraged
                                    in the pipeline

        Returns:
            model_deploy_pipeline:   The AWS CDK CodePipeline object
        """
        # Set a generic source artifact
        source_artifact = _codepipeline.Artifact()

        # Define the AWS CodeBuild project
        model_deploy_build_project = _codebuild.Project(
            self,
            "ModelDeployBuildProject",
            description="Builds the Cfn template which defines the Endpoint with specified configuration",
            project_name=f"sagemaker-{self.sagemaker_project_name}-{self.sagemaker_project_id}-modeldeploy",
            role=self.mlops_model_deploy_role,
            environment=_codebuild.BuildEnvironment(
                build_image=_codebuild.LinuxBuildImage.from_code_build_image_id(id="aws/codebuild/amazonlinux2-x86_64-standard:4.0"),
                compute_type=_codebuild.ComputeType.SMALL,
                privileged=True
            ),
            environment_variables={
                "SAGEMAKER_PROJECT_NAME": _codebuild.BuildEnvironmentVariable(value=self.sagemaker_project_name),
                "SAGEMAKER_PROJECT_ID": _codebuild.BuildEnvironmentVariable(value=self.sagemaker_project_id),
                "ARTIFACT_BUCKET": _codebuild.BuildEnvironmentVariable(value=self.mlops_artifacts_bucket.bucket_name),
                "MODEL_EXECUTION_ROLE_ARN": _codebuild.BuildEnvironmentVariable(value=self.mlops_model_deploy_role.role_arn),
                "SOURCE_MODEL_PACKAGE_GROUP_NAME": _codebuild.BuildEnvironmentVariable(value=f"{self.sagemaker_project_name}-{self.sagemaker_project_id}"),
                "AWS_REGION": _codebuild.BuildEnvironmentVariable(value=self.aws_region),
                "EXPORT_TEMPLATE_NAME": _codebuild.BuildEnvironmentVariable(value="template-export.yml"),
                "EXPORT_TEMPLATE_STAGING_CONFIG": _codebuild.BuildEnvironmentVariable(value="staging-config-export.json"),
                "EXPORT_TEMPLATE_PROD_CONFIG": _codebuild.BuildEnvironmentVariable(value="prod-config-export.json"),
            },
            source=_codebuild.Source.code_commit(repository=repository),
            timeout=Duration.minutes(30),)

        # Defines the AWS CodePipeline
        model_deploy_pipeline = _codepipeline.Pipeline(
            self,
            "ModelDeployPipeline",
            pipeline_name=f"sagemaker-{self.sagemaker_project_name}-{self.sagemaker_project_id}-modeldeploy",
            artifact_bucket=self.mlops_artifacts_bucket,)

        # Define actions that are executed in CodePipeline:
        # - `Source` leverages the AWS CodeCommit `repository` and copies that into AWS CodeBuild
        # - `Deploy` will run AWS CodeBuild leveragin the pulled `repository`
        source_action = _actions.CodeCommitSourceAction(
            action_name="Source",
            output=source_artifact,
            branch="main",
            repository=repository,
            code_build_clone_output=True)

        build_action = _actions.CodeBuildAction(
            action_name="Deploy",
            project=model_deploy_build_project,
            input=source_artifact,
            outputs=[_codepipeline.Artifact()])

        # Add the stages defined above to the pipeline
        model_deploy_pipeline.add_stage(
            stage_name="Source",
            actions=[source_action])
        model_deploy_pipeline.add_stage(
            stage_name="Deploy",
            actions=[build_action])

        return model_deploy_pipeline

    def create_image_type_projects(
        self,
        construct_id: str,
        image_type: str,
        model_build_pipeline: _codepipeline.Pipeline,
        retraining_days: int,
        container_image_tag: str="latest",
    ) -> None:
        """For each possible encapsulated Docker image (processing, training and inference)
            this method creates an end-to-end AWS CodePipeline together with a AWS CodeCommit
            repository and an underlying AWS CodeBuild. Each repository is a separate "microservice"
            oriented automated pipeline that helps build your Docker containers and push them to
            Amazon ECR. These images are registered in Amazon SageMaker and then made available
            to your Amazon SageMaker Pipeline that is run in `modelbuild`

        Args:
            construct_id:           The construct ID visible on the CloudFormation console for this resource
            image_type:             The image type you want to create, options: `processing`, `training`
                                    and `inference`
            model_build_pipeline:   The `modelbuild` pipeline that will be triggered each time a new container is
                                    created and pushed to Amazon ECR
            retraining_days:        The number of days these containers will get re-build. This also indicates
                                    how often your `modelbuil`, i.e. your model re-training cycle, will be
            container_image_tag:    The Amazon ECR image tag that indicates a new container was released, default: `latest`

        Returns:
            No return
        """

        if f"{image_type}"=="training":
            image_build_role = self.mlops_training_image_build_role
        else:
            image_build_role = self.mlops_processing_image_build_role
        # Create Amazon SageMaker Image
        sagemaker_image = _sagemaker.CfnImage(
            self,
            f"SageMakerImage-{image_type}",
            image_name=f"sagemaker-{self.sagemaker_project_id}-{image_type}-imagebuild",
            image_role_arn=image_build_role.role_arn,
        )

        # Create Amazon ECR repository
        training_ecr_repo = _ecr.Repository(
            self,
            f"DestinationEcrRepository-{image_type}",
            repository_name=f"sagemaker-{self.sagemaker_project_id}-{image_type}-imagebuild",
            image_scan_on_push = True)

        # Create AWS CodeCommit repository
        image_build_repository = _codecommit.Repository(
            self,
            f"ImageBuildCodeCommitRepository-{image_type}",
            repository_name=f"sagemaker-{self.sagemaker_project_name}-{self.sagemaker_project_id}-{image_type}-imagebuild",
            description=f"SageMaker {image_type} image-building workflow infrastructure as code for the Project {self.sagemaker_project_name}",
            code=_codecommit.Code.from_zip_file(
                file_path=f"pipeline/assets/{image_type}-imagebuild.zip",
                branch="main"))

        # Set a generic source artifact
        source_artifact = _codepipeline.Artifact()

        # Define the AWS CodeBuild project
        image_build_project = _codebuild.Project(
            self,
            f"ImageBuildProject-{image_type}",
            project_name=f"sagemaker-{self.sagemaker_project_name}-{self.sagemaker_project_id}-{image_type}-imagebuild",
            description=f"sagemaker-{self.sagemaker_project_name}-{self.sagemaker_project_id}-{image_type}-imagebuild",
#           role=image_build_role,
            environment=_codebuild.BuildEnvironment(
                build_image=_codebuild.LinuxBuildImage.from_code_build_image_id(id="aws/codebuild/amazonlinux2-x86_64-standard:4.0"),
                compute_type=_codebuild.ComputeType.SMALL,
                privileged=True
            ),
            environment_variables={
                "IMAGE_REPO_NAME": _codebuild.BuildEnvironmentVariable(value=training_ecr_repo.repository_name),
                "IMAGE_TAG": _codebuild.BuildEnvironmentVariable(value=container_image_tag),
                "AWS_ACCOUNT_ID": _codebuild.BuildEnvironmentVariable(value=self.aws_account_id),
                "AWS_REGION": _codebuild.BuildEnvironmentVariable(value=self.aws_region),
            },
            source=_codebuild.Source.code_commit(repository=image_build_repository),
            timeout=Duration.hours(2),)

        #grant the default CodeBuild role with access to pull and push ecr images to ecr repo
        training_ecr_repo.grant_pull_push(image_build_project.role)
        # Defines the AWS CodePipeline
        image_build_pipeline = _codepipeline.Pipeline(
            self,
            f"ImageBuildPipeline-{image_type}",
            pipeline_name=f"sagemaker-{self.sagemaker_project_name}-{self.sagemaker_project_id}-{image_type}-imagebuild",
            artifact_bucket=self.mlops_artifacts_bucket)

        # Define actions that are executed in CodePipeline:
        # - `Source` leverages the AWS CodeCommit `repository` and copies that into AWS CodeBuild
        # - `Deploy` will run AWS CodeBuild leveragin the pulled `repository`
        source_action = _actions.CodeCommitSourceAction(
            action_name="Source",
            output=source_artifact,
            branch="main",
            repository=image_build_repository,
            code_build_clone_output=True)

        build_action = _actions.CodeBuildAction(
            action_name="Deploy",
            project=image_build_project,
            input=source_artifact,
            outputs=[_codepipeline.Artifact()])

        # Add the stages defined above to the pipeline
        image_build_pipeline.add_stage(
            stage_name="Source",
            actions=[source_action])
        image_build_pipeline.add_stage(
            stage_name="Deploy",
            actions=[build_action])

        # Create Event rule for image build when AWS CodeCommit is updated
        # on `main` branch
        image_build_codecommit_event_rule = _events.Rule(
            self,
            f"ImageBuildCodeCommitEventRule-{image_type}",
            rule_name=f"sagemaker-{self.sagemaker_project_id}-{image_type}",
            description="Rule to trigger an image build when CodeCommit repository is updated",
            event_pattern=_events.EventPattern(
                source=["aws.codecommit"],
                resources=[image_build_pipeline.pipeline_arn],
                detail={
                    "referenceType": ["branch"],
                    "referenceName": ["main"]
                },
                detail_type=["CodeCommit Repository State Change"]),
            enabled=True,)

        # Create Event rule for re-building the Docker image
        image_build_schedule_event_rule = _events.Rule(
            self,
            f"ImageBuildScheduleRule-{image_type}",
            rule_name=f"sagemaker-{self.sagemaker_project_id}-{image_type}-image-time",
            description=f"The rule to run the {image_type} pipeline on schedule",
            schedule=_events.Schedule.rate(Duration.days(retraining_days)),
            enabled=True,)

        # Create Event rule for modelbuild when a new version of the image is created.
        trigger_model_build_rule = _events.Rule(
            self,
            f"TriggerModelBuildRule-{image_type}",
            rule_name=f"sagemaker-{self.sagemaker_project_id}-{image_type}-image-version",
            description="Rule to run the model build pipeline when new versions of the image is created.",
            event_pattern=_events.EventPattern(
                source=["aws.sagemaker"],
                detail={
                    "ImageArn": [f"arn:aws:sagemaker:{self.aws_region}:{self.aws_account_id}:image/sagemaker-{self.sagemaker_project_id}-{image_type}-imagebuild"],
                    "ImageVersionStatus": ["CREATED"],
                },
                detail_type=["SageMaker Image Version State Change"]),
            enabled=True,)

        # Add targets to Event rules
        image_build_codecommit_event_rule.add_target(
            target=_targets.CodePipeline(
                pipeline=image_build_pipeline,))

        image_build_schedule_event_rule.add_target(
            target=_targets.CodePipeline(
                pipeline=image_build_pipeline,))

        trigger_model_build_rule.add_target(
            target=_targets.CodePipeline(
                pipeline=model_build_pipeline,))

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        sagemaker_project_name: str,
        sagemaker_project_id: str,
        enable_processing_image_build_pipeline: bool,
        enable_training_image_build_pipeline: bool,
        enable_inference_image_build_pipeline: bool,
        aws_account_id: int,
        aws_region: str="eu-central-1",
        container_image_tag: str="latest",
        **kwargs,
    ) -> None:
        """Initialize the class.

        Args:
            scope:                                      The AWS CDK app that is deployed
            construct_id:                               The construct ID visible on the CloudFormation console for this resource
            sagemaker_project_name:                     The name of the Amazon SageMaker project
            sagemaker_project_id:                       The unique Amazon SageMaker project ID
            enable_processing_image_build_pipeline:     Indicate whether you want a `processing` image build pipeline
            enable_training_image_build_pipeline:       Indicate whether you want a `training` image build pipeline
            enable_inference_image_build_pipeline:      Indicate whether you want a `inference` image build pipeline
            aws_account_id:                             The AWS account the solution gets deployed in
            aws_region:                                 The region this stack will be deployed to
            container_image_tag:                        The Amazon ECR image tag that indicates a new container was released, default: `latest`

        Returns:
            No return
        """
        super().__init__(scope, construct_id, **kwargs)
        # Set attributes
        self.sagemaker_project_name = sagemaker_project_name
        self.sagemaker_project_id = sagemaker_project_id
        self.aws_account_id = aws_account_id
        self.aws_region = aws_region

        # Create IAM role wiht IAM policy and set attribute
        self.create_iam_role()

        # Create MLOps S3 artifact bucket
        mlops_artifacts_bucket = self.create_s3_artifact_bucket()

        # Create build repository
        model_build_repository = self.create_codecommit_repository(
            construct_id="ModelBuildCodeCommitRepository",
            repository_tag="modelbuild")

        # Create deploy repository
        model_deploy_repository = self.create_codecommit_repository(
            construct_id="ModelDeployCodeCommitRepository",
            repository_tag="modeldeploy")

        # Create the modelbuild pipeline
        sagemaker_modelbuild_pipeline = self.create_modelbuild_pipeline(repository=model_build_repository)

        # Create the modeldeploy pipeline
        sagemaker_modeldeploy_pipeline = self.create_modeldeploy_pipeline(repository=model_deploy_repository)

        # Create event rule for model build
        model_build_codecommit_event_rule = self.create_codecommit_event_rules(
            construct_id="ModelBuildCodeCommitEventRule",
            rule_tag="build",
            resource=sagemaker_modelbuild_pipeline)

        # Create event rule for model deploy
        model_deploy_codecommit_event_rule = self.create_codecommit_event_rules(
            construct_id="ModelDeployCodeCommitEventRule",
            rule_tag="code",
            resource=sagemaker_modeldeploy_pipeline)

        # Create the rule that triggers model deployment through
        # Amazon SageMaker Registry
        model_deploy_sagemaker_event_rule = self.create_sagemaker_event_rule(resource=sagemaker_modeldeploy_pipeline)

        # Check which automated image building AWS CodePipelines, including
        # AWS CodeCommit and AWS CodeBuild should be created. Options are:
        # - 'processing'
        # - 'training'
        # - 'inference'
        if enable_processing_image_build_pipeline:
            processing_stack = self.create_image_type_projects(
                "ProcessingImageStack",
                image_type="processing",
                model_build_pipeline=sagemaker_modelbuild_pipeline,
                retraining_days=90,
                container_image_tag=container_image_tag)
        if enable_training_image_build_pipeline:
            training_stack = self.create_image_type_projects(
                "TrainingImageStack",
                image_type="training",
                model_build_pipeline=sagemaker_modelbuild_pipeline,
                retraining_days=30,
                container_image_tag=container_image_tag)
        if enable_inference_image_build_pipeline:
            inference_stack = self.create_image_type_projects(
                "InferenceImageStack",
                image_type="inference",
                model_build_pipeline=sagemaker_modelbuild_pipeline,
                retraining_days=120,
                container_image_tag=container_image_tag)
