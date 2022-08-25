import aws_cdk as core
import aws_cdk.assertions as assertions
from SageMakerPipelineSourceCode.SageMakerPipelineSourceCode_stack import SageMakerPipelineSourceCodeStack
    
def test_code_commit_repos_created():
    app = core.App()
    stack = SageMakerPipelineSourceCodeStack(app,
                                             "SageMakerPipelineSourceCode",
                                             sagemaker_project_name="predima-compressor-temp",
                                             sagemaker_project_id="p-da34uwob29zf",
                                             enable_processing_image_build_pipeline=True,
                                             enable_training_image_build_pipeline=True,
                                             enable_inference_image_build_pipeline=False,
                                             aws_account_id=42,
                                             aws_region="eu-central-1",
                                             container_image_tag="latest")
    template = assertions.Template.from_stack(stack)

    template.resource_count_is("AWS::CodeCommit::Repository", 4)
    
    
def test_code_pipelines_created():
    app = core.App()
    stack = SageMakerPipelineSourceCodeStack(app,
                                             "SageMakerPipelineSourceCode",
                                             sagemaker_project_name="predima-compressor-temp",
                                             sagemaker_project_id="p-da34uwob29zf",
                                             enable_processing_image_build_pipeline=True,
                                             enable_training_image_build_pipeline=True,
                                             enable_inference_image_build_pipeline=False,
                                             aws_account_id=42,
                                             aws_region="eu-central-1",
                                             container_image_tag="latest")
    template = assertions.Template.from_stack(stack)

    template.resource_count_is("AWS::CodePipeline::Pipeline", 4)

def test_code_build_projects_created():
    app = core.App()
    stack = SageMakerPipelineSourceCodeStack(app,
                                             "SageMakerPipelineSourceCode",
                                             sagemaker_project_name="predima-compressor-temp",
                                             sagemaker_project_id="p-da34uwob29zf",
                                             enable_processing_image_build_pipeline=True,
                                             enable_training_image_build_pipeline=True,
                                             enable_inference_image_build_pipeline=False,
                                             aws_account_id=42,
                                             aws_region="eu-central-1",
                                             container_image_tag="latest")
    template = assertions.Template.from_stack(stack)

    template.resource_count_is("AWS::CodeBuild::Project", 4)

def test_ecr_repositories_created():
    app = core.App()
    stack = SageMakerPipelineSourceCodeStack(app,
                                             "SageMakerPipelineSourceCode",
                                             sagemaker_project_name="predima-compressor-temp",
                                             sagemaker_project_id="p-da34uwob29zf",
                                             enable_processing_image_build_pipeline=True,
                                             enable_training_image_build_pipeline=True,
                                             enable_inference_image_build_pipeline=False,
                                             aws_account_id=42,
                                             aws_region="eu-central-1",
                                             container_image_tag="latest")
    template = assertions.Template.from_stack(stack)

    template.resource_count_is("AWS::ECR::Repository", 2)


def test_sagemaker_images_created():
    app = core.App()
    stack = SageMakerPipelineSourceCodeStack(app,
                                             "SageMakerPipelineSourceCode",
                                             sagemaker_project_name="predima-compressor-temp",
                                             sagemaker_project_id="p-da34uwob29zf",
                                             enable_processing_image_build_pipeline=True,
                                             enable_training_image_build_pipeline=True,
                                             enable_inference_image_build_pipeline=False,
                                             aws_account_id=42,
                                             aws_region="eu-central-1",
                                             container_image_tag="latest")
    template = assertions.Template.from_stack(stack)

    template.resource_count_is("AWS::SageMaker::Image", 2)

def test_s3_bucket_created():
    app = core.App()
    stack = SageMakerPipelineSourceCodeStack(app,
                                             "SageMakerPipelineSourceCode",
                                             sagemaker_project_name="predima-compressor-temp",
                                             sagemaker_project_id="p-da34uwob29zf",
                                             enable_processing_image_build_pipeline=True,
                                             enable_training_image_build_pipeline=True,
                                             enable_inference_image_build_pipeline=False,
                                             aws_account_id=42,
                                             aws_region="eu-central-1",
                                             container_image_tag="latest")
    template = assertions.Template.from_stack(stack)

    template.resource_count_is("AWS::S3::Bucket", 1)

def test_event_rules_created():
    app = core.App()
    stack = SageMakerPipelineSourceCodeStack(app,
                                             "SageMakerPipelineSourceCode",
                                             sagemaker_project_name="predima-compressor-temp",
                                             sagemaker_project_id="p-da34uwob29zf",
                                             enable_processing_image_build_pipeline=True,
                                             enable_training_image_build_pipeline=True,
                                             enable_inference_image_build_pipeline=False,
                                             aws_account_id=42,
                                             aws_region="eu-central-1",
                                             container_image_tag="latest")
    template = assertions.Template.from_stack(stack)

    template.resource_count_is("AWS::Events::Rule", 13)

def test_iam_policies_created():
    app = core.App()
    stack = SageMakerPipelineSourceCodeStack(app,
                                             "SageMakerPipelineSourceCode",
                                             sagemaker_project_name="predima-compressor-temp",
                                             sagemaker_project_id="p-da34uwob29zf",
                                             enable_processing_image_build_pipeline=True,
                                             enable_training_image_build_pipeline=True,
                                             enable_inference_image_build_pipeline=False,
                                             aws_account_id=42,
                                             aws_region="eu-central-1",
                                             container_image_tag="latest")
    template = assertions.Template.from_stack(stack)

    template.resource_count_is("AWS::IAM::Policy", 15)


def test_iam_roles_created():
    app = core.App()
    stack = SageMakerPipelineSourceCodeStack(app,
                                             "SageMakerPipelineSourceCode",
                                             sagemaker_project_name="predima-compressor-temp",
                                             sagemaker_project_id="p-da34uwob29zf",
                                             enable_processing_image_build_pipeline=True,
                                             enable_training_image_build_pipeline=True,
                                             enable_inference_image_build_pipeline=False,
                                             aws_account_id=42,
                                             aws_region="eu-central-1",
                                             container_image_tag="latest")
    template = assertions.Template.from_stack(stack)

    template.resource_count_is("AWS::IAM::Role", 15)