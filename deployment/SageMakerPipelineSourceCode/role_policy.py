from aws_cdk import aws_iam as _iam

role_policy_ecr_image_build = [
    # _iam.PolicyStatement(
    #     actions=[
    #         "iam:GetRole",
    #         "iam:GetRolePolicy",
    #         "iam:CreateRole",
    #         "iam:DetachRolePolicy",
    #         "iam:AttachRolePolicy",
    #         "iam:DeleteRole",
    #         "iam:PutRolePolicy",
    #         "iam:DeleteRolePolicy",
    #         "iam:PassRole",
    #         "iam:CreatePolicy",
    #     ],
    #     resources=[
    #         "arn:aws:iam::*:role/*",
    #         "arn:aws:iam::*:policy/*",
    #     ],
    # ),
    _iam.PolicyStatement(
        actions=[
            "ecr:CompleteLayerUpload",
            "ecr:UploadLayerPart",
            "ecr:InitiateLayerUpload",
            "ecr:BatchCheckLayerAvailability",
            "ecr:PutImage",

        ],
        resources=[
            "arn:aws:ecr:*:*:repository/sagemaker-*",
            "arn:aws:ecr:*:*:repository/cdk-*",
        ],
    ),
    _iam.PolicyStatement(
        actions=[
            "ecr:BatchGetImage",
            "ecr:Describe*",
            "ecr:GetDownloadUrlForLayer",
        ],
        resources=[
            "arn:aws:ecr:*:*:repository/sagemaker-*",
            "arn:aws:ecr:*:*:repository/cdk-*",
        ],
    ),
    _iam.PolicyStatement(
        actions=[
            "ecr:GetAuthorizationToken",
        ],
        resources=[
            "*",
        ],
    ),
    _iam.PolicyStatement(
        actions=[
            "sagemaker:CreateImageVersion",
        ],
        resources=[
            "arn:aws:sagemaker:*:*:image/sagemaker-*-imagebuild",
        ],
    ),
]

role_policy_model_build = [
    # _iam.PolicyStatement(
    #     actions=[
    #         "iam:GetRole",
    #         "iam:GetRolePolicy",
    #         "iam:CreateRole",
    #         "iam:DetachRolePolicy",
    #         "iam:AttachRolePolicy",
    #         "iam:DeleteRole",
    #         "iam:PutRolePolicy",
    #         "iam:DeleteRolePolicy",
    #         "iam:PassRole",
    #         "iam:CreatePolicy",
    #     ],
    #     resources=[
    #         "arn:aws:iam::*:role/lambda-update-manifest-role",
    #         "arn:aws:iam::*:role/EnergyOptimization-SageMakerMLOpsSagemakerPipeline*",
    #     ],
    # ),
    _iam.PolicyStatement(
        actions=[
            "sagemaker:DescribeImageVersion",
            "sagemaker:DescribePipeline",
            "sagemaker:AddTags",
            "sagemaker:ListTags",
            "sagemaker:CreatePipeline",
            "sagemaker:StartPipelineExecution",
            "sagemaker:DescribePipelineExecution",
            "sagemaker:UpdatePipeline"
        ],
        resources=[
            "arn:aws:sagemaker:*:*:pipeline/enopt-project-cdk-*",
            "arn:aws:sagemaker:*:*:image/sagemaker-*-imagebuild*",
            "arn:aws:sagemaker:*:*:image-version/*/*"
        ],
    ),
    _iam.PolicyStatement(
        actions=[
            "lambda:CreateFunction",
            "lambda:InvokeFunction",
            "lambda:UpdateFunctionCode",
        ],
        resources=[
            "arn:aws:lambda:*:*:function:lambda-update-manifest",
        ],
    ),
]
role_policy_sagemaker_pipeline_execution = [
    _iam.PolicyStatement(
        actions=[
                "iam:PassRole"
        ],
        resources=[
            # "arn:aws:iam::*:role/lambda-update-manifest-role",
            "arn:aws:iam::*:role/EnergyOptimization-SageMakerMLOpsSagemakerPipeline*",
        ],
    ),
    # _iam.PolicyStatement(
    #     actions=[
    #             "lambda:CreateFunction",
    #             "lambda:DeleteFunction",
    #             "lambda:GetFunction",
    #             "lambda:InvokeFunction",
    #             "lambda:UpdateFunctionCode"
    #     ],
    #     resources=[
    #         "arn:aws:lambda:*:*:function:lambda-update-manifest",
    #     ]
    # ),
     _iam.PolicyStatement(
        actions=[
            "ecr:GetAuthorizationToken",
        ],
        resources=[
            "*",
        ],
    ),   
    _iam.PolicyStatement(
        actions=[
            "sagemaker:CreateProcessingJob",
            "sagemaker:StopProcessingJob",
            "sagemaker:DescribeProcessingJob",
            "sagemaker:CreateTrainingJob",
            "sagemaker:StopTrainingJob",
            "sagemaker:DescribeTrainingJob",
            "sagemaker:CreateArtifact",
            "sagemaker:AddTags",
            "sagemaker:DescribeModelPackage",
            "sagemaker:CreateModelPackage",
            "sagemaker:DescribeModelPackageGroup",   
            "sagemaker:CreateModelPackageGroup",
            "sagemaker:DescribeImageVersion",
        ],
        resources=[
            "arn:aws:sagemaker:*:*:processing-job/pipelines-*",
            "arn:aws:sagemaker:*:*:training-job/pipelines-*",
            "arn:aws:sagemaker:*:*:model-package/*",
            "arn:aws:sagemaker:*:*:model-package-group/*",    
            "arn:aws:sagemaker:*:*:image-version/*/*",
            "arn:aws:sagemaker:*:*:image/sagemaker-*-imagebuild*"        
        ],
    ),
    _iam.PolicyStatement(
        actions=[
            "s3:GetObject",
            "s3:ListBucket",
            "s3:PutObject",
            "s3:DeleteObject",
            "s3:AbortMultipartUpload"
        ],
        resources=[
            "arn:aws:s3:::*sagemaker*",
        ],
    ),
    _iam.PolicyStatement(
        actions=[
            "ecr:BatchGetImage",
            "ecr:Describe*",
            "ecr:GetDownloadUrlForLayer",
        ],
        resources=[
            "arn:aws:ecr:*:*:repository/sagemaker-*",
            "arn:aws:ecr:*:*:repository/cdk-*",
        ],
    ),
    _iam.PolicyStatement(
        actions=[
            "logs:CreateLogGroup",
            "logs:CreateLogStream",
            "logs:PutLogEvents"
        ],
        resources=[
            "arn:aws:logs:*:*:log-group:*",
            "arn:aws:logs:*:*:log-group:*:log-stream:*",
        ],
    ),

]


role_policy_model_deploy = [
    _iam.PolicyStatement(
        actions=["cloudformation:DescribeStacks"],
        resources=[
            "arn:aws:cloudformation:*:*:stack/CDKToolkit/*",
        ],
    ),
    # _iam.PolicyStatement(
    #     actions=[
    #         "iam:GetRole",
    #         "iam:CreateRole",
    #         "iam:DeleteRole",
    #         "iam:AttachRolePolicy",
    #         "iam:PutRolePolicy",
    #         "iam:CreatePolicy",
    #         "iam:DetachRolePolicy",
    #         "iam:DeleteRolePolicy",
    #         "iam:GetRolePolicy",
    #         "iam:PassRole",
    #         "sts:AssumeRole",
    #         "ssm:GetParameter",
    #     ],
    #     resources=[
    #         "arn:aws:iam::*:role/*deploy-role*",
    #         "arn:aws:iam::*:role/*file-publishing*",
    #         "arn:aws:iam::*:role/*image-publishing-role*",
    #         "arn:aws:ssm:*:*:parameter/cdk-bootstrap*",
    #     ],
    # ),
    _iam.PolicyStatement(
        actions=[
            "sagemaker:ListTags",
            "sagemaker:DescribeModelPackage",
            "sagemaker:ListModelPackages",
            "sagemaker:DescribeImageVersion",
            "sagemaker:AddTags",
        ],
        resources=[
            "arn:aws:sagemaker:*:*:image-version/sagemaker-*-imagebuild/*",
            "arn:aws:sagemaker:*:*:model-package/enopt-project*",
        ],
    ),
    _iam.PolicyStatement(
        actions=[
            "lambda:CreateFunction",
            "lambda:InvokeFunction",
            "lambda:UpdateFunctionCode",
        ],
        resources=[
            "arn:aws:lambda:*:*:function:lambda-update-manifest",
        ],
    ),
    _iam.PolicyStatement(
        actions=[
            "ecr:DescribeImages",
            "ecr:DescribeRepositories",
            "ecr:CompleteLayerUpload",
            "ecr:UploadLayerPart",
            "ecr:InitiateLayerUpload",
            "ecr:BatchCheckLayerAvailability",
            "ecr:PutImage"
        ],
        resources=[
            "arn:aws:ecr:*:*:repository/sagemaker-*",
            "arn:aws:ecr:*:*:repository/cdk-*",
        ],
    ), 
    _iam.PolicyStatement(
        actions=["ecr:GetAuthorizationToken"],
        resources=["*"],
    ),
]
