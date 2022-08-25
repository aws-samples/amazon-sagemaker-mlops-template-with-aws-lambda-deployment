#!/usr/bin/env python3

import os
import random
import string

import aws_cdk as _cdk
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

SageMakerPipelineSourceCodeStack(
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

app.synth()
