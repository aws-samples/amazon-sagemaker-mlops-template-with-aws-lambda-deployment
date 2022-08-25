import argparse
import json
import logging
import os

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")
s3 = boto3.client("s3")
paginator = sm_client.get_paginator('list_training_jobs')

def get_approved_package(model_package_group_name):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group_name: The model package group name.

    Returns:
        The SageMaker Model Package Creation Time.
    """
    try:
        # Get the latest approved model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            logger.debug(
                "Getting more packages for token: {}".format(response["NextToken"])
            )
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
            logger.error(error_message)
            raise Exception(error_message)

        desc = sm_client.describe_model_package(
            ModelPackageName=approved_packages[0]["ModelPackageArn"]
        )
        print("Initial ARN: ", approved_packages[0]["ModelPackageArn"])
        model = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
        dttm = desc["LastModifiedTime"]
        for package in approved_packages:
            desc = sm_client.describe_model_package(
                ModelPackageName=package["ModelPackageArn"]
            )
            if desc["LastModifiedTime"] > dttm:
                print("Updated ARN: ", approved_packages[0]["ModelPackageArn"])
                model = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

        return model
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)

def download_model_tar_from_s3(model_package_group_name):
    s3_url = get_approved_package(model_package_group_name=model_package_group_name)
    with open('model.tar.gz', 'wb') as f:
        s3.download_fileobj(s3_url.split("/")[2], "/".join(s3_url.split("/")[3:]), f)
    return "Success"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper()
    )
    parser.add_argument("--model-package-group-name", type=str, required=True)
    args, _ = parser.parse_known_args()

    # Configure logging to output the line number and message
    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=args.log_level)
    
    status = download_model_tar_from_s3(model_package_group_name=args.model_package_group_name)
    logging.info(f"Status = {status}")
    
    
