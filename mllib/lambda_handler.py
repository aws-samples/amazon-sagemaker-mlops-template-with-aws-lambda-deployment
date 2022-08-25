"""This Lambda function updates codebuild with the latest model artifact
and then triggers the codepipeline.
"""

import json

import boto3

s3 = boto3.client("s3")


def put_manifest(uri, bucket, key):
    """Get the codebuild project name based on key

    Args:
        key: the key searched for in your codebuild projects

    Returns:
        project name or empty string
    """
    manifest = {
        "fileLocations": [
            {
                "URIs": [
                    uri
                ]
            }
        ],
        "globalUploadSettings": {
            "format": "CSV",
            "delimiter": ",",
            "textqualifier": "'",
            "containsHeader": "true"
        }
    }

    s3.put_object(
        Body=json.dumps(manifest),
        Bucket=bucket,
        Key=key)


def lambda_handler(event, context):
    """Your Lambda entry function

    Args:
        event: the event sent to the lambda
        context: the context passed into lambda

    Returns:
        dictionary with status code
    """
    training = event["TRAINING"]
    simulation = event["SIMULATION"]
    bucket = event["BUCKET"]

    put_manifest(
        uri=f"{training}/dataset.csv",
        bucket=bucket,
        key="quicksight/resampled_data.json")

    put_manifest(
        uri=f"{simulation}/simulation.csv",
        bucket=bucket,
        key="quicksight/simulation.json")
    return {
        "statusCode": 200,
        "body": json.dumps("Manifest files in place!"),
    }
