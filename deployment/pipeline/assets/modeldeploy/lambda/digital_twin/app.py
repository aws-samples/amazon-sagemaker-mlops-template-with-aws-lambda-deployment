"""This is the main app that runs in your AWS Lambda function."""

# Import libraries
import tempfile
import logging
import joblib
import base64
import json
import os

import boto3
import botocore
import io

import numpy as np
import pandas as pd

import optuna
from digital_twin import DigitalTwin
from quantile_regression import QuantileRegression

features = [
    "flow",
    "pressure",
]
target = "power"
config_col = "config"

twin = joblib.load("model.joblib")


def lambda_handler(event, context):
    """Main entry point for the AWS Lambda function.

    :event: Must contain 'body' and 'key' where
        'body' = the observed number from your machine
        'key' = the model location in your S3 bucket
    :context: The AWS Lambda context sent to the function.

    :return: The dict with status code and output_body

    """
    boundary = {
        "c1c2": [180, 390],
        "c1c2c4": [180, 390],
        "c1c4": [180, 390],
        "c1c3c4": [180, 460],
        "c1c2c3c4": [180, 460],
        "c1c3": [180, 390],
        "c1c2c3": [180, 460],
        "c3c4": [180, 390],
        "c2c3c4": [180, 460],
        "c2c4": [180, 390],
        "c2c3": [180, 390],
    }

    clf = [
        "c1c2",
        "c1c2c4",
        "c1c4",
        "c1c3c4",
        "c1c2c3c4",
        "c1c3",
        "c1c2c3",
        "c3c4",
        "c2c3c4",
        "c2c4",
        "c2c3",
    ]
    flow = json.loads(event["flow"])
    press = json.loads(event["pressure"])
    n = int(json.loads(event["simulations"]))
    no_of_trials = int(json.loads(event["no_of_trials"]))
    train_error_weight = float(json.loads(event["train_error_weight"]))

    results = twin.run_monte_carlo(
        clf=clf,
        flow=flow,
        press=press,
        n=n,
        no_of_trials=no_of_trials,
        boundary=boundary,
        train_error_weight=train_error_weight,
        algorithm="keras_regression"
    )

    # Return your response
    results = [{list(x.keys())[0]: float(x[list(x.keys())[0]]/n)} for x in results]
    
    logging.info("Return...")
    return {
        "statusCode": 200,
        "body": json.dumps(results),
    }
