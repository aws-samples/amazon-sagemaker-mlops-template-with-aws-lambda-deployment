#!/usr/bin/env python3

import aws_cdk as cdk
import cdk_nag as cdknag
from DigitalTwin.DigitalTwin_stack import DigitalTwinStack

app = cdk.App()

stack = DigitalTwinStack(
    app,
    "DigitalTwinStack")
cdk.Aspects.of(app).add(cdknag.AwsSolutionsChecks())
cdknag.NagSuppressions.add_stack_suppressions(
    stack,
    [
        cdknag.NagPackSuppression(
            id="AwsSolutions-IAM4",
            reason="Use AWS managed poclicies AWSLambdaBasicExecutionRole",
        )
    ],
)
app.synth()
