#!/usr/bin/env python3

import aws_cdk as cdk
import cdk_nag as cdknag
from DigitalTwin.DigitalTwin_stack import DigitalTwinStack

app = cdk.App()

DigitalTwinStack(
    app,
    "DigitalTwinStack")
cdk.Aspects.of(app).add(cdknag.AwsSolutionsChecks())
app.synth()
