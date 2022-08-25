#!/usr/bin/env python3

import aws_cdk as cdk

from DigitalTwin.DigitalTwin_stack import DigitalTwinStack

app = cdk.App()

DigitalTwinStack(
    app,
    "DigitalTwinStack")

app.synth()
