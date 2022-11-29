from aws_cdk import Duration, Stack
from aws_cdk import aws_iam as _iam
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_lambda_event_sources as _event_sources
from aws_cdk import aws_s3 as _s3
from aws_cdk import Size
from constructs import Construct

class DigitalTwinStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Defines an AWS Lambda resource
        _lambda.DockerImageFunction(
            self, 'DigitalTwin',
            code=_lambda.DockerImageCode.from_image_asset("lambda/digital_twin"),
            memory_size=1024,
            ephemeral_storage_size=Size.mebibytes(1024),
            timeout=Duration.seconds(120),
        )