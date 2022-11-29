import setuptools

setuptools.setup(
    name="SageMakerPipelineSourceCode",
    version="0.0.1",
    description="This is the stack that deploys the core code for the SM pipeline.",
    long_description_content_type="text/markdown",
    author="Michael Wallner",
    package_dir={"": "SageMakerPipelineSourceCode"},
    packages=setuptools.find_packages(where="SageMakerPipelineSourceCode"),
    install_requires=[
        "aws-cdk-lib==2.11.0",
        "constructs>=10.0.0,<11.0.0",
        "pytest==6.2.5",
        "boto3",
    ],
    python_requires=">=3.6",
)
