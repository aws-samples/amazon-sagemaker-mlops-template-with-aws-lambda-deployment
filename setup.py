import setuptools

setuptools.setup(
    name="MLLib",
    version="0.0.1",
    description="This repository serves as the one-stop-shop for the mlops template.",
    long_description_content_type="text/markdown",
    author="Michael Wallner",
    install_requires=[
        "pre-commit",
        "scikit-learn",
        "imblearn",
        "pandas",
        "numpy",
        "pytest==6.2.5",
        "boto3",
    ],
    python_requires=">=3.6",
)
