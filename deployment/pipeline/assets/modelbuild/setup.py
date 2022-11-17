import os

import setuptools
from pipelines import __version__ as version

with open("README.md", "r") as f:
    readme = f.read()


required_packages = [
    "pip==22.1.2",
    "boto3==1.24.22",
    "shap==0.41.0",
    "imbalanced-learn==0.9.1",
    "werkzeug==2.0.3",
    "pandas==1.2.4",
    "numpy==1.22.0",
    "xgboost==1.5.2",
    "scikit-learn==1.1.1",
    "sagemaker",
    "pre-commit",
    "pytest==6.2.5",
    "optuna==2.10.1",
    "scikeras==0.8.0",
    "tensorflow-cpu==2.9.1",
    "keras==2.9.0",
]
extras = {
    "test": [
        "black",
        "coverage",
        "flake8",
        "mock",
        "pydocstyle",
        "pytest",
        "pytest-cov",
        "sagemaker",
        "tox",
    ]
}
setuptools.setup(
    name=version.__title__,
    description=version._description__,
    version=version.__version__,
    author=version.__author__,
    author_email=version.__author_email__,
    long_description=readme,
    long_description_content_type="text/markdown",
    url=version.__url__,
    license=version.__license__,
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=required_packages,
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "get-pipeline-definition=pipelines.get_pipeline_definition:main",
            "run-pipeline=pipelines.run_pipeline:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
