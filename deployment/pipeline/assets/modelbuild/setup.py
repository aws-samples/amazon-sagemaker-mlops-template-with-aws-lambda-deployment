import os

import setuptools

about = {}
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "pipelines", "__version__.py")) as f:
    exec(f.read(), about)


with open("README.md", "r") as f:
    readme = f.read()


required_packages = [
    "pip==22.1.2",
    "boto3==1.24.22",
    "shap==0.41.0",
    "imbalanced-learn==0.9.1",
    "werkzeug==2.0.3",
    "pandas==1.2.4",
    "numpy==1.21.0",
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
    name=about["__title__"],
    description=about["__description__"],
    version=about["__version__"],
    author=about["__author__"],
    author_email=["__author_email__"],
    long_description=readme,
    long_description_content_type="text/markdown",
    url=about["__url__"],
    license=about["__license__"],
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
