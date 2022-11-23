#!/usr/bin/env bash

create_virtual_env () {
    # This function creates a virtual environment and activates it
    echo "Update Python and create virtual environment..."
    python3 -m ensurepip --upgrade
    python3 -m pip install --upgrade pip
    python3 -m pip install --upgrade virtualenv
    python3 -m venv .venv

    echo "Source virtual environment..."
    source .venv/bin/activate
}


install_cdk () {
    # This function installs CDK, the requirements file and bootstraps the environment
    echo "Install CDK and Python requirements..."
    npm install -g aws-cdk
    pip install -r requirements.txt

    cd ..
}

copy_inference_files () {
    # Copy the files mandatory for inference imagebuild
    echo "Copy Python files to inference imagebuild repo"
    cp mllib/digital_twin.py deployment/pipeline/assets/inference-imagebuild/
    cp mllib/quantile_regression.py deployment/pipeline/assets/inference-imagebuild/
    cp mllib/serve.py deployment/pipeline/assets/inference-imagebuild/
}

copy_processing_files () {
    # Copy the files mandatory for processing imagebuild
    echo "Copy Python files to processing imagebuild repo"
    cp mllib/digital_twin.py deployment/pipeline/assets/processing-imagebuild/
    cp mllib/quantile_regression.py deployment/pipeline/assets/processing-imagebuild/
}

copy_training_files () {
    # Copy the files mandatory for training imagebuild
    echo "Copy Python files to training imagebuild repo"
    cp mllib/digital_twin.py deployment/pipeline/assets/training-imagebuild/
    cp mllib/quantile_regression.py deployment/pipeline/assets/training-imagebuild/
    cp mllib/train.py deployment/pipeline/assets/training-imagebuild/
}

copy_lambda_deploy_files () {
    # Copy the files mandatory for lambda model deploy
    echo "Copy Python files to lambda model deploy repo"
    # digital_twin = Simulation Lambda
    cp mllib/digital_twin.py deployment/pipeline/assets/modeldeploy/lambda/digital_twin/
    cp mllib/quantile_regression.py deployment/pipeline/assets/modeldeploy/lambda/digital_twin/
}

copy_pipeline_files () {
    # Copy the files mandatory for model building
    echo "Copy Python files to model building repo"
    cp mllib/preprocess.py deployment/pipeline/assets/modelbuild/pipelines/modelbuildpipeline/
    cp mllib/simulate.py deployment/pipeline/assets/modelbuild/pipelines/modelbuildpipeline/
    cp mllib/digital_twin.py deployment/pipeline/assets/modelbuild/pipelines/modelbuildpipeline/
    cp mllib/lambda_handler.py deployment/pipeline/assets/modelbuild/pipelines/modelbuildpipeline/
}

cleanup_repository () {
    # Remove unwanted files from repository structure
    find . -type f -name ".DS_Store" -delete
    find . -type f -name ".ipynb_checkpoints" -delete
    find . -type f -name "__pycache__" -delete
    find . -type f -name "*.egg-info" -delete
}

zip_files() {
    # This function zips the files, moves them up one level and goes back 3 levels
    # Note: In order to land where you started submit 3 levels to cd into...
    echo "ZIP files in $1"
    cd $1
    zip -r $2 *
    mv $2 ../
    cd ../../../
}

cleanup_files () {
    # Remove all copied files to make repo clean again
    rm pipeline/assets/inference-imagebuild/digital_twin.py
    rm pipeline/assets/inference-imagebuild/quantile_regression.py
    rm pipeline/assets/inference-imagebuild/serve.py
    rm pipeline/assets/processing-imagebuild/digital_twin.py
    rm pipeline/assets/processing-imagebuild/quantile_regression.py
    rm pipeline/assets/training-imagebuild/digital_twin.py
    rm pipeline/assets/training-imagebuild/quantile_regression.py
    rm pipeline/assets/training-imagebuild/train.py
    rm pipeline/assets/modeldeploy/lambda/digital_twin/digital_twin.py
    rm pipeline/assets/modeldeploy/lambda/digital_twin/quantile_regression.py
    rm pipeline/assets/modelbuild/pipelines/modelbuildpipeline/preprocess.py
    rm pipeline/assets/modelbuild/pipelines/modelbuildpipeline/simulate.py
    rm pipeline/assets/modelbuild/pipelines/modelbuildpipeline/lambda_handler.py

    rm pipeline/assets/inference-imagebuild.zip
    rm pipeline/assets/modelbuild.zip
    rm pipeline/assets/processing-imagebuild.zip
    rm pipeline/assets/training-imagebuild.zip
    rm pipeline/assets/modeldeploy.zip
}

create_virtual_env
install_cdk
copy_inference_files
copy_processing_files
copy_training_files
copy_lambda_deploy_files
copy_pipeline_files
cleanup_repository

cd deployment

zip_files "pipeline/assets/inference-imagebuild" "inference-imagebuild.zip"
zip_files "pipeline/assets/modelbuild" "modelbuild.zip"
zip_files "pipeline/assets/processing-imagebuild" "processing-imagebuild.zip"
zip_files "pipeline/assets/training-imagebuild" "training-imagebuild.zip"
zip_files "pipeline/assets/modeldeploy" "modeldeploy.zip"

echo "Bootstrap account"
cdk bootstrap

echo "Deploy CDK Stack..."
cdk deploy --require-approval never

cleanup_files
