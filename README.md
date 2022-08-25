# Compressor Optimization

## Setup

Once you cloned the repository create a virtual environment using

```
python3 -m venv .venv
```

Activate the environment:

```
source .venv/bin/activate
```

Next install the required libraries using:

```
pip install -r requirements.txt
```

Finally, initialize pre-commit using

```
pre-commit install
```

You can use the `experiments` folder to start your journey with Jupyter notebooks and the regular data science cycle. Once you have developed your code, model, etc. you can integrate it into the files located in `mllib`. These files will be copied into the main application and leveraged by the entire automation mechanism.

The most important files in `mllib` are:

* **preprocess.py**: This is the entry point for the Amazon SageMaker processing job and leverages the Docker container built and pushed to Amazon ECR.
* **train.py**: This is the entry point for the Amazon SageMaker training job and leverages the Docker container built and pushed to Amazon ECR.
* **serve.py**: This is the entry point for the Amazon SageMaker endpoint and leverages the Docker container built and pushed to Amazon ECR. (Note: This project deploys the models in an AWS Lambda function, i.e. this file won't be used but can be if hosting is done in Amazon SageMaker)

In order to deploy your solution navigate into `deployment` folder and run

```
bash deploy.sh
```

you'll find a separate README that will give you guidance.