ARG ECR_ENDPOINT
FROM $ECR_ENDPOINT/sagemaker-xgboost:1.5-1-cpu-py3

COPY requirements.txt requirements.txt
COPY digital_twin.py /opt/ml/code/digital_twin.py
COPY quantile_regression.py /opt/ml/code/quantile_regression.py
COPY train.py /opt/ml/code/train.py

RUN python3 -m pip install -r requirements.txt

ENV SAGEMAKER_PROGRAM train.py
