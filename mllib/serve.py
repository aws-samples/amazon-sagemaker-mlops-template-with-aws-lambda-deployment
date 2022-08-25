from __future__ import print_function

import argparse
import logging
import os
from io import StringIO

import joblib
import numpy as np
import pandas as pd
from digital_twin import DigitalTwin
from quantile_regression import QuantileRegression

features = [
    "flow",
    "pressure",
]
target = "power"
config_col = "config"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def input_fn(input_data, content_type="text/csv"):
    """Parse input data payload.

    Args:
        input_data (pandas.core.frame.DataFrame): A pandas.core.frame.DataFrame.
        content_type (str): A string expected to be 'text/csv'.

    Returns:
        df: pandas.core.frame.DataFrame
    """
    try:
        if "text/csv" in content_type:
            df = pd.read_csv(StringIO(input_data), header=None)
            return df
        elif "application/json" in content_type:
            df = pd.read_json(StringIO(input_data.decode("utf-8")))
            return df
        else:
            df = pd.read_csv(StringIO(input_data.decode("utf-8")), header=None)
            return df
    except ValueError as e:
        raise logger.error(f"ValueError {e}")


def output_fn(prediction, accept="text/csv"):
    """Format prediction output.

    Args:
        prediction (pandas.core.frame.DataFrame): A DataFrame with predictions.
        accept (str): A string expected to be 'text/csv'.

    Returns:
        df: str (in CSV format)
    """
    return prediction.to_csv(index=False)


def predict_fn(input_data, model):
    """Preprocess input data.

    Args:
        input_data (pandas.core.frame.DataFrame): A pandas.core.frame.DataFrame.
        model: A regression model

    Returns:
        output: pandas.core.frame.DataFrame
    """
    # Read your model and config file
    output = model.predict_efficiency(df=input_data, algorithm="keras_regression")
    return output


def model_fn(model_dir):
    """Deserialize fitted model.

    This simple function takes the path of the model, loads it,
    deserializes it and returns it for prediction.

    Args:
        model_dir (str): A string that indicates where the model is located.

    Returns:
        model:
    """
    # Load the model and deserialize
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model
