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

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


features = [
    "flow",
    "pressure",
]
target = "power"
config_col = "config"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Hyperparameters are described here
    # In this simple example only 4 Hyperparamters are permitted
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--dual_model", type=bool, default=False)
    parser.add_argument("--remove_outliers", type=bool, default=False)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--optimizer__learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2000)

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    train_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
    if len(train_files) == 0:
        raise ValueError(
            (
                "There are no files in {}.\n"
                + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                + "the data specification in S3 was incorrectly specified or the role specified\n"
                + "does not have permission to access the data."
            ).format(args.train, "train")
        )

    # Read DataFrames into array and concatenate them into one DF
    train_data = [pd.read_csv(file) for file in train_files]
    train_data = pd.concat(train_data)

    configurations = train_data[config_col].unique().tolist()
    # Set train and validation
    twin = DigitalTwin(
        configurations=configurations,
        features=features,
        target=target,
        config_col=config_col,
        quantile=args.quantile,
    )

    if args.dual_model:
    
        twin = twin.train(
            df=train_data,
            test_size=args.test_size,
            random_state=args.random_state,
            degree=2,
            verbose=args.verbose,
            remove_outliers=args.remove_outliers,
            algorithm="quantile_regression",
        )

        for configs in configurations:
            print(f"Train_{configs}_quantile_mae={twin.train_mae_errors[configs]};")
            print(f"Validation_{configs}_quantile_mae={twin.test_mae_errors[configs]};")
    
    twin = twin.train(
        df=train_data,
        test_size=args.test_size,
        random_state=args.random_state,
        degree=args.degree,
        verbose=args.verbose,
        optimizer=args.optimizer,
        optimizer__learning_rate=args.optimizer__learning_rate,
        epochs=args.epochs,
        algorithm="keras_regression",
        remove_outliers=args.remove_outliers,
    )

    for configs in configurations:
        print(f"Train_{configs}_keras_mae={twin.train_mae_errors[configs]};")
        print(f"Validation_{configs}_keras_mae={twin.test_mae_errors[configs]};")

    # Save the model and config_data to the model_dir so that it can be loaded by model_fn
    joblib.dump(twin, os.path.join(args.model_dir, "model.joblib"))

    # Print Success
    logger.info("Saved model!")


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
