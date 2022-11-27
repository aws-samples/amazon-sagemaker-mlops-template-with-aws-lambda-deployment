import argparse
import logging
import pathlib
import sys
from datetime import datetime, timedelta

import boto3
import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

stats = {
  "example_flow_mean": {
    "c1c2": 283.27,
    "c1c2c3": 473.01,
    "c1c2c3c4": 476.51,
    "c1c2c4": 358.86,
    "c1c3": 258.11,
    "c1c3c4": 450.12,
    "c1c4": 258.61,
    "c2c3": 254.56,
    "c2c3c4": 449.53,
    "c2c4": 258.27,
    "c3c4": 294.25
  },
  "example_flow_std": {
    "c1c2": 48.69,
    "c1c2c3": 76.38,
    "c1c2c3c4": 90.91,
    "c1c2c4": 65.88,
    "c1c3": 47.17,
    "c1c3c4": 83.56,
    "c1c4": 49.13,
    "c2c3": 52.34,
    "c2c3c4": 73.12,
    "c2c4": 42.48,
    "c3c4": 56.63
  },
  "example_pressure_mean": {
    "c1c2": 70.48,
    "c1c2c3": 70.9,
    "c1c2c3c4": 70.88,
    "c1c2c4": 70.99,
    "c1c3": 70.8,
    "c1c3c4": 70.81,
    "c1c4": 70.34,
    "c2c3": 70.41,
    "c2c3c4": 71.02,
    "c2c4": 70.82,
    "c3c4": 70.57
  },
  "example_pressure_std": {
    "c1c2": 0.43,
    "c1c2c3": 0.59,
    "c1c2c3c4": 0.75,
    "c1c2c4": 0.6,
    "c1c3": 0.79,
    "c1c3c4": 0.69,
    "c1c4": 0.43,
    "c2c3": 0.52,
    "c2c3c4": 0.58,
    "c2c4": 0.67,
    "c3c4": 0.77
  },
  "example_power_mean": {
    "c1c2": 61.74,
    "c1c2c3": 64.07,
    "c1c2c3c4": 62.91,
    "c1c2c4": 61.33,
    "c1c3": 63.36,
    "c1c3c4": 59.55,
    "c1c4": 54.36,
    "c2c3": 61.97,
    "c2c3c4": 59.89,
    "c2c4": 56.04,
    "c3c4": 54.68
  },
  "example_power_std": {
    "c1c2": 0.91,
    "c1c2c3": 0.56,
    "c1c2c3c4": 0.97,
    "c1c2c4": 1.17,
    "c1c3": 1.28,
    "c1c3c4": 1.53,
    "c1c4": 1.14,
    "c2c3": 1.21,
    "c2c3c4": 1.34,
    "c2c4": 1.32,
    "c3c4": 1.01
  },
}

def create_data_points(stats, setting, size, next_start="2020-01-01"):
    flow = np.random.normal(loc=stats["example_flow_mean"][setting], scale=stats["example_flow_std"][setting], size=size).reshape(-1, 1)
    pressure = np.random.normal(loc=stats["example_pressure_mean"][setting], scale=stats["example_pressure_std"][setting], size=size).reshape(-1, 1)
    power = np.random.normal(loc=stats["example_power_mean"][setting], scale=stats["example_power_std"][setting], size=size).reshape(-1, 1)
    dttm = pd.date_range(start=next_start, freq="H", periods=size)
    return [dttm, flow, pressure, power, np.repeat(a=setting, repeats=size)]

if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-name", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)

    low = 12
    high = 48
    compressor_settings = ["c1c2", "c1c3", "c1c4", "c2c3", "c2c4", "c3c4", "c1c2c3", "c1c2c4", "c1c3c4", "c2c3c4", "c1c2c3c4"]
    setting = np.random.choice(compressor_settings)
    size = int(np.random.uniform(low=low, high=high))

    data = [pd.DataFrame(data=create_data_points(stats=stats, setting=setting, size=size)).transpose()]
    next_start = data[0].iloc[-1, 0] + timedelta(minutes=1)

    while next_start < datetime.now():
        setting = np.random.choice(compressor_settings)
        size = int(np.random.uniform(low=low, high=high))
        data.append(pd.DataFrame(data=create_data_points(stats=stats, setting=setting, size=size, next_start=next_start)).transpose())
        next_start = data[-1].iloc[-1, 0] + timedelta(minutes=1)

    df = pd.concat(objs=data)
    df.columns = ["dttm", "flow", "pressure", "power", "config"]
    df.index = df.dttm
    df = df.drop("dttm", axis=1)
    df[["flow", "pressure", "power"]] = df[["flow", "pressure", "power"]].astype(float)

    # Convert time_in_seconds to a dttm and sort by this column
    df = df.sort_values(by="dttm", ascending=True)

    # Save new dataset to S3
    logger.info("Writing out datasets to %s.", base_dir)
    df.to_csv(
        f"{base_dir}/train/dataset.csv",
        index=True,
    )
