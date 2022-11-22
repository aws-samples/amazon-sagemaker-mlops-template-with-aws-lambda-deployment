import argparse
import json
import logging
import multiprocessing as mp
import pathlib
# Make library accessible
import sys
import tarfile
from datetime import datetime

import boto3
import joblib
import numpy as np
import pandas as pd
#from digital_twin import DigitalTwin

sys.path.append("/opt/ml/code/")


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

s3 = boto3.client("s3")

clf = [
    "c1c2",
    "c1c2c4",
    "c1c4",
    "c1c3c4",
    "c1c2c3c4",
    "c1c3",
    "c1c2c3",
    "c3c4",
    "c2c3c4",
    "c2c4",
    "c2c3",
]

def parallel_func(data):
    results = {}
    flow = data[:2]
    press = data[2:]
    clf = ["c1c2","c1c2c4","c1c4","c1c3c4","c1c2c3c4","c1c3","c1c2c3","c3c4","c2c3c4","c2c4","c2c3"]
    boundary = {
        "c1c2": [180, 390],
        "c1c2c4": [180, 390],
        "c1c4": [180, 390],
        "c1c3c4": [180, 460],
        "c1c2c3c4": [180, 460],
        "c1c3": [180, 390],
        "c1c2c3": [180, 460],
        "c3c4": [180, 390],
        "c2c3c4": [180, 460],
        "c2c4": [180, 390],
        "c2c3": [180, 390],
    }
    n = 100
    no_of_trials = 10
    train_error_weight = 1.0
    res = twin.run_monte_carlo(
        clf=clf,
        flow=flow,
        press=press,
        n=n,
        no_of_trials=no_of_trials,
        boundary=boundary,
        train_error_weight=train_error_weight,
        algorithm="keras_regression")
    for r in res:
        key = list(r.keys())[0]
        results[key] = r[key]
    results["flow_low"] = flow[0]
    results["flow_high"] = flow[1]
    results["pressure_low"] = press[0]
    results["pressure_high"] = press[1]
    return results

def extract_key(x):
    vec = list(json.loads(x.to_json()).values())
    vec = [y if y is not None else np.nan for y in vec]
    argmin = np.nanargmin(vec)
    key = list(x.keys())[argmin]
    return key

def extract_value(x):
    vec = list(json.loads(x.to_json()).values())
    vec = [y if y is not None else np.nan for y in vec]
    argmin = np.nanmin(vec)
    return argmin

if __name__ == "__main__":
    logger.debug("Starting simulation.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artefact", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)

    x = args.model_artefact
    bucket = x.split("/")[2]
    key = "/".join(x.split("/")[3:])
    with open('model.tar.gz', 'wb') as f:
        s3.download_fileobj(bucket, key, f)

    tar = tarfile.open("model.tar.gz")
    tar.extractall()

    twin = joblib.load("model.joblib")

    low = 180
    high = 450
    flows = []
    for i in range(low, high, 20):
        for j in range(low, high, 20):
            flows.append([i, j])

    flows = pd.DataFrame(data=flows, columns=["flow_low", "flow_high"]).sort_values(by=["flow_low", "flow_high"])
    flows["filter"] = flows.apply(lambda x: True if x.iloc[0] < x.iloc[1] else False, axis=1)
    flows = flows[flows["filter"] == True].reset_index(drop=True).drop("filter", axis=1)

    low = 50
    high = 75
    pressures = []
    for i in range(low, high, 5):
        for j in range(low, high, 5):
            pressures.append([i, j])

    pressures = pd.DataFrame(data=pressures, columns=["pressure_low", "pressure_high"]).sort_values(by=["pressure_low", "pressure_high"])
    pressures["filter"] = pressures.apply(lambda x: True if x.iloc[0] < x.iloc[1] else False, axis=1)
    pressures = pressures[pressures["filter"] == True].reset_index(drop=True).drop("filter", axis=1)

    data = pd.DataFrame()
    for i in range(0, pressures.shape[0]):
        tmp = flows.copy("deep")
        tmp[pressures.columns.tolist()] = pressures.iloc[i, :]
        data = pd.concat(objs=[data, tmp], axis=0)
    data = data.reset_index(drop=True)

    pool = mp.Pool(mp.cpu_count())
    predictions = [pool.apply(parallel_func, args=(dat, )) for dat in np.array(data)]
    pool.close()

    df = pd.DataFrame(predictions)
    df.loc[:, clf] = df.loc[:, clf].apply(lambda x: np.round(x/100, 0), axis=1)
    df["minimum_key"] = df.loc[:, clf].apply(lambda x: extract_key(x=x), axis=1)
    df["minimum_value"] = df.loc[:, clf].apply(lambda x: extract_value(x=x), axis=1)

    # Save new dataset to S3
    logger.info("Writing out datasets to %s.", base_dir)
    df.to_csv(
        f"{base_dir}/train/simulation.csv",
        index=False,
    )
