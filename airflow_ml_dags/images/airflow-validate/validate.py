import os
import pandas as pd
import click
import pickle
import json

from sklearn.metrics import accuracy_score, f1_score


@click.command("validate")
@click.option("--data-dir")
@click.option("--artifacts-dir")
@click.option("--output-dir")
def validate(data_dir: str, artifacts_dir: str, output_dir: str):
    test_data = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
    test_target = pd.read_csv(os.path.join(data_dir, "test_target.csv"))

    with open(os.path.join(artifacts_dir, "transformer.pkl"), "rb") as f:
        scaler = pickle.load(f)
    X_test_scaled = scaler.transform(test_data)

    with open(os.path.join(artifacts_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test_scaled)

    metrics = {'accuracy': accuracy_score(test_target, y_pred),
               "f1": f1_score(test_target, y_pred)}

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)



if __name__ == '__main__':
    validate()