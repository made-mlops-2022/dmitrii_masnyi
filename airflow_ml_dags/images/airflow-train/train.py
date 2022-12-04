import os
import pandas as pd
import click
import pickle
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--data-dir")
@click.option("--artifacts-dir")
@click.option("--output-dir")
def train(data_dir: str, artifacts_dir: str, output_dir: str):

    train_data = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    train_target = pd.read_csv(os.path.join(data_dir, "train_target.csv"))
    with open(os.path.join(artifacts_dir, "transformer.pkl"), 'rb') as f:
         scaler = pickle.load(f)

    X_scaled = scaler.transform(train_data)

    model = LogisticRegression()
    model.fit(X_scaled, train_target)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)



if __name__ == '__main__':
    train()