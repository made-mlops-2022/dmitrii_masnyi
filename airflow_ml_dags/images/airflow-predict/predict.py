import os
import pandas as pd
import pickle
import click


@click.command("predict")
@click.option("--data-dir")
@click.option("--artifacts-dir")
@click.option("--output-dir")
def predict(data_dir: str, artifacts_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(data_dir, "data.csv"))
    with open(os.path.join(artifacts_dir, "transformer.pkl"), "rb") as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(data)

    with open(os.path.join(artifacts_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_scaled)

    os.makedirs(output_dir, exist_ok=True)
    preds = pd.DataFrame(y_pred, columns=['target'])
    preds.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)



if __name__ == '__main__':
    predict()