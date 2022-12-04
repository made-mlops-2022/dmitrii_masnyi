import os
from sklearn.datasets import make_classification
import pandas as pd
import click

BATCH_SIZE = 1000
N_FEATS = 5
N_CLASSES = 2

@click.command("download")
@click.argument("output_dir")
def get_data(output_dir):
    X, y = make_classification(n_samples=BATCH_SIZE, 
                               n_features=N_FEATS, 
                               n_informative=N_FEATS-2, 
                               n_classes=N_CLASSES)

    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(N_FEATS)])
    y = pd.DataFrame(y, columns=['target'])

    os.makedirs(output_dir, exist_ok=True)
    X.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "target.csv"), index=False)



if __name__ == "__main__":
    get_data()