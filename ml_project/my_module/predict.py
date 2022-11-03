import click
import pandas as pd
from my_module.data import read_data
from my_module.features import create_transformer
from my_module.models import load_model, predict_model
from my_module.entities import PredictPipelineParams, read_predict_pipeline_params


def predict_pipeline(predict_pipeline_params: PredictPipelineParams) -> str:
    # logger.info(f"start predict pipeline with params {predict_pipeline_params}")
    data = read_data(predict_pipeline_params.input_data_path)
    # logger.info(f"data.shape is {data.shape}")

    transformer = create_transformer(predict_pipeline_params.feature_params)
    features = transformer.fit_transform(data)
    # logger.info(f"train_features.shape is {features.shape}")

    model = load_model(predict_pipeline_params.model_path)

    predicts = predict_model(features, model)
    pd.DataFrame(predicts).to_csv(predict_pipeline_params.output_data_path, index=False)
    # logger.info(f"predict saved by {predict_pipeline_params.path_to_output}")

    return predict_pipeline_params.output_data_path


@click.command
@click.argument("path_to_config", type=click.Path(exists=True))
def predict_pipeline_command(path_to_config: str = ""):
    params = read_predict_pipeline_params(path_to_config)
    predict_pipeline(params)


if __name__ == "__main__":
    predict_pipeline_command()
