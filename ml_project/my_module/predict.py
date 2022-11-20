import click
import pandas as pd
import logging
from my_module.data import read_data
from my_module.features import create_transformer, split_data_and_target
from my_module.models import load_model, predict_model
from my_module.entities import PredictPipelineParams, read_predict_pipeline_params

logger = logging.getLogger("Train")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(predict_pipeline_params: PredictPipelineParams) -> str:
    logger.info(f"start predict pipeline with params {predict_pipeline_params}")
    data = read_data(predict_pipeline_params.input_data_path)
    ### нужно чтобы преобразование тестовых данных было детерменировано
    train_data = read_data(predict_pipeline_params.data_for_transform)
    train_data, _ = split_data_and_target(
        train_data, predict_pipeline_params.feature_params
    )

    transformer = create_transformer(predict_pipeline_params.feature_params)
    transformer.fit(train_data)
    ###
    logger.info(f"data.shape is {data.shape}")

    features = transformer.transform(data)

    logger.info(f"test_features.shape is {features.shape}")

    model = load_model(predict_pipeline_params.model_path)
    predicts = predict_model(features, model)
    pd.DataFrame(predicts).to_csv(predict_pipeline_params.output_data_path, index=False)
    logger.info(f"predict saved in {predict_pipeline_params.output_data_path}")

    return predict_pipeline_params.output_data_path


@click.command
@click.argument("path_to_config", type=click.Path(exists=True))
def predict_pipeline_command(path_to_config: str = ""):
    params = read_predict_pipeline_params(path_to_config)
    predict_pipeline(params)


if __name__ == "__main__":
    predict_pipeline_command()
