import json
from typing import Tuple, Dict
import click
import logging

from my_module.data import read_data, split_train_val_data
from my_module.features import create_transformer, split_data_and_target
from my_module.models import train_model, predict_model, save_model, evaluate_model
from my_module.entities import read_training_pipeline_params, TrainingPipelineParams

logger = logging.getLogger("Train")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(
    train_pipeline_params: TrainingPipelineParams,
) -> Tuple[str, Dict[str, float]]:
    logger.info(f"start train pipeline with params {train_pipeline_params}")
    data = read_data(train_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    train_df, val_df = split_train_val_data(
        data, train_pipeline_params.splitting_params
    )

    train_df, train_target = split_data_and_target(
        train_df, train_pipeline_params.feature_params
    )
    val_df, val_target = split_data_and_target(
        val_df, train_pipeline_params.feature_params
    )

    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"val_df.shape is {val_df.shape}")

    transformer = create_transformer(train_pipeline_params.feature_params)
    train_features = transformer.fit_transform(train_df)
    logger.info(f"train_features.shape is {train_features.shape}")

    model = train_model(
        train_features, train_target, train_pipeline_params.model_params
    )

    val_features = transformer.transform(val_df)
    predicts = predict_model(val_features, model)

    metrics = evaluate_model(predicts, val_target)
    logger.info(f"metrics is {metrics}")

    with open(train_pipeline_params.metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f)
        logger.info(f"metrics saved in {train_pipeline_params.metric_path}")

    path_to_model = save_model(model, train_pipeline_params.output_model_path)
    logger.info(f"model saved in {train_pipeline_params.output_model_path}")
    return path_to_model, metrics


@click.command
@click.argument("path_to_config", type=click.Path(exists=True))
def train_pipeline_command(path_to_config: str = ""):
    params = read_training_pipeline_params(path_to_config)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
