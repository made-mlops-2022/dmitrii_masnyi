from dataclasses import dataclass
import yaml
from marshmallow_dataclass import class_schema

from .feature_params import FeatureParams


@dataclass
class PredictPipelineParams:
    input_data_path: str
    output_data_path: str
    model_path: str
    feature_params: FeatureParams
    data_for_transform: str


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(config_path: str) -> PredictPipelineParams:
    with open(config_path, "r", encoding="utf-8") as config:
        schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(config))
