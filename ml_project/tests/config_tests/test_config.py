import unittest
from my_module.entities import read_training_pipeline_params
import my_module


class TestConfig(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.good_config_path = "configs/train_configs/log_reg_train_config.yaml"
        self.bad_config_path = "tests/config_tests/broken.yml"

    def test_read_bad_config(self):
        with self.assertRaises(ValueError):
            read_training_pipeline_params(self.bad_config_path)

    def test_read_good_conf(self):
        params = read_training_pipeline_params(self.good_config_path)
        self.assertIsInstance(
            params, my_module.entities.train_params.TrainingPipelineParams
        )


if __name__ == "__main__":
    unittest.main()
