** Архитектурные решения по проекту **
``` bash
.
├── README.md
├── configs # all configs
│   ├── predict_configs # inference configs
│   │   ├── lr_custom_pred_config.yml
│   │   ├── lr_pred_config.yaml
│   │   └── rf_pred_config.yaml
│   └── train_configs # train configs
│       ├── log_reg_train_config.yaml
│       ├── log_reg_train_custom_transf_config.yaml
│       └── rf_train_config.yaml
├── data # directory with data
│   ├── heart_cleveland_upload.csv
│   └── test_data # directory with data for inference
│       ├── synt_data.csv
│       └── test.csv
├── models # directory where models and their metrics are saved
│   ├── log_reg_custom_model.pkl
│   ├── log_reg_model.pkl
│   ├── log_reg_model_test_custom_metrics.json
│   ├── log_reg_model_test_metrics.json
│   ├── rf_model.pkl
│   └── rf_model_test_metrics.json
├── my_module #my module with useful tools
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── predict.cpython-310.pyc
│   │   └── train.cpython-310.pyc
│   ├── data # data tools
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   └── create_dataset.cpython-310.pyc
│   │   └── create_dataset.py # script for dataset creation from csv
│   ├── entities # entities of dataclasses
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── feature_params.cpython-310.pyc
│   │   │   ├── model_params.cpython-310.pyc
│   │   │   ├── predict_params.cpython-310.pyc
│   │   │   ├── split_params.cpython-310.pyc
│   │   │   └── train_params.cpython-310.pyc
│   │   ├── feature_params.py # define features and their preprocessing params from config
│   │   ├── model_params.py # define params for model creation -//-
│   │   ├── predict_params.py define params for inference -//-
│   │   ├── split_params.py # define data split params -//-
│   │   └── train_params.py # define params of model training  -//-
│   ├── features # scrips for feature preprocessing
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   └── prepare_features.cpython-310.pyc
│   │   └── prepare_features.py
│   ├── models # scrips for model training and inference
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   └── model_tools.cpython-310.pyc
│   │   └── model_tools.py
│   ├── predict.py # inference pipeline
│   ├── train.py # training pipeline
│   └── transformer # custom implementation of standard scaler
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-310.pyc
│       │   └── custom_transformer.cpython-310.pyc
│       └── custom_transformer.py
├── my_module.egg-info # data from my_module
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   ├── entry_points.txt
│   ├── requires.txt
│   └── top_level.txt
├── notebooks # notebooks with eda and model prototyping
│   └── eda_and_model_prototype.ipynb
├── predicts
│   ├── log_reg_custom_predictions.csv
│   ├── log_reg_predictions.csv
│   ├── rf_predictions.csv
│   └── synt_data.csv
├── requirements.txt # reqs
├── setup.py # my_module setup
└── tests # tests for modules
    ├── __init__.py
    ├── __pycache__
    │   └── __init__.cpython-310.pyc
    ├── config_tests
    │   ├── __init__.py
    │   ├── broken.yml
    │   └── test_config.py
    ├── create_dataset
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-310.pyc
    │   │   └── test_create_dataset.cpython-310.pyc
    │   └── test_create_dataset.py
    ├── custom_transformer
    │   ├── __init__.py
    │   └── test_custom_transformer.py
    ├── features_preproc
    │   ├── __init__.py
    │   └── test_features_preproc.py
    ├── predict_model
    │   ├── __init__.py
    │   └── test_predict_model.py
    ├── syn_data_tests
    │   ├── __init__.py
    │   └── test_syn_data.py
    └── train_model
        ├── __init__.py
        └── test_train_model.py
```
