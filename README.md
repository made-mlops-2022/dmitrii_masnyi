**Архитектурные решения по проекту**
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
**Критерии (указаны максимальные баллы, по каждому критерию ревьюер может поставить баллы частично):**

- [X] В описании к пулл реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. В общем, описание того, что именно вы сделали и для чего, чтобы вашим ревьюерам было легче понять ваш код (1 балл)
- [X] В пулл-реквесте проведена самооценка, распишите по каждому пункту выполнен ли критерий или нет и на сколько баллов(частично или полностью) (1 балл)

- [X] Выполнено EDA, закоммитьте ноутбук в папку с ноутбуками (1 балл)
   Вы так же можете построить в ноутбуке прототип(если это вписывается в ваш стиль работы)

- [ ] Можете использовать не ноутбук, а скрипт, который сгенерит отчет, закоммитьте и скрипт и отчет (за это + 1 балл)

- [X] Написана функция/класс для тренировки модели, вызов оформлен как утилита командной строки, записана в readme инструкцию по запуску (3 балла)
- [X] Написана функция/класс predict (вызов оформлен как утилита командной строки), которая примет на вход артефакт/ы от обучения, тестовую выборку (без меток) и запишет предикт по заданному пути, инструкция по вызову записана в readme (3 балла)

- [X] Проект имеет модульную структуру (2 балла)
- [X] Использованы логгеры (2 балла)

- [X] Написаны тесты на отдельные модули и на прогон обучения и predict (3 балла)

- [X] Для тестов генерируются синтетические данные, приближенные к реальным (2 балла)
   - можно посмотреть на библиотеки https://faker.readthedocs.io/, https://feature-forge.readthedocs.io/en/latest/
   - можно просто руками посоздавать данных, собственноручно написанными функциями.

- [X] Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing) (3 балла)
- [X] Используются датаклассы для сущностей из конфига, а не голые dict (2 балла)

- [X] Напишите кастомный трансформер и протестируйте его (3 балла)
   https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156

- [X] В проекте зафиксированы все зависимости (1 балл)
- [X] Настроен CI для прогона тестов, линтера на основе github actions (3 балла).
Пример с пары: https://github.com/demo-ml-cicd/ml-python-package

PS: Можно использовать cookiecutter-data-science  https://drivendata.github.io/cookiecutter-data-science/ , но поудаляйте папки, в которые вы не вносили изменения, чтобы не затруднять ревью

Дополнительные баллы=)
- [ ] Используйте hydra для конфигурирования (https://hydra.cc/docs/intro/) - 3 балла

Mlflow
- [ ] разверните локально mlflow или на какой-нибудь виртуалке (1 балл)
- [ ] залогируйте метрики (1 балл)
- [ ] воспользуйтесь Model Registry для регистрации модели(1 балл)
  Приложите скриншот с вашим mlflow run
  DVC
- [ ] выделите в своем проекте несколько entrypoints в виде консольных утилит (1 балл).
  Пример: https://github.com/made-ml-in-prod-2021/ml_project_example/blob/main/setup.py#L16
  Но если у вас нет пакета, то можно и просто несколько скриптов

- [ ] добавьте датасет под контроль версий (1 балл)
  
- [ ] сделайте dvc пайплайн(связывающий запуск нескольких entrypoints) для изготовления модели(1 балл)

Для большего удовольствия в выполнении этих частей рекомендуется попробовать подключить удаленное S3 хранилище(например в Yandex Cloud, VK Cloud Solutions или Selectel)
