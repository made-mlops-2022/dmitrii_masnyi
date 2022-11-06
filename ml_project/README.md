**Инструкция по запуску:**
Создаем виртуальное окружение, клонируем репозиторий.  
Загружаем датасет с сайта https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci и помещаем данные в директорию ```data/```
(P.S данные уже там лежат)

**Установка:**
Находясь в директории ml_project:
```
install -r requirements.txt
pip install -e .
```

**Обучение:**
```
python my_module/train.py configs/train_configs/log_reg_train_config.yaml 
```
Или любойдругой конфиг файл

**Инференс:**
```
python my_module/predict.py configs/predict_configs/lr_pred_config.yaml 
```
Или любой другой конфиг файл

**Комментарии**
* Обученые модели и их метрики сохраняются в директорию ```models/``` 
* Данные для инференса необходимо помещать в директорию ```data/test_data/```
* Результат инференса сохраняется в директории ```predicts/```
