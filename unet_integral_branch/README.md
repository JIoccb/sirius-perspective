# Barcode Segmentation

Запуск тренировки:
```
python src/train.py
```

### Тестирование
Веса обученной модели `./checkpoints/{Имя_эксперимента}/best_clf_{Имя_эксперимента}.ckpt`

Запуск инференса:
1. Задайте в конфиге имя нужного эксперимента в поле `experiment_name`:
2. Выполните:
```
python src/test.py
```

Результат прогноза можно посомтреть в `notebooks/EDA.ipynb`


