import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Загружаем обработанные данные
print("Загрузка обработанных данных...")

# Проверяем существование файлов
x_train_file = 'X_train.pkl'  # Заглавная X
y_train_file = 'y_train.pkl'

# Показываем текущую директорию
print(f"Текущая директория: {os.getcwd()}")
print("Доступные файлы:")
for file in os.listdir('.'):
    if file.endswith('.pkl'):
        print(f"  - {file}")

if not os.path.exists(x_train_file):
    print(f"Ошибка: файл {x_train_file} не найден!")
    raise FileNotFoundError(f"Не найден файл {x_train_file}")

if not os.path.exists(y_train_file):
    print(f"Ошибка: файл {y_train_file} не найден!")
    raise FileNotFoundError(f"Не найден файл {y_train_file}")

# Загружаем данные
with open(x_train_file, 'rb') as f:
    X_train = pickle.load(f)
with open(y_train_file, 'rb') as f:
    y_train = pickle.load(f)

print(f"Данные загружены успешно:")
print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  Тип X_train: {type(X_train)}")

# Создаем и обучаем модель
print("\nОбучение модели...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оцениваем модель на тренировочных данных
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

print(f"\nМодель обучена!")
print(f"  MSE на тренировочных данных: {mse:.4f}")
print(f"  R2 на тренировочных данных: {r2:.4f}")
print(f"  Важность признаков: {model.feature_importances_}")

# Сохраняем модель
joblib.dump(model, 'model.pkl')
print(f"\nМодель сохранена в model.pkl")

# Сохраняем информацию о модели
model_info = {
    'type': type(model).__name__,
    'n_estimators': model.n_estimators,
    'feature_importances': model.feature_importances_.tolist(),
    'train_mse': mse,
    'train_r2': r2
}

with open('model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("Информация о модели сохранена в model_info.pkl")
