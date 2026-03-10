import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Загружаем обработанные данные
print("Загрузка обработанных данных...")
with open('X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

# Создаем и обучаем модель
print("Обучение модели...")
# Используем случайный лес для лучшего учета нелинейностей
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оцениваем модель на тренировочных данных
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

print(f"Модель обучена!")
print(f"  MSE на тренировочных данных: {mse:.4f}")
print(f"  R2 на тренировочных данных: {r2:.4f}")

# Сохраняем модель
joblib.dump(model, 'model.pkl')
print("Модель сохранена в model.pkl")
