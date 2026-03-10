import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
import glob

def preprocess_data(data_path, scaler=None, fit_scaler=False):
    """
    Предобработка данных: масштабирование признаков
    """
    # Загружаем все CSV файлы из директории
    csv_files = glob.glob(os.path.join(data_path, '*.csv'))
    
    if not csv_files:
        print(f"Нет CSV файлов в {data_path}")
        return None, None
    
    all_data = []
    for file in csv_files:
        df = pd.read_csv(file)
        df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear
        all_data.append(df)
    
    # Объединяем все данные
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Создаем признаки
    X = combined_data[['day_of_year']].values
    y = combined_data['temperature'].values
    
    # Масштабируем признаки
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("Для тестовых данных необходимо передать обученный scaler")
        X_scaled = scaler.transform(X)
    
    return X_scaled, y, scaler

# Обработка тренировочных данных
print("Предобработка тренировочных данных...")
X_train, y_train, scaler = preprocess_data('train', fit_scaler=True)

# Сохраняем scaler для использования на тестовых данных
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Сохраняем обработанные данные
with open('X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

print(f"Тренировочные данные обработаны: X shape {X_train.shape}")
print("Предобработка завершена!")
