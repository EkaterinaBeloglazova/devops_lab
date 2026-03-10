import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler


def preprocess_data(data_path, scaler=None, fit_scaler=Falsяe):
    """
    Предобработка данных: масштабирование признаков.
    Ищет все CSV-файлы внутри указанной директории и её подпапок.
    """
    csv_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        raise FileNotFoundError(f"Нет CSV-файлов в папке: {data_path}")

    print(f"Найдено CSV-файлов: {len(csv_files)}")
    for file in csv_files:
        print(f"  - {file}")

    all_data = []
    for file in csv_files:
        df = pd.read_csv(file)

        if "date" not in df.columns or "temperature" not in df.columns:
            print(f"Предупреждение: файл {file} пропущен, нет нужных колонок")
            continue

        df["date"] = pd.to_datetime(df["date"]).dt.dayofyear
        all_data.append(df)

    if not all_data:
        raise ValueError("Нет корректных данных для предобработки")

    combined_data = pd.concat(all_data, ignore_index=True)

    X = combined_data[["date"]].values
    y = combined_data["temperature"].values

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"Создан новый scaler на {len(X)} объектах")
    else:
        if scaler is None:
            raise ValueError("Для transform нужен обученный scaler")
        X_scaled = scaler.transform(X)
        print(f"Применён существующий scaler к {len(X)} объектам")

    return X_scaled, y, scaler


def main():
    print("=" * 50)
    print("ПРЕДОБРАБОТКА ТРЕНИРОВОЧНЫХ ДАННЫХ")
    print("=" * 50)

    train_path = "data/train"
    X_train, y_train, scaler = preprocess_data(train_path, fit_scaler=True)

    print("\nСохранение обработанных данных...")

    with open("X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    print(f"  - X_train.pkl сохранён (shape: {X_train.shape})")

    with open("y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
    print(f"  - y_train.pkl сохранён (shape: {y_train.shape})")

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("  - scaler.pkl сохранён")

    print("\nСтатистика данных:")
    print(f"  X_train: min={X_train.min():.2f}, max={X_train.max():.2f}, mean={X_train.mean():.2f}")
    print(f"  y_train: min={y_train.min():.2f}, max={y_train.max():.2f}, mean={y_train.mean():.2f}")

    print("\nПредобработка завершена успешно!")


if __name__ == "__main__":
    main()
