import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(data_path, scaler=None, fit_scaler=False):
    csv_files = []

    for root, _, files in os.walk(data_path):
        for file_name in files:
            if file_name.endswith(".csv"):
                csv_files.append(os.path.join(root, file_name))

    csv_files = sorted(csv_files)

    if not csv_files:
        raise FileNotFoundError(f"Нет CSV-файлов в папке: {data_path}")

    all_data = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        if "date" not in df.columns or "temperature" not in df.columns:
            print(f"Файл пропущен: {file_path}")
            continue

        df["date"] = pd.to_datetime(df["date"])
        df["day_of_year"] = df["date"].dt.dayofyear
        df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        all_data.append(df)

    if not all_data:
        raise ValueError("Нет корректных данных")

    combined_data = pd.concat(all_data, ignore_index=True)

    X = combined_data[["sin_day", "cos_day"]].values
    y = combined_data["temperature"].values

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("Scaler не передан")
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler


def main():
    X_train, y_train, scaler = preprocess_data("train", fit_scaler=True)

    with open("X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)

    with open("y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Предобработка завершена.")


if __name__ == "__main__":
    main()
