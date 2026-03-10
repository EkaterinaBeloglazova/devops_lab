import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_test_files(test_dir="test"):
    test_files = []

    for root, _, files in os.walk(test_dir):
        for file_name in files:
            if file_name.endswith(".csv"):
                test_files.append(os.path.join(root, file_name))

    test_files = sorted(test_files)

    if not test_files:
        raise FileNotFoundError("Тестовые файлы не найдены")

    return test_files


def build_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df


def main():
    model = joblib.load("model.pkl")

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    test_files = load_test_files("test")

    results = []

    for test_file in test_files:
        df = pd.read_csv(test_file)
        df = build_features(df)

        X_test = df[["sin_day", "cos_day"]].values
        y_test = df["temperature"].values

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append((test_file, mse, mae, r2))

        print(f"\nФайл: {test_file}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2:  {r2:.4f}")

    print("\nСредние метрики:")
    print(f"MSE: {np.mean([r[1] for r in results]):.4f}")
    print(f"MAE: {np.mean([r[2] for r in results]):.4f}")
    print(f"R2:  {np.mean([r[3] for r in results]):.4f}")

    first_file = test_files[0]
    df = pd.read_csv(first_file)
    df = build_features(df)

    X_test = df[["sin_day", "cos_day"]].values
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["temperature"], label="Фактическая температура", linewidth=2)
    plt.plot(df["date"], y_pred, "--", label="Предсказанная температура", linewidth=2)

    if "is_anomaly" in df.columns:
        anomalies = df[df["is_anomaly"] == 1]
        if not anomalies.empty:
            plt.scatter(anomalies["date"], anomalies["temperature"], label="Аномалии", s=80)

    plt.xlabel("Дата")
    plt.ylabel("Температура")
    plt.title("Сравнение фактических и предсказанных значений")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("test_results.png", dpi=120)
    plt.show()

    print("\nГрафик сохранён в test_results.png")


if __name__ == "__main__":
    main()
