import os
import numpy as np
import pandas as pd

np.random.seed(42)

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"


def create_temperature_dataset(start_date, days=30, anomaly_fraction=0.1):
    dates = pd.date_range(start=start_date, periods=days, freq="D")

    # Базовая температура: плавное изменение + шум
    base = 10 + 8 * np.sin(np.linspace(0, 3 * np.pi, days))
    noise = np.random.normal(0, 1.5, days)
    temperature = base + noise

    is_anomaly = np.zeros(days, dtype=int)

    # Добавляем аномалии
    n_anomalies = max(1, int(days * anomaly_fraction))
    anomaly_indices = np.random.choice(days, n_anomalies, replace=False)

    for idx in anomaly_indices:
        jump = np.random.choice([10, -10, 12, -12])
        temperature[idx] += jump
        is_anomaly[idx] = 1

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "temperature": np.round(temperature, 1),
        "is_anomaly": is_anomaly
    })

    return df


def main():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # train наборы
    for i in range(5):
        df = create_temperature_dataset(start_date=f"2024-01-{1+i:02d}", days=30)
        df.to_csv(f"{TRAIN_DIR}/weather_train_{i}.csv", index=False)

    # test наборы
    for i in range(2):
        df = create_temperature_dataset(start_date=f"2024-03-{1+i:02d}", days=30)
        df.to_csv(f"{TEST_DIR}/weather_test_{i}.csv", index=False)

    print("Данные созданы.")


if __name__ == "__main__":
    main()
