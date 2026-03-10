import os
import numpy as np
import pandas as pd

np.random.seed(42)

TRAIN_DIR = "train"
TEST_DIR = "test"


def create_temperature_dataset(start_date, days=30, add_anomalies=False, noise_level=1.5):
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    day_of_year = dates.dayofyear.values

    # Температура зависит от РЕАЛЬНОГО дня года
    temperature = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365)

    # Добавляем шум
    noise = np.random.normal(0, noise_level, days)
    temperature = temperature + noise

    is_anomaly = np.zeros(days, dtype=int)

    if add_anomalies:
        anomaly_count = 3
        anomaly_indices = np.random.choice(days, anomaly_count, replace=False)

        for idx in anomaly_indices:
            jump = np.random.choice([10, -10, 12, -12])
            temperature[idx] += jump
            is_anomaly[idx] = 1

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "temperature": np.round(temperature, 2),
        "is_anomaly": is_anomaly
    })

    return df


def prepare_folder(folder):
    os.makedirs(folder, exist_ok=True)
    for file_name in os.listdir(folder):
        if file_name.endswith(".csv"):
            os.remove(os.path.join(folder, file_name))


def main():
    prepare_folder(TRAIN_DIR)
    prepare_folder(TEST_DIR)

    train_configs = [
        ("2024-01-01", False, 1.0),
        ("2024-02-01", True, 1.5),
        ("2024-03-01", False, 1.2),
        ("2024-04-01", True, 1.3),
    ]

    for i, (start_date, add_anomalies, noise_level) in enumerate(train_configs):
        df = create_temperature_dataset(
            start_date=start_date,
            days=30,
            add_anomalies=add_anomalies,
            noise_level=noise_level
        )
        df.to_csv(os.path.join(TRAIN_DIR, f"weather_train_{i}.csv"), index=False)

    test_configs = [
        ("2024-05-01", False, 1.1),
        ("2024-06-01", True, 1.4),
    ]

    for i, (start_date, add_anomalies, noise_level) in enumerate(test_configs):
        df = create_temperature_dataset(
            start_date=start_date,
            days=30,
            add_anomalies=add_anomalies,
            noise_level=noise_level
        )
        df.to_csv(os.path.join(TEST_DIR, f"weather_test_{i}.csv"), index=False)

    print("Данные успешно созданы.")
    print(f"Train файлов: {len(train_configs)}")
    print(f"Test файлов: {len(test_configs)}")


if __name__ == "__main__":
    main()
