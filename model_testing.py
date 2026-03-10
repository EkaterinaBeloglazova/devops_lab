import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_test_files(test_dir="data/test"):
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(".csv"):
                test_files.append(os.path.join(root, file))

    if not test_files:
        raise FileNotFoundError(f"Тестовые CSV-файлы не найдены в папке: {test_dir}")

    return sorted(test_files)


def main():
    print("Загрузка модели...")
    model = joblib.load("model.pkl")

    print("Загрузка scaler...")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print("Загрузка тестовых данных...")
    test_files = load_test_files("data/test")
    print(f"Найдено тестовых файлов: {len(test_files)}")

    file_results = []
    all_predictions = []
    all_actual = []

    for test_file in test_files:
        print(f"\nОбработка файла: {test_file}")

        df = pd.read_csv(test_file)

        if "date" not in df.columns or "temperature" not in df.columns:
            print(f"Файл {test_file} пропущен: нет нужных колонок")
            continue

        df["day_of_year"] = pd.to_datetime(df["date"]).dt.dayofyear

        X_test = df[["day_of_year"]].values
        y_test = df["temperature"].values

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        file_results.append({
            "file": os.path.basename(test_file),
            "path": test_file,
            "mse": mse,
            "mae": mae,
            "r2": r2
        })

        all_predictions.extend(y_pred)
        all_actual.extend(y_test)

    if not file_results:
        raise ValueError("Не удалось обработать ни одного тестового файла")

    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 50)

    print("\nРезультаты по отдельным файлам:")
    for result in file_results:
        print(f"\n{result['path']}:")
        print(f"  MSE: {result['mse']:.4f}")
        print(f"  MAE: {result['mae']:.4f}")
        print(f"  R2:  {result['r2']:.4f}")

    print("\n" + "=" * 50)
    print("ОБЩИЕ МЕТРИКИ:")
    print(f"Средний MSE: {np.mean([r['mse'] for r in file_results]):.4f}")
    print(f"Средний MAE: {np.mean([r['mae'] for r in file_results]):.4f}")
    print(f"Средний R2:  {np.mean([r['r2'] for r in file_results]):.4f}")
    print("=" * 50)

    try:
        import matplotlib.pyplot as plt

        first_file = test_files[0]
        df_first = pd.read_csv(first_file)
        df_first["day_of_year"] = pd.to_datetime(df_first["date"]).dt.dayofyear

        X_first = df_first[["day_of_year"]].values
        X_first_scaled = scaler.transform(X_first)
        y_pred_first = model.predict(X_first_scaled)

        plt.figure(figsize=(12, 6))
        plt.plot(df_first["date"], df_first["temperature"], label="Фактическая температура")
        plt.plot(df_first["date"], y_pred_first, linestyle="--", label="Предсказанная температура")
        plt.xlabel("Дата")
        plt.ylabel("Температура")
        plt.title(f"Сравнение предсказаний и фактических значений ({os.path.basename(first_file)})")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("test_results.png", dpi=100)

        print("\nГрафик сохранён в test_results.png")

    except ImportError:
        print("\nmatplotlib не установлен, визуализация пропущена")
    except Exception as e:
        print(f"\nОшибка при построении графика: {e}")


if __name__ == "__main__":
    main()
