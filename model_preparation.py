import os
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    if not os.path.exists("X_train.pkl"):
        raise FileNotFoundError("X_train.pkl не найден")

    if not os.path.exists("y_train.pkl"):
        raise FileNotFoundError("y_train.pkl не найден")

    with open("X_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    with open("y_train.pkl", "rb") as f:
        y_train = pickle.load(f)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)

    print("Результаты на train:")
    print(f"MSE: {mean_squared_error(y_train, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_train, y_pred):.4f}")
    print(f"R2:  {r2_score(y_train, y_pred):.4f}")

    joblib.dump(model, "model.pkl")
    print("Модель сохранена в model.pkl")


if __name__ == "__main__":
    main()
