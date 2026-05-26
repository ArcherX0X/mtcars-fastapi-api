"""Train a linear regression model on mtcars.csv and save it to models/model.pkl."""

import pathlib
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

DATA_PATH = pathlib.Path(__file__).parent.parent / "mtcars.csv"
MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "model.pkl"

PREDICTORS = ["wt", "hp"]
RESPONSE = "mpg"


def train() -> None:
    df = pd.read_csv(DATA_PATH)

    X = df[PREDICTORS]
    y = df[RESPONSE]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred) ** 0.5:.4f}")
    for name, coef in zip(PREDICTORS, model.coef_):
        print(f"  {name}: {coef:.4f}")
    print(f"  intercept: {model.intercept_:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "predictors": PREDICTORS}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
