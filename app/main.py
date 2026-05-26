import logging
import os
import pathlib
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = pathlib.Path(
    os.getenv("MODEL_PATH", pathlib.Path(__file__).parent.parent / "models" / "model.pkl")
)

_model_bundle: dict | None = None


def load_model() -> dict:
    global _model_bundle
    if _model_bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _model_bundle = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    return _model_bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except FileNotFoundError as e:
        logger.warning("Model not loaded on startup: %s", e)
    yield


app = FastAPI(
    title="MTCARS MPG Predictor",
    description="Predicts fuel efficiency (mpg) from vehicle weight and horsepower.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    wt: float = Field(..., gt=0, description="Vehicle weight (1000 lbs)")
    hp: float = Field(..., gt=0, description="Gross horsepower")


class PredictionResponse(BaseModel):
    predicted_mpg: float
    predictors: dict[str, float]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model file not found")
    try:
        load_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        bundle = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    model = bundle["model"]
    predictors = bundle["predictors"]

    features = pd.DataFrame([{p: getattr(request, p) for p in predictors}])
    predicted_mpg = float(model.predict(features)[0])

    logger.info("Prediction: wt=%.3f hp=%.1f -> mpg=%.2f", request.wt, request.hp, predicted_mpg)

    return PredictionResponse(
        predicted_mpg=round(predicted_mpg, 2),
        predictors={"wt": request.wt, "hp": request.hp},
    )
