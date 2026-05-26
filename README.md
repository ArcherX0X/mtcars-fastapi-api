# MTCARS MPG Predictor API

A FastAPI service that predicts vehicle fuel efficiency (`mpg`) from **weight** (`wt`) and **horsepower** (`hp`) using a linear regression model trained on the classic `mtcars` dataset.

## Model

| Detail | Value |
|---|---|
| Algorithm | Linear Regression (scikit-learn) |
| Response | `mpg` (miles per gallon) |
| Predictors | `wt` (weight, 1000 lbs), `hp` (gross horsepower) |
| Training R² | 0.79 |
| Training RMSE | 2.91 |

Coefficients: `mpg = 36.78 − 3.75·wt − 0.034·hp`

## Repo Structure

```
mtcars-fastapi-api/
├── app/
│   └── main.py          # FastAPI application
├── models/
│   └── model.pkl        # Trained model artifact
├── scripts/
│   └── train_model.py   # Training script
├── tests/
│   └── test_api.py      # Automated API tests
├── mtcars.csv           # Dataset
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## Local Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (or pip)
- Podman (for container builds)

### Install and run

```bash
# clone the repo
git clone https://github.com/ArcherX0X/mtcars-fastapi-api.git
cd mtcars-fastapi-api

# create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# (re)train the model
python scripts/train_model.py

# start the API
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

The interactive docs are available at [http://localhost:8080/docs](http://localhost:8080/docs).

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check — always returns 200 |
| GET | `/ready` | Readiness check — 503 if model not loaded |
| POST | `/predict` | Return predicted mpg for given wt and hp |

### Example requests

```bash
# health
curl http://localhost:8080/health

# readiness
curl http://localhost:8080/ready

# predict
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"wt": 2.62, "hp": 110}'
```

Example response:

```json
{
  "predicted_mpg": 22.56,
  "predictors": {"wt": 2.62, "hp": 110.0}
}
```

## Run with Podman

```bash
# build
podman build -t mtcars-fastapi .

# run locally
podman run --rm -p 8080:8080 mtcars-fastapi

# test it
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"wt": 3.0, "hp": 150}'
```

## Deploy to Google Cloud Run

```bash
# authenticate
gcloud auth login
gcloud config set project mtcars-fastapi-zach
gcloud auth configure-docker us-central1-docker.pkg.dev

# create artifact registry repo (one-time)
gcloud artifacts repositories create mtcars-repo \
  --repository-format=docker \
  --location=us-central1

# build for linux/amd64 (required for Cloud Run, especially on Apple Silicon)
podman build --platform linux/amd64 \
  -t us-central1-docker.pkg.dev/mtcars-fastapi-zach/mtcars-repo/mtcars-fastapi-api:latest .

# push image
podman push us-central1-docker.pkg.dev/mtcars-fastapi-zach/mtcars-repo/mtcars-fastapi-api:latest

# deploy
gcloud run deploy mtcars-fastapi-api \
  --image us-central1-docker.pkg.dev/mtcars-fastapi-zach/mtcars-repo/mtcars-fastapi-api:latest \
  --platform managed \
  --region us-central1 \
  --port 8080 \
  --allow-unauthenticated
```

**Deployed API URL:** https://mtcars-fastapi-api-886054929408.us-central1.run.app

## Tests

```bash
uv pip install pytest httpx
pytest tests/ -v
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/model.pkl` | Path to the trained model artifact |
