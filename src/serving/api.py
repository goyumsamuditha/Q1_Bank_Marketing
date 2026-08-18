import os
import pickle
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.serving.inference import predict_single, prepare_single_record
from src.serving.schema import ClientPredictionRequest, ClientPredictionResponse

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

MODEL_PATH = os.environ.get("MODEL_PATH", "models/final_pipeline.joblib")
THRESHOLD_PATH = os.environ.get("THRESHOLD_PATH", "models/decision_threshold.txt")
SEASONAL_LOOKUP_PATH = os.environ.get("SEASONAL_LOOKUP_PATH", "models/seasonal_conversion_prior.csv")
FREQ_ENCODERS_PATH = os.environ.get("FREQ_ENCODERS_PATH", "models/frequency_encoders.pkl")

_model = None
_threshold = None
_seasonal_lookup = None
_frequency_encoders = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and its supporting artefacts ONCE when the
    container starts, not on every request - loading a model file is
    slow, so doing it per-request would make every prediction slow too.
    """
    global _model, _threshold, _seasonal_lookup, _frequency_encoders
    _model = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH) as f:
        _threshold = float(f.read().strip())
    _seasonal_lookup = pd.read_csv(SEASONAL_LOOKUP_PATH, index_col=0).iloc[:, 0]
    with open(FREQ_ENCODERS_PATH, "rb") as f:
        _frequency_encoders = pickle.load(f)
    yield
    
app = FastAPI(
    title="Bank Term Deposit Prediction API",
    description="Predicts the probability a client subscribes to a term deposit.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict:
    """Simple liveness check - used by Azure/Databricks/load balancers to
    confirm the container is up before routing traffic to it."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=ClientPredictionResponse)
def predict(request: ClientPredictionRequest) -> ClientPredictionResponse:
    """Score one client and return their subscription probability."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    prepared_row = prepare_single_record(request.model_dump(), _seasonal_lookup, _frequency_encoders)
    result = predict_single(_model, prepared_row, _threshold)
    return ClientPredictionResponse(**result)

app.mount("/static", StaticFiles(directory="src/serving/static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("src/serving/static/index.html")