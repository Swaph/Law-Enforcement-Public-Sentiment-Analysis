from __future__ import annotations

import os
from pathlib import Path

import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL = None
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))

app = FastAPI(title="Law Enforcement Sentiment API", version="0.1.0")


class PredictRequest(BaseModel):
    text: str = Field(min_length=3, max_length=10000)


class PredictResponse(BaseModel):
    label: str
    confidence: float | None = None


def _load_model() -> None:
    global MODEL
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{MODEL_PATH}'. Run training first."
        )
    MODEL = joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup_event() -> None:
    _load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    label = str(MODEL.predict([payload.text])[0])
    confidence = None
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba([payload.text])[0]
        confidence = float(max(probs))

    return PredictResponse(label=label, confidence=confidence)


def run() -> None:
    uvicorn.run("sentiment_demo.api:app", host="0.0.0.0", port=8000, reload=False)
