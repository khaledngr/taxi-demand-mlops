"""Taxi demand API: next-hour pickups for one zone.

    uv run uvicorn app.main:app --port 8000
Loads models:/taxi-demand@production (name + alias, never a path). /health answers 503 while no model
is loaded, so a load balancer stops sending traffic. Prometheus metrics on /metrics cover three of the
four monitoring layers: infrastructure (latency, rate), input (PSI per feature over the last requests),
prediction (score distribution). Layer 4, outcome, is the actual count one hour later, joined by request_id.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from threading import Lock
from typing import ClassVar

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, ConfigDict, Field

from taxi.common import FEATURES
from taxi.drift import psi
from taxi.model import load_alias, predict

ALIAS = os.environ.get("TAXI_MODEL_ALIAS", "production")
PSI_WINDOW = int(os.environ.get("TAXI_PSI_WINDOW", "300"))
MONITORED = ["lag_1", "lag_24", "lag_168", "prediction"]

REQUESTS = Counter("taxi_requests_total", "Prediction requests", ["status"])
LATENCY = Histogram(
    "taxi_request_seconds", "Request latency", buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5)
)
PREDICTION = Histogram("taxi_prediction", "Predicted pickups", buckets=(0, 10, 25, 50, 100, 200, 400, 800, 1600))
INPUT_PSI = Gauge("taxi_input_psi", "PSI of the last requests against the training reference", ["feature"])
MODEL_VERSION = Gauge("taxi_model_version", "Registry version currently served")

log = logging.getLogger("taxi.api")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class State:
    model = None
    version: int | None = None
    reference: dict | None = None
    window: ClassVar[deque] = deque(maxlen=PSI_WINDOW)
    lock: ClassVar[Lock] = Lock()


def load() -> None:
    try:
        model, version, ref = load_alias(ALIAS)
    except Exception as exc:  # noqa: BLE001  no registry, no alias yet: stay up, answer 503
        log.warning(json.dumps({"event": "model_load_failed", "alias": ALIAS, "error": str(exc)[:200]}))
        return
    with State.lock:
        State.model, State.version, State.reference = model, version, ref
        State.window.clear()
    MODEL_VERSION.set(version)
    log.info(json.dumps({"event": "model_loaded", "alias": ALIAS, "version": version}))


@asynccontextmanager
async def lifespan(_: FastAPI):
    load()
    yield


app = FastAPI(title="Taxi demand API", version="1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class PredictionRequest(BaseModel):
    """The contract: a wrong request is refused (422), never scored."""

    model_config = ConfigDict(extra="forbid")
    zone: int = Field(ge=1, le=265, description="TLC pickup zone id", examples=[237])
    ts: datetime = Field(description="start of the hour to forecast", examples=["2020-02-20T18:00:00"])
    lag_1: int = Field(ge=0, description="pickups in this zone 1 hour before ts", examples=[310])
    lag_2: int = Field(ge=0, examples=[295])
    lag_3: int = Field(ge=0, examples=[280])
    lag_24: int = Field(ge=0, description="same hour yesterday", examples=[320])
    lag_168: int = Field(ge=0, description="same hour last week", examples=[330])


class PredictionResponse(BaseModel):
    request_id: str
    zone: int
    ts: datetime
    predicted_pickups: float
    model_version: int


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    if request.url.path.startswith("/v1/predict"):
        LATENCY.observe(time.perf_counter() - start)
        REQUESTS.labels(status=str(response.status_code)).inc()
    return response


@app.get("/health")
def health():
    if State.model is None:
        raise HTTPException(503, "no model loaded")
    return {"status": "ok", "alias": ALIAS, "model_version": State.version}


@app.post("/v1/predict", response_model=PredictionResponse)
def predict_one(req: PredictionRequest):
    if State.model is None:
        raise HTTPException(503, "no model loaded")
    row = req.model_dump()
    ts = pd.Timestamp(req.ts)
    frame = pd.DataFrame([{**row, "hour": ts.hour, "dow": ts.dayofweek}])[FEATURES]
    value = float(predict(State.model, frame)[0])
    request_id = str(uuid.uuid4())
    PREDICTION.observe(value)
    with State.lock:
        State.window.append({"lag_1": req.lag_1, "lag_24": req.lag_24, "lag_168": req.lag_168, "prediction": value})
        if State.reference and len(State.window) >= 50 and len(State.window) % 10 == 0:
            live = pd.DataFrame(State.window)
            for feat in MONITORED:
                INPUT_PSI.labels(feature=feat).set(psi(State.reference[feat], live[feat].to_numpy(dtype=float)))
    # allow-list logging: what we need to join the label later, nothing about who asked
    log.info(json.dumps({"request_id": request_id, "zone": req.zone, "ts": ts.isoformat(),
                         "prediction": round(value, 2), "model_version": State.version}))
    return PredictionResponse(request_id=request_id, zone=req.zone, ts=req.ts,
                              predicted_pickups=round(value, 2), model_version=State.version)


@app.post("/v1/admin/reload")
def reload():
    """Re-resolve the alias after a promotion or a rollback, without a redeploy."""
    load()
    if State.model is None:
        raise HTTPException(503, "no model loaded")
    return {"alias": ALIAS, "model_version": State.version}


@app.get("/v1/model")
def model_info():
    return {"alias": ALIAS, "model_version": State.version, "features": FEATURES,
            "psi_window": len(State.window), "reference_features": sorted(State.reference or {})}
