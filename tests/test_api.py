from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import main
from taxi import registry as reg
from taxi.model import register, train_and_log

GOOD = {"zone": 132, "ts": "2020-01-20T18:00:00", "lag_1": 310, "lag_2": 295, "lag_3": 280, "lag_24": 320, "lag_168": 330}


@pytest.fixture()
def client(registry):
    with TestClient(main.app) as c:
        yield c


def test_health_ok_with_model(client, registry):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_version"] == registry["version"]


def test_predict_returns_contract(client):
    r = client.post("/v1/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert body["zone"] == 132 and body["predicted_pickups"] >= 0
    assert body["request_id"] and body["model_version"] >= 1


@pytest.mark.parametrize(
    "patch",
    [{"lag_1": -1}, {"zone": 999}, {"surprise": 1}, {"ts": "not a date"}],
    ids=["negative_count", "unknown_zone", "extra_field", "bad_timestamp"],
)
def test_invalid_requests_are_refused_not_scored(client, patch):
    assert client.post("/v1/predict", json={**GOOD, **patch}).status_code == 422


def test_missing_field_is_422(client):
    body = {k: v for k, v in GOOD.items() if k != "lag_24"}
    assert client.post("/v1/predict", json=body).status_code == 422


def test_metrics_exposed_after_traffic(client):
    for _ in range(60):
        client.post("/v1/predict", json=GOOD)
    text = client.get("/metrics/").text
    assert "taxi_request_seconds_bucket" in text
    assert 'taxi_input_psi{feature="lag_24"}' in text
    assert "taxi_model_version" in text


def test_no_model_means_503(registry, monkeypatch):
    monkeypatch.setattr(main, "ALIAS", "does-not-exist")
    monkeypatch.setattr(main.State, "model", None)
    with TestClient(main.app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/v1/predict", json=GOOD).status_code == 503


def test_promote_then_rollback_moves_alias_and_api_follows(client, registry):
    df, v1 = registry["df"], registry["version"]
    _, _, uri, run_id = train_and_log(df[df.ts < "2020-01-10"], df[df.ts >= "2020-01-15"], run_name="candidate")
    v2 = register(uri, run_id, "staging")

    assert reg.promote() == v2
    assert client.post("/v1/admin/reload").json()["model_version"] == v2

    assert reg.rollback() == v1
    assert client.post("/v1/admin/reload").json()["model_version"] == v1
