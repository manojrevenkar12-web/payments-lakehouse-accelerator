import pytest
from fastapi.testclient import TestClient

import paylake.app as appmod


@pytest.fixture()
def client(built, monkeypatch):
    monkeypatch.setattr(appmod, "_settings", built["settings"])
    appmod._cache.clear()
    return TestClient(appmod.app)


def test_health_and_index(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "fraud alerts" in client.get("/").text


def test_alerts_flow(client):
    r = client.get("/alerts", params={"limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["count"] <= 5
    first = body["alerts"][0]
    assert first["alert_rank"] == 1 and "debtor_iban" not in first
    d = client.get(f"/alerts/{first['uetr']}").json()
    assert d["payment"]["uetr"] == first["uetr"]
    assert client.post(f"/alerts/{first['uetr']}/review", json={"decision": "CONFIRMED_FRAUD", "analyst": "mk"}).status_code == 201
    assert client.post(f"/alerts/{first['uetr']}/review", json={"decision": "MAYBE", "analyst": "mk"}).status_code == 422
    assert client.get("/alerts/00000000-0000-4000-8000-000000000000").status_code == 404


def test_reporting_endpoints(client):
    assert len(client.get("/kpis/daily").json()) >= 10
    assert any(r["rule"] == "amount_positive" for r in client.get("/dq").json())
    assert "pr_auc" in client.get("/model").json()
