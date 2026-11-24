"""FastAPI 接口层测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _payload(budget: int, waits=None, tensors=None) -> dict:
    return {
        "streams": {
            "stream_a": ["a1", "a2", "a3"],
            "stream_b": ["b1", "b2", "b3"],
        },
        "waits": waits or [],
        "tensors": tensors
        or [
            {"name": "TA", "producer": "a1", "consumers": ["a2"], "size_bytes": 100, "alignment": 1},
            {"name": "TB", "producer": "b1", "consumers": ["b2"], "size_bytes": 100, "alignment": 1},
        ],
        "budget_bytes": budget,
    }


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_plan_ok_no_wait():
    r = client.post("/plan", json=_payload(budget=200))
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is True
    assert body["added_wait"] is None
    assert body["total_bytes"] == 200
    assert set(body["offsets"]) == {"TA", "TB"}


def test_plan_ok_adds_wait():
    r = client.post("/plan", json=_payload(budget=100))
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is True
    assert body["total_bytes"] == 100
    assert body["added_wait"] == {"waiter": "a1", "waitee": "b2"}


def test_plan_infeasible():
    r = client.post("/plan", json=_payload(budget=50))
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is False
    assert body["offsets"] == {}
    assert body["best_effort_total_bytes"] == 100


def test_semantic_422():
    # 生产不先于消费（跨流且无依赖）-> 422
    payload = _payload(
        budget=1000,
        tensors=[
            {"name": "T", "producer": "b2", "consumers": ["a1"], "size_bytes": 10, "alignment": 1}
        ],
    )
    r = client.post("/plan", json=payload)
    assert r.status_code == 422
    assert "数据竞争" in r.json()["detail"]


def test_pydantic_422_on_bad_shape():
    payload = _payload(budget=100)
    payload["tensors"][0]["size_bytes"] = -3
    r = client.post("/plan", json=payload)
    assert r.status_code == 422


def test_openapi_docs_available():
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
