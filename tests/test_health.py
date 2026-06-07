from fastapi.testclient import TestClient

from inferlite.api.app import app


def test_healthz() -> None:
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["details"]["phase"] == "5"
