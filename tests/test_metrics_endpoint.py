from fastapi.testclient import TestClient

from inferlite.api.app import app


def test_metrics_endpoint_exposes_prometheus_text() -> None:
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "inferlite_request_latency_seconds" in response.text
    assert "inferlite_ttft_seconds" in response.text

