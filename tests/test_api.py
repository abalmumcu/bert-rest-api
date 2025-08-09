import os
import sys
from unittest.mock import patch

import pytest

# Ensure the application package is importable when tests are executed from the
# "tests" directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def client():
    # Patch the Hugging Face pipelines used in ``BERTModel`` so tests do not
    # trigger model downloads.
    def dummy_pipeline(task):
        if task == "summarization":
            return lambda texts: [{"summary_text": "summary"} for _ in texts]
        return lambda x: x

    with patch('app.model.pipeline', side_effect=dummy_pipeline):
        from app.api import app
        with app.test_client() as client:
            yield client


def test_health_endpoint(client):
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert "model_name" in data


def test_summarize_endpoint(client):
    resp = client.post('/api/summarize', json={"texts": ["one", "two"]})
    assert resp.status_code == 200
    assert resp.get_json() == {"summaries": ["summary", "summary"]}


def test_predict_missing_texts(client):
    resp = client.post('/api/predict', json={})
    assert resp.status_code == 400
    assert "message" in resp.get_json()
