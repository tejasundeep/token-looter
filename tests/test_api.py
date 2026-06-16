import os
import sys
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import init_db, get_unified_api_key
from app.main import app

class TestTokenLooterHeadless(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize SQLite database in memory for testing analytics logging
        os.environ["DATABASE_PATH"] = ":memory:"
        init_db(":memory:")
        cls.client = TestClient(app)
        cls.unified_key = get_unified_api_key()
        cls.headers = {"Authorization": f"Bearer {cls.unified_key}"}

    def test_auth_required(self):
        # Verify 401 is returned when API key is missing or invalid
        res = self.client.get("/v1/models")
        self.assertEqual(res.status_code, 401)
        
        res = self.client.get("/v1/models", headers={"Authorization": "Bearer invalid"})
        self.assertEqual(res.status_code, 401)

    def test_get_models(self):
        res = self.client.get("/v1/models", headers=self.headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["object"], "list")
        
        # Verify "auto" is in the model list
        model_ids = [m["id"] for m in data["data"]]
        self.assertIn("auto", model_ids)

    def test_chat_completions_validation(self):
        # Missing messages
        res = self.client.post("/v1/chat/completions", json={}, headers=self.headers)
        self.assertEqual(res.status_code, 400)
        
        # Empty messages
        res = self.client.post("/v1/chat/completions", json={"messages": []}, headers=self.headers)
        self.assertEqual(res.status_code, 400)

    def test_chat_completions_invalid_model(self):
        res = self.client.post("/v1/chat/completions", json={
            "model": "non-existent-model-1234",
            "messages": [{"role": "user", "content": "hi"}]
        }, headers=self.headers)
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["error"]["code"], "model_not_found")

    @patch('app.v1_endpoints.route_request')
    def test_chat_completions_routing_success(self, mock_route):
        # Setup mock route
        mock_provider = AsyncMock()
        mock_provider.chat_completion.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "Hello! I am a mocked model."}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        }
        
        mock_route.return_value = {
            "provider": mock_provider,
            "modelId": "mock-model",
            "modelDbId": 0,
            "apiKey": "mock-key",
            "keyId": "mock-key-id",
            "platform": "groq",
            "displayName": "Mock Groq Model",
            "rpdLimit": None,
            "tpdLimit": None
        }

        with patch('app.v1_endpoints.record_request'), \
             patch('app.v1_endpoints.record_tokens'), \
             patch('app.v1_endpoints.record_success'):
            
            res = self.client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hello"}]
            }, headers=self.headers)
            
            self.assertEqual(res.status_code, 200)
            self.assertIn("Hello! I am a mocked model.", res.json()["choices"][0]["message"]["content"])

    def test_embeddings_validation(self):
        # Missing input
        res = self.client.post("/v1/embeddings", json={}, headers=self.headers)
        self.assertEqual(res.status_code, 400)


    def test_responses_validation(self):
        # Missing input
        res = self.client.post("/v1/responses", json={}, headers=self.headers)
        self.assertEqual(res.status_code, 400)

if __name__ == "__main__":
    unittest.main()
