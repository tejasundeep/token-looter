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

    def test_in_flight_tracking(self):
        from app.ratelimit import increment_in_flight, decrement_in_flight, get_in_flight_count
        platform, model, key = "test_platform", "test_model", "test_key"
        
        # Test initial count
        self.assertEqual(get_in_flight_count(platform, model, key), 0)
        
        # Increment
        increment_in_flight(platform, model, key)
        self.assertEqual(get_in_flight_count(platform, model, key), 1)
        
        # Decrement
        decrement_in_flight(platform, model, key)
        self.assertEqual(get_in_flight_count(platform, model, key), 0)

    def test_adaptive_cooldown_parsing(self):
        import httpx
        from app.providers.base import make_provider_http_error
        
        # Simulate response with x-ratelimit-reset header
        headers = httpx.Headers({
            "x-ratelimit-reset-requests": "12.5"
        })
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = headers
        mock_response.reason_phrase = "Too Many Requests"
        
        err = make_provider_http_error(mock_response, "Rate limit hit")
        self.assertEqual(err.retry_after_ms, 12500.0)

    def test_persistent_key_states(self):
        from app.ratelimit import set_cooldown, is_on_cooldown, set_global_key_disabled, is_key_globally_disabled
        
        # Test cooldown persistence
        set_cooldown("test_plat", "test_mod", "test_key", 5000)
        self.assertTrue(is_on_cooldown("test_plat", "test_mod", "test_key"))
        
        # Test global key disable persistence
        set_global_key_disabled("test_plat", "test_key", 5000)
        self.assertTrue(is_key_globally_disabled("test_plat", "test_key"))

    def test_in_flight_token_reservations(self):
        from app.ratelimit import increment_in_flight, decrement_in_flight, can_use_tokens
        platform, model, key = "test_res_platform", "test_res_model", "test_res_key"
        
        increment_in_flight(platform, model, key, tokens=100)
        
        # Verify can_use_tokens checks in-flight tokens
        # If limit is 150, used is 0, in-flight is 100, asking for 60 should fail (100 + 60 > 150)
        limits = {"rpm": None, "rpd": None, "tpm": 150, "tpd": None}
        self.assertFalse(can_use_tokens(platform, model, key, estimated_tokens=60, limits=limits))
        self.assertTrue(can_use_tokens(platform, model, key, estimated_tokens=40, limits=limits))
        
        decrement_in_flight(platform, model, key, tokens=100)
        self.assertTrue(can_use_tokens(platform, model, key, estimated_tokens=60, limits=limits))

    def test_get_next_chunk_safely_disconnect(self):
        import asyncio
        from app.v1_endpoints import get_next_chunk_safely
        
        # Setup mock request
        mock_request = MagicMock()
        mock_request.is_disconnected = AsyncMock(return_value=True)
        
        # Setup slow mock generator
        async def dummy_gen():
            await asyncio.sleep(5)
            yield {"text": "hello"}
            
        gen = dummy_gen()
        
        async def run_test():
            await get_next_chunk_safely(gen, mock_request)
            
        # Expect CancelledError
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()

