from typing import AsyncGenerator, Dict, Any, Optional, List
import httpx
from app.providers.base import BaseProvider, make_provider_http_error
from app.lib.content import flatten_message_content

API_BASE = 'https://api.cohere.ai/compatibility/v1'

class CohereProvider(BaseProvider):
    @property
    def platform(self) -> str:
        return "cohere"

    @property
    def name(self) -> str:
        return "Cohere"

    async def chat_completion(
        self,
        api_key: str,
        messages: List[Dict[str, Any]],
        model_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        body = {
            "model": model_id,
            "messages": flatten_message_content(messages)
        }
        if options:
            for opt in ["temperature", "max_tokens", "top_p", "tools", "tool_choice"]:
                if opt in options and options[opt] is not None:
                    body[opt] = options[opt]

        url = f"{API_BASE}/chat/completions"
        timeout = 15000.0
        if options:
            if options.get("timeoutMs") is not None:
                timeout = options["timeoutMs"]
            elif options.get("timeout_ms") is not None:
                timeout = options["timeout_ms"]
                
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        async with self.get_client(timeout) as client:
            try:
                res = await client.post(url, headers=headers, json=body)
            except Exception as e:
                raise Exception(f"Cohere connection error: {str(e)}") from e

            if res.status_code != 200:
                try:
                    err = res.json()
                except Exception:
                    err = {}
                err_msg = ""
                err_val = err.get('error') if isinstance(err, dict) else None
                if isinstance(err_val, dict):
                    err_msg = err_val.get('message')
                elif isinstance(err_val, str):
                    err_msg = err_val
                msg = err_msg or res.reason_phrase
                raise make_provider_http_error(res, f"Cohere API error {res.status_code}: {msg}")

            data = res.json()
            data["_routed_via"] = {"platform": "cohere", "model": model_id}
            return data

    async def stream_chat_completion(
        self,
        api_key: str,
        messages: List[Dict[str, Any]],
        model_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        body = {
            "model": model_id,
            "messages": flatten_message_content(messages),
            "stream": True
        }
        if options:
            for opt in ["temperature", "max_tokens", "top_p", "tools", "tool_choice"]:
                if opt in options and options[opt] is not None:
                    body[opt] = options[opt]

        url = f"{API_BASE}/chat/completions"
        timeout = 15000.0
        if options:
            if options.get("timeoutMs") is not None:
                timeout = options["timeoutMs"]
            elif options.get("timeout_ms") is not None:
                timeout = options["timeout_ms"]
                
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        async with self.get_client(timeout) as client:
            try:
                async with client.stream("POST", url, headers=headers, json=body) as res:
                    if res.status_code != 200:
                        await res.aread()
                        try:
                            err = res.json()
                        except Exception:
                            err = {}
                        err_msg = ""
                        err_val = err.get('error') if isinstance(err, dict) else None
                        if isinstance(err_val, dict):
                            err_msg = err_val.get('message')
                        elif isinstance(err_val, str):
                            err_msg = err_val
                        msg = err_msg or res.reason_phrase
                        raise make_provider_http_error(res, f"Cohere API error {res.status_code}: {msg}")

                    async for chunk in self.read_sse_stream(res):
                        yield chunk
            except Exception as e:
                if isinstance(e, Exception) and ("stalled" in str(e) or "ended unexpectedly" in str(e)):
                    raise
                raise Exception(f"Cohere stream connection error: {str(e)}") from e

    async def validate_key(self, api_key: str) -> bool:
        url = f"{API_BASE}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        async with self.get_client(10000.0) as client:
            try:
                res = await client.get(url, headers=headers)
                return res.status_code not in (401, 403)
            except Exception:
                raise
