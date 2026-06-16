from typing import AsyncGenerator, Dict, Any, Optional, List, Union
import httpx
from app.providers.base import BaseProvider, make_provider_http_error
from app.lib.content import content_to_string

class CloudflareProvider(BaseProvider):
    @property
    def platform(self) -> str:
        return "cloudflare"

    @property
    def name(self) -> str:
        return "Cloudflare Workers AI"

    def parse_key(self, api_key: str) -> Dict[str, str]:
        sep = api_key.find(":")
        if sep == -1:
            raise Exception('Cloudflare key must be in format "account_id:api_token"')
        return {
            "account_id": api_key[:sep],
            "token": api_key[sep + 1:]
        }

    def normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {**m, "content": content_to_string(m.get("content"))}
            for m in messages
        ]

    async def chat_completion(
        self,
        api_key: str,
        messages: List[Dict[str, Any]],
        model_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        kp = self.parse_key(api_key)
        account_id = kp["account_id"]
        token = kp["token"]
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
        
        body = {
            "model": model_id,
            "messages": self.normalize_messages(messages)
        }
        if options:
            for opt in ["temperature", "max_tokens", "top_p", "tools", "tool_choice", "parallel_tool_calls"]:
                if opt in options and options[opt] is not None:
                    body[opt] = options[opt]

        timeout = 15000.0
        if options:
            if options.get("timeoutMs") is not None:
                timeout = options["timeoutMs"]
            elif options.get("timeout_ms") is not None:
                timeout = options["timeout_ms"]
                
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        async with self.get_client(timeout) as client:
            try:
                res = await client.post(url, headers=headers, json=body)
            except Exception as e:
                raise Exception(f"Cloudflare connection error: {str(e)}") from e

            if res.status_code != 200:
                try:
                    err = res.json()
                except Exception:
                    err = {}
                msg = err.get("error", {}).get("message")
                if not msg and isinstance(err.get("errors"), list) and len(err["errors"]) > 0:
                    msg = err["errors"][0].get("message")
                if not msg:
                    msg = res.reason_phrase
                raise make_provider_http_error(res, f"Cloudflare API error {res.status_code}: {msg}")

            data = res.json()
            data["_routed_via"] = {"platform": "cloudflare", "model": model_id}
            return data

    async def stream_chat_completion(
        self,
        api_key: str,
        messages: List[Dict[str, Any]],
        model_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        kp = self.parse_key(api_key)
        account_id = kp["account_id"]
        token = kp["token"]
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
        
        body = {
            "model": model_id,
            "messages": self.normalize_messages(messages),
            "stream": True
        }
        if options:
            for opt in ["temperature", "max_tokens", "top_p", "tools", "tool_choice", "parallel_tool_calls"]:
                if opt in options and options[opt] is not None:
                    body[opt] = options[opt]

        timeout = 15000.0
        if options:
            if options.get("timeoutMs") is not None:
                timeout = options["timeoutMs"]
            elif options.get("timeout_ms") is not None:
                timeout = options["timeout_ms"]
                
        headers = {
            "Authorization": f"Bearer {token}",
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
                        msg = err.get("error", {}).get("message")
                        if not msg and isinstance(err.get("errors"), list) and len(err["errors"]) > 0:
                            msg = err["errors"][0].get("message")
                        if not msg:
                            msg = res.reason_phrase
                        raise make_provider_http_error(res, f"Cloudflare API error {res.status_code}: {msg}")

                    async for chunk in self.read_sse_stream(res):
                        yield chunk
            except Exception as e:
                if isinstance(e, Exception) and ("stalled" in str(e) or "ended unexpectedly" in str(e)):
                    raise
                raise Exception(f"Cloudflare stream connection error: {str(e)}") from e

    async def validate_key(self, api_key: str) -> bool:
        kp = self.parse_key(api_key)
        account_id = kp["account_id"]
        token = kp["token"]

        user_res = await self.verify_at("https://api.cloudflare.com/client/v4/user/tokens/verify", token)
        if user_res != "auth-failed":
            return user_res

        account_res = await self.verify_at(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/tokens/verify",
            token
        )
        if account_res == "auth-failed":
            return False
        return account_res

    async def verify_at(self, url: str, token: str) -> Union[bool, str]:
        headers = {"Authorization": f"Bearer {token}"}
        async with self.get_client(10000.0) as client:
            try:
                res = await client.get(url, headers=headers)
                if res.status_code in (401, 403):
                    return "auth-failed"
                if res.status_code != 200:
                    return True
                data = res.json()
                return data.get("success") is True and data.get("result", {}).get("status") == "active"
            except Exception:
                raise
