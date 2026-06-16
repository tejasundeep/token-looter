import abc
import json
import time
import email.utils
from typing import AsyncGenerator, AsyncIterator, Dict, Any, Optional, List
import httpx

from app.lib.proxy import get_httpx_client

class ProviderHttpError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, retry_after_ms: Optional[float] = None):
        super().__init__(message)
        self.status = status
        self.retry_after_ms = retry_after_ms

def parse_retry_after_ms(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    trimmed = value.strip()
    if trimmed.isdigit():
        return float(trimmed) * 1000.0
    try:
        parsed = email.utils.parsedate_to_datetime(trimmed)
        diff = (parsed.timestamp() - time.time()) * 1000.0
        return max(0.0, diff)
    except Exception:
        return None

def make_provider_http_error(response: httpx.Response, message: str) -> ProviderHttpError:
    retry_val = response.headers.get("retry-after")
    retry_ms = parse_retry_after_ms(retry_val)
    return ProviderHttpError(message, status=response.status_code, retry_after_ms=retry_ms)

class BaseProvider(abc.ABC):
    @property
    @abc.abstractmethod
    def platform(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    keyless: bool = False

    def get_client(self, timeout_ms: float = 15000.0) -> httpx.AsyncClient:
        return get_httpx_client(self.platform, timeout=timeout_ms / 1000.0)

    @abc.abstractmethod
    async def chat_completion(
        self,
        api_key: str,
        messages: List[Dict[str, Any]],
        model_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    async def stream_chat_completion(
        self,
        api_key: str,
        messages: List[Dict[str, Any]],
        model_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    async def validate_key(self, api_key: str) -> bool:
        pass

    def make_id(self) -> str:
        import secrets
        return f"chatcmpl-{int(time.time() * 1000)}-{secrets.token_hex(3)}"

    async def read_sse_stream(
        self,
        response: httpx.Response,
        inactivity_timeout_ms: float = 90000.0
    ) -> AsyncGenerator[Dict[str, Any], None]:
        saw_finish_reason = False
        try:
            async for line in response.aiter_lines():
                trimmed = line.strip()
                if not trimmed or not trimmed.startswith("data:"):
                    continue
                data = trimmed[5:].lstrip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices")
                    if choices and any(c.get("finish_reason") is not None for c in choices):
                        saw_finish_reason = True
                    yield chunk
                except Exception:
                    pass
        except httpx.ReadTimeout as e:
            raise Exception(f"{self.name} stream stalled: no data for {inactivity_timeout_ms}ms (timeout)") from e
            
        if not saw_finish_reason:
            raise Exception(f"{self.name} stream ended unexpectedly (no [DONE], no finish_reason) — connection reset or truncated upstream")
