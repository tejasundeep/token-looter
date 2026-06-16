import time
import json
from typing import AsyncGenerator, Dict, Any, Optional, List
import httpx
from app.providers.base import BaseProvider, make_provider_http_error, ProviderHttpError
from app.lib.tool_call_rescue import rescue_inline_tool_calls
from app.lib.tool_args import repair_tool_arguments, tool_schema_map

class OpenAICompatProvider(BaseProvider):
    def __init__(
        self,
        platform: str,
        name: str,
        base_url: str,
        extra_headers: Optional[Dict[str, str]] = None,
        validate_url: Optional[str] = None,
        timeout_ms: float = 15000.0,
        keyless: bool = False,
        force_single_tool_call: bool = False
    ):
        self._platform = platform
        self._name = name
        self.base_url = base_url
        self.extra_headers = extra_headers or {}
        self.validate_url = validate_url
        self.timeout_ms = timeout_ms
        self.keyless = keyless
        self.force_single_tool_call = force_single_tool_call

    @property
    def platform(self) -> str:
        return self._platform

    @property
    def name(self) -> str:
        return self._name

    def resolve_parallel_tool_calls(self, options: Optional[Dict[str, Any]]) -> Optional[bool]:
        if options and self.force_single_tool_call and options.get("tools"):
            return False
        return options.get("parallel_tool_calls") if options else None

    def rescue_failed_generation(self, err_body: Any, options: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(err_body, dict):
            return None
        failed = err_body.get("error", {}).get("failed_generation")
        if not isinstance(failed, str) or not failed:
            return None
        
        tools = options.get("tools") if options else None
        if not tools:
            return None
        tool_names = {t["function"]["name"] for t in tools if "function" in t and "name" in t["function"]}
        if not tool_names:
            return None
            
        rescue = rescue_inline_tool_calls(failed, tool_names)
        if not rescue.get("detected") or not rescue.get("calls"):
            return None
            
        schemas = tool_schema_map(tools)
        rescued_calls = []
        for i, c in enumerate(rescue["calls"]):
            name = c["name"]
            raw_args = c["arguments"]
            repaired_args = repair_tool_arguments(raw_args, schemas.get(name) or {})
            rescued_calls.append({
                "id": f"call_rescued_{i + 1}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": repaired_args
                }
            })
        return rescued_calls

    def auth_header(self, api_key: str) -> Dict[str, str]:
        return {} if self.keyless else {"Authorization": f"Bearer {api_key}"}

    async def chat_completion(
        self,
        api_key: str,
        messages: List[Dict[str, Any]],
        model_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        timeout = self.timeout_ms
        if options:
            if options.get("timeoutMs") is not None:
                timeout = options["timeoutMs"]
            elif options.get("timeout_ms") is not None:
                timeout = options["timeout_ms"]
        
        headers = {
            **self.auth_header(api_key),
            "Content-Type": "application/json",
            **self.extra_headers
        }
        
        body = {
            "model": model_id,
            "messages": messages,
        }
        if options:
            for opt in ["temperature", "max_tokens", "top_p", "tools", "tool_choice"]:
                if opt in options and options[opt] is not None:
                    body[opt] = options[opt]
            ptc = self.resolve_parallel_tool_calls(options)
            if ptc is not None:
                body["parallel_tool_calls"] = ptc

        url = f"{self.base_url}/chat/completions"
        
        async with self.get_client(timeout) as client:
            try:
                res = await client.post(url, headers=headers, json=body)
            except Exception as e:
                raise Exception(f"{self.name} connection error: {str(e)}") from e

            if res.status_code != 200:
                try:
                    err_json = res.json()
                except Exception:
                    err_json = {}
                
                rescued = self.rescue_failed_generation(err_json, options)
                if rescued:
                    print(f"[{self.name}] Rescued {len(rescued)} inline tool call(s) from a {res.status_code} tool_use_failed (#264)")
                    out = {
                        "id": f"chatcmpl-rescued-{int(time.time() * 1000)}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": rescued
                            },
                            "finish_reason": "tool_calls"
                        }],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    }
                    out["_routed_via"] = {"platform": self.platform, "model": model_id}
                    return out
                raise make_provider_http_error(res, f"{self.name} API error {res.status_code}: {err_json.get('error', {}).get('message') or res.reason_phrase}")

            try:
                data = res.json()
            except Exception as e:
                raise Exception(
                    f"{self.name} returned 200 with a non-JSON body — the endpoint is not OpenAI-compatible. "
                    f"Check the base URL (for Ollama use http://host:11434/v1, for llama.cpp/vLLM/LM Studio the /v1 path)."
                ) from e
            
            normalize_choices(data)
            data["_routed_via"] = {"platform": self.platform, "model": model_id}
            return data

    async def stream_chat_completion(
        self,
        api_key: str,
        messages: List[Dict[str, Any]],
        model_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        timeout = self.timeout_ms
        if options:
            if options.get("timeoutMs") is not None:
                timeout = options["timeoutMs"]
            elif options.get("timeout_ms") is not None:
                timeout = options["timeout_ms"]
        
        headers = {
            **self.auth_header(api_key),
            "Content-Type": "application/json",
            **self.extra_headers
        }
        
        body = {
            "model": model_id,
            "messages": messages,
            "stream": True
        }
        if options:
            for opt in ["temperature", "max_tokens", "top_p", "tools", "tool_choice"]:
                if opt in options and options[opt] is not None:
                    body[opt] = options[opt]
            ptc = self.resolve_parallel_tool_calls(options)
            if ptc is not None:
                body["parallel_tool_calls"] = ptc

        url = f"{self.base_url}/chat/completions"
        
        async with self.get_client(timeout) as client:
            try:
                async with client.stream("POST", url, headers=headers, json=body) as res:
                    if res.status_code != 200:
                        await res.aread()
                        try:
                            err_json = res.json()
                        except Exception:
                            err_json = {}
                        
                        rescued = self.rescue_failed_generation(err_json, options)
                        if rescued:
                            print(f"[{self.name}] Rescued {len(rescued)} inline tool call(s) from a {res.status_code} tool_use_failed (stream, #264)")
                            base_chunk = {
                                "id": f"chatcmpl-rescued-{int(time.time() * 1000)}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_id
                            }
                            yield {**base_chunk, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
                            yield {**base_chunk, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": idx, **c} for idx, c in enumerate(rescued)]}, "finish_reason": None}]}
                            yield {**base_chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
                            return
                        raise make_provider_http_error(res, f"{self.name} API error {res.status_code}: {err_json.get('error', {}).get('message') or res.reason_phrase}")

                    async for chunk in self.read_sse_stream(res, inactivity_timeout_ms=90000.0):
                        yield chunk
            except Exception as e:
                if isinstance(e, ProviderHttpError) or (isinstance(e, Exception) and "ended unexpectedly" in str(e)):
                    raise
                raise Exception(f"{self.name} stream connection error: {str(e)}") from e

    async def validate_key(self, api_key: str) -> bool:
        url = self.validate_url or f"{self.base_url}/models"
        headers = {
            **self.auth_header(api_key),
            **self.extra_headers
        }
        async with self.get_client(30000.0) as client:
            try:
                res = await client.get(url, headers=headers)
                return res.status_code not in (401, 403)
            except Exception:
                raise

def normalize_choices(data: Dict[str, Any]) -> None:
    choices = data.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message")
        if not isinstance(msg, dict):
            continue
            
        content = msg.get("content")
        if isinstance(content, list):
            parts = []
            for seg in content:
                if isinstance(seg, str):
                    parts.append(seg)
                elif isinstance(seg, dict):
                    parts.append(seg.get("text") or "")
            msg["content"] = "".join(parts)
            content = msg["content"]
            
        tool_calls = msg.get("tool_calls")
        has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
        if not has_tool_calls and (content == "" or content is None):
            reasoning_content = msg.get("reasoning_content")
            reasoning = msg.get("reasoning")
            fold = None
            if isinstance(reasoning_content, str) and len(reasoning_content) > 0:
                fold = reasoning_content
            elif isinstance(reasoning, str) and len(reasoning) > 0:
                fold = reasoning
            
            if fold is not None:
                msg["content"] = fold
