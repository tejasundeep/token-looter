import time
import json
import base64
from typing import AsyncGenerator, Dict, Any, Optional, List
import httpx
from app.providers.base import BaseProvider, make_provider_http_error, ProviderHttpError
from app.lib.content import content_to_string
from app.lib.proxy import get_httpx_client

API_BASE = 'https://generativelanguage.googleapis.com/v1beta'

THOUGHT_SIG_TTL_MS = 30 * 60 * 1000
THOUGHT_SIG_MAX = 5000
thought_sig_cache: Dict[str, Dict[str, Any]] = {}

def remember_thought_sig(call_id: Optional[str], sig: Optional[str]) -> None:
    if not call_id or not sig:
        return
    if len(thought_sig_cache) >= THOUGHT_SIG_MAX:
        oldest = next(iter(thought_sig_cache))
        thought_sig_cache.pop(oldest, None)
    thought_sig_cache[call_id] = {
        "sig": sig,
        "exp": (time.time() * 1000) + THOUGHT_SIG_TTL_MS
    }

def recall_thought_sig(call_id: Optional[str]) -> Optional[str]:
    if not call_id:
        return None
    hit = thought_sig_cache.get(call_id)
    if not hit:
        return None
    if hit["exp"] < (time.time() * 1000):
        thought_sig_cache.pop(call_id, None)
        return None
    return hit["sig"]

def safe_parse_object(raw: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except Exception:
        return {"value": raw}

def normalize_gemini_args(args: Any) -> str:
    if isinstance(args, str):
        return args
    return json.dumps(args or {})

def to_gemini_finish_reason(finish_reason: Optional[str]) -> str:
    r = (finish_reason or "").upper()
    if not r:
        return "stop"
    if r == "MAX_TOKENS":
        return "length"
    if r in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"):
        return "content_filter"
    return "stop"

GEMINI_UNSUPPORTED_SCHEMA_KEYS = {
    '$schema', '$id', '$ref', '$defs', '$comment',
    'definitions',
    'exclusiveMinimum', 'exclusiveMaximum',
    'patternProperties', 'unevaluatedProperties', 'unevaluatedItems',
    'if', 'then', 'else',
    'contentEncoding', 'contentMediaType', 'contentSchema',
    'dependentRequired', 'dependentSchemas', 'dependencies',
    'additionalProperties',
    'examples', 'const', 'readOnly', 'writeOnly',
    'uniqueItems',
    'not', 'allOf', 'oneOf',
    'prefixItems',
    'contains', 'minContains', 'maxContains',
    'propertyNames',
    'multipleOf',
    'deprecated',
}

def sanitize_for_gemini(schema: Any) -> Any:
    if isinstance(schema, list):
        return [sanitize_for_gemini(s) for s in schema]
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k in GEMINI_UNSUPPORTED_SCHEMA_KEYS:
                continue
            out[k] = sanitize_for_gemini(v)
        return out
    return schema

GROUNDING_TOOL_NAMES = {'google_search', 'googlesearch', 'google_search_retrieval'}

def to_gemini_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    function_declarations = []
    grounding = False
    for t in tools:
        name = t.get("function", {}).get("name", "")
        if name.lower() in GROUNDING_TOOL_NAMES:
            grounding = True
            continue
        function_declarations.append({
            "name": name,
            "description": t.get("function", {}).get("description"),
            "parameters": sanitize_for_gemini(t.get("function", {}).get("parameters"))
        })
    out = []
    if grounding:
        out.append({"google_search": {}})
    if function_declarations:
        out.append({"functionDeclarations": function_declarations})
    return out if out else None

def has_function_declarations(tools: Optional[List[Dict[str, Any]]]) -> bool:
    if not tools:
        return False
    return any("functionDeclarations" in t for t in tools)

def to_gemini_tool_config(tool_choice: Optional[Any]) -> Optional[Dict[str, Any]]:
    if not tool_choice:
        return None
    if isinstance(tool_choice, str):
        mode = "AUTO"
        if tool_choice == "none":
            mode = "NONE"
        elif tool_choice == "required":
            mode = "ANY"
        return {"functionCallingConfig": {"mode": mode}}
    
    return {
        "functionCallingConfig": {
            "mode": "ANY",
            "allowedFunctionNames": [tool_choice["function"]["name"]]
        }
    }

MAX_IMAGE_BYTES = 8 * 1024 * 1024

def extract_image_url(block: Any) -> Optional[str]:
    if not isinstance(block, dict):
        return None
    iu = block.get("image_url")
    if isinstance(iu, str):
        return iu
    if isinstance(iu, dict) and isinstance(iu.get("url"), str):
        return iu["url"]
    return None

async def image_url_to_inline_data(url: str) -> Optional[Dict[str, str]]:
    if url.startswith("data:"):
        import re
        data_match = re.match(r'^data:([^;,]+)?(;base64)?,(.*)$', url, re.DOTALL)
        if data_match:
            mime_type = data_match.group(1) or 'application/octet-stream'
            is_base64 = bool(data_match.group(2))
            payload = data_match.group(3) or ''
            if is_base64:
                data = payload
            else:
                import urllib.parse
                data = base64.b64encode(urllib.parse.unquote_to_bytes(payload)).decode('utf-8')
            return {"mimeType": mime_type, "data": data}
        return None

    if url.startswith("http://") or url.startswith("https://"):
        try:
            async with get_httpx_client('google') as client:
                res = await client.get(url)
                if res.status_code != 200:
                    return None
                buf = res.content
                if len(buf) == 0 or len(buf) > MAX_IMAGE_BYTES:
                    return None
                mime_type = res.headers.get("content-type", "").split(";")[0].strip() or "image/jpeg"
                return {"mimeType": mime_type, "data": base64.b64encode(buf).decode("utf-8")}
        except Exception:
            return None
    return None

async def user_content_to_parts(content: Any) -> List[Dict[str, Any]]:
    parts = []
    text = content_to_string(content)
    if len(text) > 0:
        parts.append({"text": text})
        
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            type_ = block.get("type")
            if type_ not in ("image_url", "image"):
                continue
            url = extract_image_url(block)
            if not url:
                continue
            inline_data = await image_url_to_inline_data(url)
            if inline_data:
                parts.append({"inlineData": inline_data})
                
    if not parts:
        parts.append({"text": ""})
    return parts

async def to_gemini_contents(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    system_messages = [
        content_to_string(m.get("content"))
        for m in messages
        if m.get("role") == "system"
    ]
    system_messages = [s for s in system_messages if len(s) > 0]
    
    tool_name_by_call_id = {}
    for m in messages:
        tool_calls = m.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                tc_id = tc.get("id")
                tc_name = tc.get("function", {}).get("name")
                if tc_id and tc_name:
                    tool_name_by_call_id[tc_id] = tc_name

    contents = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
            
        if role == "assistant":
            parts = []
            assistant_text = content_to_string(m.get("content"))
            if len(assistant_text) > 0:
                parts.append({"text": assistant_text})
                
            tool_calls = m.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    call_id = call.get("id")
                    name = call.get("function", {}).get("name")
                    raw_args = call.get("function", {}).get("arguments")
                    
                    sig = call.get("thought_signature") or recall_thought_sig(call_id)
                    part = {
                        "functionCall": {
                            "name": name,
                            "args": safe_parse_object(raw_args)
                        }
                    }
                    if call_id:
                        part["functionCall"]["id"] = call_id
                    if sig:
                        # ensure thoughtSignature is a dict to satisfy expected typing
                        if isinstance(sig, dict):
                            part["thoughtSignature"] = sig
                        else:
                            part["thoughtSignature"] = {"signature": sig}
                    parts.append(part)
                    
            if parts:
                contents.append({"role": "model", "parts": parts})
                
        elif role == "tool":
            tool_call_id = m.get("tool_call_id")
            if not tool_call_id:
                continue
            tool_name = m.get("name") or tool_name_by_call_id.get(tool_call_id) or "tool"
            response = safe_parse_object(content_to_string(m.get("content")))
            
            part = {
                "functionResponse": {
                    "id": tool_call_id,
                    "name": tool_name,
                    "response": response
                }
            }
            contents.append({"role": "user", "parts": [part]})
            
        else:
            parts = await user_content_to_parts(m.get("content"))
            contents.append({"role": "user", "parts": parts})
            
    return {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": "\n\n".join(system_messages)}]
        } if system_messages else None
    }

def extract_tool_calls(parts: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    calls = []
    if not parts:
        return calls
    fallback_index = 0
    for part in parts:
        fc = part.get("functionCall")
        if not fc or not fc.get("name"):
            continue
        id_ = fc.get("id") or f"call_{int(time.time() * 1000)}_{fallback_index}"
        fallback_index += 1
        sig = part.get("thoughtSignature")
        remember_thought_sig(id_, sig)
        call = {
            "id": id_,
            "type": "function",
            "function": {
                "name": fc["name"],
                "arguments": normalize_gemini_args(fc.get("args"))
            }
        }
        if sig:
            call["thought_signature"] = sig
        calls.append(call)
    return calls

def extract_text(parts: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    if not parts:
        return None
    text_parts = [p.get("text") or "" for p in parts]
    joined = "".join(text_parts)
    return joined if len(joined) > 0 else None

class GoogleProvider(BaseProvider):
    @property
    def platform(self) -> str:
        return "google"

    @property
    def name(self) -> str:
        return "Google AI Studio"

    async def chat_completion(
        self,
        api_key: str,
        messages: List[Dict[str, Any]],
        model_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        gemini_payload = await to_gemini_contents(messages)
        tools = to_gemini_tools(options.get("tools") if options else None)
        
        body = {
            "contents": gemini_payload["contents"],
            "generationConfig": {
                "temperature": options.get("temperature") if options else None,
                "maxOutputTokens": options.get("max_tokens") if options else None,
                "topP": options.get("top_p") if options else None,
            },
            "tools": tools,
        }
        if has_function_declarations(tools):
            body["toolConfig"] = to_gemini_tool_config(options.get("tool_choice") if options else None)
        if gemini_payload.get("systemInstruction"):
            body["systemInstruction"] = gemini_payload["systemInstruction"]

        url = f"{API_BASE}/models/{model_id}:generateContent?key={api_key}"
        timeout = 15000.0
        if options:
            if options.get("timeoutMs") is not None:
                timeout = options["timeoutMs"]
            elif options.get("timeout_ms") is not None:
                timeout = options["timeout_ms"]
        
        async with self.get_client(timeout) as client:
            try:
                res = await client.post(url, json=body, headers={"Content-Type": "application/json"})
            except Exception as e:
                raise Exception(f"Google AI Studio connection error: {str(e)}") from e
                
            if res.status_code != 200:
                try:
                    err = res.json()
                except Exception:
                    err = {}
                raise make_provider_http_error(res, f"Google API error {res.status_code}: {err.get('error', {}).get('message') or res.reason_phrase}")
                
            data = res.json()
            candidates = data.get("candidates")
            candidate = candidates[0] if candidates else {}
            parts = candidate.get("content", {}).get("parts")
            tool_calls = extract_tool_calls(parts)
            text = extract_text(parts)
            
            usage_metadata = data.get("usageMetadata", {})
            usage = {
                "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                "total_tokens": usage_metadata.get("totalTokenCount", 0),
            }
            
            choice_message = {
                "role": "assistant",
                "content": text
            }
            if tool_calls:
                choice_message["tool_calls"] = tool_calls
                
            return {
                "id": self.make_id(),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": choice_message,
                    "finish_reason": "tool_calls" if tool_calls else to_gemini_finish_reason(candidate.get("finishReason"))
                }],
                "usage": usage,
                "_routed_via": {"platform": "google", "model": model_id}
            }

    async def stream_chat_completion(
        self,
        api_key: str,
        messages: List[Dict[str, Any]],
        model_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        async def _stream() -> AsyncGenerator[Dict[str, Any], None]:
            gemini_payload = await to_gemini_contents(messages)
            tools = to_gemini_tools(options.get("tools") if options else None)
            body = {
                "contents": gemini_payload["contents"],
                "generationConfig": {
                    "temperature": options.get("temperature") if options else None,
                    "maxOutputTokens": options.get("max_tokens") if options else None,
                    "topP": options.get("top_p") if options else None,
                },
                "tools": tools,
            }
            if has_function_declarations(tools):
                body["toolConfig"] = to_gemini_tool_config(options.get("tool_choice") if options else None)
            if gemini_payload.get("systemInstruction"):
                body["systemInstruction"] = gemini_payload["systemInstruction"]

            url = f"{API_BASE}/models/{model_id}:streamGenerateContent?alt=sse&key={api_key}"
            timeout = 15000.0
            if options:
                if options.get("timeoutMs") is not None:
                    timeout = options["timeoutMs"]
                elif options.get("timeout_ms") is not None:
                    timeout = options["timeout_ms"]

            id_ = self.make_id()
            emitted_finish = False
            saw_tool_calls = False
            seen_tool_call_keys = set()

            async with self.get_client(timeout) as client:
                try:
                    async with client.stream("POST", url, json=body, headers={"Content-Type": "application/json"}) as res:
                        if res.status_code != 200:
                            await res.aread()
                            try:
                                err = res.json()
                            except Exception:
                                err = {}
                            raise make_provider_http_error(res, f"Google API error {res.status_code}: {err.get('error', {}).get('message') or res.reason_phrase}")

                        async for line in res.aiter_lines():
                            trimmed = line.strip()
                            if not trimmed or not trimmed.startswith("data: "):
                                continue
                            raw = trimmed[6:]
                            if raw == "[DONE]":
                                if not emitted_finish:
                                    emitted_finish = True
                                    yield {
                                        "id": id_,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model_id,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": "tool_calls" if saw_tool_calls else "stop"
                                        }]
                                    }
                                return

                            try:
                                chunk = json.loads(raw)
                            except Exception:
                                continue

                            candidates = chunk.get("candidates", [])
                            candidate = candidates[0] if candidates else {}
                            parts = candidate.get("content", {}).get("parts") or []

                            text = extract_text(parts)
                            tool_calls = []
                            for call in extract_tool_calls(parts):
                                key = f"{call['id']}:{call['function']['name']}:{call['function']['arguments']}"
                                if key not in seen_tool_call_keys:
                                    seen_tool_call_keys.add(key)
                                    tool_calls.append(call)

                            if (text and len(text) > 0) or tool_calls:
                                saw_tool_calls = saw_tool_calls or bool(tool_calls)
                                delta = {}
                                if text:
                                    delta["content"] = text
                                if tool_calls:
                                    delta["tool_calls"] = tool_calls
                                yield {
                                    "id": id_,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model_id,
                                    "choices": [{
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": None
                                    }]
                                }

                            if candidate.get("finishReason") and not emitted_finish:
                                emitted_finish = True
                                yield {
                                    "id": id_,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model_id,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "tool_calls" if saw_tool_calls else to_gemini_finish_reason(candidate["finishReason"])
                                    }]
                                }
                                return
                except Exception as e:
                    if isinstance(e, ProviderHttpError):
                        raise
                    raise Exception(f"Google AI Studio stream connection error: {str(e)}") from e

            if not emitted_finish:
                yield {
                    "id": id_,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if saw_tool_calls else "stop"
                    }]
                }

        return _stream()

    async def validate_key(self, api_key: str) -> bool:
        url = f"{API_BASE}/models?key={api_key}"
        async with self.get_client(10000.0) as client:
            res = await client.get(url)
            if res.status_code == 200:
                return True
            
            body = {}
            try:
                body = res.json()
            except Exception:
                pass
                
            err = body.get("error") or {}
            details = err.get("details") or []
            reason = None
            if isinstance(details, list):
                for d in details:
                    if isinstance(d, dict) and "reason" in d:
                        reason = d["reason"]
                        break
            message = err.get("message") or ""
            g_status = err.get("status")
            
            bad_credentials = (
                res.status_code == 401 or
                reason == 'API_KEY_INVALID' or
                any(x in message for x in ('API key not valid', 'API key expired', 'API_KEY_INVALID'))
            )
            
            if bad_credentials:
                print(f"[Google] validateKey: key rejected as invalid (HTTP {res.status_code}{f' {reason}' if reason else ''})")
                return False
                
            print(
                f"[Google] validateKey: inconclusive HTTP {res.status_code} ({g_status or 'UNKNOWN'}{f'/{reason}' if reason else ''}): {message[:200]} "
                f"— treating as 'error', not auto-disabling (the key may be valid but blocked by region/permission/restriction on this host)."
            )
            raise Exception(f"Google key validation inconclusive (HTTP {res.status_code}{f' {g_status}' if g_status else ''})")
