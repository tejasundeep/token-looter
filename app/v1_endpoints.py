import os
import time
import datetime
import json
import re
import math
import secrets
import asyncio
import threading
from typing import Optional, List, Dict, Any, Union, Set
from fastapi import APIRouter, Depends, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from app.database import get_db, get_unified_api_key

from app.ratelimit import (
    can_make_request, can_use_tokens, is_on_cooldown, can_use_provider,
    record_request, record_tokens, set_cooldown, get_cooldown_duration_for_limit,
    PAYMENT_REQUIRED_COOLDOWN_MS,
    increment_in_flight, decrement_in_flight
)
from app.router import (
    route_request, resolve_routing_chain, record_rate_limit_hit, record_success,
    has_enabled_vision_model, has_enabled_tools_model, get_active_chain, RouteError,
    get_model_by_id, MODELS_DATA
)
from app.lib.content import content_to_string, message_has_image, normalize_outbound_content
from app.lib.tool_args import repair_tool_arguments, tool_schema_map
from app.lib.error_redaction import sanitize_provider_error_message
from app.lib.tool_call_rescue import (
    rescue_inline_tool_calls, starts_with_dialect_marker,
    could_become_dialect_marker, contains_dialect_marker
)
from app.context_handoff import get_context_handoff_mode, record_incoming_messages, maybe_inject_context_handoff, record_successful_model, has_prior_model, HANDOFF_MAX_TOKENS
from app.embeddings import run_embeddings

v1_router = APIRouter(prefix="/v1")

MAX_RETRIES = 20
MODEL_FORBIDDEN_COOLDOWN_MS = 24 * 60 * 60 * 1000  # 24 hours

_sticky_lock = threading.Lock()
_sticky_session_map = {}
STICKY_TTL_MS = 30 * 60 * 1000

_prune_lock = threading.Lock()
_next_prune_at_ms = 0

# ── General Helpers ─────────────────────────────────────────────────────────

def extract_api_token(request: Request) -> Optional[str]:
    bearer = request.headers.get("Authorization")
    if bearer and bearer.lower().startswith("bearer "):
        return bearer[7:].strip()
    x_api_key = request.headers.get("x-api-key")
    if x_api_key:
        return x_api_key.strip()
    return None

def timing_safe_string_equal(provided: str, expected: str) -> bool:
    import hmac
    return hmac.compare_digest(provided, expected)

def require_unified_key(request: Request) -> str:
    token = extract_api_token(request)
    unified_key = get_unified_api_key()
    if not token or not timing_safe_string_equal(token, unified_key):
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid API key", "type": "authentication_error"}}
        )
    return token

def is_auto_model(model_id: Optional[str]) -> bool:
    if not model_id:
        return True
    lower = model_id.lower()
    return lower == 'auto' or lower.startswith('auto:')

def get_session_key(messages: List[Dict[str, Any]], session_id_header: Optional[str] = None, strategy_key: Optional[str] = None) -> str:
    if session_id_header:
        return f"hdr:{session_id_header}::{strategy_key}" if strategy_key else f"hdr:{session_id_header}"
    first_user = next((m for m in messages if m.get("role") == 'user'), None)
    if not first_user:
        return ""
    text = content_to_string(first_user.get("content") or "")
    if not text:
        return ""
    payload = f"{text}::{strategy_key}" if strategy_key else text
    import hashlib
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()

def get_sticky_model(messages: List[Dict[str, Any]], session_id_header: Optional[str] = None, strategy_key: Optional[str] = None) -> Optional[int]:
    has_assistant = any(m.get("role") == 'assistant' for m in messages)
    if not has_assistant:
        return None
    key = get_session_key(messages, session_id_header, strategy_key)
    if not key:
        return None
    now = time.time() * 1000
    with _sticky_lock:
        entry = _sticky_session_map.get(key)
        if not entry:
            return None
        if now - entry["lastUsed"] > STICKY_TTL_MS:
            _sticky_session_map.pop(key, None)
            return None
        return entry["modelDbId"]

def set_sticky_model(messages: List[Dict[str, Any]], model_db_id: int, session_id_header: Optional[str] = None, strategy_key: Optional[str] = None):
    key = get_session_key(messages, session_id_header, strategy_key)
    if not key:
        return
    now = time.time() * 1000
    with _sticky_lock:
        _sticky_session_map[key] = {"modelDbId": model_db_id, "lastUsed": now}
        if len(_sticky_session_map) > 500:
            for k in list(_sticky_session_map.keys()):
                if now - _sticky_session_map[k]["lastUsed"] > STICKY_TTL_MS:
                    _sticky_session_map.pop(k, None)

def get_retention_days() -> int:
    raw = os.environ.get("REQUEST_ANALYTICS_RETENTION_DAYS")
    if raw is None or raw.strip() == "":
        return 90
    try:
        return max(0, int(raw))
    except ValueError:
        return 90

def get_retention_max_rows() -> int:
    raw = os.environ.get("REQUEST_ANALYTICS_MAX_ROWS")
    if raw is None or raw.strip() == "":
        return 100000
    try:
        return max(0, int(raw))
    except ValueError:
        return 100000

def log_request(
    platform: str,
    model_id: str,
    key_id: Optional[int],
    status: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    error: Optional[str] = None,
    ttfb_ms: Optional[int] = None,
    requested_model: Optional[str] = None,
    request_type: str = "chat"
):
    global _next_prune_at_ms
    db = get_db()
    try:
        db.execute("""
            INSERT INTO requests (platform, model_id, key_id, status, input_tokens, output_tokens, latency_ms, error, ttfb_ms, requested_model, request_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (platform, model_id, key_id, status, input_tokens, output_tokens, latency_ms, error, ttfb_ms, requested_model, request_type))
        db.commit()
    except Exception as e:
        print(f"Failed to log request: {e}")
        return

    now_ms = time.time() * 1000.0
    with _prune_lock:
        if now_ms < _next_prune_at_ms:
            return
        _next_prune_at_ms = now_ms + 60000.0

    try:
        ret_days = get_retention_days()
        max_rows = get_retention_max_rows()
        if ret_days > 0:
            cutoff_sec = time.time() - (ret_days * 24 * 60 * 60)
            cutoff_str = datetime.datetime.fromtimestamp(cutoff_sec, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            db.execute("DELETE FROM requests WHERE created_at < ?", (cutoff_str,))
        if max_rows > 0:
            db.execute("""
                DELETE FROM requests
                WHERE id IN (
                    SELECT id
                    FROM requests
                    ORDER BY created_at DESC, id DESC
                    LIMIT -1 OFFSET ?
                )
            """, (max_rows,))
        
        # Prune expired key states
        now_ms = time.time() * 1000.0
        db.execute("DELETE FROM key_states WHERE expires_at < ?", (now_ms,))
        db.commit()
    except Exception as e:
        print(f"Failed to prune request analytics: {e}")

def is_model_access_forbidden_error(err: Exception) -> bool:
    status = getattr(err, "status", None)
    if status == 403:
        return True
    msg = str(err).lower()
    return "403" in msg or "forbidden" in msg

def is_payment_required_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "402" in msg or 
        "payment required" in msg or 
        "insufficient_quota" in msg or 
        "insufficient credit" in msg or 
        "insufficient balance" in msg or
        "quota exceeded" in msg or
        "billing" in msg or
        "insufficient funds" in msg
    )


def is_model_not_found_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "404" in msg or "not found" in msg or "no endpoints found" in msg

def is_retryable_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "429" in msg or "rate limit" in msg or "too many requests" in msg
        or "quota" in msg or "resource_exhausted" in msg
        or "aborted" in msg or "timeout" in msg or "etimedout" in msg
        or "econnrefused" in msg or "econnreset" in msg
        or "fetch failed" in msg
        or "503" in msg or "unavailable" in msg
        or "500" in msg or "internal server error" in msg
        or "413" in msg or "payload too large" in msg or "request body too large" in msg
        or "request entity too large" in msg or "content too large" in msg
        or "404" in msg or "not found" in msg or "no endpoints found" in msg
        or "401" in msg or "unauthorized" in msg or "invalid api key" in msg or "invalid_key" in msg
        or is_model_access_forbidden_error(err)
        or "api error 400" in msg
        or is_payment_required_error(err)
        or "empty completion" in msg
        or "in-band provider error" in msg
        or "stream ended unexpectedly" in msg
        or "stream stalled" in msg
        or "unparseable inline tool-call dialect" in msg
    )

# ── endpoint: GET /v1/models ───────────────────────────────────────────────

@v1_router.get("/models")
def get_v1_models(request: Request, available: Optional[str] = None, connected: Optional[str] = None):
    require_unified_key(request)
    
    from app.providers import get_provider
    from app.database import get_provider_keys

    processed_models = []
    for idx, m in enumerate(MODELS_DATA):
        platform = m.get("platform")
        enabled = m.get("enabled", 1) == 1
        
        # Check provider keys configured in keys.json
        keys = get_provider_keys(platform)
        provider = get_provider(platform)
        is_keyless = getattr(provider, "keyless", False) if provider else False
        
        has_keys = len(keys) > 0 or is_keyless
        is_available = enabled and has_keys
        
        processed_models.append({
            "id": m.get("id", idx),
            "platform": platform,
            "model_id": m.get("model_id"),
            "display_name": m.get("display_name"),
            "context_window": m.get("context_window"),
            "intelligence_rank": m.get("intelligence_rank", 1),
            "enabled": enabled,
            "available": is_available
        })
        
    grouped = {}
    for pm in processed_models:
        mid = pm["model_id"]
        if mid not in grouped:
            grouped[mid] = pm
        else:
            existing = grouped[mid]
            keep_new = False
            if pm["available"] and not existing["available"]:
                keep_new = True
            elif pm["available"] == existing["available"]:
                if pm["intelligence_rank"] < existing["intelligence_rank"]:
                    keep_new = True
                elif pm["intelligence_rank"] == existing["intelligence_rank"] and pm["id"] < existing["id"]:
                    keep_new = True
            if keep_new:
                grouped[mid] = pm
                
    deduped_models = list(grouped.values())
    
    deduped_models.sort(key=lambda x: (
        not x["available"],
        not x["enabled"],
        x["intelligence_rank"],
        x["id"]
    ))
    
    q = str(available or connected or '').lower()
    only_available = q in ('1', 'true', 'yes')
    listed = [m for m in deduped_models if m["available"]] if only_available else deduped_models
    
    available_context_windows = [m["context_window"] for m in deduped_models if m["available"] and m["context_window"] is not None]
    auto_context_window = max(available_context_windows) if available_context_windows else None
    
    return {
        "object": "list",
        "data": [
            {
                "id": "auto",
                "object": "model",
                "created": 0,
                "owned_by": "tokenlooter",
                "name": "Auto (router picks the best available model)",
                "context_window": auto_context_window,
                "context_length": auto_context_window,
                "available": True,
                "unavailable_reason": None
            }
        ] + [
            {
                "id": m["model_id"],
                "object": "model",
                "created": 0,
                "owned_by": m["platform"],
                "name": m["display_name"],
                "context_window": m["context_window"],
                "context_length": m["context_window"],
                "available": m["available"],
                "unavailable_reason": None if m["available"] else ("no_key" if m["enabled"] else "disabled")
            } for m in listed
        ]
    }

# ── endpoint: POST /v1/embeddings ───────────────────────────────────────────

class EmbeddingsBody(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]

@v1_router.post("/embeddings")
async def post_v1_embeddings(request: Request, body: EmbeddingsBody):
    require_unified_key(request)
    inputs = [body.input] if isinstance(body.input, str) else body.input
    try:
        result = await run_embeddings(body.model, inputs)
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": values} for i, values in enumerate(result["vectors"])],
            "model": result["family"],
            "provider": result["platform"],
            "usage": {"prompt_tokens": result["inputTokens"], "total_tokens": result["inputTokens"]}
        }
    except Exception as err:
        status_code = getattr(err, "status", 502)
        err_type = "invalid_request_error" if status_code == 400 else ("rate_limit_error" if status_code == 429 else "server_error")
        raise HTTPException(
            status_code=status_code,
            detail={"error": {"message": f"embedding error: {str(err)}", "type": err_type}}
        )

# ── endpoint: POST /v1/chat/completions ──────────────────────────────────────

class ToolFunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None

class ToolDefinition(BaseModel):
    type: Optional[str] = 'function'
    function: ToolFunctionDefinition

class ToolChoiceFunction(BaseModel):
    name: str

class ToolChoiceObject(BaseModel):
    type: str = 'function'
    function: ToolChoiceFunction

class ChatCompletionRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., min_length=1)
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, ToolChoiceObject]] = None
    parallel_tool_calls: Optional[bool] = None

    class Config:
        extra = "ignore"

async def get_next_chunk_safely(gen, request: Optional[Request]) -> Any:
    if not request:
        return await gen.__anext__()

    chunk_task = asyncio.create_task(gen.__anext__())
    
    async def watch_disconnect():
        try:
            while not chunk_task.done():
                if await request.is_disconnected():
                    chunk_task.cancel()
                    break
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            pass

    watch_task = asyncio.create_task(watch_disconnect())
    
    try:
        await asyncio.wait([chunk_task, watch_task], return_when=asyncio.FIRST_COMPLETED)
        if chunk_task.done() and not chunk_task.cancelled():
            return chunk_task.result()
        else:
            raise asyncio.CancelledError("Client disconnected")
    finally:
        if not chunk_task.done():
            chunk_task.cancel()
        if not watch_task.done():
            watch_task.cancel()

async def stream_generator(
    first_chunk: Dict[str, Any],
    gen,
    route: Dict[str, Any],
    estimated_input_tokens: int,
    injected_handoff_tokens: int,
    session_key: str,
    handoff_mode: str,
    model_key: str,
    pinned_model_id: Optional[str],
    messages: List[Dict[str, Any]],
    sessionIdHeader: Optional[str],
    strategyKey: Optional[str],
    tools: Optional[List[Dict[str, Any]]],
    start_time: float,
    request: Optional[Request] = None,
    reserved_tokens: int = 0
):
    total_output_tokens = 0
    ttfb_ms = None
    
    mode = 'undecided'
    held_text = ''
    preamble = []
    tool_call_acc = {}
    upstream_finish = None
    usage_chunk = None
    last_meta = {}
    
    chunks = [first_chunk]
    
    def format_sse(data: Any) -> str:
        return f"data: {json.dumps(data)}\n\n"
        
    try:
        idx = 0
        while True:
            if request and await request.is_disconnected():
                print(f"[Proxy] Client disconnected. Stopping stream generator.")
                return

            if idx < len(chunks):
                chunk = chunks[idx]
                idx += 1
            else:
                try:
                    chunk = await get_next_chunk_safely(gen, request)
                except (StopAsyncIteration, asyncio.CancelledError):
                    break
                    
            any_chunk = chunk
            
            if "error" in any_chunk and "choices" not in any_chunk:
                msg = any_chunk["error"].get("message") or json.dumps(any_chunk["error"])[:200]
                safe_msg = sanitize_provider_error_message(str(msg))
                yield format_sse({
                    "error": {
                        "message": f"Provider error ({route['displayName']}): {safe_msg}",
                        "type": "stream_error"
                    }
                })
                yield "data: [DONE]\n\n"
                
                latency_ms = int((time.time() - start_time) * 1000)
                log_request(route["platform"], route["modelId"], route["keyId"], 'error', 
                            estimated_input_tokens, total_output_tokens, latency_ms, 
                            f"in-band error frame: {safe_msg}", ttfb_ms, pinned_model_id)
                return
                
            if any_chunk.get("id"):
                last_meta = {
                    "id": any_chunk["id"],
                    "model": any_chunk.get("model"),
                    "created": any_chunk.get("created")
                }
                
            choices = any_chunk.get("choices", [])
            if not choices:
                if any_chunk.get("usage"):
                    usage_chunk = any_chunk
                continue
                
            choice = choices[0]
            if choice.get("finish_reason"):
                upstream_finish = choice["finish_reason"]
                
            delta = choice.get("delta", {})
            for tc in delta.get("tool_calls", []):
                tc_idx = tc.get("index", 0)
                if tc_idx not in tool_call_acc:
                    tool_call_acc[tc_idx] = {"id": None, "name": "", "args": ""}
                acc = tool_call_acc[tc_idx]
                if tc.get("id") and not acc["id"]:
                    acc["id"] = tc["id"]
                tc_func = tc.get("function", {})
                if tc_func.get("name"):
                    acc["name"] += tc_func["name"]
                if tc_func.get("arguments"):
                    acc["args"] += tc_func["arguments"]
                    
            normalize_outbound_content(any_chunk)
            text = delta.get("content") or ""
            if not isinstance(text, str):
                text = ""
                
            if len(text) == 0:
                if delta and any(k != 'content' and k != 'tool_calls' and delta.get(k) is not None for k in delta.keys()):
                    cleaned_delta = {k: v for k, v in delta.items() if k != 'tool_calls'}
                    cleaned_choice = {**choice, "delta": cleaned_delta, "finish_reason": None}
                    cleaned_chunk = {**any_chunk, "choices": [cleaned_choice]}
                    if ttfb_ms is not None:
                        yield format_sse(cleaned_chunk)
                    else:
                        preamble.append(cleaned_chunk)
                continue
                
            total_output_tokens += math.ceil(len(text) / 4)
            
            if mode == 'passthrough':
                cleaned_delta = {k: v for k, v in delta.items() if k != 'tool_calls'}
                cleaned_choice = {**choice, "delta": cleaned_delta, "finish_reason": None}
                cleaned_chunk = {**any_chunk, "choices": [cleaned_choice]}
                yield format_sse(cleaned_chunk)
                continue
                
            held_text += text
            if mode == 'dialect':
                continue
                
            probe = held_text.lstrip()
            if starts_with_dialect_marker(probe):
                mode = 'dialect'
            elif not could_become_dialect_marker(probe) or len(probe) > 256:
                mode = 'passthrough'
                if ttfb_ms is None:
                    ttfb_ms = int((time.time() - start_time) * 1000)
                for p in preamble:
                    yield format_sse(p)
                preamble.clear()
                
                def_id = last_meta.get("id") or f"chatcmpl-{int(time.time()*1000)}"
                created_t = last_meta.get("created") or int(time.time())
                yield format_sse({
                    "id": def_id,
                    "object": "chat.completion.chunk",
                    "created": created_t,
                    "model": last_meta.get("model") or route["modelId"],
                    "choices": [{"index": 0, "delta": {"content": held_text}, "finish_reason": None}]
                })
                held_text = ''
                
        schemas = tool_schema_map(tools)
        synthetic_stream_ids = 0
        completed_calls = []
        for tc_idx in sorted(tool_call_acc.keys()):
            acc = tool_call_acc[tc_idx]
            tc_id = acc["id"] if acc["id"] and len(acc["id"]) > 0 else f"call_stream_{synthetic_stream_ids + 1}"
            if not acc["id"]:
                synthetic_stream_ids += 1
            repaired_args = repair_tool_arguments(acc["args"] or "{}", schemas.get(acc["name"]))
            
            try:
                json.loads(repaired_args)
                is_valid = len(acc["name"]) > 0
            except Exception:
                is_valid = False
                
            if is_valid:
                completed_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": acc["name"],
                        "arguments": repaired_args
                    }
                })
                
        if mode == 'dialect' or (mode == 'undecided' and len(held_text) > 0 and contains_dialect_marker(held_text)):
            rescue = rescue_inline_tool_calls(held_text, set(t.function.name for t in tools) if tools else set())
            if rescue.get("detected"):
                if not rescue.get("calls"):
                    raise Exception(f"unparseable inline tool-call dialect from {route['displayName']}: {held_text[:120]}")
                rescued_ids = 0
                for c in rescue["calls"]:
                    rescued_ids += 1
                    completed_calls.append({
                        "id": f"call_rescued_{rescued_ids}",
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": repair_tool_arguments(c["arguments"], schemas.get(c["name"]))
                        }
                    })
                held_text = rescue.get("cleanText", "")
                print(f"[Proxy] Rescued {rescued_ids} inline tool call(s) from {route['displayName']} into structured tool_calls")
                
        has_text = (ttfb_ms is not None) or (len(held_text.strip()) > 0)
        if not has_text and len(completed_calls) == 0:
            raise Exception(f"empty completion from {route['displayName']} (stream produced no content and no tool calls)")
            
        if ttfb_ms is None:
            ttfb_ms = int((time.time() - start_time) * 1000)
            for p in preamble:
                yield format_sse(p)
            preamble.clear()
            
        def_id = last_meta.get("id") or f"chatcmpl-{int(time.time()*1000)}"
        created_t = last_meta.get("created") or int(time.time())
        model_name = last_meta.get("model") or route["modelId"]
        
        if len(held_text) > 0:
            yield format_sse({
                "id": def_id,
                "object": "chat.completion.chunk",
                "created": created_t,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": held_text}, "finish_reason": None}]
            })
            
        if len(completed_calls) > 0:
            yield format_sse({
                "id": def_id,
                "object": "chat.completion.chunk",
                "created": created_t,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, **c} for i, c in enumerate(completed_calls)]}, "finish_reason": None}]
            })
            total_output_tokens += math.ceil(sum(len(c["function"]["arguments"]) for c in completed_calls) / 4)
            
        finish = 'tool_calls' if len(completed_calls) > 0 else (upstream_finish if (upstream_finish and upstream_finish != 'tool_calls') else 'stop')
        yield format_sse({
            "id": def_id,
            "object": "chat.completion.chunk",
            "created": created_t,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]
        })
        
        if usage_chunk:
            yield format_sse(usage_chunk)
            
        yield "data: [DONE]\n\n"
        
        record_request(route["platform"], route["modelId"], route["keyId"])
        record_tokens(route["platform"], route["modelId"], route["keyId"], estimated_input_tokens + injected_handoff_tokens + total_output_tokens)
        record_success(route["modelDbId"])
        set_sticky_model(messages, route["modelDbId"], sessionIdHeader, strategyKey)
        if handoff_mode != 'off' and session_key:
            record_successful_model(session_key=session_key, model_key=model_key)
            
        latency_ms = int((time.time() - start_time) * 1000)
        log_request(route["platform"], route["modelId"], route["keyId"], 'success',
                    estimated_input_tokens + injected_handoff_tokens, total_output_tokens,
                    latency_ms, None, ttfb_ms, pinned_model_id)
                    
    except Exception as stream_err:
        print(f"[Proxy] Mid-stream error from {route['displayName']}: {str(stream_err)}")
        payload = {
            "error": {
                "message": f"Provider error ({route['displayName']}): stream interrupted: {str(stream_err)}",
                "type": "stream_error"
            }
        }
        yield format_sse(payload)
        yield "data: [DONE]\n\n"
        latency_ms = int((time.time() - start_time) * 1000)
        log_request(route["platform"], route["modelId"], route["keyId"], 'error',
                    estimated_input_tokens, total_output_tokens, latency_ms,
                    sanitize_provider_error_message(str(stream_err)), ttfb_ms, pinned_model_id)
    finally:
        decrement_in_flight(route["platform"], route["modelId"], route["keyId"], reserved_tokens)

@v1_router.post("/chat/completions")
async def post_v1_chat_completions(request: Request):
    start_time = time.time()
    require_unified_key(request)
    
    try:
        body_json = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"error": {"message": "Invalid request: invalid JSON body", "type": "invalid_request_error"}})
        
    try:
        body = ChatCompletionRequest.model_validate(body_json)
    except ValidationError as val_err:
        errors = []
        for err in val_err.errors():
            loc_str = ".".join(str(p) for p in err.get("loc", []))
            errors.append(f"{loc_str}: {err['msg']}")
        detail = ", ".join(errors[:5])
        print(f"[proxy] 400 invalid /chat/completions request: {detail}")
        raise HTTPException(status_code=400, detail={"error": {"message": f"Invalid request: {detail}", "type": "invalid_request_error"}})
    except Exception as val_err:
        detail = str(val_err)
        print(f"[proxy] 400 invalid /chat/completions request: {detail}")
        print(f"[proxy] 400 invalid /chat/completions request: {detail}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Invalid request: {detail}",
                    "type": "invalid_request_error"
                }
            }
        )
        
    # Message normalization
    pending_tool_call_ids = []
    synthetic_id_counter = 0
    
    def take_tool_call_id(given: Optional[str]) -> str:
        nonlocal synthetic_id_counter
        if given and len(given) > 0:
            if given in pending_tool_call_ids:
                pending_tool_call_ids.remove(given)
            return given
        if pending_tool_call_ids:
            return pending_tool_call_ids.pop(0)
        synthetic_id_counter += 1
        return f"call_auto_{synthetic_id_counter}"
        
    norm_messages = []
    for m in body.messages:
        role = m.get("role")
        content = m.get("content")
        name = m.get("name")
        
        if role == 'assistant':
            tool_calls = m.get("tool_calls")
            has_tool_calls = bool(tool_calls and len(tool_calls) > 0)
            
            is_empty_content = (content is None) or (isinstance(content, str) and len(content) == 0) or (isinstance(content, list) and len(content) == 0)
            assistant_content = None if has_tool_calls else ""
            if not is_empty_content:
                assistant_content = content
                
            msg = {
                "role": "assistant",
                "content": assistant_content
            }
            if name:
                msg["name"] = name
            reasoning_content = m.get("reasoning_content")
            if isinstance(reasoning_content, str) and len(reasoning_content) > 0:
                msg["reasoning_content"] = reasoning_content
                
            if has_tool_calls:
                norm_tcs = []
                for tc in tool_calls:
                    tc_id = tc.get("id")
                    if not tc_id or len(tc_id) == 0:
                        synthetic_id_counter += 1
                        tc_id = f"call_auto_{synthetic_id_counter}"
                    pending_tool_call_ids.append(tc_id)
                    
                    tc_func = tc.get("function", {})
                    tc_args = tc_func.get("arguments", "{}")
                    if not isinstance(tc_args, str):
                        tc_args = json.dumps(tc_args)
                        
                    tc_norm = {
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": tc_func.get("name"),
                            "arguments": tc_args
                        }
                    }
                    if "thought_signature" in tc:
                        tc_norm["thought_signature"] = tc["thought_signature"]
                    norm_tcs.append(tc_norm)
                msg["tool_calls"] = norm_tcs
            norm_messages.append(msg)
            
        elif role == 'tool':
            norm_messages.append({
                "role": "tool",
                "content": content if content is not None else "",
                "tool_call_id": take_tool_call_id(m.get("tool_call_id")),
                **({"name": name} if name else {})
            })
            
        elif role == 'function':
            norm_messages.append({
                "role": "tool",
                "content": content if content is not None else "",
                "tool_call_id": take_tool_call_id(None),
                "name": name
            })
            
        else:
            role_to_use = "system" if role == "developer" else role
            norm_messages.append({
                "role": role_to_use,
                "content": content,
                **({"name": name} if name else {})
            })
            
    # Token estimation
    estimated_input_tokens = sum(math.ceil(len(content_to_string(m.get("content") or "")) / 4) for m in norm_messages)
    has_image = message_has_image(norm_messages)
    if has_image and not has_enabled_vision_model():
        raise HTTPException(status_code=422, detail={
            "error": {
                "message": "This request includes an image, but no vision-capable model is enabled. Enable a vision model in the Fallback Chain.",
                "type": "invalid_request_error",
                "code": "no_vision_model"
            }
        })
        
    IMAGE_TOKEN_ESTIMATE = 1000
    image_count = 0
    for m in norm_messages:
        content = m.get("content")
        if isinstance(content, list):
            image_count += sum(1 for b in content if isinstance(b, dict) and b.get("type") in ('image_url', 'image'))
            
    max_tokens = body.max_tokens if (body.max_tokens is not None and body.max_tokens > 0) else None
    estimated_total = estimated_input_tokens + image_count * IMAGE_TOKEN_ESTIMATE + (max_tokens or 1000)
    
    wants_tools = bool(body.tools and len(body.tools) > 0)
    if wants_tools and not has_enabled_tools_model():
        raise HTTPException(status_code=422, detail={
            "error": {
                "message": "This request includes tools, but no tool-capable model is enabled. Enable a tool-calling model in the Fallback Chain.",
                "type": "invalid_request_error",
                "code": "no_tools_model"
            }
        })
        
    raw_session_id = request.headers.get("x-session-id")
    sessionIdHeader = raw_session_id.strip() if raw_session_id else None
    
    resolved_chain = None
    strategy_key = None
    if is_auto_model(body.model):
        resolved_chain = resolve_routing_chain(body.model)
        strategy_key = resolved_chain.get("strategyKey")
        
    is_auto_routed = not body.model or is_auto_model(body.model)
    handoff_mode = get_context_handoff_mode() if is_auto_routed else "off"
    session_key = get_session_key(norm_messages, sessionIdHeader, strategy_key) if handoff_mode != "off" else ""
    if handoff_mode != "off" and session_key:
        record_incoming_messages(session_key, norm_messages)
        
    handoff_possible = handoff_mode != "off" and bool(session_key) and has_prior_model(session_key)
    
    preferred_model = None
    if is_auto_model(body.model):
        preferred_model = get_sticky_model(norm_messages, sessionIdHeader, strategy_key)
    elif body.model:
        model_meta = get_model_by_id(body.model)
        if model_meta and model_meta.get("enabled", 1) == 1:
            preferred_model = None
            for idx, m in enumerate(MODELS_DATA):
                if m.get("model_id") == body.model:
                    preferred_model = m.get("id", idx)
                    break
        else:
            reason = "is disabled" if model_meta else "is not in the catalog"
            raise HTTPException(status_code=400, detail={
                "error": {
                    "message": f"Model '{body.model}' {reason}. Use 'auto' to auto-route, or call /v1/models for the available list.",
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            })
    else:
        preferred_model = get_sticky_model(norm_messages, sessionIdHeader, strategy_key)
        
    pinned_model_id = body.model if (body.model and not is_auto_model(body.model)) else None
    
    skip_keys = set()
    skip_models = set()
    last_error = None
    
    tools_dict_list = [t.model_dump() for t in body.tools] if body.tools else None
    tool_choice_val = body.tool_choice
    if isinstance(tool_choice_val, ToolChoiceObject):
        tool_choice_val = tool_choice_val.model_dump()
        
    for attempt in range(MAX_RETRIES):
        try:
            routing_estimate = estimated_total + HANDOFF_MAX_TOKENS if handoff_possible else estimated_total
            route = route_request(
                estimated_tokens=routing_estimate,
                skip_keys=skip_keys if skip_keys else None,
                preferred_model_db_id=preferred_model,
                require_vision=has_image,
                require_tools=wants_tools,
                skip_models=skip_models if skip_models else None,
                prefetched_chain=resolved_chain.get("chain") if resolved_chain else None
            )
        except RouteError as re_err:
            if last_error:
                safe_last_error = sanitize_provider_error_message(str(last_error))
                raise HTTPException(status_code=429, detail={
                    "error": {
                        "message": f"All models rate-limited. Last error: {safe_last_error}",
                        "type": "rate_limit_error"
                    }
                })
            else:
                raise HTTPException(status_code=re_err.status, detail={
                    "error": {"message": str(re_err), "type": "routing_error"}
                })
        except Exception as e:
            raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "routing_error"}})
            
        model_key = f"{route['platform']}:{route['modelId']}"
        outbound_messages = norm_messages
        injected_handoff_tokens = 0
        if handoff_mode != "off" and session_key:
            outbound_messages, injected, injected_handoff_tokens = maybe_inject_context_handoff(
                mode=handoff_mode,
                session_key=session_key,
                messages=norm_messages,
                selected_model_key=model_key
            )
            if injected:
                print(f"[Proxy] Context handoff injected (session {session_key[:8]}..., model switch detected)")
            
        in_flight_incremented = False
        try:
            if body.stream:
                increment_in_flight(route["platform"], route["modelId"], route["keyId"], routing_estimate)
                in_flight_incremented = True
                gen = await route["provider"].stream_chat_completion(
                    api_key=route["apiKey"],
                    messages=outbound_messages,
                    model_id=route["modelId"],
                    options={
                        "temperature": body.temperature,
                        "max_tokens": max_tokens,
                        "top_p": body.top_p,
                        "tools": tools_dict_list,
                        "tool_choice": tool_choice_val,
                        "parallel_tool_calls": body.parallel_tool_calls
                    }
                )
                
                # Fetch first chunk to verify stream start
                first_chunk = await gen.__anext__()
                
                # Return StreamingResponse with headers
                headers = {
                    "X-Routed-Via": f"{route['platform']}/{route['modelId']}",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
                if attempt > 0:
                    headers["X-Fallback-Attempts"] = str(attempt)
                    
                in_flight_incremented = False # stream_generator will clean it up
                return StreamingResponse(
                    stream_generator(
                        first_chunk=first_chunk,
                        gen=gen,
                        route=route,
                        estimated_input_tokens=estimated_input_tokens,
                        injected_handoff_tokens=injected_handoff_tokens,
                        session_key=session_key,
                        handoff_mode=handoff_mode,
                        model_key=model_key,
                        pinned_model_id=pinned_model_id,
                        messages=norm_messages,
                        sessionIdHeader=sessionIdHeader,
                        strategyKey=strategy_key,
                        tools=tools_dict_list,
                        start_time=start_time,
                        request=request,
                        reserved_tokens=routing_estimate
                    ),
                    media_type="text/event-stream",
                    headers=headers
                )
            else:
                increment_in_flight(route["platform"], route["modelId"], route["keyId"], routing_estimate)
                in_flight_incremented = True
                result = await route["provider"].chat_completion(
                    api_key=route["apiKey"],
                    messages=outbound_messages,
                    model_id=route["modelId"],
                    options={
                        "temperature": body.temperature,
                        "max_tokens": max_tokens,
                        "top_p": body.top_p,
                        "tools": tools_dict_list,
                        "tool_choice": tool_choice_val,
                        "parallel_tool_calls": body.parallel_tool_calls
                    }
                )
                
                choices = result.get("choices", [])
                finish_reason = choices[0].get("finish_reason") if choices else None
                if finish_reason == "content_filter":
                    total_tokens = result.get("usage", {}).get("total_tokens", 0)
                    record_request(route["platform"], route["modelId"], route["keyId"])
                    record_tokens(route["platform"], route["modelId"], route["keyId"], total_tokens)
                    record_success(route["modelDbId"])
                    set_sticky_model(norm_messages, route["modelDbId"], sessionIdHeader, strategy_key)
                    if handoff_mode != 'off' and session_key:
                        record_successful_model(session_key=session_key, model_key=model_key)
                    
                    latency_ms = int((time.time() - start_time) * 1000)
                    log_request(route["platform"], route["modelId"], route["keyId"], 'success',
                                result.get("usage", {}).get("prompt_tokens", 0),
                                result.get("usage", {}).get("completion_tokens", 0),
                                latency_ms, "content_filter triggered", None, pinned_model_id)
                    return JSONResponse(content=result, headers={"X-Routed-Via": f"{route['platform']}/{route['modelId']}"})

                resp_msg = choices[0].get("message") if choices else None
                resp_text = content_to_string(resp_msg.get("content") or "") if resp_msg else ""
                
                if not resp_text and (not resp_msg or len(resp_msg.get("tool_calls", [])) == 0):
                    latency_ms = int((time.time() - start_time) * 1000)
                    log_request(route["platform"], route["modelId"], route["keyId"], 'error',
                                estimated_input_tokens, 0, latency_ms, 'empty completion (no content, no tool_calls)', None, pinned_model_id)
                    skip_keys.add(f"{route['platform']}:{route['modelId']}:{route['keyId']}")
                    set_cooldown(route["platform"], route["modelId"], route["keyId"],
                                 get_cooldown_duration_for_limit(route["platform"], route["modelId"], route["keyId"],
                                                                 {"rpd": route["rpdLimit"], "tpd": route["tpdLimit"]}))
                    record_rate_limit_hit(route["modelDbId"])
                    last_error = Exception(f"empty completion from {route['displayName']}")
                    continue
                    
                if wants_tools and resp_msg and len(resp_msg.get("tool_calls", [])) == 0 and resp_text:
                    rescue = rescue_inline_tool_calls(resp_text, set(t.function.name for t in body.tools) if body.tools else set())
                    if rescue.get("detected"):
                        if not rescue.get("calls"):
                            raise Exception(f"unparseable inline tool-call dialect from {route['displayName']}: {resp_text[:120]}")
                        schemas = tool_schema_map(tools_dict_list)
                        resp_msg["tool_calls"] = [
                            {
                                "id": f"call_rescued_{i + 1}",
                                "type": "function",
                                "function": {
                                    "name": c.name,
                                    "arguments": repair_tool_arguments(c.arguments, schemas.get(c.name))
                                }
                            } for i, c in enumerate(rescue["calls"])
                        ]
                        resp_msg["content"] = rescue["cleanText"] if len(rescue["cleanText"]) > 0 else None
                        if choices:
                            choices[0]["finish_reason"] = 'tool_calls'
                        print(f"[Proxy] Rescued {len(rescue['calls'])} inline tool call(s) from {route['displayName']} into structured tool_calls")
                        
                total_tokens = result.get("usage", {}).get("total_tokens", 0)
                record_request(route["platform"], route["modelId"], route["keyId"])
                record_tokens(route["platform"], route["modelId"], route["keyId"], total_tokens)
                record_success(route["modelDbId"])
                set_sticky_model(norm_messages, route["modelDbId"], sessionIdHeader, strategy_key)
                if handoff_mode != 'off' and session_key:
                    record_successful_model(session_key=session_key, model_key=model_key)
                    
                headers = {
                    "X-Routed-Via": f"{route['platform']}/{route['modelId']}"
                }

                if attempt > 0:
                    headers["X-Fallback-Attempts"] = str(attempt)
                    
                if resp_msg and resp_msg.get("tool_calls"):
                    schemas = tool_schema_map(tools_dict_list)
                    for tc in resp_msg["tool_calls"]:
                        tc_func = tc.get("function")
                        if tc_func and tc_func.get("arguments") is not None:
                            tc_func["arguments"] = repair_tool_arguments(tc_func["arguments"], schemas.get(tc_func["name"]))
                            
                normalize_outbound_content(result)
                latency_ms = int((time.time() - start_time) * 1000)
                log_request(route["platform"], route["modelId"], route["keyId"], 'success',
                            result.get("usage", {}).get("prompt_tokens", 0),
                            result.get("usage", {}).get("completion_tokens", 0),
                            latency_ms, None, None, pinned_model_id)
                            
                return JSONResponse(content=result, headers=headers)
                
        except Exception as err:
            latency_ms = int((time.time() - start_time) * 1000)
            safe_error = sanitize_provider_error_message(str(err))
            log_request(route["platform"], route["modelId"], route["keyId"], 'error',
                        estimated_input_tokens, 0, latency_ms, safe_error, None, pinned_model_id)
                        
            if is_retryable_error(err):
                if is_model_not_found_error(err) or is_model_access_forbidden_error(err):
                    skip_models.add(route["modelDbId"])
                skip_id = f"{route['platform']}:{route['modelId']}:{route['keyId']}"
                skip_keys.add(skip_id)
                
                if is_payment_required_error(err):
                    from app.ratelimit import set_global_key_disabled
                    set_global_key_disabled(route["platform"], route["keyId"], PAYMENT_REQUIRED_COOLDOWN_MS)
                    cooldown_duration = PAYMENT_REQUIRED_COOLDOWN_MS
                elif "401" in str(err).lower() or "unauthorized" in str(err).lower() or "invalid api key" in str(err).lower() or "invalid_key" in str(err).lower():
                    from app.ratelimit import set_global_key_disabled
                    set_global_key_disabled(route["platform"], route["keyId"], 24 * 60 * 60 * 1000)
                    cooldown_duration = 24 * 60 * 60 * 1000
                elif is_model_access_forbidden_error(err):
                    cooldown_duration = MODEL_FORBIDDEN_COOLDOWN_MS
                else:
                    retry_after_ms = getattr(err, "retry_after_ms", None)
                    cooldown_duration = get_cooldown_duration_for_limit(
                        route["platform"], route["modelId"], route["keyId"],
                        {"rpd": route["rpdLimit"], "tpd": route["tpdLimit"]},
                        retry_after_ms
                    )
                set_cooldown(route["platform"], route["modelId"], route["keyId"], cooldown_duration)
                record_rate_limit_hit(route["modelDbId"])
                last_error = err
                print(f"[Proxy] {safe_error[:60]} from {route['displayName']}, falling back (attempt {attempt + 1}/{MAX_RETRIES})")
                import random
                await asyncio.sleep(random.uniform(0.05, 0.15))
                continue
                
            raise HTTPException(status_code=502, detail={
                "error": {
                    "message": f"Provider error ({route['displayName']}): {safe_error}",
                    "type": "provider_error"
                }
            })
        finally:
            if in_flight_incremented:
                decrement_in_flight(route["platform"], route["modelId"], route["keyId"], routing_estimate)

            
    if last_error:
        safe_last_error = sanitize_provider_error_message(str(last_error))
        raise HTTPException(status_code=429, detail={
            "error": {
                "message": f"All models rate-limited after {MAX_RETRIES} attempts. Last: {safe_last_error}",
                "type": "rate_limit_error"
            }
        })
    else:
        raise HTTPException(status_code=429, detail={
            "error": {
                "message": "All models exhausted. Add more API keys or wait for rate limits to reset.",
                "type": "rate_limit_error"
            }
        })

# ── endpoint: POST /v1/responses ───────────────────────────────────────────

class ResponsesRequest(BaseModel):
    model: Optional[str] = None
    instructions: Optional[str] = None
    input: Union[str, List[Dict[str, Any]]]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None

    class Config:
        extra = "allow"

def responses_input_has_image(inputs: Union[str, List[Dict[str, Any]]]) -> bool:
    if isinstance(inputs, str):
        return False
    for item in inputs:
        content = item.get("content")
        if isinstance(content, list):
            for p in content:
                if isinstance(p, dict):
                    t = p.get("type")
                    if t in ('input_image', 'image_url', 'image'):
                        return True
        elif isinstance(content, str):
            pass
    return False

def partsToString(content: Union[str, List[Dict[str, Any]]]) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if (isinstance(p, dict) and isinstance(p.get("text"), str)) else "" for p in content)

def to_chat_messages(req: ResponsesRequest) -> List[Dict[str, Any]]:
    messages = []
    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})
    if isinstance(req.input, str):
        messages.append({"role": "user", "content": req.input})
        return messages
    for item in req.input:
        item_type = item.get("type")
        if item_type == 'function_call':
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.get("call_id"),
                    "type": "function",
                    "function": {"name": item.get("name"), "arguments": item.get("arguments")}
                }]
            })
        elif item_type == 'function_call_output':
            output = item.get("output")
            if isinstance(output, str):
                output_str = output
            elif isinstance(output, list):
                output_str = partsToString(output)
            else:
                output_str = json.dumps(output)
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id"),
                "content": output_str
            })
        else:
            role = item.get("role")
            if role == 'developer':
                role = 'system'
            messages.append({
                "role": role,
                "content": partsToString(item.get("content") or "")
            })
    return messages

def to_chat_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    res = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") != 'function':
            continue
        func_src = t.get("function")
        if isinstance(func_src, dict):
            name = func_src.get("name")
            desc = func_src.get("description")
            params = func_src.get("parameters")
            strict = func_src.get("strict")
        else:
            name = t.get("name")
            desc = t.get("description")
            params = t.get("parameters")
            strict = t.get("strict")
        if not isinstance(name, str) or not name:
            continue
        func = {"name": name}
        if desc:
            func["description"] = desc
        if params:
            func["parameters"] = params
        if strict is not None:
            func["strict"] = strict
        res.append({
            "type": "function",
            "function": func
        })
    return res if res else None

def to_chat_tool_choice(tc: Optional[Union[str, Dict[str, Any]]]) -> Optional[Union[str, Dict[str, Any]]]:
    if not tc:
        return None
    if isinstance(tc, str):
        return tc
    return {"type": "function", "function": {"name": tc.get("name")}}

def build_response_object(response_id: str, model: str, text: str, tool_calls: List[Dict[str, Any]], prompt_tokens: int, completion_tokens: int) -> Dict[str, Any]:
    output = []
    if len(text) > 0:
        output.append({
            "type": "message",
            "id": f"msg_{secrets.token_hex(18)}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}]
        })
    for tc in tool_calls:
        tc_func = tc.get("function", {})
        output.append({
            "type": "function_call",
            "id": f"fc_{secrets.token_hex(18)}",
            "call_id": tc.get("id"),
            "name": tc_func.get("name"),
            "arguments": tc_func.get("arguments"),
            "status": "completed"
        })
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "output_text": text,
        "usage": {
            "input_tokens": prompt_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": prompt_tokens + completion_tokens
        }
    }

async def responses_stream_generator(
    first_chunk: Dict[str, Any],
    gen,
    route: Dict[str, Any],
    estimated_input_tokens: int,
    messages: List[Dict[str, Any]],
    sessionIdHeader: Optional[str],
    tool_schemas: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]],
    response_id: str,
    start_time: float,
    skip_keys: Set[str],
    request: Optional[Request] = None,
    reserved_tokens: int = 0
):
    seq = 0
    stream_started = False
    
    def sse(event: str, payload: Dict[str, Any]) -> str:
        nonlocal seq
        data = f"event: {event}\ndata: {json.dumps({'type': event, 'sequence_number': seq, **payload})}\n\n"
        seq += 1
        return data
        
    output_index = 0
    msg_item_id = None
    msg_text = ""
    tool_acc = {}
    total_output_tokens = 0
    dialect_mode = 'undecided'
    held_text = ""
    
    chunks = [first_chunk]
    
    def open_text_item(text: str) -> List[str]:
        nonlocal msg_item_id, msg_text
        msg_item_id = f"msg_{secrets.token_hex(18)}"
        events = []
        events.append(sse('response.output_item.added', {
            "output_index": output_index,
            "item": {"id": msg_item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}
        }))
        events.append(sse('response.content_part.added', {
            "item_id": msg_item_id, "output_index": output_index, "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []}
        }))
        if text:
            events.append(sse('response.output_text.delta', {
                "item_id": msg_item_id, "output_index": output_index, "content_index": 0, "delta": text
            }))
            msg_text += text
        return events

    try:
        idx = 0
        while True:
            if request and await request.is_disconnected():
                print(f"[Proxy] Client disconnected. Stopping responses stream generator.")
                return

            if idx < len(chunks):
                chunk = chunks[idx]
                idx += 1
            else:
                try:
                    chunk = await get_next_chunk_safely(gen, request)
                except (StopAsyncIteration, asyncio.CancelledError):
                    break
                    
            any_chunk = chunk
            
            if "error" in any_chunk and "choices" not in any_chunk:
                raise Exception(any_chunk["error"].get("message") or "provider error")
                
            if not stream_started:
                skeleton = {
                    "id": response_id, "object": "response", "created_at": int(time.time()),
                    "status": "in_progress", "model": route["modelId"], "output": [], "output_text": ""
                }
                yield sse('response.created', {"response": skeleton})
                yield sse('response.in_progress', {"response": skeleton})
                stream_started = True
                
            choices = any_chunk.get("choices", [])
            if not choices:
                continue
                
            delta = choices[0].get("delta", {})
            text = delta.get("content") or ""
            if not isinstance(text, str):
                text = ""
                
            if text:
                total_output_tokens += math.ceil(len(text) / 4)
                if dialect_mode == 'passthrough':
                    if msg_item_id is None:
                        for ev in open_text_item(''):
                            yield ev
                    yield sse('response.output_text.delta', {
                        "item_id": msg_item_id, "output_index": 0, "content_index": 0, "delta": text
                    })
                    msg_text += text
                else:
                    held_text += text
                    if dialect_mode == 'undecided':
                        probe = held_text.lstrip()
                        if starts_with_dialect_marker(probe):
                            dialect_mode = 'dialect'
                        elif not could_become_dialect_marker(probe) or len(held_text) > 256:
                            dialect_mode = 'passthrough'
                            for ev in open_text_item(held_text):
                                yield ev
                            held_text = ''
                            
            for tc in delta.get("tool_calls", []):
                tc_idx = tc.get("index", 0)
                if tc_idx not in tool_acc:
                    if msg_item_id is not None and len(msg_text) > 0:
                        yield sse('response.output_text.done', {"item_id": msg_item_id, "output_index": 0, "content_index": 0, "text": msg_text})
                        yield sse('response.content_part.done', {"item_id": msg_item_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": msg_text, "annotations": []}})
                        yield sse('response.output_item.done', {"output_index": 0, "item": {"id": msg_item_id, "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": msg_text, "annotations": []}]}})
                        msg_item_id = None
                        
                    output_index = len(tool_acc) + (1 if len(msg_text) > 0 else 0)
                    tool_acc[tc_idx] = {
                        "outputIndex": output_index,
                        "itemId": f"fc_{secrets.token_hex(18)}",
                        "callId": tc.get("id") or f"call_{secrets.token_hex(18)}",
                        "name": tc.get("function", {}).get("name") or "",
                        "args": ""
                    }
                    acc = tool_acc[tc_idx]
                    yield sse('response.output_item.added', {
                        "output_index": acc["outputIndex"],
                        "item": {"id": acc["itemId"], "type": "function_call", "status": "in_progress", "call_id": acc["callId"], "name": acc["name"], "arguments": ""}
                    })
                    
                acc = tool_acc[tc_idx]
                tc_func = tc.get("function", {})
                arg_frag = tc_func.get("arguments") or ""
                if tc_func.get("name") and not acc["name"]:
                    acc["name"] = tc_func["name"]
                if arg_frag:
                    acc["args"] += arg_frag
                    yield sse('response.function_call_arguments.delta', {
                        "item_id": acc["itemId"], "output_index": acc["outputIndex"], "delta": arg_frag
                    })
                    
        if len(held_text) > 0:
            rescue = rescue_inline_tool_calls(held_text, set(t.function.name for t in tools) if tools else set()) if (dialect_mode == 'dialect' or contains_dialect_marker(held_text)) else {"detected": False, "calls": None, "cleanText": held_text}
            if rescue.get("detected") and not rescue.get("calls"):
                log_request(route["platform"], route["modelId"], route["keyId"], 'error', estimated_input_tokens, 0, int((time.time() - start_time) * 1000), f"unparseable inline tool-call dialect: {held_text[:120]}")
                skip_keys.add(f"{route['platform']}:{route['modelId']}:{route['keyId']}")
                set_cooldown(route["platform"], route["modelId"], route["keyId"], get_cooldown_duration_for_limit(route["platform"], route["modelId"], route["keyId"], {"rpd": route["rpdLimit"], "tpd": route["tpdLimit"]}))
                record_rate_limit_hit(route["modelDbId"])
                raise Exception(f"unparseable inline tool-call dialect from {route['displayName']}")
                
            if rescue.get("detected") and rescue.get("calls"):
                print(f"[Responses] Rescued {len(rescue['calls'])} inline tool call(s) from {route['displayName']}")
                clean_text = rescue.get("cleanText", "")
                if len(clean_text) > 0 and msg_item_id is None:
                    for ev in open_text_item(clean_text):
                        yield ev
                rescued_idx = 0
                for c in rescue["calls"]:
                    idx = 1000 + rescued_idx
                    rescued_idx += 1
                    output_index = len(tool_acc) + (1 if len(msg_text) > 0 else 0)
                    tool_acc[idx] = {
                        "outputIndex": output_index,
                        "itemId": f"fc_{secrets.token_hex(18)}",
                        "callId": f"call_{secrets.token_hex(18)}",
                        "name": c["name"],
                        "args": c["arguments"]
                    }
                    acc = tool_acc[idx]
                    yield sse('response.output_item.added', {
                        "output_index": acc["outputIndex"],
                        "item": {"id": acc["itemId"], "type": "function_call", "status": "in_progress", "call_id": acc["callId"], "name": acc["name"], "arguments": ""}
                    })
            else:
                if msg_item_id is None:
                    for ev in open_text_item(held_text):
                        yield ev
            held_text = ''
            
        if len(msg_text) == 0 and len(tool_acc) == 0:
            log_request(route["platform"], route["modelId"], route["keyId"], 'error', estimated_input_tokens, 0, int((time.time() - start_time) * 1000), 'empty completion (no content, no tool_calls)')
            skip_keys.add(f"{route['platform']}:{route['modelId']}:{route['keyId']}")
            set_cooldown(route["platform"], route["modelId"], route["keyId"], get_cooldown_duration_for_limit(route["platform"], route["modelId"], route["keyId"], {"rpd": route["rpdLimit"], "tpd": route["tpdLimit"]}))
            record_rate_limit_hit(route["modelDbId"])
            raise Exception(f"empty completion from {route['displayName']}")
            
        if msg_item_id is not None:
            yield sse('response.output_text.done', {"item_id": msg_item_id, "output_index": 0, "content_index": 0, "text": msg_text})
            yield sse('response.content_part.done', {"item_id": msg_item_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": msg_text, "annotations": []}})
            yield sse('response.output_item.done', {"output_index": 0, "item": {"id": msg_item_id, "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": msg_text, "annotations": []}]}})
            
        final_tool_calls = []
        for acc in tool_acc.values():
            repaired_args = repair_tool_arguments(acc["args"], tool_schemas.get(acc["name"]))
            yield sse('response.function_call_arguments.done', {"item_id": acc["itemId"], "output_index": acc["outputIndex"], "arguments": repaired_args})
            yield sse('response.output_item.done', {"output_index": acc["outputIndex"], "item": {"id": acc["itemId"], "type": "function_call", "status": "completed", "call_id": acc["callId"], "name": acc["name"], "arguments": repaired_args}})
            final_tool_calls.append({
                "id": acc["callId"],
                "type": "function",
                "function": {
                    "name": acc["name"],
                    "arguments": repaired_args
                }
            })
            
        final_response = build_response_object(response_id, route["modelId"], msg_text, final_tool_calls, estimated_input_tokens, total_output_tokens)
        yield sse('response.completed', {"response": final_response})
        
        record_request(route["platform"], route["modelId"], route["keyId"])
        record_tokens(route["platform"], route["modelId"], route["keyId"], estimated_input_tokens + total_output_tokens)
        record_success(route["modelDbId"])
        set_sticky_model(messages, route["modelDbId"], sessionIdHeader)
        log_request(route["platform"], route["modelId"], route["keyId"], 'success', estimated_input_tokens, total_output_tokens, int((time.time() - start_time) * 1000), None)
        
    except Exception as stream_err:
        exhausted_msg = f"Provider error ({route['displayName']}): stream interrupted: {str(stream_err)}"
        if stream_started:
            yield sse('response.failed', {"response": {"id": response_id, "object": "response", "status": "failed", "error": {"message": exhausted_msg, "type": "stream_error"}}})
        else:
            raise stream_err
    finally:
        decrement_in_flight(route["platform"], route["modelId"], route["keyId"], reserved_tokens)

@v1_router.post("/responses")
async def post_v1_responses(request: Request):
    start_time = time.time()
    require_unified_key(request)
    
    try:
        body_json = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"error": {"message": "Invalid request: invalid JSON body", "type": "invalid_request_error"}})
        
    try:
        body = ResponsesRequest.model_validate(body_json)
    except Exception as val_err:
        raise HTTPException(status_code=400, detail={"error": {"message": f"Invalid request: {str(val_err)}", "type": "invalid_request_error"}})
        
    if responses_input_has_image(body.input):
        raise HTTPException(status_code=422, detail={
            "error": {
                "message": "Image input is not yet supported on /v1/responses. Use /v1/chat/completions with an image_url content part instead.",
                "type": "invalid_request_error",
                "code": "no_vision_model"
            }
        })
        
    messages = to_chat_messages(body)
    tools = to_chat_tools(body.tools)
    tool_schemas = tool_schema_map(tools)
    tool_choice = to_chat_tool_choice(body.tool_choice)
    
    completion_opts = {
        "temperature": body.temperature,
        "max_tokens": body.max_output_tokens,
        "top_p": body.top_p,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": body.parallel_tool_calls
    }
    
    estimated_input_tokens = sum(math.ceil(len(content_to_string(m.get("content") or "")) / 4) for m in messages)
    estimated_total = estimated_input_tokens + (body.max_output_tokens or 1000)
    
    raw_session_id = request.headers.get("x-session-id")
    sessionIdHeader = raw_session_id.strip() if raw_session_id else None
    preferred_model = get_sticky_model(messages, sessionIdHeader)
    
    wants_tools = bool(tools and len(tools) > 0)
    if wants_tools and not has_enabled_tools_model():
        raise HTTPException(status_code=422, detail={
            "error": {
                "message": "This request includes tools, but no tool-capable model is enabled. Enable a tool-calling model in the Fallback Chain.",
                "type": "invalid_request_error",
                "code": "no_tools_model"
            }
        })
        
    response_id = f"resp_{secrets.token_hex(18)}"
    skip_keys = set()
    skip_models = set()
    last_error = None
    stream_started = False
    
    for attempt in range(MAX_RETRIES):
        try:
            route = route_request(
                estimated_tokens=estimated_total,
                skip_keys=skip_keys if skip_keys else None,
                preferred_model_db_id=preferred_model,
                require_vision=False,
                require_tools=wants_tools,
                skip_models=skip_models if skip_models else None
            )
        except RouteError as re_err:
            status_code = 429 if last_error else re_err.status
            msg = f"All models rate-limited. Last error: {sanitize_provider_error_message(str(last_error))}" if last_error else str(re_err)
            err_type = "rate_limit_error" if last_error else "routing_error"
            raise HTTPException(status_code=status_code, detail={"error": {"message": msg, "type": err_type}})
        except Exception as e:
            raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "routing_error"}})
            
        in_flight_incremented = False
        try:
            if body.stream:
                increment_in_flight(route["platform"], route["modelId"], route["keyId"], estimated_total)
                in_flight_incremented = True
                gen = await route["provider"].stream_chat_completion(
                    api_key=route["apiKey"],
                    messages=messages,
                    model_id=route["modelId"],
                    options=completion_opts
                )
                
                # Fetch first chunk to verify stream start
                first_chunk = await gen.__anext__()
                
                # Return StreamingResponse with headers
                headers = {
                    "X-Routed-Via": f"{route['platform']}/{route['modelId']}",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
                if attempt > 0:
                    headers["X-Fallback-Attempts"] = str(attempt)
                    
                stream_started = True
                in_flight_incremented = False # responses_stream_generator will clean it up
                return StreamingResponse(
                    responses_stream_generator(
                        first_chunk=first_chunk,
                        gen=gen,
                        route=route,
                        estimated_input_tokens=estimated_input_tokens,
                        messages=messages,
                        sessionIdHeader=sessionIdHeader,
                        tool_schemas=tool_schemas,
                        tools=tools,
                        response_id=response_id,
                        start_time=start_time,
                        skip_keys=skip_keys,
                        request=request,
                        reserved_tokens=estimated_total
                    ),
                    media_type="text/event-stream",
                    headers=headers
                )
            else:
                increment_in_flight(route["platform"], route["modelId"], route["keyId"], estimated_total)
                in_flight_incremented = True
                result = await route["provider"].chat_completion(
                    api_key=route["apiKey"],
                    messages=messages,
                    model_id=route["modelId"],
                    options=completion_opts
                )
                
                choices = result.get("choices", [])
                msg_msg = choices[0].get("message") if choices else None
                text = content_to_string(msg_msg.get("content") or "") if msg_msg else ""
                tool_calls = (msg_msg.get("tool_calls") or []) if msg_msg else []
                tool_calls = [
                    {
                        **tc,
                        "function": {
                            **tc.get("function", {}),
                            "arguments": repair_tool_arguments(tc.get("function", {}).get("arguments", "{}"), tool_schemas.get(tc.get("function", {}).get("name")))
                        }
                    } for tc in tool_calls
                ]
                
                if wants_tools and len(tool_calls) == 0 and text:
                    rescue = rescue_inline_tool_calls(text, set(t.function.name for t in tools) if tools else set())
                    if rescue.get("detected"):
                        if not rescue.get("calls"):
                            raise Exception(f"unparseable inline tool-call dialect from {route['displayName']}: {text[:120]}")
                        print(f"[Responses] Rescued {len(rescue['calls'])} inline tool call(s) from {route['displayName']}")
                        tool_calls = [
                            {
                                "id": f"call_rescued_{i + 1}",
                                "type": "function",
                                "function": {
                                    "name": c.name,
                                    "arguments": repair_tool_arguments(c.arguments, tool_schemas.get(c.name))
                                }
                            } for i, c in enumerate(rescue["calls"])
                        ]
                        text = rescue.get("cleanText", "")
                        
                prompt_tokens = result.get("usage", {}).get("prompt_tokens") or estimated_input_tokens
                completion_tokens = result.get("usage", {}).get("completion_tokens") or math.ceil(len(text) / 4)
                
                if not text and len(tool_calls) == 0:
                    latency_ms = int((time.time() - start_time) * 1000)
                    log_request(route["platform"], route["modelId"], route["keyId"], 'error', estimated_input_tokens, 0, latency_ms, 'empty completion (no content, no tool_calls)')
                    skip_keys.add(f"{route['platform']}:{route['modelId']}:{route['keyId']}")
                    set_cooldown(route["platform"], route["modelId"], route["keyId"],
                                 get_cooldown_duration_for_limit(route["platform"], route["modelId"], route["keyId"],
                                                                 {"rpd": route["rpdLimit"], "tpd": route["tpdLimit"]}))
                    record_rate_limit_hit(route["modelDbId"])
                    last_error = Exception(f"empty completion from {route['displayName']}")
                    continue
                    
                record_request(route["platform"], route["modelId"], route["keyId"])
                record_tokens(route["platform"], route["modelId"], route["keyId"], result.get("usage", {}).get("total_tokens", 0))
                record_success(route["modelDbId"])
                set_sticky_model(messages, route["modelDbId"], sessionIdHeader)
                
                headers = {
                    "X-Routed-Via": f"{route['platform']}/{route['modelId']}"
                }
                if attempt > 0:
                    headers["X-Fallback-Attempts"] = str(attempt)
                    
                resp_obj = build_response_object(
                    response_id=response_id,
                    model=route["modelId"],
                    text=text,
                    tool_calls=tool_calls,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens
                )
                latency_ms = int((time.time() - start_time) * 1000)
                log_request(route["platform"], route["modelId"], route["keyId"], 'success',
                            prompt_tokens, completion_tokens, latency_ms, None)
                            
                return JSONResponse(content=resp_obj, headers=headers)
                
        except Exception as err:
            latency_ms = int((time.time() - start_time) * 1000)
            safe_error = sanitize_provider_error_message(str(err))
            log_request(route["platform"], route["modelId"], route["keyId"], 'error',
                        estimated_input_tokens, 0, latency_ms, safe_error)
                        
            if is_retryable_error(err):
                if is_model_not_found_error(err) or is_model_access_forbidden_error(err):
                    skip_models.add(route["modelDbId"])
                skip_keys.add(f"{route['platform']}:{route['modelId']}:{route['keyId']}")
                cooldown_duration = (
                    PAYMENT_REQUIRED_COOLDOWN_MS if is_payment_required_error(err)
                    else get_cooldown_duration_for_limit(route["platform"], route["modelId"], route["keyId"], {"rpd": route["rpdLimit"], "tpd": route["tpdLimit"]})
                )
                set_cooldown(route["platform"], route["modelId"], route["keyId"], cooldown_duration)
                record_rate_limit_hit(route["modelDbId"])
                last_error = err
                import random
                await asyncio.sleep(random.uniform(0.05, 0.15))
                continue
                
            raise HTTPException(status_code=502, detail={"error": {"message": f"Provider error ({route['displayName']}): {safe_error}", "type": "provider_error"}})
        finally:
            if in_flight_incremented:
                decrement_in_flight(route["platform"], route["modelId"], route["keyId"], estimated_total)

            
    exhausted_msg = f"All models rate-limited after {MAX_RETRIES} attempts. Last: {str(last_error)}"
    raise HTTPException(status_code=429, detail={"error": {"message": exhausted_msg, "type": "rate_limit_error"}})
