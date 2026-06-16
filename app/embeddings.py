import time
import math
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.database import get_db, get_provider_keys

class EmbeddingsError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status

from app.router import EMBEDDING_MODELS, ensure_models_loaded

def list_embedding_models() -> List[Dict[str, Any]]:
    ensure_models_loaded()
    return EMBEDDING_MODELS


def get_default_family() -> str:
    # Static default family for embeddings
    return 'text-embedding-3-small'

def resolve_family(model: Optional[str]) -> Optional[str]:
    if not model or model == 'auto':
        return get_default_family()
    rows = list_embedding_models()
    if any(r.get("family") == model for r in rows):
        return model
    by_model_id = next((r for r in rows if r.get("model_id") == model), None)
    return by_model_id["family"] if by_model_id else None

def estimate_tokens(inputs: List[str]) -> int:
    return math.ceil(sum(len(s) for s in inputs) / 4)

async def open_ai_style_embed(
    url: str,
    key: str,
    model_id: str,
    inputs: List[str],
    platform: str,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if extra is None:
        extra = {}
    async with get_httpx_client(platform=platform, timeout=30.0) as client:
        try:
            res = await client.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}"
                },
                json={"model": model_id, "input": inputs, **extra}
            )
        except Exception as e:
            raise EmbeddingsError(f"upstream connection error: {str(e)}", 502)
            
    if res.status_code != 200:
        raise EmbeddingsError(f"upstream {res.status_code}: {res.text[:200]}", res.status_code)
        
    try:
        j = res.json()
    except Exception:
        raise EmbeddingsError("upstream returned invalid JSON", 502)
        
    data = j.get("data", [])
    # Sort by index
    data = sorted(data, key=lambda x: x.get("index", 0))
    
    usage = j.get("usage", {})
    input_tokens = usage.get("prompt_tokens") or usage.get("total_tokens")
    
    return {
        "vectors": [d["embedding"] for d in data],
        "inputTokens": input_tokens
    }

async def call_provider(row: Dict[str, Any], key: str, inputs: List[str]) -> Dict[str, Any]:
    platform = row["platform"]
    model_id = row["model_id"]
    
    if platform == 'google':
        return await open_ai_style_embed('https://generativelanguage.googleapis.com/v1beta/openai/embeddings', key, model_id, inputs, platform)
    elif platform == 'nvidia':
        return await open_ai_style_embed('https://integrate.api.nvidia.com/v1/embeddings', key, model_id, inputs, platform, {"input_type": "query"})
    elif platform == 'openrouter':
        return await open_ai_style_embed('https://openrouter.ai/api/v1/embeddings', key, model_id, inputs, platform)
    elif platform == 'github':
        return await open_ai_style_embed('https://models.github.ai/inference/embeddings', key, model_id, inputs, platform)
    elif platform == 'cloudflare':
        sep = key.find(':')
        if sep == -1:
            raise EmbeddingsError('cloudflare key is not in account_id:token form', 500)
        account_id = key[:sep]
        token = key[sep+1:]
        return await open_ai_style_embed(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/embeddings",
            token, model_id, inputs, platform
        )
    elif platform == 'huggingface':
        async with get_httpx_client(platform=platform, timeout=30.0) as client:
            try:
                res = await client.post(
                    f"https://router.huggingface.co/hf-inference/models/{model_id}/pipeline/feature-extraction",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {key}"
                    },
                    json={"inputs": inputs}
                )
            except Exception as e:
                raise EmbeddingsError(f"upstream connection error: {str(e)}", 502)
                
        if res.status_code != 200:
            raise EmbeddingsError(f"upstream {res.status_code}: {res.text[:200]}", res.status_code)
            
        try:
            j = res.json()
        except Exception:
            raise EmbeddingsError("upstream returned invalid JSON", 502)
            
        if isinstance(j, list) and len(j) > 0 and isinstance(j[0], list):
            vectors = j
        else:
            vectors = [j]
            
        return {"vectors": vectors, "inputTokens": None}
    elif platform == 'cohere':
        async with get_httpx_client(platform=platform, timeout=30.0) as client:
            try:
                res = await client.post(
                    'https://api.cohere.com/v2/embed',
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {key}"
                    },
                    json={
                        "model": model_id,
                        "texts": inputs,
                        "input_type": "search_document",
                        "embedding_types": ["float"]
                    }
                )
            except Exception as e:
                raise EmbeddingsError(f"upstream connection error: {str(e)}", 502)
                
        if res.status_code != 200:
            raise EmbeddingsError(f"upstream {res.status_code}: {res.text[:200]}", res.status_code)
            
        try:
            j = res.json()
        except Exception:
            raise EmbeddingsError("upstream returned invalid JSON", 502)
            
        vectors = j.get("embeddings", {}).get("float", [])
        billed = j.get("meta", {}).get("billed_units", {})
        input_tokens = billed.get("input_tokens")
        
        return {"vectors": vectors, "inputTokens": input_tokens}
    else:
        raise EmbeddingsError(f"no embeddings adapter for platform '{platform}'", 500)

def log_embedding_request(
    row: Dict[str, Any],
    key_id: Optional[str],
    status: str,
    input_tokens: int,
    latency_ms: int,
    error: Optional[str]
) -> None:
    try:
        db = get_db()
        db.execute("""
            INSERT INTO requests (platform, model_id, key_id, status, input_tokens, output_tokens, latency_ms, error, request_type)
            VALUES (?, ?, ?, ?, ?, 0, ?, ?, 'embedding')
        """, (row["platform"], row["model_id"], key_id, status, input_tokens, latency_ms, error))
        db.commit()
    except Exception as e:
        print('Failed to log embedding request:', e)

async def run_embeddings(model: Optional[str], inputs: List[str]) -> Dict[str, Any]:
    # Filter EMBEDDING_MODELS statically
    chain = [m for m in EMBEDDING_MODELS if m.get("enabled", True)]
    
    # If a specific family or model ID was requested, bubble it to the front of the loop
    if model and model != 'auto':
        requested_family = resolve_family(model)
        if requested_family:
            chain = [r for r in chain if r.get("family") == requested_family] + [r for r in chain if r.get("family") != requested_family]
            
    if len(chain) == 0:
        raise EmbeddingsError("No enabled embedding models configured in the static configuration.", 503)
        
    from app.ratelimit import (
        is_on_cooldown, can_use_provider, get_in_flight_count,
        increment_in_flight, decrement_in_flight, set_cooldown, set_global_key_disabled
    )
    from app.v1_endpoints import (
        is_retryable_error, is_payment_required_error,
        is_model_access_forbidden_error, sanitize_provider_error_message,
        PAYMENT_REQUIRED_COOLDOWN_MS, MODEL_FORBIDDEN_COOLDOWN_MS
    )
    import asyncio
    import random

    last_error = None
    for row in chain:
        platform = row["platform"]
        model_id = row["model_id"]
        
        keys = get_provider_keys(platform)
        if not keys:
            continue
            
        # Sort keys primarily by in-flight count (ascending) to distribute concurrent load
        keys_with_info = []
        for idx, key in enumerate(keys):
            fake_key_id = f"{platform}:{idx}"
            inflight = get_in_flight_count(platform, model_id, fake_key_id)
            keys_with_info.append({
                "index": idx,
                "key": key,
                "fake_key_id": fake_key_id,
                "inflight": inflight
            })
        keys_with_info.sort(key=lambda x: x["inflight"])

        for k_info in keys_with_info:
            fake_key_id = k_info["fake_key_id"]
            raw_key = k_info["key"]
            
            if is_on_cooldown(platform, model_id, fake_key_id):
                continue
                
            if not can_use_provider(platform, fake_key_id):
                continue

            started = time.time()
            estimated_tokens = estimate_tokens(inputs)
            increment_in_flight(platform, model_id, fake_key_id, estimated_tokens)
            try:
                out = await call_provider(row, raw_key, inputs)
                vectors = out["vectors"]
                if len(vectors) != len(inputs) or any(not isinstance(v, list) or len(v) == 0 for v in vectors):
                    raise EmbeddingsError('upstream returned malformed embeddings', 502)
                    
                tokens = out["inputTokens"] if out["inputTokens"] is not None else estimated_tokens
                latency_ms = int((time.time() - started) * 1000)
                log_embedding_request(row, fake_key_id, 'success', tokens, latency_ms, None)
                
                decrement_in_flight(platform, model_id, fake_key_id, estimated_tokens)
                return {
                    "family": row.get("family"),
                    "platform": platform,
                    "modelId": model_id,
                    "dimensions": len(vectors[0]),
                    "vectors": vectors,
                    "inputTokens": tokens
                }
            except Exception as err:
                decrement_in_flight(platform, model_id, fake_key_id, estimated_tokens)
                
                e = err if isinstance(err, EmbeddingsError) else EmbeddingsError(str(err), 502)
                latency_ms = int((time.time() - started) * 1000)
                safe_error = sanitize_provider_error_message(e.args[0])
                log_embedding_request(row, fake_key_id, 'error', 0, latency_ms, safe_error[:300])
                last_error = e

                if is_retryable_error(e):
                    if is_payment_required_error(e):
                        set_global_key_disabled(platform, fake_key_id, PAYMENT_REQUIRED_COOLDOWN_MS)
                        cooldown_duration = PAYMENT_REQUIRED_COOLDOWN_MS
                    elif "401" in str(e).lower() or "unauthorized" in str(e).lower() or "invalid api key" in str(e).lower() or "invalid_key" in str(e).lower():
                        set_global_key_disabled(platform, fake_key_id, 24 * 60 * 60 * 1000)
                        cooldown_duration = 24 * 60 * 60 * 1000
                    elif is_model_access_forbidden_error(e):
                        cooldown_duration = MODEL_FORBIDDEN_COOLDOWN_MS
                    else:
                        retry_after_ms = getattr(e, "retry_after_ms", None)
                        from app.ratelimit import get_cooldown_duration_for_limit
                        cooldown_duration = get_cooldown_duration_for_limit(
                            platform, model_id, fake_key_id,
                            {"rpd": None, "tpd": None},
                            retry_after_ms
                        )
                    set_cooldown(platform, model_id, fake_key_id, cooldown_duration)
                    await asyncio.sleep(random.uniform(0.05, 0.15))
                
    status_code = 429 if (last_error and getattr(last_error, "status", None) == 429) else 502
    last_msg = f" (last: {last_error.args[0][:160]})" if last_error else " (no usable keys)"
    raise EmbeddingsError(
        f"All embedding models and keys exhausted/failed{last_msg}.",
        status_code
    )

from app.lib.proxy import get_httpx_client
