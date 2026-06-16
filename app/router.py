import time
import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from app.database import get_provider_keys, get_db
from app.providers import get_provider, has_provider, resolve_provider
from app.providers.base import BaseProvider
from app.ratelimit import can_make_request, can_use_tokens, is_on_cooldown, can_use_provider, get_in_flight_count


class RouteError(Exception):
    def __init__(self, message: str, status: int = 429):
        super().__init__(message)
        self.status = status

# Thread safety lock
_router_lock = threading.Lock()

MODELS_DATA: List[Dict[str, Any]] = []
EMBEDDING_MODELS: List[Dict[str, Any]] = []

_models_last_mtime = 0.0
_models_lock = threading.Lock()
DUMP_PATH = Path(__file__).parent / "dump.json"

def ensure_models_loaded() -> None:
    global _models_last_mtime
    if not DUMP_PATH.exists():
        return
    try:
        current_mtime = DUMP_PATH.stat().st_mtime
    except Exception:
        current_mtime = 0.0
        
    with _models_lock:
        if current_mtime == _models_last_mtime:
            return
            
        for attempt in range(3):
            try:
                with open(DUMP_PATH, 'r', encoding='utf-8') as f:
                    dump_data = json.load(f)
                    models_list = dump_data.get("models", [])
                    embed_list = dump_data.get("embedding_models", [])
                    
                    MODELS_DATA.clear()
                    MODELS_DATA.extend(models_list)
                    
                    EMBEDDING_MODELS.clear()
                    EMBEDDING_MODELS.extend(embed_list)
                    
                    _models_last_mtime = current_mtime
                    return
            except Exception as e:
                if attempt < 2:
                    time.sleep(0.05)
                else:
                    print("[Router] Failed to dynamically load models list after retries:", e)

# Initial load
ensure_models_loaded()


def get_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    ensure_models_loaded()
    return next((m for m in MODELS_DATA if m.get("model_id") == model_id), None)

def record_rate_limit_hit(model_db_id: int) -> None: pass
def record_success(model_db_id: int) -> None: pass

def get_routing_strategy() -> str:

    return 'priority'

def order_chain(chain: List[Dict[str, Any]], strategy: str = 'priority') -> List[Dict[str, Any]]:
    # Dynamic routing based on real-time success rate and latency metrics from SQLite (last 10 minutes)
    import datetime
    
    recent_metrics = {}
    try:
        db = get_db()
        cutoff_sec = time.time() - 600
        cutoff_str = datetime.datetime.fromtimestamp(cutoff_sec, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        cursor = db.execute("""
            SELECT platform, model_id, status, AVG(latency_ms) as avg_lat, COUNT(*) as cnt
            FROM requests
            WHERE created_at >= ?
            GROUP BY platform, model_id, status
        """, (cutoff_str,))
        rows = cursor.fetchall()
        for row in rows:
            key = f"{row['platform']}:{row['model_id']}"
            if key not in recent_metrics:
                recent_metrics[key] = {"success": 0, "error": 0, "latency": 0.0}
            if row["status"] == 'success':
                recent_metrics[key]["success"] += row["cnt"]
                recent_metrics[key]["latency"] = row["avg_lat"] or 0.0
            else:
                recent_metrics[key]["error"] += row["cnt"]
    except Exception:
        pass

    def get_sort_key(entry: Dict[str, Any]):
        base_priority = entry.get("priority", 999)
        key = f"{entry['platform']}:{entry['model_id']}"
        metrics = recent_metrics.get(key)
        if not metrics:
            return base_priority
            
        success = metrics["success"]
        error = metrics["error"]
        total = success + error
        
        penalty = 0.0
        if total > 0:
            error_rate = error / total
            penalty += error_rate * 50.0  # Add penalty for error rate
            
            avg_latency = metrics["latency"]
            if avg_latency > 3000:
                penalty += min(20.0, (avg_latency - 3000) / 1000.0) # Add penalty for high latency
                
        return base_priority + penalty

    return sorted(chain, key=get_sort_key)



def get_active_chain(db=None) -> List[Dict[str, Any]]:
    ensure_models_loaded()
    # Map the dump.json properties to match the expected dict fields
    chain = []
    for idx, m in enumerate(MODELS_DATA):

        if not m.get("enabled", 1):
            continue
        chain.append({
            "model_db_id": m.get("id", idx),
            "priority": m.get("priority", idx + 1),
            "enabled": 1,
            "platform": m.get("platform"),
            "model_id": m.get("model_id"),
            "display_name": m.get("display_name"),
            "intelligence_rank": m.get("intelligence_rank", 1),
            "size_label": m.get("size_label", ""),
            "monthly_token_budget": m.get("monthly_token_budget", ""),
            "rpm_limit": m.get("rpm_limit"),
            "rpd_limit": m.get("rpd_limit"),
            "tpm_limit": m.get("tpm_limit"),
            "tpd_limit": m.get("tpd_limit"),
            "supports_vision": m.get("supports_vision", 0),
            "supports_tools": m.get("supports_tools", 0),
            "context_window": m.get("context_window", 8192),
            "key_id": m.get("key_id")
        })
    return chain

def resolve_routing_chain(model_string: Optional[str]) -> Dict[str, Any]:
    return {"chain": get_active_chain(), "strategyKey": "auto"}

def route_request(
    estimated_tokens: int = 1000,
    skip_keys: Optional[Set[str]] = None,
    preferred_model_db_id: Optional[int] = None,
    require_vision: bool = False,
    require_tools: bool = False,
    skip_models: Optional[Set[int]] = None,
    prefetched_chain: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    chain = prefetched_chain if prefetched_chain is not None else get_active_chain()
    sorted_chain = order_chain(chain)

    # Pin preferred model to the front if requested
    if preferred_model_db_id:
        idx = -1
        for i, e in enumerate(sorted_chain):
            if e["model_db_id"] == preferred_model_db_id:
                idx = i
                break
        if idx > 0:
            preferred = sorted_chain.pop(idx)
            sorted_chain.insert(0, preferred)

    for entry in sorted_chain:
        if skip_models and entry["model_db_id"] in skip_models:
            continue
        if require_vision and not entry["supports_vision"]:
            continue
        if require_tools and not entry["supports_tools"]:
            continue
        if entry["context_window"] is not None and estimated_tokens > entry["context_window"]:
            continue
        if entry["tpm_limit"] is not None and estimated_tokens > entry["tpm_limit"]:
            continue

        if not has_provider(entry["platform"]):
            continue
        provider = get_provider(entry["platform"])

        # Fetch keys from keys.json instead of SQL
        keys = get_provider_keys(entry["platform"])
        if not keys:
            continue

        limits = {
            "rpm": entry["rpm_limit"],
            "rpd": entry["rpd_limit"],
            "tpm": entry["tpm_limit"],
            "tpd": entry["tpd_limit"]
        }

        # Sort keys primarily by in-flight count (ascending) to distribute concurrent load
        keys_with_info = []
        for idx, key in enumerate(keys):
            fake_key_id = f"{entry['platform']}:{idx}"
            inflight = get_in_flight_count(entry["platform"], entry["model_id"], fake_key_id)
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
            
            skip_id = f"{entry['platform']}:{entry['model_id']}:{fake_key_id}"
            if skip_keys and skip_id in skip_keys:
                continue

            if is_on_cooldown(entry["platform"], entry["model_id"], fake_key_id):
                continue

            if not can_use_provider(entry["platform"], fake_key_id):
                continue

            if not can_make_request(entry["platform"], entry["model_id"], fake_key_id, limits):
                continue
            if not can_use_tokens(entry["platform"], entry["model_id"], fake_key_id, estimated_tokens, limits):
                continue

            resolved_provider = provider
            apiKey = raw_key
            if entry["platform"] == 'custom':
                sep = raw_key.find('::')
                if sep != -1:
                    custom_url = raw_key[:sep]
                    apiKey = raw_key[sep+2:]
                    resolved_provider = resolve_provider('custom', custom_url)
                else:
                    resolved_provider = resolve_provider('custom', 'https://api.openai.com/v1')

            if not resolved_provider:
                continue

            return {
                "provider": resolved_provider,
                "modelId": entry["model_id"],
                "modelDbId": entry["model_db_id"],
                "apiKey": apiKey,
                "keyId": fake_key_id,
                "platform": entry["platform"],
                "displayName": entry["display_name"],
                "rpdLimit": limits["rpd"],
                "tpdLimit": limits["tpd"]
            }


    raise RouteError('All models and keys exhausted. Please check your keys.json configuration.', 429)

def get_routing_scores() -> Dict[str, Any]:
    ensure_models_loaded()
    scores = []
    for idx, m in enumerate(MODELS_DATA):
        scores.append({
            "modelDbId": m.get("id", idx),
            "platform": m.get("platform"),
            "modelId": m.get("model_id"),
            "displayName": m.get("display_name"),
            "enabled": True,
            "reliability": 1.0,
            "speed": 1.0,
            "intelligence": 1.0,
            "headroom": 1.0,
            "rateLimit": 1.0,
            "score": float(100 - idx),
            "totalRequests": 0,
        })
    return {
        "strategy": "priority",
        "weights": {"reliability": 1.0, "speed": 0.0, "intelligence": 0.0},
        "customWeights": {"reliability": 1.0, "speed": 0.0, "intelligence": 0.0},
        "scores": scores
    }

def has_enabled_vision_model() -> bool:
    ensure_models_loaded()
    return any(m.get("supports_vision") == 1 for m in MODELS_DATA)

def has_enabled_tools_model() -> bool:
    ensure_models_loaded()
    return any(m.get("supports_tools") == 1 for m in MODELS_DATA)

