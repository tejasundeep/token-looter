import time
import threading
from typing import Dict, List, Optional, Any

# Lock for in-memory ratelimit state
_lock = threading.Lock()

# Key format: "platform:modelId:keyId:type"
_windows: Dict[str, List[float]] = {}
_token_windows: Dict[str, List[Dict[str, Any]]] = {}

# Cooldown structures
_cooldowns: Dict[str, float] = {} # key -> expiry timestamp ms
_global_disabled_keys: Dict[str, float] = {} # "platform:key_id" -> expiry timestamp ms

MINUTE = 60 * 1000
PAYMENT_REQUIRED_COOLDOWN_MS = 24 * 60 * 60 * 1000
DAY = 24 * 60 * MINUTE

def clean_old_records(window_key: str, window_ms: float, now: float):
    # Keep only records within window_ms
    cutoff = now - window_ms
    if window_key in _windows:
        _windows[window_key] = [t for t in _windows[window_key] if t > cutoff]
    if window_key in _token_windows:
        _token_windows[window_key] = [t for t in _token_windows[window_key] if t["ts"] > cutoff]

def record_request(platform: str, model_id: str, key_id: Any) -> None:
    now = time.time() * 1000
    rpm_key = f"{platform}:{model_id}:{key_id}:rpm"
    rpd_key = f"{platform}:{model_id}:{key_id}:rpd"
    
    with _lock:
        if rpm_key not in _windows:
            _windows[rpm_key] = []
        _windows[rpm_key].append(now)
        
        if rpd_key not in _windows:
            _windows[rpd_key] = []
        _windows[rpd_key].append(now)

def record_tokens(platform: str, model_id: str, key_id: Any, tokens: int) -> None:
    now = time.time() * 1000
    tpm_key = f"{platform}:{model_id}:{key_id}:tpm"
    tpd_key = f"{platform}:{model_id}:{key_id}:tpd"
    
    with _lock:
        if tpm_key not in _token_windows:
            _token_windows[tpm_key] = []
        _token_windows[tpm_key].append({"ts": now, "tokens": tokens})
        
        if tpd_key not in _token_windows:
            _token_windows[tpd_key] = []
        _token_windows[tpd_key].append({"ts": now, "tokens": tokens})

def can_make_request(
    platform: str,
    model_id: str,
    key_id: Any,
    limits: Dict[str, Optional[int]]
) -> bool:
    now = time.time() * 1000
    rpm = limits.get("rpm")
    rpd = limits.get("rpd")

    with _lock:
        if rpm is not None and rpm > 0:
            rpm_key = f"{platform}:{model_id}:{key_id}:rpm"
            clean_old_records(rpm_key, MINUTE, now)
            if len(_windows.get(rpm_key, [])) >= rpm:
                return False

        if rpd is not None and rpd > 0:
            rpd_key = f"{platform}:{model_id}:{key_id}:rpd"
            clean_old_records(rpd_key, DAY, now)
            if len(_windows.get(rpd_key, [])) >= rpd:
                return False

    return True

def can_use_tokens(
    platform: str,
    model_id: str,
    key_id: Any,
    estimated_tokens: int,
    limits: Dict[str, Optional[int]]
) -> bool:
    now = time.time() * 1000
    tpm = limits.get("tpm")
    tpd = limits.get("tpd")

    with _lock:
        if tpm is not None and tpm > 0:
            tpm_key = f"{platform}:{model_id}:{key_id}:tpm"
            clean_old_records(tpm_key, MINUTE, now)
            used = sum(item["tokens"] for item in _token_windows.get(tpm_key, []))
            if used + estimated_tokens > tpm:
                return False

        if tpd is not None and tpd > 0:
            tpd_key = f"{platform}:{model_id}:{key_id}:tpd"
            clean_old_records(tpd_key, DAY, now)
            used = sum(item["tokens"] for item in _token_windows.get(tpd_key, []))
            if used + estimated_tokens > tpd:
                return False

    return True

def set_global_key_disabled(platform: str, key_id: Any, duration_ms: float) -> None:
    disabled_key = f"{platform}:{key_id}"
    now = time.time() * 1000
    with _lock:
        _global_disabled_keys[disabled_key] = now + duration_ms

def is_key_globally_disabled(platform: str, key_id: Any) -> bool:
    disabled_key = f"{platform}:{key_id}"
    now = time.time() * 1000
    with _lock:
        expiry = _global_disabled_keys.get(disabled_key)
        if expiry is not None:
            if now > expiry:
                _global_disabled_keys.pop(disabled_key, None)
                return False
            return True
    return False

def can_use_provider(platform: str, key_id: Any) -> bool:
    return not is_key_globally_disabled(platform, key_id)


def set_cooldown(platform: str, model_id: str, key_id: Any, duration_ms: float = 60000) -> None:
    cooldown_key = f"{platform}:{model_id}:{key_id}"
    now = time.time() * 1000
    with _lock:
        _cooldowns[cooldown_key] = now + duration_ms

def is_on_cooldown(platform: str, model_id: str, key_id: Any) -> bool:
    cooldown_key = f"{platform}:{model_id}:{key_id}"
    now = time.time() * 1000
    with _lock:
        expiry = _cooldowns.get(cooldown_key)
        if expiry is not None:
            if now > expiry:
                _cooldowns.pop(cooldown_key, None)
                return False
            return True
    return False

def get_cooldown_duration_for_limit(
    platform: str,
    model_id: str,
    key_id: Any,
    limits: Dict[str, Optional[int]],
    retry_after_ms: Optional[float] = None
) -> float:
    if retry_after_ms is not None:
        return min(retry_after_ms, DAY)
    return 90 * 1000 # Default 90 seconds transient cooldown

def clear_persisted_cooldown(platform: str, model_id: str, key_id: Any) -> None:
    cooldown_key = f"{platform}:{model_id}:{key_id}"
    with _lock:
        _cooldowns.pop(cooldown_key, None)

# In-flight request tracking for load balancing
_in_flight: Dict[str, int] = {}

def increment_in_flight(platform: str, model_id: str, key_id: Any) -> None:
    cooldown_key = f"{platform}:{model_id}:{key_id}"
    with _lock:
        _in_flight[cooldown_key] = _in_flight.get(cooldown_key, 0) + 1

def decrement_in_flight(platform: str, model_id: str, key_id: Any) -> None:
    cooldown_key = f"{platform}:{model_id}:{key_id}"
    with _lock:
        val = _in_flight.get(cooldown_key, 0)
        if val > 0:
            _in_flight[cooldown_key] = val - 1

def get_in_flight_count(platform: str, model_id: str, key_id: Any) -> int:
    cooldown_key = f"{platform}:{model_id}:{key_id}"
    with _lock:
        return _in_flight.get(cooldown_key, 0)

