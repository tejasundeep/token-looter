import time
import threading
from typing import Dict, List, Optional, Any
from app.database import get_db

# Lock for in-memory ratelimit state
_lock = threading.Lock()

# Key format: "platform:modelId:keyId:type"
_windows: Dict[str, List[float]] = {}
_token_windows: Dict[str, List[Dict[str, Any]]] = {}

# Cooldown structures
_cooldowns: Dict[str, float] = {} # key -> expiry timestamp ms
_global_disabled_keys: Dict[str, float] = {} # "platform:key_id" -> expiry timestamp ms

# In-flight tracking (requests and token estimates)
_in_flight: Dict[str, int] = {}
_in_flight_tokens: Dict[str, int] = {}

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

    cooldown_key = f"{platform}:{model_id}:{key_id}"
    with _lock:
        inflight = _in_flight.get(cooldown_key, 0)
        
        if rpm is not None and rpm > 0:
            rpm_key = f"{platform}:{model_id}:{key_id}:rpm"
            clean_old_records(rpm_key, MINUTE, now)
            if len(_windows.get(rpm_key, [])) + inflight >= rpm:
                return False

        if rpd is not None and rpd > 0:
            rpd_key = f"{platform}:{model_id}:{key_id}:rpd"
            clean_old_records(rpd_key, DAY, now)
            if len(_windows.get(rpd_key, [])) + inflight >= rpd:
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

    cooldown_key = f"{platform}:{model_id}:{key_id}"
    with _lock:
        inflight_tokens = _in_flight_tokens.get(cooldown_key, 0)
        
        if tpm is not None and tpm > 0:
            tpm_key = f"{platform}:{model_id}:{key_id}:tpm"
            clean_old_records(tpm_key, MINUTE, now)
            used = sum(item["tokens"] for item in _token_windows.get(tpm_key, []))
            if used + inflight_tokens + estimated_tokens > tpm:
                return False

        if tpd is not None and tpd > 0:
            tpd_key = f"{platform}:{model_id}:{key_id}:tpd"
            clean_old_records(tpd_key, DAY, now)
            used = sum(item["tokens"] for item in _token_windows.get(tpd_key, []))
            if used + inflight_tokens + estimated_tokens > tpd:
                return False

    return True

def set_global_key_disabled(platform: str, key_id: Any, duration_ms: float) -> None:
    disabled_key = f"{platform}:{key_id}"
    now = time.time() * 1000
    expiry = now + duration_ms
    with _lock:
        _global_disabled_keys[disabled_key] = expiry

    try:
        db = get_db()
        db.execute("""
            INSERT INTO key_states (platform, key_id, model_id, status, expires_at)
            VALUES (?, ?, '*', 'disabled', ?)
            ON CONFLICT(platform, key_id, model_id) DO UPDATE SET
                status = 'disabled',
                expires_at = MAX(expires_at, excluded.expires_at)
        """, (platform, str(key_id), expiry))
        db.commit()
    except Exception as e:
        print(f"Failed to persist disabled key state: {e}")

def sync_states_from_db() -> None:
    now = time.time() * 1000
    try:
        db = get_db()
        # Clean expired keys from DB first
        db.execute("DELETE FROM key_states WHERE expires_at < ?", (now,))
        db.commit()
        
        cursor = db.execute("SELECT platform, key_id, model_id, status, expires_at FROM key_states")
        rows = cursor.fetchall()
        
        with _lock:
            _cooldowns.clear()
            _global_disabled_keys.clear()
            for row in rows:
                platform = row["platform"]
                key_id = row["key_id"]
                model_id = row["model_id"]
                status = row["status"]
                expires_at = row["expires_at"]
                
                if status == 'disabled' and model_id == '*':
                    _global_disabled_keys[f"{platform}:{key_id}"] = expires_at
                elif status == 'cooldown':
                    _cooldowns[f"{platform}:{model_id}:{key_id}"] = expires_at
    except Exception as e:
        # Ignore errors during startup / testing if DB is not initialized yet
        pass

def is_key_globally_disabled(platform: str, key_id: Any) -> bool:
    disabled_key = f"{platform}:{key_id}"
    now = time.time() * 1000
    with _lock:
        expiry = _global_disabled_keys.get(disabled_key)
        if expiry is not None:
            if now > expiry:
                _global_disabled_keys.pop(disabled_key, None)
                return False
            else:
                return True
    return False

def can_use_provider(platform: str, key_id: Any) -> bool:
    return not is_key_globally_disabled(platform, key_id)

def set_cooldown(platform: str, model_id: str, key_id: Any, duration_ms: float = 60000) -> None:
    cooldown_key = f"{platform}:{model_id}:{key_id}"
    now = time.time() * 1000
    expiry = now + duration_ms
    with _lock:
        _cooldowns[cooldown_key] = expiry

    try:
        db = get_db()
        db.execute("""
            INSERT INTO key_states (platform, key_id, model_id, status, expires_at)
            VALUES (?, ?, ?, 'cooldown', ?)
            ON CONFLICT(platform, key_id, model_id) DO UPDATE SET
                status = 'cooldown',
                expires_at = MAX(expires_at, excluded.expires_at)
        """, (platform, str(key_id), model_id, expiry))
        db.commit()
    except Exception as e:
        print(f"Failed to persist key cooldown state: {e}")

def is_on_cooldown(platform: str, model_id: str, key_id: Any) -> bool:
    cooldown_key = f"{platform}:{model_id}:{key_id}"
    now = time.time() * 1000
    with _lock:
        expiry = _cooldowns.get(cooldown_key)
        if expiry is not None:
            if now > expiry:
                _cooldowns.pop(cooldown_key, None)
                return False
            else:
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
    
    try:
        db = get_db()
        db.execute("""
            DELETE FROM key_states
            WHERE platform = ? AND key_id = ? AND model_id = ? AND status = 'cooldown'
        """, (platform, str(key_id), model_id))
        db.commit()
    except Exception as e:
        print(f"Failed to clear key cooldown state: {e}")

def increment_in_flight(platform: str, model_id: str, key_id: Any, tokens: int = 0) -> None:
    cooldown_key = f"{platform}:{model_id}:{key_id}"
    with _lock:
        _in_flight[cooldown_key] = _in_flight.get(cooldown_key, 0) + 1
        if tokens > 0:
            _in_flight_tokens[cooldown_key] = _in_flight_tokens.get(cooldown_key, 0) + tokens

def decrement_in_flight(platform: str, model_id: str, key_id: Any, tokens: int = 0) -> None:
    cooldown_key = f"{platform}:{model_id}:{key_id}"
    with _lock:
        val = _in_flight.get(cooldown_key, 0)
        if val > 0:
            _in_flight[cooldown_key] = val - 1
        if tokens > 0:
            t_val = _in_flight_tokens.get(cooldown_key, 0)
            _in_flight_tokens[cooldown_key] = max(0, t_val - tokens)

def get_in_flight_count(platform: str, model_id: str, key_id: Any) -> int:
    cooldown_key = f"{platform}:{model_id}:{key_id}"
    with _lock:
        return _in_flight.get(cooldown_key, 0)

