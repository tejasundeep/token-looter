import os
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from app.lib.content import content_to_string

_lock = threading.Lock()

class TrimmedMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class SessionContext:
    def __init__(self, last_model_key: Optional[str] = None):
        self.last_model_key = last_model_key
        self.recent_messages: List[TrimmedMessage] = []
        self.updated_at = time.time() * 1000

MAX_RECENT_MESSAGES = 12
MAX_HANDOFF_CHARS = 6000
MAX_CONTENT_PER_MSG = 500
SESSION_TTL_MS = 3 * 60 * 60 * 1000 # 3 hours
MAX_STORE_SIZE = 500

HANDOFF_MAX_TOKENS = int((MAX_HANDOFF_CHARS + 400 + 3) / 4) # 1600

# Store session contexts in memory
_store: Dict[str, SessionContext] = {}

def get_context_handoff_mode() -> str:
    raw = os.environ.get("TOKENLOOTER_CONTEXT_HANDOFF", "").strip().lower()
    return "on_model_switch" if raw == "on_model_switch" else "off"

def trim_content(content: Any) -> str:
    text = content_to_string(content)
    if len(text) > MAX_CONTENT_PER_MSG:
        return text[:MAX_CONTENT_PER_MSG] + "…"
    return text

def prune_expired() -> None:
    now = time.time() * 1000
    expired_keys = [k for k, ctx in _store.items() if now - ctx.updated_at > SESSION_TTL_MS]
    for k in expired_keys:
        _store.pop(k, None)

def record_incoming_messages(session_key: str, messages: List[Dict[str, Any]]) -> None:
    if not session_key:
        return
    
    with _lock:
        prune_expired()

        trimmed = []
        for m in messages:
            role = m.get("role")
            if role in ("user", "assistant"):
                trimmed.append(TrimmedMessage(role=role, content=trim_content(m.get("content"))))
        
        trimmed = trimmed[-MAX_RECENT_MESSAGES:]

        has_assistant = any(m.get("role") == "assistant" for m in messages)
        
        if session_key in _store:
            existing = _store[session_key]
            if not has_assistant:
                existing.last_model_key = None
            existing.recent_messages = trimmed
            existing.updated_at = time.time() * 1000
        else:
            ctx = SessionContext()
            ctx.recent_messages = trimmed
            _store[session_key] = ctx

        # Enforce size limit
        if len(_store) > MAX_STORE_SIZE:
            # Re-prune expired first
            prune_expired()
            if len(_store) > MAX_STORE_SIZE:
                # Evict oldest by updatedAt
                sorted_entries = sorted(_store.items(), key=lambda item: item[1].updated_at)
                evict_count = len(_store) - MAX_STORE_SIZE
                for k, _ in sorted_entries[:evict_count]:
                    _store.pop(k, None)

def build_summary(messages: List[TrimmedMessage]) -> str:
    lines = []
    for m in messages:
        role_label = "User" if m.role == "user" else "Assistant"
        lines.append(f"{role_label}: {m.content}")
    joined = "\n".join(lines)
    if len(joined) > MAX_HANDOFF_CHARS:
        return joined[:MAX_HANDOFF_CHARS] + "\n…[truncated]"
    return joined

def has_prior_model(session_key: str) -> bool:
    if not session_key:
        return False
    with _lock:
        ctx = _store.get(session_key)
        return ctx is not None and ctx.last_model_key is not None and ctx.last_model_key != ""

def maybe_inject_context_handoff(
    mode: str,
    session_key: str,
    messages: List[Dict[str, Any]],
    selected_model_key: str
) -> Tuple[List[Dict[str, Any]], bool, int]:
    if mode == "off" or not session_key:
        return messages, False, 0

    with _lock:
        ctx = _store.get(session_key)
        if not ctx or not ctx.last_model_key or ctx.last_model_key == selected_model_key:
            return messages, False, 0

    # Skip if handoff message is already present
    already_present = False
    for m in messages:
        if m.get("role") == "system":
            text = content_to_string(m.get("content"))
            if text.startswith("TokenLooter context handoff:"):
                already_present = True
                break
    if already_present:
        return messages, False, 0

    with _lock:
        summary = build_summary(ctx.recent_messages)
        last_model = ctx.last_model_key

    handoff_content = "\n".join([
        "TokenLooter context handoff:",
        f"You are taking over an ongoing conversation from another model ({last_model} → {selected_model_key}).",
        "Continue the user's task using the conversation context already provided in this request.",
        "Do not restart the task, re-ask already answered setup questions, or discard prior tool results.",
        "Respect the user's latest message as the highest-priority instruction.",
        "",
        "Recent session summary:",
        summary
    ])

    handoff_msg = {"role": "system", "content": handoff_content}

    # Insert after any leading system messages
    insert_at = -1
    for idx, m in enumerate(messages):
        if m.get("role") != "system":
            insert_at = idx
            break
    
    pos = len(messages) if insert_at == -1 else insert_at

    new_messages = list(messages)
    new_messages.insert(pos, handoff_msg)

    return new_messages, True, int((len(handoff_content) + 3) / 4)

def record_successful_model(session_key: str, model_key: str) -> None:
    if not session_key:
        return
    with _lock:
        prune_expired()
        if session_key in _store:
            ctx = _store[session_key]
            ctx.last_model_key = model_key
            ctx.updated_at = time.time() * 1000
        else:
            ctx = SessionContext(last_model_key=model_key)
            _store[session_key] = ctx
            
            # Enforce size limit
            if len(_store) > MAX_STORE_SIZE:
                sorted_entries = sorted(_store.items(), key=lambda item: item[1].updated_at)
                evict_count = len(_store) - MAX_STORE_SIZE
                for k, _ in sorted_entries[:evict_count]:
                    _store.pop(k, None)

def _clear_store_for_testing() -> None:
    with _lock:
        _store.clear()
