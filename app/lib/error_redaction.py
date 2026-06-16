import re

MAX_PROVIDER_ERROR_LENGTH = 240

REDACTIONS = [
    (re.compile(r'\bBearer\s+[A-Za-z0-9._~+/-]+=*', re.IGNORECASE), 'Bearer [redacted]'),
    (re.compile(r"""\b(api[_-]?key|access[_-]?token|token|secret|authorization)(\s*[:=]\s*)(["']?)[^"',\s}\]]+""", re.IGNORECASE), r'\1\2\3[redacted]'),
    (re.compile(r'\bsk-[A-Za-z0-9_-]{8,}\b'), '[redacted-key]'),
    (re.compile(r'\bgsk_[A-Za-z0-9_-]{8,}\b'), '[redacted-key]'),
    (re.compile(r'\btokenlooter-[A-Za-z0-9_-]{8,}\b'), '[redacted-key]'),
    (re.compile(r'\bAIza[0-9A-Za-z_-]{20,}\b'), '[redacted-key]'),
    (re.compile(r'\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b'), '[redacted-token]'),
    (re.compile(r'\bhttps?://[^\s"\'<>)]*', re.IGNORECASE), '[redacted-url]'),
]

def sanitize_provider_error_message(message) -> str:
    sanitized = str(message) if message is not None else ''
    sanitized = sanitized.strip()
    if not sanitized:
        return 'Provider error'
        
    for pattern, replacement in REDACTIONS:
        sanitized = pattern.sub(replacement, sanitized)
        
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    if len(sanitized) > MAX_PROVIDER_ERROR_LENGTH:
        sanitized = sanitized[:MAX_PROVIDER_ERROR_LENGTH - 3].rstrip() + '...'
    return sanitized
