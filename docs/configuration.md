# Configuration & Environment Variables

This document defines all configuration settings and environment variables used to tune **token-looter**.

## Environment Variables

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `DATABASE_PATH` | String | `data/freeapi.db` | File path to the SQLite database. Used strictly for request history analytics logs. |
| `PROXY_URL` | String | `""` | Global outbound proxy URL (e.g. `http://proxy:8080`) for routing outgoing LLM calls. |
| `TOKENLOOTER_CONTEXT_HANDOFF` | String | `""` | Set to `on_model_switch` to enable system instructions context handoff when switching models. |

---

## keys.json File Schema

API keys and gateway authorization tokens are declared in a static `keys.json` file at the root folder:

```json
{
  "unified_api_key": "your_custom_gateway_access_token_here",
  "providers": {
    "google": ["AIzaSy..."],
    "groq": ["gsk_..."],
    "cerebras": [],
    "nvidia": [],
    "mistral": [],
    "openrouter": [],
    "github": [],
    "cohere": [],
    "cloudflare": [],
    "zhipu": [],
    "huggingface": [],
    "ollama": [],
    "kilo": [],
    "pollinations": [],
    "llm7": [],
    "opencode": [],
    "ovh": [],
    "custom": []
  }
}
```

---

## Sliding Window Rate Limits

Rate-limiting requests uses thread-safe, in-memory sliding windows for each configured key:
*   **RPM/TPM**: Requests and tokens checked within a rolling 60-second window.
*   **RPD/TPD**: Requests and tokens checked within a rolling 24-hour window.
*   **Cooldown Durations**: A temporary penalty of **90 seconds** is applied to keys that hit rate-limits or throw transient errors.
