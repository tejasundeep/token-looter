# Proxy API Reference

This reference details the public OpenAI-compatible `/v1/*` proxy endpoints.

---

## 1. Chat Completions

Generate completions for messages.

*   **URL**: `/v1/chat/completions`
*   **Method**: `POST`
*   **Headers**:
    *   `Authorization: Bearer <unified_api_key>`
*   **Body (JSON)**:
    *   `messages` (Array, Required, Min Length 1): List of chat message objects.
    *   `model` (String, Optional): Model ID. Use `"auto"` (or omit) to let the router pick the best model.
    *   `stream` (Boolean, Optional): Stream response chunks using SSE (`text/event-stream`).

### Response Headers
*   `X-Routed-Via`: Identifies the backend platform and model that served the request (e.g. `google/gemini-2.5-flash`).
*   `X-Fallback-Attempts`: Included if the router had to fail over from a rate-limited or broken model/key (e.g. `1`).

---

## 2. Embeddings

Generate vector representations of input text.

*   **URL**: `/v1/embeddings`
*   **Method**: `POST`
*   **Body (JSON)**:
    *   `input` (String or Array of Strings, Required): The text to embed.
    *   `model` (String, Optional): The embedding model ID (e.g. `text-embedding-3-small`).

---

## 3. List Models

List available models.

*   **URL**: `/v1/models`
*   **Method**: `GET`
*   **Query Parameters**:
    *   `available` (Boolean, Optional): Set to `true` to return only models that currently have healthy, enabled API keys configured.

---

## 4. Responses (Codex Responses API)

A specialized endpoint matching the legacy Codex Responses schema.

*   **URL**: `/v1/responses`
*   **Method**: `POST`
*   **Body (JSON)**:
    *   `input` (String, Required): The user prompt.
    *   `system_instruction` (String, Optional): System instructions.
    *   `conversation` (Array, Optional): Prior dialogue context.
    *   `tools` (Array, Optional): Function calling tool definitions.
    *   `stream` (Boolean, Optional): Stream responses via custom Codex event-stream format.
