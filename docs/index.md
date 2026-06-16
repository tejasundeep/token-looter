# Token Looter Documentation Index

Welcome to the documentation for **token-looter**, a zero-configuration, headless Python API proxy gateway for LLMs.


## Guides & Reference

*   [**Architecture & Design (`docs/architecture.md`)**](file:///c:/Users/Teja/Desktop/freellmapi/token-looter/docs/architecture.md)
    Understand the backend flow, stateless in-memory fallback, and key rotation mechanisms.
*   [**Configuration Guide (`docs/configuration.md`)**](file:///c:/Users/Teja/Desktop/freellmapi/token-looter/docs/configuration.md)
    A guide to environment variables, keys.json schema, and sliding window rate-limits.
*   [**API Reference (`docs/api-reference.md`)**](file:///c:/Users/Teja/Desktop/freellmapi/token-looter/docs/api-reference.md)
    Developer reference for public proxy endpoints (`/v1/chat/completions`, `/v1/embeddings`), tool-call rescues, and the custom `/v1/responses` shim API.

## Quickstart

To get up and running immediately:

```bash
# Clone and enter directory
cd token-looter

# Configure keys.json at root directory (see README.md for schema)

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --host 0.0.0.0 --port 3001
```
