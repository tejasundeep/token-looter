from typing import Dict, List, Optional
from app.providers.base import BaseProvider
from app.providers.google import GoogleProvider
from app.providers.openai_compat import OpenAICompatProvider
from app.providers.cohere import CohereProvider
from app.providers.cloudflare import CloudflareProvider

_providers: Dict[str, BaseProvider] = {}

def register(provider: BaseProvider) -> None:
    _providers[provider.platform] = provider

register(GoogleProvider())

register(OpenAICompatProvider(
    platform='groq',
    name='Groq',
    base_url='https://api.groq.com/openai/v1',
))

register(OpenAICompatProvider(
    platform='cerebras',
    name='Cerebras',
    base_url='https://api.cerebras.ai/v1',
))

register(OpenAICompatProvider(
    platform='nvidia',
    name='NVIDIA NIM',
    base_url='https://integrate.api.nvidia.com/v1',
    force_single_tool_call=True,
))

register(OpenAICompatProvider(
    platform='mistral',
    name='Mistral',
    base_url='https://api.mistral.ai/v1',
))

register(OpenAICompatProvider(
    platform='openrouter',
    name='OpenRouter',
    base_url='https://openrouter.ai/api/v1',
    extra_headers={
        'HTTP-Referer': 'http://localhost:3001',
        'X-Title': 'TokenLooter',
    },
))

register(OpenAICompatProvider(
    platform='github',
    name='GitHub Models',
    base_url='https://models.github.ai/inference',
))

register(CohereProvider())
register(CloudflareProvider())

register(OpenAICompatProvider(
    platform='zhipu',
    name='Zhipu AI',
    base_url='https://open.bigmodel.cn/api/paas/v4',
))

register(OpenAICompatProvider(
    platform='huggingface',
    name='HuggingFace Router',
    base_url='https://router.huggingface.co/v1',
))

register(OpenAICompatProvider(
    platform='ollama',
    name='Ollama Cloud',
    base_url='https://ollama.com/v1',
    timeout_ms=120000.0,
))

register(OpenAICompatProvider(
    platform='kilo',
    name='Kilo Gateway',
    base_url='https://api.kilo.ai/api/gateway/v1',
    validate_url='https://api.kilo.ai/api/gateway/models',
    keyless=True,
))

register(OpenAICompatProvider(
    platform='pollinations',
    name='Pollinations',
    base_url='https://text.pollinations.ai/openai/v1',
    keyless=True,
))

register(OpenAICompatProvider(
    platform='llm7',
    name='LLM7',
    base_url='https://api.llm7.io/v1',
))

register(OpenAICompatProvider(
    platform='opencode',
    name='OpenCode Zen',
    base_url='https://opencode.ai/zen/v1',
))

register(OpenAICompatProvider(
    platform='ovh',
    name='OVH AI Endpoints',
    base_url='https://oai.endpoints.kepler.ai.cloud.ovh.net/v1',
    keyless=True,
))

register(OpenAICompatProvider(
    platform='custom',
    name='Custom (OpenAI-compatible)',
    base_url='',
))

CUSTOM_PROVIDER_TIMEOUT_MS = 120000.0

def get_provider(platform: str) -> Optional[BaseProvider]:
    return _providers.get(platform)

def resolve_provider(platform: str, base_url: Optional[str] = None) -> Optional[BaseProvider]:
    if platform == 'custom':
        trimmed = base_url.strip() if base_url else ""
        if not trimmed:
            return None
        return OpenAICompatProvider(
            platform='custom',
            name='Custom (OpenAI-compatible)',
            base_url=trimmed,
            timeout_ms=CUSTOM_PROVIDER_TIMEOUT_MS,
        )
    return _providers.get(platform)

def get_all_providers() -> List[BaseProvider]:
    return list(_providers.values())

def has_provider(platform: str) -> bool:
    return platform in _providers
