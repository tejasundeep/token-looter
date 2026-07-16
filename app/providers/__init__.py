from typing import Dict, List, Optional
from app.providers.base import BaseProvider
from app.providers.google import GoogleProvider
from app.providers.openai_compat import OpenAICompatProvider
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

register(CloudflareProvider())

register(OpenAICompatProvider(
    platform='zhipu',
    name='Zhipu AI',
    base_url='https://open.bigmodel.cn/api/paas/v4',
))

register(OpenAICompatProvider(
    platform='llm7',
    name='LLM7',
    base_url='https://api.llm7.io/v1',
))

def get_provider(platform: str) -> Optional[BaseProvider]:
    return _providers.get(platform)

def resolve_provider(platform: str, base_url: Optional[str] = None) -> Optional[BaseProvider]:
    return _providers.get(platform)

def get_all_providers() -> List[BaseProvider]:
    return list(_providers.values())

def has_provider(platform: str) -> bool:
    return platform in _providers
