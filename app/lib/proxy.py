import os
import httpx
from typing import Optional

def get_httpx_client(platform: Optional[str] = None, timeout: float = 15.0) -> httpx.AsyncClient:
    proxy_url = (
        os.environ.get("PROXY_URL") or 
        os.environ.get("http_proxy") or 
        os.environ.get("https_proxy") or 
        os.environ.get("HTTP_PROXY") or 
        os.environ.get("HTTPS_PROXY")
    )
    
    kwargs = {
        "timeout": httpx.Timeout(timeout),
        "follow_redirects": True
    }
    if proxy_url:
        kwargs["proxy"] = proxy_url
        
    return httpx.AsyncClient(**kwargs)

