import os
import httpx
import threading
from typing import Optional, Dict, Tuple

class SharedAsyncClient(httpx.AsyncClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_opened = False
        self._open_lock = threading.Lock()

    async def __aenter__(self):
        with self._open_lock:
            if not self._is_opened:
                await super().__aenter__()
                self._is_opened = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Do not close the connection pool when exiting context blocks
        pass

    async def force_close(self):
        with self._open_lock:
            if self._is_opened:
                await super().__aexit__(None, None, None)
                self._is_opened = False


_clients: Dict[Tuple[Optional[str], float], SharedAsyncClient] = {}
_lock = threading.Lock()

def get_httpx_client(platform: Optional[str] = None, timeout: float = 15.0) -> httpx.AsyncClient:
    key = (platform, timeout)
    with _lock:
        if key in _clients:
            return _clients[key]
        
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
            
        client = SharedAsyncClient(**kwargs)
        _clients[key] = client
        return client

async def close_all_clients() -> None:
    with _lock:
        for client in list(_clients.values()):
            try:
                await client.force_close()
            except Exception:
                pass
        _clients.clear()


