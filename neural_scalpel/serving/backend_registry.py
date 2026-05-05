import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field

@dataclass
class BackendInfo:
    url: str
    is_healthy: bool = True
    last_check_at: float = 0.0
    failure_count: int = 0

class BackendRegistry:
    """
    Registry for mapping route_ids to external vLLM backend URLs.
    
    This registry is used by the External Proxy Fallback mode to resolve
    where a request for a specific route should be forwarded.
    """
    def __init__(self):
        # Maps route_id -> BackendInfo
        self._route_mappings: Dict[str, BackendInfo] = {}
        # Maps backend_url -> BackendInfo (for shared health tracking)
        self._backends: Dict[str, BackendInfo] = {}

    def register_backend(self, route_id: str, url: str):
        """
        Registers a backend URL for a specific route_id.
        
        The URL should include the full endpoint, e.g., 
        'http://backend-a:8000/v1/completions'
        """
        if url not in self._backends:
            self._backends[url] = BackendInfo(url=url)
        
        backend_info = self._backends[url]
        self._route_mappings[route_id] = backend_info

    def resolve_backend(self, route_id: str) -> Optional[str]:
        """
        Resolves a route_id to a backend URL if the route exists and the backend is healthy.
        Returns None if not found or unhealthy.
        """
        backend_info = self._route_mappings.get(route_id)
        if backend_info and backend_info.is_healthy:
            return backend_info.url
        return None

    def update_health(self, url: str, is_healthy: bool):
        """Updates the health status of a backend URL."""
        if url in self._backends:
            info = self._backends[url]
            info.is_healthy = is_healthy
            info.last_check_at = time.time()
            if not is_healthy:
                info.failure_count += 1
            else:
                info.failure_count = 0

    def get_route_status(self, route_id: str) -> str:
        """Returns a string description of the route's backend status."""
        backend_info = self._route_mappings.get(route_id)
        if not backend_info:
            return "NOT_FOUND"
        if not backend_info.is_healthy:
            return "UNHEALTHY_BACKEND"
        return "READY"

    def list_routes(self) -> List[str]:
        """Lists all registered route_ids."""
        return list(self._route_mappings.keys())

    def remove_route(self, route_id: str):
        """Removes a route mapping from the registry."""
        self._route_mappings.pop(route_id, None)
