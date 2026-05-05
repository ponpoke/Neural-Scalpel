import pytest
from neural_scalpel.serving.backend_registry import BackendRegistry

def test_backend_registration_and_resolution():
    registry = BackendRegistry()
    registry.register_backend("alpaca", "http://backend-a:8000")
    registry.register_backend("sql", "http://backend-b:8000")
    
    assert registry.resolve_backend("alpaca") == "http://backend-a:8000"
    assert registry.resolve_backend("sql") == "http://backend-b:8000"
    assert registry.resolve_backend("unknown") is None

def test_backend_health_filtering():
    registry = BackendRegistry()
    url = "http://backend-a:8000"
    registry.register_backend("alpaca", url)
    
    # Healthy by default
    assert registry.resolve_backend("alpaca") == url
    assert registry.get_route_status("alpaca") == "READY"
    
    # Mark as unhealthy
    registry.update_health(url, is_healthy=False)
    assert registry.resolve_backend("alpaca") is None
    assert registry.get_route_status("alpaca") == "UNHEALTHY_BACKEND"
    
    # Mark as healthy again
    registry.update_health(url, is_healthy=True)
    assert registry.resolve_backend("alpaca") == url
    assert registry.get_route_status("alpaca") == "READY"

def test_shared_backend_health():
    # If multiple routes share the same backend, health updates should affect all
    registry = BackendRegistry()
    url = "http://shared-backend:8000"
    registry.register_backend("route1", url)
    registry.register_backend("route2", url)
    
    assert registry.resolve_backend("route1") == url
    assert registry.resolve_backend("route2") == url
    
    registry.update_health(url, is_healthy=False)
    assert registry.resolve_backend("route1") is None
    assert registry.resolve_backend("route2") is None

def test_list_routes():
    registry = BackendRegistry()
    registry.register_backend("r1", "url1")
    registry.register_backend("r2", "url2")
    
    routes = registry.list_routes()
    assert len(routes) == 2
    assert "r1" in routes
    assert "r2" in routes

def test_unknown_route_status():
    registry = BackendRegistry()
    assert registry.get_route_status("nonexistent") == "NOT_FOUND"
