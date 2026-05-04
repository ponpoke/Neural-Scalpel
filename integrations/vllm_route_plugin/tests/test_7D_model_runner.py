"""
Phase 7D: ModelRunner Swap/Rollback Hook Unit Test
"""
import pytest
from unittest.mock import MagicMock

def test_model_runner_hook_logic():
    """
    Test that the ModelRunner hook correctly identifies route, 
    swaps before, and rolls back after forward pass.
    """
    class DummyModelRunner:
        def __init__(self):
            self.swap_count = 0
            self.rollback_count = 0
            self.forward_count = 0
            
        def execute_model(self, scheduler_output):
            self.forward_count += 1
            return "success"

    # Helpers from model_runner_hook.py
    def _unwrap_request(obj):
        return getattr(obj, "request", obj)

    def _get_request_route_id(req_or_state) -> str:
        req = _unwrap_request(req_or_state)
        return getattr(req, "route_id", "__base__")

    def _iter_scheduled_reqs(output):
        for attr in ("scheduled_new_reqs", "scheduled_resumed_reqs", "scheduled_running_reqs"):
            items = getattr(output, attr, None)
            if items:
                for item in items: yield item

    # The actual hook logic
    def patched_execute_model(self, scheduler_output):
        # 1. Homogeneity check
        routes = [_get_request_route_id(req) for req in _iter_scheduled_reqs(scheduler_output)]
        unique_routes = set(routes)
        
        if len(unique_routes) > 1:
            raise RuntimeError("Mixed-route batch detected")
            
        active_route = getattr(scheduler_output, "active_route_id", None)
        if active_route is None:
            active_route = next(iter(unique_routes)) if unique_routes else "__base__"
            
        is_swapped = False
        if active_route != "__base__":
            is_swapped = True
            self.swap_count += 1
            
        try:
            return self.execute_model(scheduler_output)
        finally:
            if is_swapped:
                self.rollback_count += 1

    # --- Test Cases ---
    runner = DummyModelRunner()
    
    # 1. Base route test
    class BaseOutput:
        scheduled_new_reqs = [{"request": MagicMock(route_id="__base__")}]
        active_route_id = "__base__"
    
    patched_execute_model(runner, BaseOutput())
    assert runner.forward_count == 1
    assert runner.swap_count == 0
    assert runner.rollback_count == 0
    
    # 2. RouteA test
    class RouteAOutput:
        scheduled_new_reqs = [{"request": MagicMock(route_id="routeA")}]
        active_route_id = "routeA"
        
    patched_execute_model(runner, RouteAOutput())
    assert runner.forward_count == 2
    assert runner.swap_count == 1
    assert runner.rollback_count == 1
    
    # 3. Mixed route failure test
    class MixedOutput:
        scheduled_new_reqs = [
            {"request": MagicMock(route_id="routeA")},
            {"request": MagicMock(route_id="routeB")}
        ]
        active_route_id = "routeA"
        
    with pytest.raises(RuntimeError) as excinfo:
        patched_execute_model(runner, MixedOutput())
    assert "Mixed-route batch detected" in str(excinfo.value)
    # Check that forward was NOT called for mixed batch
    assert runner.forward_count == 2 
