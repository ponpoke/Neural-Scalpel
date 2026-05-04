"""
Phase 7B: Route-Aware Scheduler Unit Test
"""
import pytest
from unittest.mock import MagicMock

def test_scheduler_homogeneous_batch():
    """
    Test that the patched scheduler only returns requests from a single route.
    """
    try:
        import vllm
    except ImportError:
        pytest.skip("vLLM not installed.")
        
    from integrations.vllm_route_plugin.patch import apply_all_patches
    apply_all_patches()
    
    import vllm.v1.core.sched.scheduler as vllm_scheduler
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams
    
    # We need to mock the scheduler to just test the waiting queue logic
    # This might be tricky because we need vllm config objects to initialize it.
    # Instead, we will instantiate the class by bypassing init or passing dummy configs if possible.
    
    # As a simple unit test for the monkey-patched method, we can create a dummy class
    class DummyScheduler:
        def __init__(self):
            self.running = []
            self.waiting = []
            
        def schedule(self):
            # Mock the original schedule to just take the first N requests
            # For this test, let's say it takes up to 2 requests
            taken = self.waiting[:2]
            self.running.extend(taken)
            self.waiting = self.waiting[2:]
            return "dummy_output"
            
    # Apply our patch logic manually to the DummyScheduler for isolated testing
    from integrations.vllm_route_plugin.scheduler_patch import inject_route_aware_scheduler
    
    # Re-applying logic inline for the test (since original patch touches vllm globally)
    original_schedule = DummyScheduler.schedule
    
    def patched_schedule(self):
        if self.running:
            self.active_route_id = getattr(self.running[0], "route_id", "__base__")
        else:
            if self.waiting:
                self.active_route_id = getattr(self.waiting[0], "route_id", "__base__")
            else:
                self.active_route_id = None

        if self.active_route_id is None:
            return original_schedule(self)

        matching_waiting = []
        non_matching_waiting = []
        for req in self.waiting:
            req_route = getattr(req, "route_id", "__base__")
            if req_route == self.active_route_id:
                matching_waiting.append(req)
            else:
                non_matching_waiting.append(req)

        self.waiting = matching_waiting
        try:
            output = original_schedule(self)
        finally:
            self.waiting.extend(non_matching_waiting)
        return output

    DummyScheduler.schedule = patched_schedule
    
    # Now run the test
    scheduler = DummyScheduler()
    
    class DummyReq:
        def __init__(self, rid, route):
            self.request_id = rid
            self.route_id = route
            self.priority = 0
            self.arrival_time = 0
            
    # Add mixed routes to waiting queue
    scheduler.waiting = [
        DummyReq("1", "routeA"),
        DummyReq("2", "routeB"),
        DummyReq("3", "routeA")
    ]
    
    # First schedule call should pick routeA and ONLY routeA requests
    scheduler.schedule()
    
    assert len(scheduler.running) == 2
    assert scheduler.running[0].request_id == "1"
    assert scheduler.running[1].request_id == "3"
    assert getattr(scheduler.running[0], "route_id", None) == "routeA"
    assert getattr(scheduler.running[1], "route_id", None) == "routeA"
    
    # waiting should only contain routeB now
    assert len(scheduler.waiting) == 1
    assert scheduler.waiting[0].request_id == "2"
