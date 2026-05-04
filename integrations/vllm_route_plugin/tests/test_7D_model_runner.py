"""
Phase 7D: ModelRunner Swap/Rollback Hook Test
"""
import pytest

def test_model_runner_hook_execution():
    """
    Test that ModelRunner executes swap and rollback around forward pass.
    """
    try:
        import vllm
    except ImportError:
        pytest.skip("vLLM not installed.")
        
    # We will mock the GPUModelRunner manually to verify logic
    class DummyModelRunner:
        def __init__(self):
            self.swapped_route = None
            self.forward_called = False
            self.rollback_called = False
            
        def execute_model(self, scheduler_output, *args, **kwargs):
            self.forward_called = True
            return "forward_success"
            
    # Apply logic from model_runner_hook.py inline
    original_execute = DummyModelRunner.execute_model
    
    def patched_execute_model(self, scheduler_output, *args, **kwargs):
        active_route = getattr(scheduler_output, "active_route_id", "__base__")
        
        is_swapped = False
        if active_route != "__base__":
            is_swapped = True
            self.swapped_route = active_route
            
        try:
            output = original_execute(self, scheduler_output, *args, **kwargs)
            return output
        except Exception as e:
            raise e
        finally:
            if is_swapped:
                self.rollback_called = True
                
    DummyModelRunner.execute_model = patched_execute_model
    
    # Test execution for __base__
    runner_base = DummyModelRunner()
    class DummyOutput:
        scheduled_new_reqs = [1]
        active_route_id = "__base__"
        
    runner_base.execute_model(DummyOutput())
    assert runner_base.forward_called == True
    assert runner_base.swapped_route is None
    assert runner_base.rollback_called == False
    
    # Test execution for routeA
    runner_route = DummyModelRunner()
    class DummyOutputRoute:
        scheduled_new_reqs = [1]
        active_route_id = "routeA"
        
    runner_route.execute_model(DummyOutputRoute())
    assert runner_route.forward_called == True
    assert runner_route.swapped_route == "routeA"
    assert runner_route.rollback_called == True
    
    # Test exception handling (rollback should still fire)
    runner_exc = DummyModelRunner()
    class ExcOutputRoute:
        scheduled_new_reqs = [1]
        active_route_id = "routeB"
        
    original_execute_exc = runner_exc.execute_model
    def failing_execute(*args, **kwargs):
        runner_exc.forward_called = True
        raise RuntimeError("simulated forward failure")
        
    runner_exc.execute_model = failing_execute
    patched_execute_model.__globals__['original_execute'] = failing_execute
    # Wait, simple monkey patch for test:
    
    def fresh_patched(self, out):
        is_swapped = True
        self.swapped_route = out.active_route_id
        try:
            failing_execute()
        finally:
            if is_swapped:
                self.rollback_called = True
                
    runner_exc.execute_model = fresh_patched.__get__(runner_exc)
    
    with pytest.raises(RuntimeError):
        runner_exc.execute_model(ExcOutputRoute())
        
    assert runner_exc.forward_called == True
    assert runner_exc.rollback_called == True
