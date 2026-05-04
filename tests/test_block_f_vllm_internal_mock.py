"""
Tests for vLLM Internal Integration Mock (Step 4B)
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from neural_scalpel.serving.vllm_internal_mock import (
    MockRequest, RouteAwareBlockAllocator, RouteAwareScheduler, RouteAwareModelRunner,
    UnsafeMixedRouteBatchError, RouteLeakageError
)

def test_scheduler_groups_by_route():
    scheduler = RouteAwareScheduler(max_batch_size=4)
    req1 = MockRequest("1", "Hello", "route-A")
    req2 = MockRequest("2", "Hi", "route-B")
    req3 = MockRequest("3", "Hey", "route-A")
    
    scheduler.add_request(req1)
    scheduler.add_request(req2)
    scheduler.add_request(req3)
    
    # First batch should only contain route-A requests (since req1 is first)
    batch1 = scheduler.schedule_next_batch()
    assert len(batch1) == 2
    assert all(r.route_id == "route-A" for r in batch1)
    
    # Second batch should contain route-B
    batch2 = scheduler.schedule_next_batch()
    assert len(batch2) == 1
    assert batch2[0].route_id == "route-B"

def test_scheduler_rejects_unsafe_mixed_route_without_isolation():
    allocator = RouteAwareBlockAllocator()
    runner = RouteAwareModelRunner(allocator)
    
    # Manually create an unsafe batch
    req1 = MockRequest("1", "Hello", "route-A")
    req2 = MockRequest("2", "Hi", "route-B")
    unsafe_batch = [req1, req2]
    
    with pytest.raises(UnsafeMixedRouteBatchError):
        runner.execute_forward_pass(unsafe_batch)

def test_kv_blocks_are_tagged_with_route_id():
    allocator = RouteAwareBlockAllocator()
    block_id_A = allocator.allocate("route-A")
    block_id_B = allocator.allocate("route-B")
    
    assert allocator.blocks[block_id_A].route_id == "route-A"
    assert allocator.blocks[block_id_B].route_id == "route-B"

def test_kv_cache_rejects_cross_route_reuse():
    allocator = RouteAwareBlockAllocator()
    runner = RouteAwareModelRunner(allocator)
    
    reqA = MockRequest("1", "A", "route-A")
    reqA.kv_blocks.append(allocator.allocate("route-A"))
    
    # Maliciously change route_id without clearing KV cache
    reqA.route_id = "route-B"
    
    with pytest.raises(RouteLeakageError):
        runner.execute_forward_pass([reqA])
        
    assert runner.metrics["route_leakage_events"] == 1

def test_model_runner_swaps_before_forward_and_rolls_back():
    allocator = RouteAwareBlockAllocator()
    runner = RouteAwareModelRunner(allocator)
    
    req = MockRequest("1", "A", "route-A")
    
    assert runner.active_weights_route is None
    runner.execute_forward_pass([req])
    
    # Rollback happens automatically after forward
    assert runner.active_weights_route is None
    assert runner.metrics["swaps_performed"] == 1

def test_failure_during_forward_triggers_rollback():
    allocator = RouteAwareBlockAllocator()
    runner = RouteAwareModelRunner(allocator)
    
    req = MockRequest("1", "A", "route-A")
    # Force a failure in allocator to trigger exception during forward
    allocator.allocate = lambda x: 1/0
    
    with pytest.raises(ZeroDivisionError):
        runner.execute_forward_pass([req])
        
    # Active weights should be rolled back to None
    assert runner.active_weights_route is None

def test_route_leakage_counter_stays_zero():
    allocator = RouteAwareBlockAllocator()
    runner = RouteAwareModelRunner(allocator)
    
    scheduler = RouteAwareScheduler(max_batch_size=2)
    for i in range(10):
        scheduler.add_request(MockRequest(f"A{i}", "A", "route-A"))
        scheduler.add_request(MockRequest(f"B{i}", "B", "route-B"))
        
    while scheduler.pending_requests:
        batch = scheduler.schedule_next_batch()
        runner.execute_forward_pass(batch)
        
    assert runner.metrics["route_leakage_events"] == 0
    assert runner.metrics["rollback_failures"] == 0
    assert runner.metrics["swaps_performed"] == 10 # 5 batches of A, 5 batches of B
