import os
import json
import time
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from neural_scalpel.route.registry import RouteRegistry
from neural_scalpel.experimental.runtime import HotSwapRuntime, RuntimeState
from neural_scalpel.route.crypto import RouteSigner
from neural_scalpel.route.tenant import TenantContext

def setup_registry_and_routes():
    registry_dir = os.path.join(os.path.dirname(__file__), ".stress_registry")
    signer = RouteSigner({"stress-key": "secret"})
    registry = RouteRegistry(storage_dir=registry_dir, signer=signer)
    
    # We will create two dummy routes in memory, save them, and register them.
    # Route A injects +0.1, Route B injects -0.1
    route_a = {
        "route_schema_version": "0.1.0",
        "route_id": "stress-route-A",
        "source_model": "dummy",
        "target_model": "dummy",
        "source_adapter_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "target_model_sha256": "8b1a9953c4611296a827abf8c47804d7e4cf39d88532ee43640243e86ac58e77",
        "tenant_id": "stress-tenant",
        "license": "MIT",
        "projection_method": "DUMMY",
        "calibration": {"forward_passes": 64},
        "diagnostics": {"verdict": "PASS", "ppl_degradation": 0, "kl_divergence": 0, "portability_score": 100},
        "layers": [{"name": "layer_x", "shape": [10, 10], "dtype": "float32", "delta_sha256": "a"*64}]
    }
    
    route_b = dict(route_a)
    route_b["route_id"] = "stress-route-B"
    route_b["layers"] = [{"name": "layer_x", "shape": [10, 10], "dtype": "float32", "delta_sha256": "b"*64}]
    
    route_a = signer.sign(route_a, "stress-key")
    route_b = signer.sign(route_b, "stress-key")
    
    path_a = os.path.join(registry_dir, "route_a.json")
    path_b = os.path.join(registry_dir, "route_b.json")
    
    os.makedirs(registry_dir, exist_ok=True)
    with open(path_a, "w") as f: json.dump(route_a, f)
    with open(path_b, "w") as f: json.dump(route_b, f)
    
    registry.register_route(path_a)
    registry.register_route(path_b)
    
    return registry

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Target layer for Hot-Swap
        self.layer_x = torch.nn.Parameter(torch.zeros(10, 10, dtype=torch.float32))
        
    def forward(self, x):
        return torch.matmul(x, self.layer_x)

class StressTestRuntime(HotSwapRuntime):
    def swap(self, route_data: dict):
        self.transition(RuntimeState.SWAPPING)
        state = self._get_state_dict()
        with torch.no_grad():
            for layer in route_data.get("layers", []):
                name = layer["name"]
                live_tensor = state[name]
                # Route A adds 0.1, Route B subtracts 0.1
                if route_data["route_id"] == "stress-route-A":
                    live_tensor.add_(0.1)
                elif route_data["route_id"] == "stress-route-B":
                    live_tensor.sub_(0.1)

def worker_task(runtime, route_id, tenant, thread_id):
    """Simulates an incoming inference request for a specific route."""
    def dummy_inference(model):
        # We perform inference. The input is all 1s.
        # Original weight is 0. 
        # If Route A is swapped, weight is 0.1. Output should be exactly 10 * 0.1 = 1.0.
        # If Route B is swapped, weight is -0.1. Output should be exactly 10 * -0.1 = -1.0.
        x = torch.ones(1, 10, dtype=torch.float32, device=model.layer_x.device)
        out = model(x)
        # Random sleep to force overlapping request queues
        time.sleep(0.001)
        return out.mean().item()
        
    try:
        req_id = f"req-{route_id}-{thread_id}"
        result = runtime.infer(route_id, tenant, req_id, dummy_inference, runtime.model)
        # Validation
        if route_id == "stress-route-A":
            if not (0.99 < result < 1.01):
                return f"LEAKAGE_DETECTED: Route A got {result}"
        elif route_id == "stress-route-B":
            if not (-1.01 < result < -0.99):
                return f"LEAKAGE_DETECTED: Route B got {result}"
                
        return "SUCCESS"
    except Exception as e:
        return f"ERROR: {str(e)}"

def run_stress_test(threads, num_requests, device="cpu"):
    registry = setup_registry_and_routes()
    runtime_model_hash = "8b1a9953c4611296a827abf8c47804d7e4cf39d88532ee43640243e86ac58e77"
    tenant = TenantContext("stress-tenant")
    
    model = MockModel().to(device)
    runtime = StressTestRuntime(target_model=model, registry=registry, runtime_model_hash=runtime_model_hash)
    
    print(f"\n--- Stress Test: {threads} Threads, {num_requests} Requests ({device}) ---")
    
    results = []
    start_time = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for i in range(num_requests):
            route_id = "stress-route-A" if i % 2 == 0 else "stress-route-B"
            futures.append(executor.submit(worker_task, runtime, route_id, tenant, i))

            
        for future in as_completed(futures):
            results.append(future.result())
            
    elapsed = time.perf_counter() - start_time
    
    successes = results.count("SUCCESS")
    errors = [r for r in results if r.startswith("ERROR")]
    leakages = [r for r in results if r.startswith("LEAKAGE")]
    
    print(f"Elapsed Time : {elapsed:.2f}s")
    print(f"Successes    : {successes}")
    print(f"Errors       : {len(errors)}")
    print(f"Leakages     : {len(leakages)}")
    
    if len(errors) > 0:
        print(f"Sample Error : {errors[0]}")
    if len(leakages) > 0:
        print(f"Sample Leak  : {leakages[0]}")
        
    assert successes == num_requests, "Stress test failed due to errors or leakages."
    assert runtime.state.name == "IDLE", "Runtime did not return to IDLE state."
    
    # Verify baseline is completely unmodified
    base_weight_mean = model.layer_x.detach().mean().item()
    assert abs(base_weight_mean) < 1e-5, f"Torn State Detected! Baseline weight mutated to {base_weight_mean}"

def main():
    print("Running Concurrency Stress and Isolation Test")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for workers in [1, 2, 4, 8, 16, 32]:
        # Limit requests on higher thread counts to keep test time reasonable, 
        # but high enough to force overlapping queue lock contention.
        run_stress_test(threads=workers, num_requests=1000, device=device)
        
    print("\nAll stress tests passed. Zero leakages, zero deadlocks, zero torn states.")

if __name__ == "__main__":
    main()
