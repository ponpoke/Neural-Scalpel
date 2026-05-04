import os
import json
import time
import torch
import numpy as np
from pathlib import Path

from neural_scalpel.route.registry import RouteRegistry
from neural_scalpel.experimental.runtime import HotSwapRuntime
from neural_scalpel.route.crypto import RouteSigner
from neural_scalpel.route.tenant import TenantContext

def percentile(data, p):
    return np.percentile(data, p)

def dummy_inference(model, tokens=10):
    """Simulates an inference pass using mock matmuls to exercise GPU."""
    start_time = time.perf_counter()
    
    # Simulate TTFT (Time To First Token)
    x = torch.randn(1, 1024, device=model['model.layers.0.self_attn.q_proj.weight'].device, dtype=torch.float16)
    q_proj = model['model.layers.0.self_attn.q_proj.weight']
    _ = torch.matmul(x, q_proj.t())
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    ttft = time.perf_counter() - start_time
    
    # Simulate remaining tokens
    for _ in range(tokens - 1):
        _ = torch.matmul(x, q_proj.t())
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    tokens_per_sec = tokens / (time.perf_counter() - start_time)
    return {"ttft": ttft, "tokens_per_sec": tokens_per_sec}

def main():
    print("Starting Hot-Swap Latency Benchmark...")
    
    # 1. Setup mock environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    mock_model = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(1024, 1024, dtype=torch.float16, device=device),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(256, 1024, dtype=torch.float16, device=device)
    }
    runtime_model_hash = "8b1a9953c4611296a827abf8c47804d7e4cf39d88532ee43640243e86ac58e77"
    tenant = TenantContext("tenant-xyz")
    
    registry_dir = os.path.join(os.path.dirname(__file__), ".registry")
    signer = RouteSigner({"dev-key-01": "sig_mock_value_998877_secret"})
    registry = RouteRegistry(storage_dir=registry_dir, signer=signer)
    
    # Sign and register the route
    route_path = os.path.join(os.path.dirname(__file__), "../examples/routes/example.scalpel_route.json")
    with open(route_path, "r") as f:
        route_data = json.load(f)
    route_data = signer.sign(route_data, "dev-key-01")
    
    signed_path = os.path.join(registry_dir, "signed_example.json")
    os.makedirs(registry_dir, exist_ok=True)
    with open(signed_path, "w") as f:
        json.dump(route_data, f)
        
    route_id = registry.register_route(signed_path)
    
    runtime = HotSwapRuntime(target_model=mock_model, registry=registry, runtime_model_hash=runtime_model_hash)
    
    # 2. Benchmark parameters
    num_runs = 500
    metrics = {
        "end_to_end_latency": [],
        "lock_wait_time": [],
        "swap_latency": [],
        "rollback_latency": [],
        "swap_plus_rollback_latency": [],
        "ttft": [],
        "tokens_per_sec": []
    }
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
    
    # 3. Execution Loop
    print(f"Running {num_runs} inference loops...")
    for i in range(num_runs):
        start_e2e = time.perf_counter()
        
        out = runtime.infer(route_id, tenant, f"req-{i}", dummy_inference, model=mock_model, tokens=20)
        
        e2e_latency = time.perf_counter() - start_e2e
        
        metrics["end_to_end_latency"].append(e2e_latency * 1000) # ms
        metrics["lock_wait_time"].append(runtime.last_timings["lock_wait_time"] * 1000)
        metrics["swap_latency"].append(runtime.last_timings["swap_latency"] * 1000)
        metrics["rollback_latency"].append(runtime.last_timings["rollback_latency"] * 1000)
        metrics["swap_plus_rollback_latency"].append((runtime.last_timings["swap_latency"] + runtime.last_timings["rollback_latency"]) * 1000)
        metrics["ttft"].append(out["ttft"] * 1000)
        metrics["tokens_per_sec"].append(out["tokens_per_sec"])

    # 4. Memory Profiling
    vram_peak = 0
    if torch.cuda.is_available():
        vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
        mem_after = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"VRAM Peak: {vram_peak:.2f} MB")
    
    # 5. Statistical Analysis
    results = {}
    for key, values in metrics.items():
        arr = np.array(values)
        results[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p50": float(percentile(arr, 50)),
            "p90": float(percentile(arr, 90)),
            "p95": float(percentile(arr, 95)),
            "p99": float(percentile(arr, 99)),
            "max": float(np.max(arr))
        }

    # 6. Save Outputs
    out_dir = Path(__file__).parent
    
    config = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "model": "Mock-Qwen2.5-0.5B-Layers",
        "route_count": 1,
        "num_runs": num_runs,
        "precision": "fp16",
        "vram_peak_mb": vram_peak
    }
    
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
        
    with open(out_dir / "latency_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    with open(out_dir / "latency_report.md", "w") as f:
        f.write("# Hot-Swap Runtime Latency Report\n\n")
        f.write("## Configuration\n")
        for k, v in config.items():
            f.write(f"- **{k}**: {v}\n")
            
        f.write("\n## Metrics (in ms, except tokens_per_sec)\n")
        f.write("| Metric | Mean | P50 | P90 | P95 | P99 | Max |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for key, res in results.items():
            f.write(f"| {key} | {res['mean']:.2f} | {res['p50']:.2f} | {res['p90']:.2f} | {res['p95']:.2f} | {res['p99']:.2f} | {res['max']:.2f} |\n")
            
    print("Benchmark complete. Results saved to latency_report.md.")

if __name__ == "__main__":
    main()
