import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from neural_scalpel.route.registry import RouteRegistry
from neural_scalpel.experimental.runtime import HotSwapRuntime
from neural_scalpel.route.crypto import RouteSigner
from neural_scalpel.route.tenant import TenantContext
from neural_scalpel.experimental.audit import AuditLogger

# ---------------------------------------------------------------------------
# 1. Define a Realistic (but tiny) Transformer Model
# ---------------------------------------------------------------------------
class TinyAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        # Extremely simplified attention purely for compute simulation
        q = self.q_proj(x)
        return self.o_proj(q)

class TinyTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.self_attn = TinyAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias=False)
        )
        
    def forward(self, x):
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x

class TinyQwen(nn.Module):
    def __init__(self, num_layers=4, d_model=256, n_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([TinyTransformerLayer(d_model, n_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# ---------------------------------------------------------------------------
# 2. Setup Runtime Environment
# ---------------------------------------------------------------------------
def setup_environment(model, device):
    registry_dir = Path(__file__).parent / ".real_registry"
    registry_dir.mkdir(exist_ok=True)
    
    signer = RouteSigner({"bench-key": "secret-key-123"})
    registry = RouteRegistry(storage_dir=str(registry_dir), signer=signer)
    audit_logger = AuditLogger(str(registry_dir / "bench_audit.jsonl"))
    
    # Calculate dummy hash for model
    runtime_model_hash = "a" * 64
    
    # Define route targeting the first two layers' q_proj
    route_data = {
        "route_schema_version": "0.1.0",
        "route_id": "real-bench-route",
        "source_model": "llama-3",
        "target_model": "tiny-qwen",
        "source_adapter_sha256": "b" * 64,
        "target_model_sha256": runtime_model_hash,
        "tenant_id": "bench-tenant",
        "license": "MIT",
        "projection_method": "JTSA_WDR_CALIBRATED",
        "calibration": {"forward_passes": 64},
        "diagnostics": {"verdict": "PASS", "ppl_degradation": 0.05, "kl_divergence": 0.01, "portability_score": 95},
        "layers": [
            {
                "name": "layers.0.self_attn.q_proj.weight",
                "shape": list(model.layers[0].self_attn.q_proj.weight.shape),
                "dtype": str(model.layers[0].self_attn.q_proj.weight.dtype).replace('torch.', ''),
                "delta_sha256": "c" * 64
            },
            {
                "name": "layers.1.self_attn.q_proj.weight",
                "shape": list(model.layers[1].self_attn.q_proj.weight.shape),
                "dtype": str(model.layers[1].self_attn.q_proj.weight.dtype).replace('torch.', ''),
                "delta_sha256": "d" * 64
            }
        ]
    }
    
    route_data = signer.sign(route_data, "bench-key")
    route_path = registry_dir / "real_route.json"
    with open(route_path, "w") as f:
        json.dump(route_data, f)
        
    route_id = registry.register_route(str(route_path))
    
    runtime = HotSwapRuntime(target_model=model, registry=registry, runtime_model_hash=runtime_model_hash, audit_logger=audit_logger)
    return runtime, route_id, TenantContext("bench-tenant")

# ---------------------------------------------------------------------------
# 3. Execution
# ---------------------------------------------------------------------------
def dummy_generation(model, device, seq_len=128, tokens_to_generate=10):
    start_time = time.perf_counter()
    x = torch.randn(1, seq_len, 256, device=device)
    
    # Pre-fill / TTFT
    _ = model(x)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    ttft = time.perf_counter() - start_time
    
    # Generation phase
    for _ in range(tokens_to_generate - 1):
        x = torch.randn(1, 1, 256, device=device)
        _ = model(x)
        
    if torch.cuda.is_available(): torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    tokens_per_sec = tokens_to_generate / total_time
    
    return {"ttft": ttft, "tokens_per_sec": tokens_per_sec, "total_time": total_time}

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Real-Model Benchmark on {device.upper()}...")
    
    model = TinyQwen().to(device)
    # Warmup
    _ = model(torch.randn(1, 128, 256, device=device))
    
    runtime, route_id, tenant = setup_environment(model, device)
    
    metrics = {"e2e": [], "swap": [], "rollback": [], "ttft": [], "tps": []}
    num_requests = 100
    
    print(f"Running {num_requests} requests through HotSwapRuntime...")
    for i in range(num_requests):
        start_e2e = time.perf_counter()
        
        # Infer
        out = runtime.infer(route_id, tenant, f"req-{i}", dummy_generation, model=model, device=device)
        
        e2e = time.perf_counter() - start_e2e
        
        metrics["e2e"].append(e2e * 1000)
        metrics["swap"].append(runtime.last_timings["swap_latency"] * 1000)
        metrics["rollback"].append(runtime.last_timings["rollback_latency"] * 1000)
        metrics["ttft"].append(out["ttft"] * 1000)
        metrics["tps"].append(out["tokens_per_sec"])
        
    print("\n--- Real Model Benchmark Results ---")
    for key, values in metrics.items():
        arr = np.array(values)
        if key == "tps":
            print(f"{key.upper()}: Mean={arr.mean():.2f} tokens/s")
        else:
            print(f"{key.upper()} (ms): Mean={arr.mean():.2f} | p50={np.percentile(arr, 50):.2f} | p99={np.percentile(arr, 99):.2f} | Max={arr.max():.2f}")

if __name__ == "__main__":
    run_benchmark()
