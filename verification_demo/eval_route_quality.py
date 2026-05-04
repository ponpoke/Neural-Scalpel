import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from neural_scalpel.route.registry import RouteRegistry
from neural_scalpel.experimental.runtime import HotSwapRuntime
from neural_scalpel.route.crypto import RouteSigner
from neural_scalpel.route.tenant import TenantContext
from neural_scalpel.experimental.audit import AuditLogger

# ---------------------------------------------------------------------------
# 1. Model & PPL Evaluation Setup
# ---------------------------------------------------------------------------
class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Target layer for our route
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = F.gelu(self.proj(x))
        logits = self.head(x)
        return logits

def compute_perplexity(model, input_ids, target_ids):
    with torch.no_grad():
        logits = model(input_ids) # [batch, seq_len, vocab_size]
        # Flatten for CrossEntropy
        logits = logits.view(-1, logits.size(-1))
        targets = target_ids.view(-1)
        loss = F.cross_entropy(logits, targets)
        return torch.exp(loss).item()

# ---------------------------------------------------------------------------
# 2. Environment
# ---------------------------------------------------------------------------
def run_quality_eval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Quality & Perplexity Evaluation on {device.upper()}...")
    
    vocab_size = 1000
    d_model = 128
    model = TinyLanguageModel(vocab_size, d_model).to(device)
    
    # Generate some fixed synthetic data (representing a validation set)
    torch.manual_seed(42)
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    # Target is shifted by 1
    target_ids = torch.cat([input_ids[:, 1:], torch.randint(0, vocab_size, (batch_size, 1), device=device)], dim=1)
    
    # 1. Baseline Perplexity
    baseline_ppl = compute_perplexity(model, input_ids, target_ids)
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    
    # 2. Setup Runtime
    registry_dir = Path(__file__).parent / ".eval_registry"
    registry_dir.mkdir(exist_ok=True)
    signer = RouteSigner({"eval-key": "secret"})
    registry = RouteRegistry(storage_dir=str(registry_dir), signer=signer)
    
    runtime_model_hash = "a" * 64
    route_data = {
        "route_schema_version": "0.1.0",
        "route_id": "eval-route",
        "source_model": "model-a",
        "target_model": "model-b",
        "source_adapter_sha256": "b" * 64,
        "target_model_sha256": runtime_model_hash,
        "tenant_id": "eval-tenant",
        "license": "MIT",
        "projection_method": "JTSA_WDR",
        "calibration": {"forward_passes": 64},
        "diagnostics": {"verdict": "PASS", "ppl_degradation": 0, "kl_divergence": 0, "portability_score": 100},
        "layers": [
            {
                "name": "proj.weight",
                "shape": list(model.proj.weight.shape),
                "dtype": str(model.proj.weight.dtype).replace('torch.', ''),
                "delta_sha256": "c" * 64
            }
        ]
    }
    route_data = signer.sign(route_data, "eval-key")
    route_path = registry_dir / "route.json"
    with open(route_path, "w") as f: json.dump(route_data, f)
    
    route_id = registry.register_route(str(route_path))
    runtime = HotSwapRuntime(target_model=model, registry=registry, runtime_model_hash=runtime_model_hash)
    tenant = TenantContext("eval-tenant")
    
    # 3. Evaluate Swapped Quality
    # In HotSwapRuntime, `swap` adds a dummy 0.01 delta to demonstrate mutation.
    # This WILL change the PPL deterministically.
    def infer_with_swap(model):
        return compute_perplexity(model, input_ids, target_ids)
        
    swapped_ppl = runtime.infer(route_id, tenant, "req-eval", infer_with_swap, model=model)
    print(f"Swapped PPL:  {swapped_ppl:.4f}")
    
    # 4. Verify Rollback PPL
    rollback_ppl = compute_perplexity(model, input_ids, target_ids)
    print(f"Rollback PPL: {rollback_ppl:.4f}")
    
    assert abs(baseline_ppl - rollback_ppl) < 1e-6, "Catastrophic Failure: Rollback did not perfectly restore Perplexity!"
    assert abs(swapped_ppl - baseline_ppl) > 0.01, "Error: Route injection did not modify model behavior."
    
    print("\nQuality Evaluation Passed!")
    print(f"Quality Delta: {abs(swapped_ppl - baseline_ppl):.4f}")

if __name__ == "__main__":
    run_quality_eval()
