# Neural-Scalpel Model Compatibility Matrix

## Overview

This matrix documents the validation status of each model architecture
with the Neural-Scalpel hot-swap runtime.

## Compatibility Status

| Model | Architecture | Status | Layer Map | QKV Structure | Swap Test | Mixed-Route | Endurance | Rollback |
|-------|-------------|--------|-----------|---------------|-----------|-------------|-----------|----------|
| OPT-125M | OPT | ✅ Supported | ✅ | Fused QKV in tested vLLM representation | ✅ 100 | ✅ 100 | ✅ 10K | ✅ |
| Qwen2.5-0.5B | Qwen2 | ✅ Supported | ✅ | Transformers: separate Q/K/V; tested vLLM V1: fused `qkv_proj` / `gate_up_proj` | ✅ 100 | ✅ 100 | ✅ 1K | ✅ |
| Qwen2.5-0.5B + Alpaca LoRA projected payload | vLLM V1 | ✅ Phase 5-F Controlled Validation PASS | ✅ | Supports fused `gate_up_proj` / `qkv_proj` conversion | ✅ | ✅ two-route / ⏳ 3+ route | ✅ 50 prompts × 3 runs + 1K two-route mixed | ✅ checksum + text/top-token trace |
| Qwen2.5-Coder-0.5B | Qwen2 | ✅ Supported | ✅ | Separate Q/K/V | ✅ 100 | ✅ 100 | ✅ 1K | ✅ |
| Qwen2.5-1.5B | Qwen2 | 🧪 Experimental | ✅ | Separate Q/K/V | ✅ 100 | ⏳ | ⏳ | ✅ |
| Llama-3.2-1B | Llama | 🧪 Experimental | ✅ | Separate Q/K/V | ✅ 100 | ⏳ | ⏳ | ✅ |
| Llama-3.1-8B | Llama | ❌ Unsupported | ✅ | Separate Q/K/V | ⏳ | ⏳ | ⏳ | ⏳ |

> OPT-125M also passed a 6-hour mixed-route extended soak with 1,956,000 requests, 1,114,920 swap/rollback cycles, 0 violations, 0 errors, and 0.0MB VRAM growth. Final 24h soak remains pending.

## Legend

- ✅ Validated and passing
- 🧪 Experimental — basic tests passing, full endurance pending
- ⏳ Not yet tested
- ❌ Not supported (resource constraints or known incompatibility)

## Per-Model Details

### OPT-125M
- **Layer naming in tested vLLM representation**: `model.decoder.layers.{i}.self_attn.qkv_proj.weight`
- **QKV structure**: Fused QKV in tested vLLM representation
- **MLP naming**: `model.decoder.layers.{i}.fc1.weight`, `fc2.weight`
- **Swappable layers**: 12 per block (6 blocks)
- **Notes**: Primary development reference model; non-vLLM/HuggingFace representations may expose attention projections differently.

### Qwen2.5-0.5B / Qwen2.5-Coder-0.5B
- **Layer naming**: `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
- **MLP naming**: `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
- **Swappable layers**: 7 per block (24 blocks)
- **Notes**: Primary production validation target

### Llama-3.2-1B
- **Layer naming**: Same as Qwen2 pattern
- **MLP naming**: Same as Qwen2 pattern
- **Swappable layers**: 7 per block (16 blocks)
- **Notes**: Cross-architecture validation

## Layer Discovery

The `model_layer_discovery` module automatically detects:
1. Architecture type from parameter naming patterns
2. Transformer block count
3. Attention projection structure (separate vs fused QKV)
4. MLP projection structure
5. Swappable vs non-swappable layer classification

```python
from neural_scalpel.serving.model_layer_discovery import discover_layers
layer_map = discover_layers(model)
print(f"Architecture: {layer_map.model_type}")
print(f"Blocks: {layer_map.num_blocks}")
print(f"Swappable: {len(layer_map.swappable_layers)}")
```

## Adding New Models

To validate a new model:
1. Run layer discovery to generate the layer map
2. Create a `.scalpel_route` manifest targeting the model's layers
3. Run the following test sequence:
   - Same-route 100 requests
   - Mixed-route 100 requests
   - 1K endurance test
   - Rollback checksum verification
4. Update this matrix with results
> Qwen2.5 exposes separate q/k/v and gate/up projections in the Transformers/PEFT representation, while the tested vLLM V1 runtime packs them into fused qkv_proj and gate_up_proj tensors.

> Phase 5-E-1 validates two-route mixed-batch behavior for `__base__` ↔ Alpaca. 3+ route and worst-case alternating-route validation remain pending.
lti-route mixed-batch validation for projected payloads remains pending.
