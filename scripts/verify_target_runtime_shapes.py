import os
import sys
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file

def verify_shapes(payload_path: str, target_model_id: str):
    print(f"\n[PHASE 1] Loading Payload & Target Model...")
    payload_path = Path(payload_path)
    if not payload_path.exists():
        print(f"[ERROR] Payload not found: {payload_path}")
        return

    payload_sd = load_file(str(payload_path))
    
    # Load model (meta device to save memory/time)
    print(f"  Fetching target state_dict (via meta device)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            target_model_id, 
            device_map="meta", 
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        target_sd = model.state_dict()
    except Exception as e:
        print(f"[ERROR] Failed to load target model: {e}")
        return

    print(f"\n[PHASE 2] Cross-Referencing Tensors...")
    matched = 0
    mismatched = 0
    missing = 0
    unexpected = []

    results_audit = []
    # Check payload against target
    for key, p_tensor in payload_sd.items():
        # Fusion awareness: map fused keys back to expected components
        is_fused = False
        audit_entry = {"payload_key": key, "payload_shape": list(p_tensor.shape), "status": "UNKNOWN"}
        if ".mlp.gate_up_proj.weight" in key:
            target_keys = [key.replace("gate_up_proj", "gate_proj"), key.replace("gate_up_proj", "up_proj")]
            is_fused = True
        elif ".self_attn.qkv_proj.weight" in key:
            target_keys = [key.replace("qkv_proj", "q_proj"), key.replace("qkv_proj", "k_proj"), key.replace("qkv_proj", "v_proj")]
            is_fused = True
        
        if is_fused:
            audit_entry["target_components"] = target_keys
            # For fused, we check if all components have same input dim and their output dims sum up
            try:
                valid_fusion = True
                p_shape = list(p_tensor.shape)
                sum_out = 0
                for tk in target_keys:
                    if tk in target_sd:
                        t_shape = list(target_sd[tk].shape)
                        if t_shape[1] != p_shape[1]: 
                            valid_fusion = False
                        sum_out += t_shape[0]
                    else:
                        valid_fusion = False
                
                if valid_fusion and sum_out == p_shape[0]:
                    matched += 1; audit_entry["status"] = "MATCH"
                else:
                    mismatched += 1; audit_entry["status"] = "MISMATCH"
            except Exception as e:
                mismatched += 1; audit_entry["status"] = "ERROR"
        elif key in target_sd:
            t_shape = list(target_sd[key].shape)
            p_shape = list(p_tensor.shape)
            if t_shape == p_shape:
                matched += 1; audit_entry["status"] = "MATCH"
            else:
                mismatched += 1; audit_entry["status"] = "MISMATCH"
        else:
            unexpected.append(key); audit_entry["status"] = "UNEXPECTED"
        results_audit.append(audit_entry)

    # Check for missing tensors (only for relevant layers)
    for key in target_sd.keys():
        if ".layers." in key and (".self_attn." in key or ".mlp." in key) and ".weight" in key:
            if key not in payload_sd:
                # Note: vLLM uses fused layers, so some target keys might be missing in fused payload
                # but represented by fused keys. This logic needs to be aware of fusion.
                is_fused = any(f in key for f in ["gate_proj", "up_proj", "q_proj", "k_proj", "v_proj"])
                if not is_fused:
                    missing += 1
                    print(f"  [MISSING] {key} not found in payload.")

    report = {
        "status": "FAIL",
        "validation_type": "transformers_meta_state_dict_fused_shape_validation",
        "target_model": target_model_id,
        "payload_path": str(payload_path),
        "summary": {
            "matched": matched,
            "mismatched": mismatched,
            "missing": missing,
            "unexpected": len(unexpected)
        },
        "tensors": results_audit,
        "does_not_validate": [
            "vLLM internal tensor ordering",
            "successful route application",
            "behavioral transfer",
            "generation quality"
        ],
        "note": "Structural shape alignment only. Behavioral validation is PENDING."
    }

    # Stricter status check: everything must match, no unexpected tensors allowed.
    if mismatched == 0 and missing == 0 and len(unexpected) == 0:
        report["status"] = "PASS"

    report_path = payload_path.parent / "runtime_shape_verification.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[RESULT] Verification: {report['status']} (RUNTIME_SHAPE_VERIFIED)")
    print(f"  Matched: {matched}")
    print(f"  Mismatched: {mismatched}")
    print(f"  Missing: {missing}")
    print(f"  Unexpected: {len(unexpected)}")
    print(f"Report saved to {report_path}")

    return report["status"] == "PASS"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--target-model", required=True)
    args = parser.parse_args()
    verify_shapes(args.payload, args.target_model)
