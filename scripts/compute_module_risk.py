import json
import os

REGISTRY_PATH = "reports/regression/v291_interference_map.json"
REPORT_PATH = "reports/regression/v291_module_risk_report.json"

def calculate_risk(stats):
    fixed_count = len(stats["fixed"])
    regressed_count = len(stats["regressed"])
    has_sentinel_regression = stats["sentinel_regressed"]
    
    # Heuristic Risk Scoring
    if has_sentinel_regression:
        score = 0.9 if regressed_count > 0 else 0.7
    else:
        score = regressed_count / (fixed_count + regressed_count + 1e-6)
        
    if score > 0.8:
        level = "CRITICAL"
        rec = "EXCLUDE"
    elif score > 0.5 or has_sentinel_regression:
        level = "HIGH"
        rec = "EXCLUDE_OR_ULTRA_LOW_ALPHA"
    elif score > 0.1:
        level = "MEDIUM"
        rec = "REDUCE_ALPHA"
    else:
        level = "LOW"
        rec = "ALLOW_LOW_ALPHA"
        
    return {
        "risk_level": level,
        "risk_score": round(score, 3),
        "recommendation": rec,
        "fixed": fixed_count,
        "regressed": regressed_count,
        "sentinel_regression": has_sentinel_regression
    }

def main():
    if not os.path.exists(REGISTRY_PATH):
        print(f"Error: Registry not found at {REGISTRY_PATH}")
        return

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    # Map ablation runs to conceptual modules
    # v2.9.1 specific mapping
    module_mapping = {
        "attention": "attention_a4",
        "mlp": "mlp_a4",
        "down_proj": "down_proj_a4"
    }
    
    report = {}
    for module, run_name in module_mapping.items():
        if run_name in registry["runs"]:
            report[module] = calculate_risk(registry["runs"][run_name])
    
    # Save Report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n[v2.10 Diagnostic] Module Risk Report Generated at {REPORT_PATH}")
    print("-" * 60)
    print(f"{'Module':<12} | {'Risk':<10} | {'Score':<6} | {'Rec':<25}")
    print("-" * 60)
    for mod, data in report.items():
        print(f"{mod:<12} | {data['risk_level']:<10} | {data['risk_score']:<6} | {data['recommendation']}")
    print("-" * 60)

if __name__ == "__main__":
    main()
