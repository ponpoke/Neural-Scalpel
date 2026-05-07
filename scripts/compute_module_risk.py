import json
import os

REGISTRY_PATH = "reports/regression/v291_interference_map.json"
REPORT_PATH = "reports/regression/v291_module_risk_report.json"

def calculate_risk(stats):
    fixed_count = len(stats["fixed"])
    regressed_count = len(stats["regressed"])
    has_sentinel_regression = stats["sentinel_regressed"]
    
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
        
    return level, score, rec

def main():
    if not os.path.exists(REGISTRY_PATH):
        print(f"Error: Registry not found at {REGISTRY_PATH}")
        return

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    report = {}
    
    # 1. Analyze Attention Sweep
    attn_runs = {name: data for name, data in registry["runs"].items() if name.startswith("attention_a")}
    if attn_runs:
        tested_alphas = sorted([int(name.split("_a")[1]) for name in attn_runs.keys()])
        safe_alphas = []
        first_regression = None
        
        for alpha in tested_alphas:
            run_name = f"attention_a{alpha}"
            stats = attn_runs[run_name]
            if not stats["sentinel_regressed"] and len(stats["regressed"]) == 0:
                safe_alphas.append(alpha)
            elif first_regression is None:
                first_regression = alpha
        
        safe_max = max(safe_alphas) if safe_alphas else 0
        
        # Use a4 as the representative for general stats if available
        rep_stats = attn_runs.get(f"attention_a{safe_max}", next(iter(attn_runs.values())))
        level, score, rec = calculate_risk(rep_stats)
        
        report["attention"] = {
            "risk_level": "CONDITIONAL_LOW" if safe_max > 0 else "HIGH",
            "safe_alpha_max": safe_max,
            "tested_alphas": tested_alphas,
            "first_sentinel_regression_alpha": first_regression,
            "recommendation": f"ALLOW_ALPHA_LE_{safe_max}" if safe_max > 0 else "EXCLUDE",
            "risk_score": score,
            "fixed": len(rep_stats["fixed"]),
            "regressed": len(rep_stats["regressed"]),
            "sentinel_regression": rep_stats["sentinel_regressed"]
        }

    # 2. Analyze MLP components (fixed at alpha 4)
    if "mlp_a4" in registry["runs"]:
        level, score, rec = calculate_risk(registry["runs"]["mlp_a4"])
        report["mlp"] = {
            "risk_level": level,
            "risk_score": score,
            "recommendation": "EXCLUDE_OR_ALPHA_LE_1",
            "fixed": len(registry["runs"]["mlp_a4"]["fixed"]),
            "regressed": len(registry["runs"]["mlp_a4"]["regressed"]),
            "sentinel_regression": registry["runs"]["mlp_a4"]["sentinel_regressed"]
        }
        
    if "down_proj_a4" in registry["runs"]:
        level, score, rec = calculate_risk(registry["runs"]["down_proj_a4"])
        report["down_proj"] = {
            "risk_level": level,
            "risk_score": score,
            "recommendation": "EXCLUDE",
            "fixed": len(registry["runs"]["down_proj_a4"]["fixed"]),
            "regressed": len(registry["runs"]["down_proj_a4"]["regressed"]),
            "sentinel_regression": registry["runs"]["down_proj_a4"]["sentinel_regressed"]
        }
    
    # Save Report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\n[v2.10 Diagnostic] Enhanced Module Risk Report Generated at {REPORT_PATH}")
    print("-" * 80)
    print(f"{'Module':<12} | {'Risk':<15} | {'Safe Max':<8} | {'Recommendation'}")
    print("-" * 80)
    for mod, data in report.items():
        s_max = data.get("safe_alpha_max", "N/A")
        print(f"{mod:<12} | {data['risk_level']:<15} | {s_max:<8} | {data['recommendation']}")
    print("-" * 80)

if __name__ == "__main__":
    main()
