import json
import hashlib
import sys
from pathlib import Path

def calculate_sha256(filepath: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def run_package_validate(args):
    print(f"[Validate] Verifying package integrity: {args.package_dir}")
    
    package_dir = Path(args.package_dir)
    meta_path = package_dir / "projection_metadata.json"
    
    if not meta_path.exists():
        print(" [!] ERROR: projection_metadata.json not found. This is not a valid Neural-Scalpel package.")
        sys.exit(1)
        
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    print(f" [i] Framework: {meta.get('framework', 'Unknown')}")
    print(f" [i] Diagnostic Verdict: {meta.get('diagnostic_verdict', 'N/A')}")
    
    hashes = meta.get("integrity_hashes", {})
    if not hashes:
        print(" [!] WARNING: No integrity hashes found in metadata. Full verification not possible.")
    
    mismatches = []
    missing = []
    
    for filename, expected_hash in hashes.items():
        filepath = package_dir / filename
        if not filepath.exists():
            missing.append(filename)
            continue
            
        actual_hash = calculate_sha256(filepath)
        if actual_hash != expected_hash:
            mismatches.append(filename)
            
    print(f"\n--- Integrity Check Results ---")
    
    integrity_ok = not missing and not mismatches
    if integrity_ok:
        print(" [+] ALL FILES VERIFIED: Integrity hashes match.")
    else:
        if missing:
            print(f" [!] MISSING FILES: {', '.join(missing)}")
        if mismatches:
            print(f" [X] CORRUPTED/MODIFIED FILES: {', '.join(mismatches)}")
            
    # Check Diagnostic Consistency
    diagnostic_mismatch = False
    report_path = package_dir / "diagnostic_report.json"
    if report_path.exists():
        with open(report_path, "r") as f:
            report = json.load(f)
        report_verdict = report.get("release_decision_gate", {}).get("verdict")
        if report_verdict != meta.get("diagnostic_verdict"):
            diagnostic_mismatch = True
            print(" [X] CRITICAL: Diagnostic report verdict mismatch with metadata!")
        else:
            print(f" [+] Consistency: Metadata verdict matches embedded diagnostic report.")
            
    if integrity_ok and not diagnostic_mismatch:
        print(f"\n[Validate] SUCCESS: AUTHENTIC AND INTACT.")
    else:
        print(f"\n[Validate] FAILURE: Package integrity is compromised.")
        sys.exit(1)

def add_package_validate_parser(subparsers):
    parser = subparsers.add_parser(
        "package-validate",
        help="Verify the integrity and authenticity of a Neural-Scalpel release package."
    )
    parser.add_argument("--package-dir", required=True, help="Directory containing the release package")
    parser.set_defaults(func=run_package_validate)
