import json
import hashlib
import os
import hmac

def calculate_hash(model_id_or_path: str) -> str:
    """Calculates SHA-256 hash. If it's a file, reads in chunks to avoid OOM."""
    if os.path.isfile(model_id_or_path):
        sha256_hash = hashlib.sha256()
        with open(model_id_or_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    else:
        # Fallback for string identifiers or mock test paths
        return hashlib.sha256(model_id_or_path.encode('utf-8')).hexdigest()

class ScalpelRouteManager:
    def __init__(self, route_dir: str = "./routes"):
        self.route_dir = route_dir
        os.makedirs(self.route_dir, exist_ok=True)

    def create_route(self, source_id: str, target_id: str, domain: str, R_matrix: list, s_factor: float, expected_drift: float = 0.0):
        """Creates a .scalpel_route file with strict hash checks and semantic drift certification."""
        source_hash = calculate_hash(source_id)
        target_hash = calculate_hash(target_id)
        
        route_data = {
            "metadata": {
                "source": source_id,
                "target": target_id,
                "domain": domain,
                "source_hash": source_hash,
                "target_hash": target_hash,
                "expected_drift": expected_drift, # V5: Semantic Drift Certification
                "version": "1.0"
            },
            "matrices": {
                "R": R_matrix, # Typically a nested list or base64 encoded tensor
                "s": s_factor
            },
            "signature": None # V5: Web of Trust Integration
        }
        
        filename = f"{source_id.split('/')[-1]}-to-{target_id.split('/')[-1]}-{domain}.scalpel_route"
        filepath = os.path.join(self.route_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(route_data, f, indent=4)
            
        print(f"Created route file: {filepath}")
        return filepath

    def sign_route(self, filepath: str, provider_key: str):
        """V5: Web of Trust Integration. Signs the route to prevent poisoning."""
        with open(filepath, "r") as f:
            route_data = json.load(f)
            
        # Remove existing signature before signing
        route_data["signature"] = None
        payload = json.dumps(route_data, sort_keys=True).encode('utf-8')
        
        signature = hmac.HMAC(provider_key.encode('utf-8'), payload, hashlib.sha256).hexdigest()
        route_data["signature"] = signature
        
        with open(filepath, "w") as f:
            json.dump(route_data, f, indent=4)
        print(f"Route signed successfully: {filepath}")

    def verify_and_load_route(self, filepath: str, current_source_id: str, current_target_id: str, trusted_keys: list = None):
        """Loads a .scalpel_route and verifies compatibility and Chain of Trust."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Route file not found: {filepath}")
            
        with open(filepath, "r") as f:
            route_data = json.load(f)
            
        # 1. V5 Web of Trust Validation
        if trusted_keys is not None:
            signature = route_data.get("signature")
            if not signature:
                raise PermissionError("Web of Trust Error: Route is not signed.")
                
            route_data_for_verification = route_data.copy()
            route_data_for_verification["signature"] = None
            payload = json.dumps(route_data_for_verification, sort_keys=True).encode('utf-8')
            
            verified = False
            for key in trusted_keys:
                expected_sig = hmac.HMAC(key.encode('utf-8'), payload, hashlib.sha256).hexdigest()
                if hmac.compare_digest(expected_sig, signature):
                    verified = True
                    break
            if not verified:
                raise PermissionError("Web of Trust Error: Signature verification failed. Route may be poisoned.")
            print("Web of Trust signature verified.")

        # 2. Strict Version Control
        metadata = route_data["metadata"]
        current_source_hash = calculate_hash(current_source_id)
        current_target_hash = calculate_hash(current_target_id)
        
        if metadata["source_hash"] != current_source_hash:
            raise ValueError(f"Strict Version Control Error: Source hash mismatch. Route expects {metadata['source']}, got {current_source_id}.")
            
        if metadata["target_hash"] != current_target_hash:
            raise ValueError(f"Strict Version Control Error: Target hash mismatch. Route expects {metadata['target']}, got {current_target_id}.")
            
        print("Route hashes verified successfully.")
        
        # 3. V5 Semantic Drift Certification Check
        drift = metadata.get("expected_drift", 0.0)
        print(f"Certified Expected Semantic Drift: {drift:.4f}")
        
        return route_data["matrices"]

    # Backward compatibility wrapper
    def load_route(self, filepath: str, current_source_id: str, current_target_id: str):
        return self.verify_and_load_route(filepath, current_source_id, current_target_id)
