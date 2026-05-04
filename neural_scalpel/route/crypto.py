import hmac
import hashlib
import json

class RouteSigner:
    """
    Provides HMAC-SHA256 signing and verification for Scalpel Routes.
    In a real enterprise environment, this would integrate with a KMS or use Ed25519.
    """
    def __init__(self, secret_keys: dict):
        """
        secret_keys: A mapping of key_id to the actual secret string.
        """
        self.secret_keys = secret_keys

    def _canonicalize(self, route_data: dict) -> str:
        """
        Removes the signature block and returns a canonical JSON string for signing.
        """
        payload = dict(route_data)
        if "signature" in payload:
            del payload["signature"]
        return json.dumps(payload, sort_keys=True, separators=(',', ':'))

    def sign(self, route_data: dict, key_id: str) -> dict:
        """
        Signs the route data and appends the signature block.
        """
        if key_id not in self.secret_keys:
            raise ValueError(f"Signing key '{key_id}' not found.")
            
        canonical_str = self._canonicalize(route_data)
        secret = self.secret_keys[key_id].encode('utf-8')
        
        signature = hmac.new(secret, canonical_str.encode('utf-8'), hashlib.sha256).hexdigest()
        
        route_data["signature"] = {
            "algorithm": "hmac-sha256",
            "key_id": key_id,
            "value": signature
        }
        return route_data

    def verify(self, route_data: dict):
        """
        Verifies the HMAC-SHA256 signature of the route.
        Throws a ValueError if the signature is invalid or missing.
        """
        sig_block = route_data.get("signature")
        if not sig_block:
            raise ValueError("Missing signature block.")
            
        algorithm = sig_block.get("algorithm")
        key_id = sig_block.get("key_id")
        provided_signature = sig_block.get("value")
        
        if algorithm != "hmac-sha256":
            raise ValueError(f"Unsupported signature algorithm: {algorithm}")
            
        if not key_id or key_id not in self.secret_keys:
            raise ValueError(f"Unknown signing key_id: {key_id}")
            
        canonical_str = self._canonicalize(route_data)
        secret = self.secret_keys[key_id].encode('utf-8')
        
        expected_signature = hmac.new(secret, canonical_str.encode('utf-8'), hashlib.sha256).hexdigest()
        
        if not hmac.compare_digest(expected_signature, provided_signature):
            raise ValueError("Cryptographic signature verification failed: Hash mismatch.")
