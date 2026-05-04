import json
import os
from typing import Dict, List, Optional

from neural_scalpel.route.policy import RouteStatus
from neural_scalpel.route.crypto import RouteSigner
from neural_scalpel.route.tenant import TenantContext
from neural_scalpel.route.verifier import (
    verify_schema, verify_hash, verify_signature, 
    verify_runtime_compatibility, verify_tenant_access, verify_license_policy
)

class RouteRegistry:
    def __init__(self, storage_dir: str, signer: RouteSigner):
        self.storage_dir = storage_dir
        self.signer = signer
        self.routes: Dict[str, dict] = {}
        self.statuses: Dict[str, RouteStatus] = {}
        self.quarantine_reasons: Dict[str, str] = {}
        
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            
    def _load_route_file(self, filepath: str) -> dict:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def register_route(self, filepath: str) -> str:
        """
        Loads, verifies, and registers a route into the registry.
        Throws ValueError or PermissionError if the route is invalid or unsafe.
        Uses a strict Fail-Close policy: any exception completely aborts registration.
        """
        try:
            route_data = self._load_route_file(filepath)
            
            # 1. Structural Verification
            verify_schema(route_data)
            verify_hash(route_data)
            
            # 2. Cryptographic Verification
            verify_signature(route_data, self.signer)
            
            # 3. Policy & Logic Verification
            verify_license_policy(route_data)
            
            diagnostics = route_data.get("diagnostics", {})
            if diagnostics.get("verdict") == "FAIL":
                raise ValueError("Cannot register route with FAIL verdict.")
                
            route_id = route_data["route_id"]
            
            # If we pass everything, register
            self.routes[route_id] = route_data
            self.statuses[route_id] = RouteStatus.PRODUCTION 
            
            return route_id
            
        except Exception as e:
            # Explicit Fail-Close: Re-raise immediately, leaving registry unmodified.
            raise e

    def get_verified_route(self, route_id: str, runtime_model_hash: str, current_tenant: TenantContext) -> dict:
        """
        Retrieves a route only if it is completely verified, compatible with the running model,
        and accessible by the current tenant.
        Throws ValueError or PermissionError if unsafe.
        """
        try:
            route_data = self.get_route(route_id)
            if not route_data:
                raise ValueError(f"Route {route_id} not found in registry.")
                
            status = self.get_route_status(route_id)
            if status in (RouteStatus.REVOKED, RouteStatus.QUARANTINED):
                raise PermissionError(f"Route {route_id} is {status.value} and cannot be executed.")
                
            # Verify runtime constraints
            verify_runtime_compatibility(route_data, runtime_model_hash)
            verify_tenant_access(route_data, current_tenant)
            
            return route_data
            
        except Exception as e:
            # Fail-Close on retrieval
            raise e

    def get_route(self, route_id: str) -> Optional[dict]:
        return self.routes.get(route_id)

    def list_routes(self) -> List[str]:
        return list(self.routes.keys())

    def get_route_status(self, route_id: str) -> Optional[RouteStatus]:
        return self.statuses.get(route_id)

    def is_revoked(self, route_id: str) -> bool:
        return self.get_route_status(route_id) == RouteStatus.REVOKED

    def revoke_route(self, route_id: str):
        if route_id not in self.routes:
            raise ValueError(f"Route {route_id} not found.")
        self.statuses[route_id] = RouteStatus.REVOKED

    def quarantine_route(self, route_id: str, reason: str):
        if route_id not in self.routes:
            raise ValueError(f"Route {route_id} not found.")
        self.statuses[route_id] = RouteStatus.QUARANTINED
        self.quarantine_reasons[route_id] = reason
