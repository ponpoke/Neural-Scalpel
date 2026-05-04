import json
import os
import jsonschema
import re

from neural_scalpel.route.crypto import RouteSigner
from neural_scalpel.route.policy import evaluate_license_risk, PolicyDecision
from neural_scalpel.route.tenant import TenantContext

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "scalpel_route.schema.json")

def _load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def verify_schema(route_data: dict):
    schema = _load_schema()
    try:
        jsonschema.validate(instance=route_data, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        raise ValueError(f"Route manifest schema validation failed: {e.message}")

def verify_hash(route_data: dict):
    sha256_pattern = re.compile(r"^[a-fA-F0-9]{64}$")
    
    source_hash = route_data.get("source_adapter_sha256", "")
    target_hash = route_data.get("target_model_sha256", "")
    
    if not sha256_pattern.match(source_hash):
        raise ValueError(f"Invalid source_adapter_sha256 format: {source_hash}")
    if not sha256_pattern.match(target_hash):
        raise ValueError(f"Invalid target_model_sha256 format: {target_hash}")
    
    for layer in route_data.get("layers", []):
        delta_hash = layer.get("delta_sha256", "")
        if not sha256_pattern.match(delta_hash):
            raise ValueError(f"Invalid delta_sha256 format for layer {layer.get('name')}: {delta_hash}")

def verify_signature(route_data: dict, signer: RouteSigner):
    """
    Verifies the cryptographic signature of the route using the provided signer.
    """
    signer.verify(route_data)

def verify_runtime_compatibility(route_data: dict, current_model_hash: str):
    if route_data.get("target_model_sha256") != current_model_hash:
        raise ValueError(
            f"Route target model hash ({route_data.get('target_model_sha256')}) "
            f"does not match current runtime model hash ({current_model_hash})."
        )

def verify_tenant_access(route_data: dict, current_tenant: TenantContext):
    route_tenant_id = route_data.get("tenant_id")
    if route_tenant_id and route_tenant_id != current_tenant.tenant_id:
        raise PermissionError(
            f"Tenant mismatch. Route is bound to tenant '{route_tenant_id}', "
            f"but current context is '{current_tenant.tenant_id}'."
        )

def verify_license_policy(route_data: dict):
    """
    Verifies the license and throws PermissionError if denied or requires manual review.
    """
    license_name = route_data.get("license", "")
    decision = evaluate_license_risk(license_name)
    
    if decision == PolicyDecision.DENY:
        raise PermissionError(f"Route rejected due to high-risk license policy: {license_name}")
    elif decision == PolicyDecision.MANUAL_REVIEW:
        raise PermissionError(f"Route rejected: License '{license_name}' requires manual review.")
    # ALLOW passes through silently
