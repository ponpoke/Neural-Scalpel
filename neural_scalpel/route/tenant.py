class TenantContext:
    """
    Represents the active tenant making an inference request.
    """
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

def verify_tenant_access(route_data: dict, current_tenant: TenantContext):
    """
    Ensures that the current tenant is authorized to use this route.
    If 'tenant_id' is missing from the route, it might be considered a public/global route,
    but for strict enterprise isolation, we enforce exact matches.
    """
    route_tenant_id = route_data.get("tenant_id")
    
    # If route specifies a tenant, it must match
    if route_tenant_id and route_tenant_id != current_tenant.tenant_id:
        raise PermissionError(
            f"Tenant mismatch. Route is bound to tenant '{route_tenant_id}', "
            f"but current context is '{current_tenant.tenant_id}'."
        )
