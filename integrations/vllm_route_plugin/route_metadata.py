"""
Route Metadata Extraction for vLLM internal integration.
"""

from typing import Optional, Dict, Any
from fastapi import Request

def extract_route_id_from_header(request: Request) -> Optional[str]:
    """
    Extract X-Scalpel-Route-ID from HTTP headers.
    """
    return request.headers.get("X-Scalpel-Route-ID")

def inject_route_id_to_vllm_request():
    """
    Monkey patch vLLM's Request object to accept and retain route_id.
    """
    import vllm.v1.request as vllm_request
    
    if getattr(vllm_request.Request, "_scalpel_route_patched", False):
        return

    # Save the original init
    original_init = vllm_request.Request.__init__

    def patched_init(self, *args, **kwargs):
        route_id = kwargs.pop("route_id", None)

        sampling_params = kwargs.get("sampling_params", None)
        if route_id is None and sampling_params is not None:
            extra_args = getattr(sampling_params, "extra_args", None)
            if isinstance(extra_args, dict):
                route_id = extra_args.get("route_id")

        if route_id is None:
            route_id = "__base__"

        original_init(self, *args, **kwargs)
        self.route_id = route_id

    # Apply patch
    vllm_request.Request.__init__ = patched_init
    vllm_request.Request._scalpel_route_patched = True
