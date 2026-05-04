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
    
    # Save the original init
    original_init = vllm_request.Request.__init__

    def patched_init(self, *args, **kwargs):
        # Extract route_id from kwargs if it exists, otherwise default to __base__
        route_id = kwargs.pop("route_id", "__base__")
        
        # Check if route_id was passed in via sampling_params
        sampling_params = kwargs.get("sampling_params")
        if sampling_params is None and len(args) > 2:
            # If passed as positional argument (3rd argument is usually sampling_params)
            sampling_params = args[2]
            
        if sampling_params and hasattr(sampling_params, "extra_args") and sampling_params.extra_args:
            extra_route = sampling_params.extra_args.get("route_id")
            if extra_route:
                route_id = extra_route
                
        # Call original init
        original_init(self, *args, **kwargs)
        
        # Set route_id after init to ensure it's not overridden
        if not hasattr(self, "route_id"):
            self.route_id = route_id

    # Apply patch
    vllm_request.Request.__init__ = patched_init
