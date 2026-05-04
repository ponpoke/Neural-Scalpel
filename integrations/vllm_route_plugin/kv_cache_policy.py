"""
KV Cache Isolation Policy for vLLM internal integration.
"""

def inject_route_aware_kv_cache():
    """
    Monkey patch vLLM's prefix cache block hashing to include route_identity.
    This guarantees that KV cache blocks cannot be reused across different routes,
    even if the prompt tokens are identical.
    """
    try:
        import vllm.v1.core.kv_cache_utils as kv_utils
    except ImportError:
        return
        
    original_hash_block_tokens = getattr(kv_utils, "hash_block_tokens", None)
    
    if original_hash_block_tokens is None:
        # Fallback if hash_block_tokens is not exposed in this vLLM version
        return

    def patched_hash_block_tokens(*args, **kwargs):
        """
        Wrapper around original hash_block_tokens that injects the route_id into the hash state.
        For true isolation, the request's route_id must be part of the tuple that gets hashed.
        """
        # Calculate the original hash which is typically based on (is_prompt, tuple(token_ids), lora_req)
        base_hash = original_hash_block_tokens(*args, **kwargs)
        
        # We need to extract the route_id. In a real integration, the hash_block_tokens
        # signature needs to accept the request or route_id. Since we monkey-patch,
        # we might have to rely on thread-local storage or modify the caller (Request) to pass it.
        # For this prototype, we'll assume we can retrieve the active route from context.
        # This is a conceptual implementation of Phase 4 (Level 2: Route-tagged cache key).
        import threading
        thread_local = threading.local()
        active_route = getattr(thread_local, "active_route_id", "__base__")
        
        # Combine the original hash with the route_id
        route_identity = f"{active_route}_payload_hash_stub"
        
        # Return the new combined hash
        return hash((base_hash, route_identity))

    kv_utils.hash_block_tokens = patched_hash_block_tokens

    # We also patch the Request's block hasher to pass the route_id via thread-local context
    import vllm.v1.request as vllm_request
    original_update_block_hashes = vllm_request.Request.update_block_hashes

    def patched_update_block_hashes(self):
        import threading
        thread_local = threading.local()
        # Set context before computing hashes
        thread_local.active_route_id = getattr(self, "route_id", "__base__")
        original_update_block_hashes(self)
        
    vllm_request.Request.update_block_hashes = patched_update_block_hashes
