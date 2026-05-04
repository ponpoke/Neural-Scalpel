"""
vLLM Route Plugin - Main Entry Point for patching.
"""

def apply_all_patches():
    """
    Apply all monkey patches required for the Neural-Scalpel internal vLLM integration.
    Must be called BEFORE starting the vLLM engine or API server.
    """
    import logging
    logger = logging.getLogger("Neural-Scalpel-vLLM")
    
    logger.info("Applying Neural-Scalpel Route-Aware patches to vLLM...")
    
    # 1. Phase 2: Route Metadata
    try:
        from .route_metadata import inject_route_id_to_vllm_request
        inject_route_id_to_vllm_request()
        logger.info("-> Patched vLLM Request for route_id injection.")
    except Exception as e:
        logger.error(f"Failed to patch route metadata: {e}")
        
    # 2. Phase 3: Route-Aware Scheduler
    try:
        from .scheduler_patch import inject_route_aware_scheduler
        inject_route_aware_scheduler()
        logger.info("-> Patched vLLM Scheduler for Route-Homogeneous Batching.")
    except Exception as e:
        logger.error(f"Failed to patch scheduler: {e}")
        
    # 3. Phase 4: KV Cache Isolation
    try:
        from .kv_cache_policy import inject_route_aware_kv_cache
        inject_route_aware_kv_cache()
        logger.info("-> Patched vLLM KV Cache hashing for route isolation.")
    except Exception as e:
        logger.error(f"Failed to patch KV cache: {e}")
        
    # 4. Phase 5: Model Runner Hook
    try:
        from .model_runner_hook import inject_model_runner_hook
        inject_model_runner_hook()
        logger.info("-> Patched vLLM GPUModelRunner for Hot-Swap / Rollback.")
    except Exception as e:
        logger.error(f"Failed to patch model runner: {e}")
        
    logger.info("Neural-Scalpel patches applied successfully.")

if __name__ == "__main__":
    # If run directly, apply patches and start an example engine (mock)
    apply_all_patches()
