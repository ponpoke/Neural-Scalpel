import os
import torch
from typing import Optional
from neural_scalpel.io.base import BaseIOBridge
from neural_scalpel.io.safetensors_bridge import SafetensorsBridge

class IOBridgeFactory:
    """
    Factory to select the correct I/O bridge based on file extensions.
    """
    @staticmethod
    def get_bridge(path: str, calibration_data: Optional[torch.Tensor] = None) -> BaseIOBridge:
        # Check extensions
        if path.endswith(".gguf"):
            # Import GGUFBridge lazily to avoid dependency issues if not installed
            from neural_scalpel.io.gguf_bridge import GGUFBridge
            return GGUFBridge()
            
        if ".awq" in path.lower():
            from neural_scalpel.io.awq_bridge import AWQBridge
            return AWQBridge(calibration_data=calibration_data)
            
        # Default to Safetensors for now (handles directories and .safetensors)
        return SafetensorsBridge()
