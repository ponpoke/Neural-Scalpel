import torch
from typing import Dict, Optional

class BaseIOBridge:
    """
    Abstract interface for model weight I/O.
    Ensures all formats are converted to high-precision PyTorch tensors for surgery.
    """
    def load_weights(self, path: str) -> Dict[str, torch.Tensor]:
        """Loads all weights (Legacy/Small Models)."""
        raise NotImplementedError("Subclasses must implement load_weights")

    def iter_layers(self, path: str):
        """
        Streaming Iterator: Yields (layer_name, tensor) one by one.
        Essential for processing large models (7B+) on constrained hardware.
        """
        raise NotImplementedError("Subclasses must implement iter_layers")

    def save_weights(self, path: str, state_dict: Dict[str, torch.Tensor], metadata: Optional[Dict] = None):
        """Saves all weights (Legacy/Small Models)."""
        raise NotImplementedError("Subclasses must implement save_weights")

    def open_writer(self, path: str, metadata: Optional[Dict] = None):
        """Initializes an incremental writer."""
        pass

    def write_layer(self, name: str, tensor: torch.Tensor):
        """Writes a single layer and allows memory reclamation."""
        raise NotImplementedError("Subclasses must implement write_layer")

    def close_writer(self):
        """Finalizes the writing process and closes file handles."""
        pass
