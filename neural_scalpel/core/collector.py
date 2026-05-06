import torch
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager

class HookCollector:
    """
    Utility for collecting activations from specific modules using forward hooks.
    """
    def __init__(self, model: torch.nn.Module, target_modules: List[str]):
        self.model = model
        self.target_modules = target_modules
        self.activations: Dict[str, List[torch.Tensor]] = {m: [] for m in target_modules}
        self.hooks = []

    def _get_hook(self, name: str) -> Callable:
        def hook(module, input, output):
            # Input is usually a tuple: (hidden_states, ...)
            if isinstance(input, tuple):
                data = input[0].detach()
            else:
                data = input.detach()
            
            # Handle 3D (batch, seq, dim) vs 2D (seq, dim)
            if data.dim() == 3:
                # Take last token only for efficient alignment/solving
                self.activations[name].append(data[:, -1, :].cpu())
            else:
                self.activations[name].append(data.cpu())
        return hook

    def register(self):
        for name, module in self.model.named_modules():
            if name in self.target_modules:
                self.hooks.append(module.register_forward_hook(self._get_hook(name)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_stacked(self) -> Dict[str, torch.Tensor]:
        """Returns collected activations stacked into (N, dim) tensors."""
        return {name: torch.cat(acts, dim=0) for name, acts in self.activations.items() if acts}

@contextmanager
def collection_context(model: torch.nn.Module, target_modules: List[str]):
    collector = HookCollector(model, target_modules)
    collector.register()
    try:
        yield collector
    finally:
        collector.remove()
