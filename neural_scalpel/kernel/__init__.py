import warnings

try:
    from .scalpel_cuda_kernel import atomic_swap
    HAS_CUDA_KERNEL = True
except ImportError:
    HAS_CUDA_KERNEL = False
    
    def atomic_swap(target, source):
        """Fallback CPU implementation"""
        warnings.warn("Scalpel CUDA Kernel not found. Falling back to native PyTorch copy_(). This is not hardware atomic.", RuntimeWarning)
        target.copy_(source)

__all__ = ["atomic_swap", "HAS_CUDA_KERNEL"]
