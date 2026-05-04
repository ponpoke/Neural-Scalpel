#include <torch/extension.h>

// Forward declaration of the CUDA function
void atomic_swap_cuda(at::Tensor target, at::Tensor source);

// Python wrapper
void atomic_swap(at::Tensor target, at::Tensor source) {
    if (target.is_cuda() && source.is_cuda()) {
        atomic_swap_cuda(target, source);
    } else {
        // Fallback to PyTorch's native copy for CPU
        target.copy_(source);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("atomic_swap", &atomic_swap, "Atomically swap tensor contents (CUDA)");
}
