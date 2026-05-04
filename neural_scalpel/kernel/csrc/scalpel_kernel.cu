#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// Atomic memory swap using CUDA Stream Synchronization
// This ensures that the swap happens without any other kernel
// reading partial states.
void atomic_swap_cuda(at::Tensor target, at::Tensor source) {
    TORCH_CHECK(target.is_cuda(), "Target tensor must be a CUDA tensor");
    TORCH_CHECK(source.is_cuda(), "Source tensor must be a CUDA tensor");
    TORCH_CHECK(target.sizes() == source.sizes(), "Tensors must have the same shape");
    TORCH_CHECK(target.dtype() == source.dtype(), "Tensors must have the same dtype");
    TORCH_CHECK(target.is_contiguous(), "Target tensor must be contiguous");
    TORCH_CHECK(source.is_contiguous(), "Source tensor must be contiguous");

    // Get the current CUDA stream from PyTorch
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    // Calculate memory size
    size_t num_bytes = target.numel() * target.element_size();

    // 1. Wait for all pending operations on the current stream to finish
    cudaStreamSynchronize(stream);

    // 2. Perform lightning-fast device-to-device memory copy
    // We use cudaMemcpyAsync on the same stream to ensure it's queued immediately
    // after the sync, preventing other operations from slipping in.
    cudaError_t err = cudaMemcpyAsync(
        target.data_ptr(), 
        source.data_ptr(), 
        num_bytes, 
        cudaMemcpyDeviceToDevice, 
        stream
    );
    
    TORCH_CHECK(err == cudaSuccess, "CUDA memory copy failed: ", cudaGetErrorString(err));

    // 3. Ensure the copy is fully complete before continuing execution
    cudaStreamSynchronize(stream);
}
