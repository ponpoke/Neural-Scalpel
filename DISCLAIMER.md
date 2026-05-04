# Neural-Scalpel Research Disclaimer

**Neural-Scalpel is an experimental research project.**

## 1. No Production Guarantees
This software is provided "as-is", without any express or implied warranty. It is currently in alpha (`v1.0.0-alpha`). The Hot-Swap Runtime, projection algorithms, and diagnostic tools are not guaranteed to be safe for production environments, enterprise SLAs, or untrusted public endpoints. 

## 2. Gradient-Free does NOT mean Data-Free
While Neural-Scalpel performs no gradient-based retraining (backpropagation), it relies on forward-pass activation calibrations to compute projection manifolds (e.g., JTSA). Without proper calibration, projecting language model weights will likely destroy the model's emergent properties and result in gibberish.

## 3. Empirical Task Performance is Not Guaranteed
A "PASS" verdict from the Neural-Scalpel diagnostic tools simply means that the structural integrity of the model (PPL and KL divergence) remains within acceptable statistical bounds. **It does not mathematically guarantee that the model can perform the complex downstream logic (e.g., coding, math, reasoning) learned by the source LoRA.** Users MUST perform their own downstream task evaluations.

## 4. Hardware and Stability
The PyTorch-native Hot-Swap Runtime uses synchronized tensor swapping to achieve real-time weight injection. This bypasses typical memory-management safeguards and can cause VRAM fragmentation, latency spikes, or CUDA OOM errors under heavy concurrent load if not properly architected. The internal vLLM integration is currently provided as a conceptual mock and is not production-ready.
