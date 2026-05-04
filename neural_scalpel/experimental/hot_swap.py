import threading
import time
import torch
from neural_scalpel.kernel import atomic_swap, HAS_CUDA_KERNEL

class VRAMHotSwapAPI:
    """
    Experimental API for safely modifying VRAM weights during runtime.
    This fulfills the Layer 4 requirements in the PRD.
    """
    def __init__(self, target_model=None):
        self.model = target_model
        self.lock = threading.Lock()
        self.baseline_norms = {}
        self.shadow_buffers = {}
        
    def _get_state_dict(self):
        """Helper to handle both nn.Module and dictionary-based mock models."""
        if self.model is None:
            return None
        if hasattr(self.model, 'state_dict'):
            return self.model.state_dict()
        if isinstance(self.model, dict):
            return self.model
        return None

    def inject_concept(self, task_vector, layer_name: str):
        """
        Concurrency Safe-Injection (V4 Legacy):
        Performs a Micro-Pause (mutex lock) to safely inject the task vector into VRAM.
        """
        state = self._get_state_dict()
        if state is None or layer_name not in state:
            return
            
        print(f"[Hot-Swap] Acquiring VRAM lock for layer: {layer_name}")
        with self.lock:
            time.sleep(0.01) 
            print(f"[Hot-Swap] Injecting task vector into {layer_name}...")
            with torch.no_grad():
                state[layer_name].add_(task_vector.to(state[layer_name].device))
            print(f"[Hot-Swap] Injection complete. Lock released.")
            
    def remove_concept(self, task_vector, layer_name: str):
        """
        Unlearning support (V4 Legacy): Subtracts the task vector.
        """
        state = self._get_state_dict()
        if state is None or layer_name not in state:
            return
            
        print(f"[Hot-Swap] Acquiring VRAM lock for Unlearning layer: {layer_name}")
        with self.lock:
            time.sleep(0.01) 
            print(f"[Hot-Swap] Removing task vector from {layer_name}...")
            with torch.no_grad():
                state[layer_name].sub_(task_vector.to(state[layer_name].device))
            print(f"[Hot-Swap] Unlearning complete. Lock released.")

    # ==============================================================================
    # V5 Enterprise Upgrades: Shadow Registering & Transactional Rollback
    # ==============================================================================
    
    def inject_concept_shadow(self, task_vector, layer_name: str):
        """
        Shadow Registering (Double Buffering) with CUDA-Synchronized Fallback:
        Copies the live tensor to a shadow buffer, applies the task vector, 
        and atomically swaps the pointer.
        """
        state = self._get_state_dict()
        if state is None or layer_name not in state:
            return
            
        # 1. Allocate Shadow Buffer and Compute (No Lock Required)
        print(f"[Hot-Swap V5] Allocating shadow buffer for {layer_name}...")
        live_tensor = state[layer_name]
        
        # Save original state for potential strict rollback
        self.shadow_buffers[layer_name] = live_tensor.clone()
        
        # Compute in shadow memory
        shadow_tensor = live_tensor.clone() + task_vector.to(live_tensor.device)
        
        # 2. Atomic Pointer Swap
        kernel_status = "Native CUDA" if HAS_CUDA_KERNEL else "CUDA-Synchronized Fallback"
        print(f"[Hot-Swap V5] Acquiring lock for pointer swap [{kernel_status}]...")
        with self.lock:
            with torch.no_grad():
                if HAS_CUDA_KERNEL:
                    atomic_swap(state[layer_name], shadow_tensor)
                else:
                    # Strict ACID Fallback for Enterprise
                    if shadow_tensor.is_cuda:
                        torch.cuda.synchronize(shadow_tensor.device)
                    # Python-level atomic pointer swap after C++ streams are drained
                    state[layer_name].data = shadow_tensor.data
        print(f"[Hot-Swap V5] Swap complete.")
        
    def transactional_rollback(self, layer_name: str):
        """
        Strict Transactional Rollback:
        Instantly reverts the tensor to the exact state preserved in the shadow buffer.
        """
        if layer_name not in self.shadow_buffers:
            print(f"[Rollback] No shadow buffer found for {layer_name}. Cannot rollback.")
            return False
            
        state = self._get_state_dict()
        if state is None or layer_name not in state:
            return False
            
        print(f"[Rollback] CRITICAL: Reverting {layer_name} to pristine shadow state...")
        with self.lock:
            with torch.no_grad():
                if HAS_CUDA_KERNEL:
                    atomic_swap(state[layer_name], self.shadow_buffers[layer_name])
                else:
                    if state[layer_name].is_cuda:
                        torch.cuda.synchronize(state[layer_name].device)
                    state[layer_name].data = self.shadow_buffers[layer_name].data
                    
        # Clear buffer after use
        del self.shadow_buffers[layer_name]
        print(f"[Rollback] Restoration complete.")
        return True

    def register_baseline(self, layer_name: str, norm_value: float):
        self.baseline_norms[layer_name] = norm_value

    def monitor_drift(self, layer_name: str, current_norm: float, threshold: float = 0.05):
        """
        Drift Monitor (Dual-Monitor Guardrail): 
        Checks if L2 norm drifted too far due to multiple swaps.
        """
        if layer_name not in self.baseline_norms:
            return True
            
        baseline_norm = self.baseline_norms[layer_name]
        drift = abs(current_norm - baseline_norm) / max(baseline_norm, 1e-9)
        
        if drift > threshold:
            print(f"[Warning] Drift threshold exceeded for {layer_name}: {drift:.2%}. Recommend baseline reset or rollback.")
            return False
            
        return True
        
    def ppl_gateway_monitor(self, current_ppl: float, baseline_ppl: float, threshold_ratio: float = 1.1):
        """
        PPL Gateway Monitor:
        Detects Catastrophic Forgetting after an unlearning event.
        If PPL spikes beyond the threshold ratio (e.g., 10% worse), trigger rollback.
        """
        if current_ppl > baseline_ppl * threshold_ratio:
            print(f"[CRITICAL] Catastrophic Forgetting Detected! PPL spiked from {baseline_ppl:.2f} to {current_ppl:.2f}.")
            print(f"[CRITICAL] Triggering automatic transaction rollback...")
            return False
        return True
