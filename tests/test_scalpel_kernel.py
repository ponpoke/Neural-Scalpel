import pytest
import threading
import time
import torch
from neural_scalpel.experimental.hot_swap import VRAMHotSwapAPI
from neural_scalpel.kernel import atomic_swap, HAS_CUDA_KERNEL

def test_atomic_swap_basic():
    """Test basic functionality of atomic_swap (either CUDA or fallback)."""
    target = torch.zeros(10, 10)
    source = torch.ones(10, 10)
    
    atomic_swap(target, source)
    
    assert torch.all(target == 1), "atomic_swap failed to copy data correctly."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_atomic_swap_cuda():
    """Test CUDA specific functionality if CUDA is available."""
    target = torch.zeros(10, 10, device='cuda')
    source = torch.ones(10, 10, device='cuda')
    
    atomic_swap(target, source)
    
    assert torch.all(target == 1), "atomic_swap failed on CUDA tensors."
    assert target.is_cuda, "Target tensor lost CUDA designation after swap."

def test_hot_swap_api_integration():
    """Ensure VRAMHotSwapAPI can inject and rollback using the new kernel."""
    mock_state = {'layer1.weight': torch.zeros(5, 5)}
    api = VRAMHotSwapAPI(target_model=mock_state)
    
    task_vector = torch.ones(5, 5)
    
    api.inject_concept_shadow(task_vector, 'layer1.weight')
    assert torch.all(mock_state['layer1.weight'] == 1), "Shadow injection failed."
    
    success = api.transactional_rollback('layer1.weight')
    assert success is True, "Rollback reported failure."
    assert torch.all(mock_state['layer1.weight'] == 0), "Rollback failed to restore pristine state."

def test_kernel_status_flag():
    """Verify HAS_CUDA_KERNEL is a boolean."""
    assert isinstance(HAS_CUDA_KERNEL, bool)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for hardware-level stress test")
def test_vram_hot_swap_stress_concurrency():
    """
    Stress test for VRAM Hot-Swap ACID compliance.
    """
    tensor_size = (4096, 4096)
    device = 'cuda'
    
    baseline_tensor = torch.zeros(tensor_size, device=device)
    mock_state = {'layer1.weight': baseline_tensor}
    api = VRAMHotSwapAPI(target_model=mock_state)
    
    task_vector = torch.ones(tensor_size, device=device)
    
    expected_sum_baseline = 0.0
    expected_sum_injected = float(tensor_size[0] * tensor_size[1])
    
    read_errors = []
    reads_completed = 0
    is_running = True

    def inference_loop():
        nonlocal reads_completed, is_running
        while is_running:
            current_tensor = mock_state['layer1.weight']
            current_sum = current_tensor.sum().item()
            
            if current_sum != expected_sum_baseline and current_sum != expected_sum_injected:
                read_errors.append(current_sum)
                
            reads_completed += 1
            time.sleep(0.0001) 

    reader_thread = threading.Thread(target=inference_loop)
    reader_thread.start()
    
    try:
        for _ in range(30):
            api.inject_concept_shadow(task_vector, 'layer1.weight')
            time.sleep(0.002)
            
            api.transactional_rollback('layer1.weight')
            time.sleep(0.002)
            
    finally:
        is_running = False
        reader_thread.join()

    assert len(read_errors) == 0, f"ACID Violation! Read corrupted partial states: {read_errors[:5]}"
    assert reads_completed > 50, f"Not enough reads performed ({reads_completed}) to validate stress test."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory tracking")
def test_vram_memory_leak():
    """
    1. VRAM Memory Leak & Fragmentation (Resource Exhaustion)
    Ensures O(1) memory footprint over 1,000 swaps.
    """
    device = 'cuda'
    tensor_size = (1024, 1024)
    baseline_tensor = torch.zeros(tensor_size, device=device)
    mock_state = {'layer1.weight': baseline_tensor}
    api = VRAMHotSwapAPI(target_model=mock_state)
    task_vector = torch.ones(tensor_size, device=device)
    
    # Warmup to initialize CUDA context completely
    api.inject_concept_shadow(task_vector, 'layer1.weight')
    api.transactional_rollback('layer1.weight')
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    initial_memory = torch.cuda.memory_allocated(device)
    
    for _ in range(1000):
        api.inject_concept_shadow(task_vector, 'layer1.weight')
        api.transactional_rollback('layer1.weight')
        
    torch.cuda.synchronize()
    final_memory = torch.cuda.memory_allocated(device)
    
    assert final_memory == initial_memory, f"Memory leak detected! Initial: {initial_memory}, Final: {final_memory}"

def test_multi_layer_contention():
    """
    2. Multi-Layer Contention
    Tests simultaneous hot-swaps on multiple layers from multiple threads.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    layers = [f'layer{i}.weight' for i in range(5)]
    mock_state = {layer: torch.zeros(128, 128, device=device) for layer in layers}
    api = VRAMHotSwapAPI(target_model=mock_state)
    task_vector = torch.ones(128, 128, device=device)
    
    errors = []
    
    def worker(layer_name):
        try:
            for _ in range(20):
                api.inject_concept_shadow(task_vector, layer_name)
                time.sleep(0.001)
                api.transactional_rollback(layer_name)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(layer,)) for layer in layers]
    for t in threads: t.start()
    for t in threads: t.join()
    
    assert len(errors) == 0, f"Exceptions occurred during multi-layer contention: {errors}"
    for layer in layers:
        assert torch.all(mock_state[layer] == 0), f"Layer {layer} did not correctly restore to baseline."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for latency testing")
def test_latency_impact():
    """
    3. Latency Impact & TPOT
    Ensures hot-swapping does not block the inference stream unreasonably.
    """
    device = 'cuda'
    tensor_size = (4096, 4096)
    baseline_tensor = torch.zeros(tensor_size, device=device)
    mock_state = {'layer1.weight': baseline_tensor}
    api = VRAMHotSwapAPI(target_model=mock_state)
    task_vector = torch.ones(tensor_size, device=device)
    
    latencies = []
    is_running = True
    
    def inference_matmul_loop():
        nonlocal is_running
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        while is_running:
            start_time = time.perf_counter()
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start_time)
            
    inf_thread = threading.Thread(target=inference_matmul_loop)
    inf_thread.start()
    
    time.sleep(0.1) # Let inference stabilize
    
    # Perform hot-swaps
    for _ in range(10):
        api.inject_concept_shadow(task_vector, 'layer1.weight')
        time.sleep(0.01)
        api.transactional_rollback('layer1.weight')
        
    is_running = False
    inf_thread.join()
    
    if len(latencies) == 0:
        return
        
    # Calculate p99 latency
    latencies.sort()
    p99_idx = int(len(latencies) * 0.99)
    p99_latency = latencies[p99_idx]
    
    # Ensure p99 latency is under 50ms (generous margin for local test, proves no hard freeze)
    assert p99_latency < 0.05, f"Latency impact too high! p99 latency: {p99_latency:.4f}s"

def test_fault_tolerance():
    """
    4. Graceful Degradation
    Ensures system handles Shape Mismatch cleanly without corrupting state.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    baseline_tensor = torch.zeros(128, 128, device=device)
    mock_state = {'layer1.weight': baseline_tensor}
    api = VRAMHotSwapAPI(target_model=mock_state)
    
    # Intentional shape mismatch
    bad_task_vector = torch.ones(64, 64, device=device)
    
    # Should raise RuntimeError from PyTorch broadcasting/copying
    with pytest.raises(RuntimeError):
        api.inject_concept_shadow(bad_task_vector, 'layer1.weight')
        
    # Verify baseline is untouched
    assert torch.all(mock_state['layer1.weight'] == 0), "State corrupted after failed injection."
    
    # Ensure lock was released (we can do another operation)
    good_task_vector = torch.ones(128, 128, device=device)
    api.inject_concept_shadow(good_task_vector, 'layer1.weight')
    assert torch.all(mock_state['layer1.weight'] == 1), "Lock deadlocked after exception."
