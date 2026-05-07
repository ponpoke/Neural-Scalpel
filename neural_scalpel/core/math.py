import torch
import math
from typing import Tuple, Optional, Union

def head_wise_orthogonal_procrustes(
    A: torch.Tensor,
    B: torch.Tensor,
    num_heads: int,
    bias_A: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Head-wise Scaling Orthogonal Procrustes
    
    Solves the optimization problem: min_{s, R} || s * A * R - B ||_F for each head independently.
    Aligns the representations of a source model (A) to a target model (B).
    
    Args:
        A (torch.Tensor): Source activations/embeddings of shape (N, d).
        B (torch.Tensor): Target activations/embeddings of shape (N, d).
        num_heads (int): Number of attention heads to split the dimension `d` into.
        bias_A (torch.Tensor, optional): Bias vector of shape (d,) associated with the source space to be translated.
        
    Returns:
        A_transformed (torch.Tensor): The transformed source tensor.
        bias_A_transformed (torch.Tensor, optional): The transformed bias vector.
        R_stacked (torch.Tensor): Stacked rotation matrices of shape (num_heads, head_dim, head_dim).
        s_stacked (torch.Tensor): Stacked scaling factors of shape (num_heads,).
    """
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: A {A.shape} and B {B.shape} must have the same shape.")
        
    N, d = A.shape
    if d % num_heads != 0:
        raise ValueError(f"Dimension d ({d}) must be divisible by num_heads ({num_heads}).")
        
    head_dim = d // num_heads
    
    # Reshape into (N, num_heads, head_dim)
    A_heads = A.view(N, num_heads, head_dim)
    B_heads = B.view(N, num_heads, head_dim)
    
    R_list = []
    s_list = []
    
    for i in range(num_heads):
        A_i = A_heads[:, i, :]  # (N, head_dim)
        B_i = B_heads[:, i, :]  # (N, head_dim)
        
        # Cross-covariance matrix M = A_i^T @ B_i
        M = torch.matmul(A_i.t(), B_i)  # (head_dim, head_dim)
        
        # SVD of M
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        
        # Optimal rotation R = U @ V^T
        R = torch.matmul(U, Vh)
        
        # Optimal scaling s = Tr(S) / Tr(A_i^T @ A_i)
        trace_S = torch.sum(S)
        # Tr(A_i^T @ A_i) is the sum of squared elements of A_i
        trace_A_A = torch.sum(A_i * A_i)
        
        # Avoid division by zero
        s = trace_S / torch.clamp(trace_A_A, min=1e-12)
        
        R_list.append(R)
        s_list.append(s)
        
    # Stack for batched operations
    R_stacked = torch.stack(R_list)  # (num_heads, head_dim, head_dim)
    s_stacked = torch.stack(s_list)  # (num_heads,)
    
    # Transform A: s_i * (A_i @ R_i)
    # Using einsum to apply R_i to A_i for each head
    A_transformed = torch.einsum('nhi,hij->nhj', A_heads, R_stacked)
    A_transformed = A_transformed * s_stacked.view(1, num_heads, 1)
    A_transformed = A_transformed.reshape(N, d)
    
    # Transform Bias if provided
    bias_A_transformed = None
    if bias_A is not None:
        if bias_A.shape[0] != d:
            raise ValueError(f"Bias dimension {bias_A.shape[0]} does not match d {d}")
        bias_A_heads = bias_A.view(num_heads, head_dim)
        # b_new = s * (b @ R)
        b_trans = torch.einsum('hi,hij->hj', bias_A_heads, R_stacked)
        b_trans = b_trans * s_stacked.view(num_heads, 1)
        bias_A_transformed = b_trans.reshape(d)
        
    return A_transformed, bias_A_transformed, R_stacked, s_stacked


def create_sparse_task_vector(
    W_trained: torch.Tensor,
    W_base: torch.Tensor,
    trim_ratio: float = 0.2
) -> torch.Tensor:
    """
    Physical Sparse Memory Hack
    
    Calculates the task vector (tau = W_trained - W_base), performs pre-SVD trimming 
    to remove noise (bottom `trim_ratio` absolute values), and casts to a CSR sparse tensor 
    to prevent memory explosion.
    
    Args:
        W_trained (torch.Tensor): Weights of the trained/fine-tuned model.
        W_base (torch.Tensor): Weights of the base model.
        trim_ratio (float): The fraction of smallest magnitude weights to drop (default: 0.2).
        
    Returns:
        tau_sparse (torch.Tensor): The trimmed task vector as a CSR sparse tensor.
    """
    if W_trained.shape != W_base.shape:
        raise ValueError("W_trained and W_base must have the same shape.")
        
    if not (0.0 <= trim_ratio < 1.0):
        raise ValueError("trim_ratio must be in [0, 1).")
        
    # Calculate task vector
    tau = W_trained - W_base
    
    # 1. Pre-SVD Trimming
    if trim_ratio > 0.0:
        # Flatten and compute absolute values to find the threshold
        tau_abs = torch.abs(tau).view(-1)
        # For huge tensors on 16GB RAM, quantile is computationally safer than full sorting
        # Explicit conversion to float32 might be needed for quantile on MPS/CPU depending on pytorch version
        threshold = torch.quantile(tau_abs.float(), trim_ratio).to(tau.dtype)
        
        # Apply mask (values strictly less than threshold become 0)
        mask = torch.abs(tau) >= threshold
        tau = tau * mask

    # 2. Sparse Casting
    if tau.dim() != 2:
        # SVD and CSR formats require 2D tensors. Flatten out extra dims if any.
        tau = tau.view(tau.shape[0], -1)
        
    tau_sparse = tau.to_sparse_csr()
    return tau_sparse


def adaptive_rsvd_bootstrap(
    M: Union[torch.Tensor, torch.sparse.Tensor],
    epsilon: float = 1e-3,
    block_size: int = 10,
    max_blocks: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adaptive Randomized SVD with Bootstrap Stopping
    
    Dynamically expands the SVD rank block-by-block until the largest singular value 
    of the newly extracted subspace falls below `epsilon` relative to the overall 
    maximum singular value (sigma_1_hat) established in the first block.
    
    Args:
        M (torch.Tensor or sparse tensor): The target matrix to decompose (m x n).
        epsilon (float): Stopping threshold ratio.
        block_size (int): Number of random projections per iteration step.
        max_blocks (int): Maximum number of block iterations.
        
    Returns:
        U (torch.Tensor): Left singular vectors.
        S (torch.Tensor): Singular values.
        V (torch.Tensor): Right singular vectors (transposed, as in full SVD).
    """
    m, n = M.shape
    device = M.device
    dtype = M.dtype if not M.is_sparse else M.values().dtype
    
    Q_total = None
    sigma_1_hat = None
    
    # For sparse handling, we might need M^T to project leftwards
    is_sparse = M.is_sparse_csr or getattr(M, 'is_sparse', False)
    if is_sparse:
        try:
            M_t = M.t()
        except RuntimeError:
            # Fallback if CSR t() is unsupported in the current PyTorch build
            M_t = M.to_sparse_coo().t().coalesce()
    else:
        M_t = M.t()

    for k in range(max_blocks):
        # 1. Random projection block
        Omega = torch.randn(n, block_size, device=device, dtype=dtype)
        
        # Y = M @ Omega
        if is_sparse:
            # For CSR * Dense, matmul is supported
            Y = torch.matmul(M, Omega)
        else:
            Y = torch.matmul(M, Omega)
            
        # 2. Orthogonalize against previously accumulated basis Q_total
        if Q_total is not None:
            # Modified Gram-Schmidt or double classical orthogonalization for stability
            Y = Y - torch.matmul(Q_total, torch.matmul(Q_total.t(), Y))
            Y = Y - torch.matmul(Q_total, torch.matmul(Q_total.t(), Y))
            
        # 3. QR Decomposition to orthonormalize the new block
        # For PyTorch QR, we require a 2D dense tensor
        Q_new, _ = torch.linalg.qr(Y, mode='reduced')
        
        # 4. Project M onto Q_new to check new singular values
        # B_new = Q_new^T M => (M^T Q_new)^T
        B_new_t = torch.matmul(M_t, Q_new)
        B_new = B_new_t.t()
        
        # Compute local SVD for the new block to check stopping criterion
        _, S_b, _ = torch.linalg.svd(B_new, full_matrices=False)
        sigma_new_max = S_b[0].item()
        
        if k == 0:
            sigma_1_hat = sigma_new_max
            Q_total = Q_new
            
            # Global SVD on the first block
            U_b, S_total, V_total = torch.linalg.svd(B_new, full_matrices=False)
            U_total = torch.matmul(Q_total, U_b)
        else:
            # Stopping check: Is the new max singular value negligible?
            if sigma_new_max / sigma_1_hat < epsilon:
                print(f"[Adaptive rSVD] Stopping criterion met at block {k+1}. "
                      f"sigma_new_max/sigma_1_hat = {sigma_new_max/sigma_1_hat:.4e} < {epsilon}")
                break
                
            # Append new orthonormal basis
            Q_total = torch.cat([Q_total, Q_new], dim=1)
            
            # Recompute global SVD over the entire accumulated subspace
            B_total_t = torch.matmul(M_t, Q_total)
            B_total = B_total_t.t()
            
            U_b, S_total, V_total = torch.linalg.svd(B_total, full_matrices=False)
            U_total = torch.matmul(Q_total, U_b)

    return U_total, S_total, V_total

# ==============================================================================
# V3 Advanced Mathematical Upgrades
# ==============================================================================

def adaptive_variance_preserving_sparsity(
    W_tuned: torch.Tensor,
    W_base: torch.Tensor,
    variance_preservation: float = 0.99
) -> torch.Tensor:
    """
    Adaptive Variance-Preserving Sparsity (AVPS)
    Dynamically thresholds the task vector to preserve a specific percentage (e.g., 99%)
    of the total L2 energy, avoiding arbitrary magnitude pruning.
    """
    tau = W_tuned - W_base
    tau_abs = torch.abs(tau).view(-1)
    
    # Square for energy
    energy = tau_abs ** 2
    total_energy = torch.sum(energy)
    
    # Sort energy to find threshold
    sorted_energy, sorted_indices = torch.sort(energy, descending=True)
    cum_energy = torch.cumsum(sorted_energy, dim=0)
    
    # Find the index where cumulative energy exceeds the preservation threshold
    threshold_idx = torch.searchsorted(cum_energy, total_energy * variance_preservation)
    if threshold_idx >= len(sorted_energy):
        threshold_idx = len(sorted_energy) - 1
        
    threshold_value = tau_abs[sorted_indices[threshold_idx]]
    
    # Apply threshold
    mask = torch.abs(tau) >= threshold_value
    tau_sparse = (tau * mask).to_sparse_csr()
    return tau_sparse

def pca_guided_subspace_injection(
    source_tensor: torch.Tensor, 
    target_activations: torch.Tensor 
) -> torch.Tensor:
    """
    Principal Component Subspace Injection (PCSI)
    Projects the lower-dimensional source concept onto the principal components of the 
    higher-dimensional target activation space, rather than naive zero-padding.
    """
    # Assuming source_tensor is (N, dim_S) and target_activations is (M, dim_T)
    # PCA on target activations
    U, S, Vh = torch.linalg.svd(target_activations, full_matrices=False)
    
    dim_S = source_tensor.shape[-1]
    available_components = Vh.shape[0]  # min(M, dim_T)
    
    if dim_S <= available_components:
        # Standard case: enough components available
        top_components = Vh[:dim_S, :]  # (dim_S, dim_T)
        injected_tensor = torch.matmul(source_tensor, top_components)
    else:
        # Not enough SVD components: project what we can, zero-pad the rest
        top_components = Vh  # (available, dim_T)
        # Truncate source to available dims, project, then add remainder via zero-pad
        injected_partial = torch.matmul(source_tensor[:, :available_components], top_components)
        injected_tensor = injected_partial
    
    return injected_tensor

def soft_routing_head_pooling(
    source_heads: torch.Tensor, 
    num_target_heads: int
) -> torch.Tensor:
    """
    Soft-Routing Head Pooling (SRHP)
    Compresses N source attention heads into M target attention heads using SVD.
    Preserves reasoning logic without severing (robotomizing) heads.
    """
    # source_heads shape: (batch_size, num_source_heads, head_dim)
    B, s_heads, h_dim = source_heads.shape
    
    if s_heads <= num_target_heads:
        return source_heads
        
    # Reshape to treat batch and head_dim as samples, heads as features
    # M: (B * h_dim, s_heads)
    M = source_heads.transpose(1, 2).reshape(-1, s_heads)
    
    # SVD to extract the most important combinatorial routing patterns
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    
    # Reduce to target heads
    M_reduced = U[:, :num_target_heads] * S[:num_target_heads] # (B * h_dim, num_target_heads)
    
    # Reshape back
    target_heads = M_reduced.view(B, h_dim, num_target_heads).transpose(1, 2)
    return target_heads

# ==============================================================================
# V6 Advanced Math: Wasserstein Discrete Routing
# ==============================================================================

def sinkhorn_knopp(
    C: torch.Tensor, 
    epsilon: float = 0.01, 
    n_iter: int = 100,
    stop_thr: float = 1e-9
) -> torch.Tensor:
    """
    Log-domain Sinkhorn-Knopp algorithm for Entropic Regularized Optimal Transport.
    Improved with marginal residual stopping criteria.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
        
    device = C.device
    dtype = C.dtype
    n, m = C.shape
    
    K_log = -C / epsilon
    
    # Uniform marginals in log domain
    log_a = torch.full((n,), -math.log(n), device=device, dtype=dtype)
    log_b = torch.full((m,), -math.log(m), device=device, dtype=dtype)
    
    # Initial dual variables in log domain
    log_u = torch.zeros(n, device=device, dtype=dtype)
    log_v = torch.zeros(m, device=device, dtype=dtype)

    for i in range(n_iter):
        log_u = log_a - torch.logsumexp(K_log + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(K_log.t() + log_u.unsqueeze(0), dim=1)
        
        # Marginal Residual Stopping Criteria
        if i % 10 == 0:
            log_P = log_u.unsqueeze(1) + K_log + log_v.unsqueeze(0)
            row_err = (torch.logsumexp(log_P, dim=1) - log_a).abs().max()
            col_err = (torch.logsumexp(log_P, dim=0) - log_b).abs().max()
            
            if max(row_err.item(), col_err.item()) < stop_thr:
                break
                
    log_P = log_u.unsqueeze(1) + K_log + log_v.unsqueeze(0)
    return torch.exp(log_P)

def wasserstein_discrete_routing(
    source_heads: torch.Tensor,
    target_heads: torch.Tensor,
    epsilon: float = 0.01,
    alpha: float = 0.1,
    mode: str = "hard",
    return_diagnostics: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
    """
    Wasserstein Discrete Routing (WDR) with Soft-Merge Fallback.
    Numerically stabilized for Baseline v2 with diagnostic reporting.
    """
    if mode not in {"soft", "hard"}:
        raise ValueError("mode must be 'soft' or 'hard'")
        
    S_heads = source_heads.shape[1]
    T_heads = target_heads.shape[1]
    dtype = source_heads.dtype
    
    # 1. Compute Cost Matrix (Normalized Squared L2 Distance)
    H_s = source_heads.transpose(0, 1).reshape(S_heads, -1).to(torch.float32)
    H_t = target_heads.transpose(0, 1).reshape(T_heads, -1).to(torch.float32)
    
    H_s_norm = (H_s ** 2).sum(dim=1, keepdim=True) 
    H_t_norm = (H_t ** 2).sum(dim=1, keepdim=True).t() 
    C = torch.clamp(H_s_norm + H_t_norm - 2 * torch.matmul(H_s, H_t.t()), min=0.0)
    C = C / (C.max() + 1e-12)
    
    # 2. Solve Soft Optimal Transport (Log-domain Sinkhorn)
    P_soft = sinkhorn_knopp(C, epsilon=epsilon)
    
    col_sums = P_soft.sum(dim=0, keepdim=True)
    tiny = torch.finfo(P_soft.dtype).tiny
    fallback_used = bool(torch.any(col_sums <= tiny).item())
    
    if mode == "soft":
        if fallback_used:
            P_col = torch.softmax(-C / max(epsilon, 1e-6), dim=0)
            P_final = (P_col / P_col.sum(dim=0, keepdim=True)).to(dtype)
        else:
            P_final = (P_soft / col_sums.clamp_min(tiny)).to(dtype)
            
        if return_diagnostics:
            diag = {"sinkhorn_fallback_used": fallback_used, "epsilon": epsilon, "mode": mode}
            return P_final, diag
        return P_final

    # 3. Hard-WDR with Soft-Merge Fallback
    winner_indices = torch.argmax(P_soft, dim=0) 
    P_hard = torch.zeros_like(P_soft)
    P_hard[winner_indices, torch.arange(T_heads, device=P_soft.device)] = 1.0
    
    all_source_indices = set(range(S_heads))
    matched_source_indices = set(winner_indices.tolist())
    remnant_indices = list(all_source_indices - matched_source_indices)
    
    if remnant_indices and alpha > 0:
        for i in remnant_indices:
            j = torch.argmin(C[i, :])
            P_hard[i, j] = alpha
            
    P_final = (P_hard / (P_hard.sum(dim=0, keepdim=True) + 1e-12)).to(dtype)
    
    if return_diagnostics:
        diag = {"sinkhorn_fallback_used": False, "epsilon": epsilon, "mode": mode}
        return P_final, diag
    return P_final

# ==============================================================================
# V5 Enterprise Upgrades: Quantization & MoE
# ==============================================================================

def vram_decoupled_decoding(quantized_tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, dtype=torch.float16) -> torch.Tensor:
    """
    VRAM-Decoupled Decoding Pipeline
    Dynamically inflates a quantized tensor (e.g., INT4/INT8) back to a high-precision 
    floating point representation (FP16/BF16) purely in VRAM, avoiding disk I/O bottlenecks.
    
    Args:
        quantized_tensor: The low-precision tensor.
        scale: The quantization scaling factor.
        zero_point: The quantization zero point.
        dtype: The target high-precision dtype.
        
    Returns:
        The de-quantized, high-precision tensor ready for mathematical projection.
    """
    # Simple linear de-quantization simulation: Dequantized = (Quantized - ZeroPoint) * Scale
    # In a real engine (like bitsandbytes or AutoGPTQ), this calls custom CUDA kernels.
    dequantized = (quantized_tensor.to(dtype) - zero_point.to(dtype)) * scale.to(dtype)
    return dequantized

def quantization_aware_procrustes(
    A: torch.Tensor, 
    B: torch.Tensor, 
    num_heads: int, 
    quantization_bits: int = 4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantization-Aware Procrustes (QAP)
    Calculates the alignment rotation matrix while accounting for the inevitable information loss 
    that occurs when the projected vector is re-quantized by the target engine.
    
    Args:
        A: Source activations/embeddings.
        B: Target activations/embeddings.
        num_heads: Number of attention heads.
        quantization_bits: The bit-width of the target quantization grid (default: 4).
        
    Returns:
        A_transformed, R_stacked, s_stacked
    """
    # Standard Procrustes first
    A_transformed, _, R_stacked, s_stacked = head_wise_orthogonal_procrustes(A, B, num_heads)
    
    # QAP Penalty: If the target space is heavily quantized, extreme rotations might map 
    # continuous values into the same discrete quantization bucket. 
    # We apply a simulated dampening factor to the scaling 's' based on the bit-width resolution.
    # The lower the bits, the more we dampen extreme values to prevent overflow in the grid.
    
    grid_resolution = 2 ** quantization_bits
    # A crude simulation of grid-awareness dampening. In practice, this involves rounding 
    # to nearest quantization steps and recalculating the optimal R.
    dampening_factor = 1.0 - (1.0 / grid_resolution)
    
    s_stacked_qap = s_stacked * dampening_factor
    A_transformed_qap = A_transformed * dampening_factor
    
    return A_transformed_qap, R_stacked, s_stacked_qap

def expert_wise_procrustes(
    A_experts: list[torch.Tensor], 
    B_experts: list[torch.Tensor]
) -> Tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Expert-wise Procrustes Alignment for MoE models.
    Calculates independent rotation and scaling matrices for each Expert in an MoE layer.
    
    Args:
        A_experts: List of source expert activations.
        B_experts: List of target expert activations.
        
    Returns:
        transformed_experts, R_experts, s_experts
    """
    if len(A_experts) != len(B_experts):
        raise ValueError("Number of source and target experts must match.")
        
    transformed_experts = []
    R_experts = []
    s_experts = []
    
    for A, B in zip(A_experts, B_experts):
        # Treat each expert as a single 'head' for alignment
        A_trans, _, R, s = head_wise_orthogonal_procrustes(A, B, num_heads=1)
        transformed_experts.append(A_trans)
        R_experts.append(R)
        s_experts.append(s)
        
    return transformed_experts, R_experts, s_experts

# ==============================================================================
# V7 Experimental: Kernel Orthogonal Procrustes (KOP)
# ==============================================================================

def kernel_orthogonal_procrustes(
    A: torch.Tensor,
    B: torch.Tensor,
    num_heads: int,
    sigma: float = 1.0,
    n_components: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Kernel Orthogonal Procrustes (KOP) with Nyström Approximation.
    
    Addresses non-linear distortion (GeGLU/SwiGLU) by aligning activations 
    in a high-dimensional Hilbert space. This targets <0.5% PPL degradation.
    
    Args:
        A, B: Activations (N, d).
        num_heads: Number of heads for alignment.
        sigma: RBF kernel bandwidth.
        n_components: Nyström landmark points.
        
    Returns:
        A_transformed, R_kernel_stacked, s_stacked
    """
    N, d = A.shape
    head_dim = d // num_heads
    A_heads = A.view(N, num_heads, head_dim)
    B_heads = B.view(N, num_heads, head_dim)
    
    R_list = []
    s_list = []
    A_trans_list = []
    
    for i in range(num_heads):
        A_i = A_heads[:, i, :] # (N, h_dim)
        B_i = B_heads[:, i, :] # (N, h_dim)
        
        # 1. Nyström Kernel Mapping (Approximates Non-linear Feature Map)
        # Select landmarks
        indices = torch.randperm(N)[:min(n_components, N)]
        landmarks = A_i[indices] # (m, h_dim)
        
        # Compute Kernel Matrix K(A_i, landmarks)
        dist = torch.cdist(A_i, landmarks) ** 2
        K_am = torch.exp(-dist / (2 * sigma ** 2)) # (N, m)
        
        # Compute K_mm^-1/2
        dist_mm = torch.cdist(landmarks, landmarks) ** 2
        K_mm = torch.exp(-dist_mm / (2 * sigma ** 2)) + 1e-6 * torch.eye(len(indices), device=A.device)
        L, Q = torch.linalg.eigh(K_mm)
        K_mm_inv_half = Q @ torch.diag(1.0 / torch.sqrt(torch.clamp(L, min=1e-7))) @ Q.t()
        
        # Feature map Phi(A_i) = K_am @ K_mm^-1/2
        Phi_A = torch.matmul(K_am, K_mm_inv_half) # (N, m)
        
        # 2. Linear Procrustes in Kernel Space
        # Since Phi_A is (N, m), we need a target in the same space or project B_i.
        # For simplicity in this surgical context, we align Phi_A to a projected B_i
        Phi_B = torch.matmul(torch.matmul(B_i, torch.linalg.pinv(B_i)), Phi_A) # (N, m)
        
        U, S, Vh = torch.linalg.svd(torch.matmul(Phi_A.t(), Phi_B), full_matrices=False)
        R_k = torch.matmul(U, Vh)
        
        # 3. Transform and project back to original space
        Phi_A_trans = torch.matmul(Phi_A, R_k)
        
        # Map kernel space back to head_dim using pseudo-inverse of Phi_A
        W_reconstruct = torch.matmul(torch.linalg.pinv(Phi_A), B_i)
        A_i_trans = torch.matmul(Phi_A_trans, W_reconstruct)
        
        R_list.append(R_k) # Note: This is a kernel-space rotation
        s_list.append(torch.tensor(1.0, device=A.device))
        A_trans_list.append(A_i_trans)
        
    A_transformed = torch.stack(A_trans_list, dim=1).reshape(N, d)
    return A_transformed, torch.stack(R_list), torch.stack(s_list)

# ==============================================================================
# V8 God-Tier Math: Jacobian & Hessian Tangent Space Alignment (JTSA/HAMA)
# ==============================================================================

def swiglu_jacobian(x: torch.Tensor) -> torch.Tensor:
    """Calculates the diagonal Jacobian of the SiLU/Swish component of SwiGLU."""
    sig = torch.sigmoid(x)
    return sig * (1 + x * (1 - sig))

def geglu_jacobian(x: torch.Tensor) -> torch.Tensor:
    """Calculates the diagonal Jacobian of the GELU component of GeGLU."""
    return 0.5 * (1 + torch.erf(x / math.sqrt(2))) + (x / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * x**2)

def swiglu_hessian(x: torch.Tensor) -> torch.Tensor:
    """Calculates the diagonal Hessian (2nd derivative) of the SiLU/Swish component."""
    sig = torch.sigmoid(x)
    return sig * (1 - sig) * (2 + x * (1 - 2 * sig))

def geglu_hessian(x: torch.Tensor) -> torch.Tensor:
    """Calculates the diagonal Hessian of the GELU component."""
    return (2 - x**2) / math.sqrt(2 * math.pi) * torch.exp(-0.5 * x**2)

def jacobian_tangent_space_alignment(
    A: torch.Tensor,
    B: torch.Tensor,
    num_heads: int,
    activation_type: str = "swiglu",
    activation_states: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Jacobian Tangent Space Alignment (JTSA).
    
    Uses structural knowledge of the target's non-linear activation function (f) 
    to pre-compensate for distortion via the Jacobian J_f.
    Requires activation_states (calibration data) to accurately capture emergent outliers in LLMs.
    If not provided, falls back to deriving synthetic activation states purely from weights (Risky).
    """
    N, d = A.shape
    head_dim = d // num_heads
    A_heads = A.view(N, num_heads, head_dim)
    B_heads = B.view(N, num_heads, head_dim)
    
    if activation_states is not None:
        if activation_states.shape != (N, d):
            activation_states = activation_states.expand(N, d)
        state_heads = activation_states.view(N, num_heads, head_dim)
    else:
        print("[WARNING] No calibration data provided for JTSA. Falling back to Synthetic States from weights. Outliers may be lost.")
        state_heads = None
    
    if activation_type == "swiglu":
        j_func = swiglu_jacobian
    else:
        j_func = geglu_jacobian
        
    R_list = []
    s_list = []
    A_trans_list = []
    
    for i in range(num_heads):
        A_i = A_heads[:, i, :]
        B_i = B_heads[:, i, :]
        
        if state_heads is not None:
            state = state_heads[:, i, :]
        else:
            # Zero-Dataset State Synthesis Fallback
            state = B_i / (torch.norm(B_i, dim=0, keepdim=True) + 1e-6)
        
        J = j_func(state)
        
        A_weighted = J * A_i
        
        M = torch.matmul(A_weighted.t(), B_i)
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        R = torch.matmul(U, Vh)
        
        A_i_trans = torch.matmul(A_i, R)
        
        R_list.append(R)
        s_list.append(torch.tensor(1.0, device=A.device))
        A_trans_list.append(A_i_trans)
        
    A_transformed = torch.stack(A_trans_list, dim=1).reshape(N, d)
    return A_transformed, torch.stack(R_list), torch.stack(s_list)

def hessian_aware_manifold_alignment(
    A: torch.Tensor,
    B: torch.Tensor,
    num_heads: int,
    activation_type: str = "swiglu",
    alpha: float = 0.5,
    activation_states: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hessian-Aware Manifold Alignment (HAMA).
    
    A 2nd-order Taylor expansion pre-compensating for extreme curvature in OOD regions.
    Requires activation_states (calibration data) to accurately capture emergent outliers in LLMs.
    If not provided, falls back to deriving synthetic activation states purely from weights (Risky).
    """
    N, d = A.shape
    head_dim = d // num_heads
    A_heads = A.view(N, num_heads, head_dim)
    B_heads = B.view(N, num_heads, head_dim)
    
    if activation_states is not None:
        if activation_states.shape != (N, d):
            activation_states = activation_states.expand(N, d)
        state_heads = activation_states.view(N, num_heads, head_dim)
    else:
        print("[WARNING] No calibration data provided for HAMA. Falling back to Synthetic States from weights. Outliers may be lost.")
        state_heads = None
    
    if activation_type == "swiglu":
        j_func = swiglu_jacobian
        h_func = swiglu_hessian
    else:
        j_func = geglu_jacobian
        h_func = geglu_hessian
        
    R_list = []
    s_list = []
    A_trans_list = []
    
    for i in range(num_heads):
        A_i = A_heads[:, i, :]
        B_i = B_heads[:, i, :]
        
        if state_heads is not None:
            state = state_heads[:, i, :]
        else:
            # Zero-Dataset State Synthesis Fallback
            state = B_i / (torch.norm(B_i, dim=0, keepdim=True) + 1e-6)
        
        J = j_func(state)
        H = h_func(state)
        
        # 2nd-order curvature compensator
        curvature_factor = J + 0.5 * alpha * H * state
        
        A_weighted = curvature_factor * A_i
        
        M = torch.matmul(A_weighted.t(), B_i)
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        R = torch.matmul(U, Vh)
        
        A_i_trans = torch.matmul(A_i, R)
        
        R_list.append(R)
        s_list.append(torch.tensor(1.0, device=A.device))
        A_trans_list.append(A_i_trans)
        
    A_transformed = torch.stack(A_trans_list, dim=1).reshape(N, d)
    return A_transformed, torch.stack(R_list), torch.stack(s_list)

def router_logic_preservation_mapping(
    source_gate: torch.Tensor, 
    target_gate: torch.Tensor
) -> torch.Tensor:
    """
    Router Logic Preservation Mapping.
    Projects the MoE gating network (router) weights such that the transplanted concept 
    correctly triggers the corresponding expert paths in the target architecture.
    """
    # A simplified simulation: align the gating logic via PCA subspace injection
    # In reality, this requires evaluating the routing probability distribution.
    return pca_guided_subspace_injection(source_gate, target_gate)

def solve_ridge(X: torch.Tensor, Y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Solves for W in the equation X @ W = Y using Ridge Regression (L2 regularization).
    
    Args:
        X (torch.Tensor): Input activations of shape (n_samples, d_in).
        Y (torch.Tensor): Target activations/deltas of shape (n_samples, d_out).
        alpha (float): L2 regularization strength.
        
    Returns:
        W (torch.Tensor): The solved weight matrix of shape (d_in, d_out).
    """
    # X: (n, d_in), Y: (n, d_out)
    n, d_in = X.shape
    device = X.device
    dtype = X.dtype
    
    # Use float64 for stability during matrix inversion
    X_f64 = X.to(torch.float64)
    Y_f64 = Y.to(torch.float64)
    
    # (X.T @ X + alpha * I) @ W = X.T @ Y
    XTX = torch.matmul(X_f64.t(), X_f64)
    reg = alpha * torch.eye(d_in, device=device, dtype=torch.float64)
    
    # Solve via Cholesky or pseudo-inverse
    W_f64 = torch.linalg.solve(XTX + reg, torch.matmul(X_f64.t(), Y_f64))
    
    return W_f64.to(dtype)

def low_rank_decompose_for_peft(W: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decomposes a full-rank delta matrix W into low-rank matrices A and B using SVD.
    A: (rank, d_in), B: (d_out, rank) such that W approx A.T @ B.T.
    Matches the PEFT/LoRA weight format for linear layers.
    """
    # W: (d_in, d_out)
    # We want W approx A^T @ B^T
    # SVD: W = U S V^T
    # W_r = U_r S_r V_r^T = (U_r sqrt(S_r)) (sqrt(S_r) V_r^T)
    # A^T = U_r sqrt(S_r) -> A = (U_r sqrt(S_r))^T = sqrt(S_r) U_r^T
    # B^T = sqrt(S_r) V_r^T -> B = (sqrt(S_r) V_r^T)^T = V_r sqrt(S_r)
    
    U, S, Vh = torch.linalg.svd(W.to(torch.float64), full_matrices=False)
    
    # Truncate to rank
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    
    sqrtS = torch.sqrt(S_r)
    
    # A: (rank, d_in)
    A = (sqrtS[:, None] * U_r.T).to(W.dtype)
    # B: (d_out, rank) -> Vh_r is (rank, d_out), so Vh_r.T is (d_out, rank)
    B = (Vh_r.T * sqrtS[None, :]).to(W.dtype)
    
    return A, B

def piecewise_svd_projection(
    W_delta: torch.Tensor,
    r_target: int,
    high_energy_ratio: float = 0.3,
    mid_energy_ratio: float = 0.5,
    high_scale: float = 1.0,
    mid_scale: float = 0.9,
    low_scale: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Piecewise Component Projection (v2.8)
    Splits the delta weight into different energy components via SVD.
    """
    U, S, Vh = torch.linalg.svd(W_delta.float(), full_matrices=False)
    num_s = len(S)
    k_high = max(1, int(num_s * high_energy_ratio))
    k_mid = max(1, int(num_s * mid_energy_ratio))
    S_new = S.clone()
    S_new[:k_high] *= high_scale
    S_new[k_high:k_high+k_mid] *= mid_scale
    S_new[k_high+k_mid:] *= low_scale
    U_trunc = U[:, :r_target]
    S_trunc = S_new[:r_target]
    Vh_trunc = Vh[:r_target, :]
    return U_trunc, S_trunc, Vh_trunc

def kernel_orthogonal_procrustes(
    A: torch.Tensor,
    B: torch.Tensor,
    num_heads: int,
    sigma: float = 1.0,
    n_components: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kernel Orthogonal Procrustes (KOP) with Nyström Approximation (v2.9)."""
    N, d = A.shape
    head_dim = d // num_heads
    A_heads = A.view(N, num_heads, head_dim)
    B_heads = B.view(N, num_heads, head_dim)
    R_list, s_list, A_trans_list = [], [], []
    for i in range(num_heads):
        A_i, B_i = A_heads[:, i, :], B_heads[:, i, :]
        indices = torch.randperm(N)[:min(n_components, N)]
        landmarks = A_i[indices]
        dist = torch.cdist(A_i, landmarks) ** 2
        K_am = torch.exp(-dist / (2 * sigma ** 2))
        dist_mm = torch.cdist(landmarks, landmarks) ** 2
        K_mm = torch.exp(-dist_mm / (2 * sigma ** 2)) + 1e-6 * torch.eye(len(indices), device=A.device)
        L, Q = torch.linalg.eigh(K_mm)
        K_mm_inv_half = Q @ torch.diag(1.0 / torch.sqrt(torch.clamp(L, min=1e-7))) @ Q.t()
        Phi_A = torch.matmul(K_am, K_mm_inv_half)
        Phi_B = torch.matmul(torch.matmul(B_i, torch.linalg.pinv(B_i)), Phi_A)
        U, S, Vh = torch.linalg.svd(torch.matmul(Phi_A.t(), Phi_B), full_matrices=False)
        R_k = torch.matmul(U, Vh)
        Phi_A_trans = torch.matmul(Phi_A, R_k)
        W_reconstruct = torch.matmul(torch.linalg.pinv(Phi_A), B_i)
        A_trans_list.append(torch.matmul(Phi_A_trans, W_reconstruct))
        R_list.append(R_k)
        s_list.append(torch.tensor(1.0, device=A.device))
    A_transformed = torch.stack(A_trans_list, dim=1).reshape(N, d)
    return A_transformed, torch.stack(R_list), torch.stack(s_list)

def swiglu_jacobian(x: torch.Tensor) -> torch.Tensor:
    """Calculates the diagonal Jacobian of the SiLU/Swish component of SwiGLU."""
    sig = torch.sigmoid(x)
    return sig * (1 + x * (1 - sig))

def geglu_jacobian(x: torch.Tensor) -> torch.Tensor:
    """Calculates the diagonal Jacobian of the GELU component of GeGLU."""
    return 0.5 * (1 + torch.erf(x / math.sqrt(2))) + (x / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * x**2)

def swiglu_hessian(x: torch.Tensor) -> torch.Tensor:
    """Calculates the diagonal Hessian (2nd derivative) of the SiLU/Swish component."""
    sig = torch.sigmoid(x)
    return sig * (1 - sig) * (2 + x * (1 - 2 * sig))

def geglu_hessian(x: torch.Tensor) -> torch.Tensor:
    """Calculates the diagonal Hessian of the GELU component."""
    return (2 - x**2) / math.sqrt(2 * math.pi) * torch.exp(-0.5 * x**2)

def jacobian_tangent_space_alignment(
    A: torch.Tensor,
    B: torch.Tensor,
    num_heads: int,
    activation_type: str = "swiglu",
    activation_states: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Jacobian Tangent Space Alignment (JTSA) (v2.9)."""
    N, d = A.shape
    head_dim = d // num_heads
    A_heads, B_heads = A.view(N, num_heads, head_dim), B.view(N, num_heads, head_dim)
    state_heads = activation_states.view(N, num_heads, head_dim) if activation_states is not None else None
    j_func = swiglu_jacobian if activation_type == "swiglu" else geglu_jacobian
    R_list, s_list, A_trans_list = [], [], []
    for i in range(num_heads):
        A_i, B_i = A_heads[:, i, :], B_heads[:, i, :]
        state = state_heads[:, i, :] if state_heads is not None else B_i / (torch.norm(B_i, dim=0, keepdim=True) + 1e-6)
        J = j_func(state)
        M = torch.matmul((J * A_i).t(), B_i)
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        R = torch.matmul(U, Vh)
        A_trans_list.append(torch.matmul(A_i, R))
        R_list.append(R)
        s_list.append(torch.tensor(1.0, device=A.device))
    return torch.stack(A_trans_list, dim=1).reshape(N, d), torch.stack(R_list), torch.stack(s_list)

def hessian_aware_manifold_alignment(
    A: torch.Tensor,
    B: torch.Tensor,
    num_heads: int,
    activation_type: str = "swiglu",
    alpha: float = 0.5,
    activation_states: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hessian-Aware Manifold Alignment (HAMA) (v2.9)."""
    N, d = A.shape
    head_dim = d // num_heads
    A_heads, B_heads = A.view(N, num_heads, head_dim), B.view(N, num_heads, head_dim)
    state_heads = activation_states.view(N, num_heads, head_dim) if activation_states is not None else None
    j_func = swiglu_jacobian if activation_type == "swiglu" else geglu_jacobian
    h_func = swiglu_hessian if activation_type == "swiglu" else geglu_hessian
    R_list, s_list, A_trans_list = [], [], []
    for i in range(num_heads):
        A_i, B_i = A_heads[:, i, :], B_heads[:, i, :]
        state = state_heads[:, i, :] if state_heads is not None else B_i / (torch.norm(B_i, dim=0, keepdim=True) + 1e-6)
        J, H = j_func(state), h_func(state)
        curvature_factor = J + 0.5 * alpha * H * state
        M = torch.matmul((curvature_factor * A_i).t(), B_i)
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        R = torch.matmul(U, Vh)
        A_trans_list.append(torch.matmul(A_i, R))
        R_list.append(R)
        s_list.append(torch.tensor(1.0, device=A.device))
    return torch.stack(A_trans_list, dim=1).reshape(N, d), torch.stack(R_list), torch.stack(s_list)

if __name__ == "__main__":
    print("Task Vector Projection Core Algorithms Loaded Successfully.")
