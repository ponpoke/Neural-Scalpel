# Methodology: Cross-Architecture Adapter Projection

The projection of a LoRA adapter from a source model (S) to a target model (T) involves several key stages:

## 1. Structural Homology Mapping
We identify the corresponding layers between the Source and Target architectures. For Qwen-family models, the layers are largely homologous, but dimensions (d_model, num_heads) differ.

## 2. Weight-Space Projection
We use **Head-wise Orthogonal Procrustes** to align the activation spaces.
The task vector $\tau = W_{tuned} - W_{base}$ is extracted and projected:
$W_{projected} = \text{Project}(\tau, S, T)$

## 3. Subspace Compression
Using **Adaptive Randomized SVD (rSVD)**, we compress the projected task vector to fit the target's rank (r) while preserving >95% of the L2 energy.

## 4. Non-Linear Compensation (JTSA)
To handle non-linearities (GeGLU/SwiGLU), we use **Jacobian Tangent Space Alignment**. This requires a small calibration dataset to estimate the activation manifold and preserve emergent outliers.

## 5. Fusion Handling
vLLM often fuses Q/K/V or Gate/Up layers into a single matrix. Our projection pipeline automatically handles the splitting, projecting, and re-fusing of these tensors.
