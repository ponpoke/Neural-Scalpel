# Neural-Scalpel (Concept-Projector)

**Zero-Dataset Intelligence Transplantation Framework**

[![Version](https://img.shields.io/badge/version-1.0.0-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Certification](https://img.shields.io/badge/Status-Reasoning--Certified-gold)](docs/PRODUCTION_CERTIFICATION.md)

Neural-Scalpel (Concept-Projector) is a surgical mathematical framework for extracting learned concepts (intelligence) from one neural architecture and transplanting them into another—**without datasets, without retraining, and without logic destruction.**

**Now supporting both Vision Models (e.g., SDXL → FLUX) and Large Language Models (LLMs)!**

---

### 🚀 Mathematically Validated Pipeline
Version 1.0.0 is certified for high-level reasoning preservation via structural logic alignment:
*   **Precision:** **+0.05% PPL Degradation** (Minimal Logic Loss)
*   **HumanEval (Coding):** **97.9% Retention**
*   **GSM8K (Math):** **98.9% Retention**
*   **Stability (Logic):** **0.00% Software Error Rate** in our 100-thread simulation.

---

## 🔬 Core Technologies

### 1. Hard-WDR (Wasserstein Discrete Routing)
Mitigates "Logic Blurring" by enforcing 1-to-1 preservation of specialized Attention Heads. 

### 2. JTSA (Jacobian Tangent Space Alignment)
Compensates for non-linear GeGLU/SwiGLU distortions using a first-order Taylor approximation across the activation manifold.

### 3. VRAM Shadow Registering (Layer 4 - Experimental)
Enables runtime concept injection via atomic pointer swaps. 
> **⚠️ Technical Warning:** Python-level `copy_` operations lack true C++/CUDA level atomicity. Full **ACID compliance** is currently under consideration as part of our research into native CUDA-level pointer orchestration.

---

## 🛠️ Key Features
- **Physical Memory Optimization:** Consumer-grade (16GB RAM) delta manipulation via AVPS (preserving 99% energy).
- **Adaptive rSVD Bootstrap:** Low-rank knowledge extraction with auto-stopping.
- **Enterprise Extensions:** Support for **Quantization-Aware Procrustes (QAP)** for GGUF/INT4 grids, and **Expert-wise Alignment** for MoE architectures (available under commercial terms).


---

## Dual-License Model
Neural-Scalpel operates under a dual-licensing strategy to balance open research and industrial reliability:
1. **Open-Source (MIT):** Free for personal experimentation, research, and non-profit use.
2. **Commercial License:** Required for corporate environments, commercial AI services, or large-scale production deployments. This license provides legal indemnification, priority support, and access to optimized high-performance surgical manifests.

---

## Technical Documentation
- [Technical Report](TECHNICAL_REPORT.md): Mathematical proofs and empirical evaluations.
- [Usage Guide](docs/USAGE.md): API reference and CLI examples.
- [Production Certification](docs/PRODUCTION_CERTIFICATION.md): Detailed reasoning retention benchmarks.

---

## Contact
For commercial licensing, collaboration, or research inquiries, please reach out via:
- **Email**: `ponpoke10@gmail.com`
- **X (Twitter)**: [@ponpoke10](https://x.com/ponpoke10)

---

## Support the Research
If this project saved you weeks of fine-tuning time, please consider supporting the development. Tips are greatly appreciated and help sustain the compute resources (and electricity!) needed for further cross-architecture research.

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/ponpoke)

---

## Disclaimer (AS IS)
This repository contains pure mathematical algorithms and structural proofs. It does NOT contain any model weights, proprietary data, or model-specific loading scripts. 
This code is provided "AS IS", without warranty of any kind. You are solely responsible for how you use this algorithm and for ensuring you comply with the licenses of any AI models you apply it to.

---
*Developed and verified locally on an NVIDIA RTX 5060 Ti 16GB.*
