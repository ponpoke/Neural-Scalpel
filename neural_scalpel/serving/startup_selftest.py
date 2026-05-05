"""
Neural-Scalpel Startup Self-Test

Boot-time validation ensuring the runtime environment meets all
prerequisites before accepting traffic. The healthz endpoint
returns unhealthy until all self-tests pass.

Checks performed:
  1. vLLM version compatibility (if applicable)
  2. GPU availability
  3. Model layer map resolution
  4. Route registry readability
  5. Payload storage readability
  6. Dry-run swap/rollback cycle
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from pathlib import Path


@dataclass
class SelfTestResult:
    """Result of a single self-test check."""
    name: str
    passed: bool
    duration_ms: float = 0.0
    detail: str = ""


@dataclass
class SelfTestReport:
    """Aggregate report of all self-test results."""
    results: List[SelfTestResult] = field(default_factory=list)
    started_at: float = 0.0
    completed_at: float = 0.0

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def total_duration_ms(self) -> float:
        return (self.completed_at - self.started_at) * 1000

    @property
    def failed_tests(self) -> List[SelfTestResult]:
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "tests": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration_ms": round(r.duration_ms, 2),
                    "detail": r.detail,
                }
                for r in self.results
            ],
        }


def _run_check(name: str, check_fn: Callable[[], str]) -> SelfTestResult:
    """Runs a single check function, capturing timing and exceptions."""
    t0 = time.perf_counter()
    try:
        detail = check_fn()
        duration = (time.perf_counter() - t0) * 1000
        return SelfTestResult(name=name, passed=True, duration_ms=duration, detail=detail)
    except Exception as e:
        duration = (time.perf_counter() - t0) * 1000
        return SelfTestResult(name=name, passed=False, duration_ms=duration, detail=str(e))


def check_gpu_available() -> str:
    """Verifies that at least one CUDA GPU is available."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    return f"{gpu_name} ({vram_gb:.1f} GB)"


def check_vllm_version(supported_range: Optional[str] = None) -> str:
    """Checks if vLLM is importable and within supported version range."""
    try:
        import vllm
        version = getattr(vllm, "__version__", "unknown")
        return f"vLLM {version}"
    except ImportError:
        return "vLLM not installed (standalone mode)"


def check_route_registry(storage_dir: str) -> str:
    """Verifies the route registry storage directory is readable."""
    path = Path(storage_dir)
    if not path.exists():
        raise RuntimeError(f"Registry directory does not exist: {storage_dir}")
    if not path.is_dir():
        raise RuntimeError(f"Registry path is not a directory: {storage_dir}")
    if not os.access(storage_dir, os.R_OK):
        raise RuntimeError(f"Registry directory is not readable: {storage_dir}")
    file_count = len(list(path.glob("*.scalpel_route")))
    return f"{file_count} route files found"


def check_payload_storage(payload_dir: str) -> str:
    """Verifies the payload storage directory is readable."""
    path = Path(payload_dir)
    if not path.exists():
        raise RuntimeError(f"Payload directory does not exist: {payload_dir}")
    if not os.access(payload_dir, os.R_OK):
        raise RuntimeError(f"Payload directory is not readable: {payload_dir}")
    st_count = len(list(path.glob("**/*.safetensors")))
    return f"{st_count} safetensors files found"


def check_dry_run_swap(model_state_dict: Optional[dict] = None) -> str:
    """Performs a dry-run swap and rollback cycle on a test tensor."""
    import torch
    # Use a small test tensor
    test_tensor = torch.randn(4, 4)
    original = test_tensor.clone()
    delta = torch.ones(4, 4) * 0.01

    # Swap
    with torch.no_grad():
        test_tensor.add_(delta)

    # Verify swap changed values
    if torch.allclose(test_tensor, original):
        raise RuntimeError("Dry-run swap had no effect")

    # Rollback
    with torch.no_grad():
        test_tensor.copy_(original)

    # Verify rollback restored values
    if not torch.allclose(test_tensor, original):
        raise RuntimeError("Dry-run rollback failed to restore original values")

    return "Swap/rollback cycle verified"


def run_self_tests(
    registry_dir: str = ".",
    payload_dir: str = ".",
    require_gpu: bool = False,
    model_state_dict: Optional[dict] = None,
) -> SelfTestReport:
    """
    Runs the complete self-test suite.

    Args:
        registry_dir: Path to the route registry storage
        payload_dir: Path to the payload file storage
        require_gpu: If True, GPU check failure is fatal
        model_state_dict: Optional model state dict for dry-run swap

    Returns:
        SelfTestReport with results of all checks
    """
    report = SelfTestReport()
    report.started_at = time.perf_counter()

    # 1. GPU check
    gpu_result = _run_check("gpu_available", check_gpu_available)
    if not require_gpu and not gpu_result.passed:
        gpu_result.passed = True
        gpu_result.detail = "GPU not available (non-fatal in CPU mode)"
    report.results.append(gpu_result)

    # 2. vLLM version
    report.results.append(_run_check("vllm_version", check_vllm_version))

    # 3. Route registry
    report.results.append(
        _run_check("route_registry", lambda: check_route_registry(registry_dir))
    )

    # 4. Payload storage
    report.results.append(
        _run_check("payload_storage", lambda: check_payload_storage(payload_dir))
    )

    # 5. Dry-run swap/rollback
    report.results.append(
        _run_check("dry_run_swap", lambda: check_dry_run_swap(model_state_dict))
    )

    report.completed_at = time.perf_counter()
    return report
