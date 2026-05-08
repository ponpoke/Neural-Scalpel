import os
from dataclasses import dataclass
from typing import Optional
from .engine import ServingMode


@dataclass
class ModeSelectionResult:
    requested_mode: ServingMode
    selected_mode: Optional[ServingMode]
    reason: str
    should_start: bool


def parse_serving_mode(value: Optional[str]) -> ServingMode:
    raw = (value or "fail_closed").strip().lower()
    try:
        return ServingMode(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid SCALPEL_SERVING_MODE: {raw}") from exc


def select_serving_mode(
    requested: ServingMode,
    internal_compatible: bool,
    external_proxy_configured: bool,
) -> ModeSelectionResult:
    if requested == ServingMode.INTERNAL:
        if internal_compatible:
            return ModeSelectionResult(requested, ServingMode.INTERNAL, "internal compatible", True)
        return ModeSelectionResult(requested, None, "internal incompatible; fail closed", False)

    if requested == ServingMode.EXTERNAL_PROXY:
        if external_proxy_configured:
            return ModeSelectionResult(requested, ServingMode.EXTERNAL_PROXY, "external proxy configured", True)
        return ModeSelectionResult(requested, None, "external proxy not configured; fail closed", False)

    if requested == ServingMode.AUTO:
        if internal_compatible:
            return ModeSelectionResult(requested, ServingMode.INTERNAL, "auto selected internal", True)
        if external_proxy_configured:
            return ModeSelectionResult(requested, ServingMode.EXTERNAL_PROXY, "auto fallback to external proxy", True)
        return ModeSelectionResult(requested, None, "auto found no safe serving mode; fail closed", False)

    if requested == ServingMode.NATIVE_LORA:
        return ModeSelectionResult(requested, None, "native_lora fallback not implemented; fail closed", False)

    return ModeSelectionResult(requested, None, "fail_closed requested", False)


def select_from_environment(
    internal_compatible: bool,
    external_proxy_configured: bool,
) -> ModeSelectionResult:
    requested = parse_serving_mode(os.environ.get("SCALPEL_SERVING_MODE"))
    return select_serving_mode(requested, internal_compatible, external_proxy_configured)
