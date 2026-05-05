import pytest
import os
from neural_scalpel.serving.engine import ServingMode
from neural_scalpel.serving.mode_selector import select_serving_mode, parse_serving_mode

def test_parse_serving_mode():
    assert parse_serving_mode("internal") == ServingMode.INTERNAL
    assert parse_serving_mode("EXTERNAL_PROXY") == ServingMode.EXTERNAL_PROXY
    assert parse_serving_mode("auto") == ServingMode.AUTO
    assert parse_serving_mode(None) == ServingMode.FAIL_CLOSED
    
    with pytest.raises(RuntimeError, match="Invalid SCALPEL_SERVING_MODE"):
        parse_serving_mode("invalid_mode")

def test_select_serving_mode_internal():
    # Internal requested and compatible
    res = select_serving_mode(ServingMode.INTERNAL, internal_compatible=True, external_proxy_configured=False)
    assert res.selected_mode == ServingMode.INTERNAL
    assert res.should_start is True
    
    # Internal requested but incompatible
    res = select_serving_mode(ServingMode.INTERNAL, internal_compatible=False, external_proxy_configured=True)
    assert res.selected_mode is None
    assert res.should_start is False

def test_select_serving_mode_external():
    # External proxy requested and configured
    res = select_serving_mode(ServingMode.EXTERNAL_PROXY, internal_compatible=False, external_proxy_configured=True)
    assert res.selected_mode == ServingMode.EXTERNAL_PROXY
    assert res.should_start is True
    
    # External proxy requested but not configured
    res = select_serving_mode(ServingMode.EXTERNAL_PROXY, internal_compatible=True, external_proxy_configured=False)
    assert res.selected_mode is None
    assert res.should_start is False

def test_select_serving_mode_auto():
    # Auto: Internal is compatible -> Use Internal
    res = select_serving_mode(ServingMode.AUTO, internal_compatible=True, external_proxy_configured=True)
    assert res.selected_mode == ServingMode.INTERNAL
    assert res.should_start is True
    
    # Auto: Internal incompatible, Proxy configured -> Use Proxy
    res = select_serving_mode(ServingMode.AUTO, internal_compatible=False, external_proxy_configured=True)
    assert res.selected_mode == ServingMode.EXTERNAL_PROXY
    assert res.should_start is True
    
    # Auto: Both incompatible/unconfigured -> Fail Closed
    res = select_serving_mode(ServingMode.AUTO, internal_compatible=False, external_proxy_configured=False)
    assert res.selected_mode is None
    assert res.should_start is False

def test_select_serving_mode_fail_closed():
    res = select_serving_mode(ServingMode.FAIL_CLOSED, internal_compatible=True, external_proxy_configured=True)
    assert res.selected_mode is None
    assert res.should_start is False

def test_select_serving_mode_native_lora():
    # Currently not implemented
    res = select_serving_mode(ServingMode.NATIVE_LORA, internal_compatible=True, external_proxy_configured=True)
    assert res.selected_mode is None
    assert res.should_start is False

def test_select_from_environment(monkeypatch):
    from neural_scalpel.serving.mode_selector import select_from_environment

    monkeypatch.setenv("SCALPEL_SERVING_MODE", "auto")
    res = select_from_environment(
        internal_compatible=False,
        external_proxy_configured=True,
    )
    assert res.selected_mode == ServingMode.EXTERNAL_PROXY
    assert res.should_start is True

    monkeypatch.delenv("SCALPEL_SERVING_MODE", raising=False)
    res = select_from_environment(
        internal_compatible=True,
        external_proxy_configured=True,
    )
    # Default to fail_closed if env var is missing
    assert res.selected_mode is None
    assert res.should_start is False
