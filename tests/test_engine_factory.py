import pytest
from unittest.mock import MagicMock, patch
from neural_scalpel.serving.engine_factory import EngineFactory, InternalPluginEngine
from neural_scalpel.serving.proxy_engine import ProxyServingEngine
from neural_scalpel.serving.backend_registry import BackendRegistry
from neural_scalpel.serving.engine import ServingMode

@pytest.fixture
def mock_runtime_factory():
    def _factory():
        return MagicMock()
    return _factory

def test_factory_selects_internal_when_requested_and_compatible(monkeypatch, mock_runtime_factory):
    monkeypatch.setenv("SCALPEL_SERVING_MODE", "internal")
    
    with patch("neural_scalpel.serving.engine_factory.run_self_tests") as mock_test:
        # Mock passing self-tests
        mock_report = MagicMock()
        mock_report.all_passed = True
        mock_test.return_value = mock_report
        
        engine = EngineFactory.create_engine(
            registry_dir=".", 
            payload_dir=".",
            internal_runtime_factory=mock_runtime_factory
        )
        
        assert isinstance(engine, InternalPluginEngine)
        assert engine.get_health()["engine_type"] == "internal_plugin"

def test_factory_fails_internal_when_incompatible(monkeypatch, mock_runtime_factory):
    monkeypatch.setenv("SCALPEL_SERVING_MODE", "internal")
    
    with patch("neural_scalpel.serving.engine_factory.run_self_tests") as mock_test:
        # Mock failing self-tests
        mock_report = MagicMock()
        mock_report.all_passed = False
        mock_test.return_value = mock_report
        
        with pytest.raises(RuntimeError, match="internal incompatible; fail closed"):
            EngineFactory.create_engine(
                registry_dir=".", 
                payload_dir=".",
                internal_runtime_factory=mock_runtime_factory
            )

def test_factory_falls_back_to_proxy_in_auto_mode(monkeypatch, mock_runtime_factory):
    monkeypatch.setenv("SCALPEL_SERVING_MODE", "auto")
    
    # 1. Mock internal failure
    with patch("neural_scalpel.serving.engine_factory.run_self_tests") as mock_test:
        mock_report = MagicMock()
        mock_report.all_passed = False
        mock_test.return_value = mock_report
        
        # 2. Provide a backend registry with routes
        reg = BackendRegistry()
        reg.register_backend("r1", "http://backend/v1/completions")
        
        engine = EngineFactory.create_engine(
            registry_dir=".", 
            payload_dir=".",
            backend_registry=reg,
            internal_runtime_factory=mock_runtime_factory
        )
        
        assert isinstance(engine, ProxyServingEngine)
        assert engine.get_health()["engine_type"] == "external_proxy"

def test_factory_fails_when_mode_is_fail_closed(monkeypatch):
    monkeypatch.setenv("SCALPEL_SERVING_MODE", "fail_closed")
    
    with pytest.raises(RuntimeError, match="fail_closed requested"):
        EngineFactory.create_engine(registry_dir=".", payload_dir=".")

def test_factory_fails_when_auto_has_no_options(monkeypatch):
    monkeypatch.setenv("SCALPEL_SERVING_MODE", "auto")
    
    with patch("neural_scalpel.serving.engine_factory.run_self_tests") as mock_test:
        mock_report = MagicMock()
        mock_report.all_passed = False
        mock_test.return_value = mock_report
        
        # No backend registry provided
        with pytest.raises(RuntimeError, match="auto found no safe serving mode"):
            EngineFactory.create_engine(registry_dir=".", payload_dir=".")

def test_factory_selects_proxy_when_requested(monkeypatch):
    monkeypatch.setenv("SCALPEL_SERVING_MODE", "external_proxy")

    reg = BackendRegistry()
    reg.register_backend("r1", "http://backend/v1/completions")

    with patch("neural_scalpel.serving.engine_factory.run_self_tests") as mock_test:
        mock_report = MagicMock()
        mock_report.all_passed = False
        mock_test.return_value = mock_report

        engine = EngineFactory.create_engine(
            registry_dir=".",
            payload_dir=".",
            backend_registry=reg,
        )

    assert isinstance(engine, ProxyServingEngine)
    assert engine.get_health()["engine_type"] == "external_proxy"

def test_factory_fails_proxy_when_unconfigured(monkeypatch):
    monkeypatch.setenv("SCALPEL_SERVING_MODE", "external_proxy")

    with pytest.raises(RuntimeError, match="external proxy not configured"):
        EngineFactory.create_engine(
            registry_dir=".",
            payload_dir=".",
            backend_registry=BackendRegistry(),
        )
