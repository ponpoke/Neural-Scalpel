import pytest
from unittest.mock import patch, MagicMock
from neural_scalpel.commands.safe_project import run_safe_project

class MockArgs:
    def __init__(self, **kwargs):
        self.source_base_model = "sb"
        self.source_adapter = "sa"
        self.target_model = "tm"
        self.benchmark = "sql_50"
        self.output_dir = "runs/test"
        self.rank = 16
        self.alpha = 16
        self.positive_delta_threshold = 0.0
        self.max_regression_rate = 0.05
        self.force = False
        self.projection_mode = "linear"
        for k, v in kwargs.items():
            setattr(self, k, v)

@patch("neural_scalpel.commands.safe_project.DiagnosticRunner.execute")
@patch("neural_scalpel.commands.safe_project.run_project")
@patch("neural_scalpel.commands.safe_project.run_evaluate")
def test_safe_project_flow_success(mock_eval, mock_proj, mock_diag):
    # Setup mock diagnostic report
    report = MagicMock()
    report.release_decision_gate.verdict = "PROJECTION_CANDIDATE"
    mock_diag.return_value = report
    
    args = MockArgs()
    run_safe_project(args)
    
    # Should call project and evaluate
    mock_proj.assert_called_once()
    mock_eval.assert_called_once()

@patch("neural_scalpel.commands.safe_project.DiagnosticRunner.execute")
@patch("neural_scalpel.commands.safe_project.run_project")
@patch("neural_scalpel.commands.safe_project.run_evaluate")
def test_safe_project_abort_on_inconclusive(mock_eval, mock_proj, mock_diag):
    # Setup mock diagnostic report
    report = MagicMock()
    report.release_decision_gate.verdict = "INCONCLUSIVE"
    mock_diag.return_value = report
    
    args = MockArgs()
    run_safe_project(args)
    
    # Should NOT call project or evaluate
    mock_proj.assert_not_called()
    mock_eval.assert_not_called()

@patch("neural_scalpel.commands.safe_project.DiagnosticRunner.execute")
@patch("neural_scalpel.commands.safe_project.run_project")
@patch("neural_scalpel.commands.safe_project.run_evaluate")
def test_safe_project_force_continue(mock_eval, mock_proj, mock_diag):
    # Setup mock diagnostic report
    report = MagicMock()
    report.release_decision_gate.verdict = "INCONCLUSIVE"
    mock_diag.return_value = report
    
    args = MockArgs(force=True)
    run_safe_project(args)
    
    # Should call project and evaluate DESPITE inconclusive verdict because of --force
    mock_proj.assert_called_once()
    mock_eval.assert_called_once()
