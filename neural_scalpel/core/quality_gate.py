from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json

@dataclass
class QualityGateConfig:
    """Configuration for Quality Gate thresholds."""
    primary_metric: str = "execution_accuracy"
    positive_delta_threshold: float = 0.03
    weak_positive_threshold: float = 0.00
    negative_delta_threshold: float = -0.01
    max_regression_rate: float = 0.10
    max_empty_output_rate: float = 0.05
    max_repetition_rate: float = 0.10

@dataclass
class SourceAdapterQualityReport:
    """Report summarizing the quality of a source adapter on its own base model."""
    schema_version: str = "source_adapter_quality_gate.v1.1"
    base_model: str = ""
    adapter_path: str = ""
    benchmark: str = ""
    task_type: str = "sql"
    
    base_metrics: Dict[str, float] = field(default_factory=dict)
    adapter_metrics: Dict[str, float] = field(default_factory=dict)
    delta: Dict[str, float] = field(default_factory=dict)
    
    total_cases: int = 0
    failure_classification: Dict[str, int] = field(default_factory=dict)
    regression_rate: float = 0.0
    
    stability: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    verdict: str = "INCONCLUSIVE"
    gate_status: str = "WARNING"
    recommendation: str = "WAITING_FOR_EVAL"
    notes: List[str] = field(default_factory=list)

    def to_json(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)

    def generate_verdict(self, config: Optional[QualityGateConfig] = None):
        """Generate verdict based on metrics and thresholds."""
        if config is None:
            config = QualityGateConfig()

        if config.primary_metric not in self.delta:
            self.verdict = "INCONCLUSIVE"
            self.gate_status = "WARNING"
            self.recommendation = "PRIMARY_METRIC_MISSING"
            self.notes.append(f"Primary metric '{config.primary_metric}' not found in evaluation results.")
            return

        diff = self.delta[config.primary_metric]
        
        # 1. Stability Check (Failing)
        # Check for model collapse
        if self.stability.get("collapse_detected", 0.0) > 0.5:
            self.verdict = "UNSTABLE_TEACHER"
            self.gate_status = "FAIL"
            self.recommendation = "DO_NOT_PROJECT"
            self.notes.append("Severe model collapse detected (Adapter output is significantly worse than base).")
            return

        # Check for empty output
        if self.stability.get("empty_output_rate", 0.0) > config.max_empty_output_rate:
            self.verdict = "UNSTABLE_TEACHER"
            self.gate_status = "FAIL"
            self.recommendation = "DO_NOT_PROJECT"
            self.notes.append(f"High empty output rate ({self.stability['empty_output_rate']*100:.1f}%) exceeds threshold ({config.max_empty_output_rate*100:.1f}%).")
            return

        # 2. Regression Rate Check (Failing)
        if self.regression_rate > config.max_regression_rate:
            self.verdict = "NEGATIVE_TEACHER"
            self.gate_status = "FAIL"
            self.recommendation = "DO_NOT_PROJECT"
            self.notes.append(f"High regression rate ({self.regression_rate*100:.1f}%) exceeds threshold ({config.max_regression_rate*100:.1f}%).")
            return

        # 3. Score based check (Verdict Determination)
        if diff >= config.positive_delta_threshold:
            self.verdict = "POSITIVE_TEACHER"
            self.gate_status = "PASS"
            self.recommendation = "PROCEED_TO_PROJECTION"
        elif diff > config.weak_positive_threshold:
            self.verdict = "WEAK_POSITIVE_TEACHER"
            self.gate_status = "WARNING"
            self.recommendation = "PROCEED_WITH_CAUTION"
        elif diff <= config.negative_delta_threshold:
            self.verdict = "NEGATIVE_TEACHER"
            self.gate_status = "FAIL"
            self.recommendation = "DO_NOT_PROJECT"
            self.notes.append(f"Adapter degrades base model performance by {abs(diff)*100:.1f}%.")
        else:
            self.verdict = "NEUTRAL_TEACHER"
            self.gate_status = "WARNING"
            self.recommendation = "PROJECTION_NOT_PRIORITIZED"

        # 4. Final Status Polishing (Downgrading status based on non-failing issues)
        if self.stability.get("repetition_rate", 0.0) > config.max_repetition_rate:
             self.notes.append(f"High repetition detected ({self.stability['repetition_rate']*100:.1f}%).")
             if self.gate_status == "PASS":
                 self.gate_status = "WARNING"
                 self.recommendation = "PROCEED_WITH_CAUTION"

    def to_markdown(self) -> str:
        """Generate a human-readable markdown summary."""
        md = f"# Source Adapter Quality Gate Report\n\n"
        md += f"## Summary\n\n"
        md += f"| Field | Value |\n"
        md += f"|---|---|\n"
        md += f"| Base Model | {self.base_model} |\n"
        md += f"| Adapter | {self.adapter_path} |\n"
        md += f"| Benchmark | {self.benchmark} |\n"
        md += f"| **Verdict** | **{self.verdict}** |\n"
        md += f"| **Status** | **{self.gate_status}** |\n"
        md += f"| Recommendation | {self.recommendation} |\n\n"
        
        md += f"## Metrics\n\n"
        md += f"| Metric | Base | Adapter | Delta |\n"
        md += f"|---|---:|---:|---:|\n"
        for m in self.base_metrics:
            b = self.base_metrics[m]
            a = self.adapter_metrics.get(m, 0.0)
            d = self.delta.get(m, 0.0)
            md += f"| {m} | {b*100:.1f}% | {a*100:.1f}% | {d*100:+.1f}% |\n"
        
        md += f"\n## Diagnostic Analysis\n\n"
        md += f"| Analysis | Value |\n"
        md += f"|---|---:|\n"
        md += f"| Total Cases | {self.total_cases} |\n"
        md += f"| Regression Rate | {self.regression_rate*100:.1f}% |\n"
        for k, v in self.stability.items():
            md += f"| {k.replace('_', ' ').capitalize()} | {v*100:.1f}% |\n"

        md += f"\n## Failure Classification\n\n"
        md += f"| Type | Count |\n"
        md += f"|---|---:|\n"
        for k, v in self.failure_classification.items():
            md += f"| {k.replace('_', ' ').capitalize()} | {v} |\n"
            
        if self.notes:
            md += f"\n## Notes\n\n"
            for note in self.notes:
                md += f"- {note}\n"
        
        md += f"\n## Metadata\n\n"
        md += f"```json\n{json.dumps(self.metadata, indent=2)}\n```\n"
                
        return md
