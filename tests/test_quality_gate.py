import unittest
import json
import os
from neural_scalpel.core.quality_gate import SourceAdapterQualityReport, QualityGateConfig

class TestSourceAdapterQualityGate(unittest.TestCase):
    def test_positive_teacher(self):
        report = SourceAdapterQualityReport(
            delta={"execution_accuracy": 0.16},
            regression_rate=0.02
        )
        report.generate_verdict()
        self.assertEqual(report.verdict, "POSITIVE_TEACHER")
        self.assertEqual(report.gate_status, "PASS")

    def test_negative_teacher_by_delta(self):
        report = SourceAdapterQualityReport(
            delta={"execution_accuracy": -0.06},
            regression_rate=0.05
        )
        report.generate_verdict()
        self.assertEqual(report.verdict, "NEGATIVE_TEACHER")
        self.assertEqual(report.gate_status, "FAIL")

    def test_negative_teacher_by_regression(self):
        report = SourceAdapterQualityReport(
            delta={"execution_accuracy": 0.05},
            regression_rate=0.15 # Above 0.10 threshold
        )
        report.generate_verdict()
        self.assertEqual(report.verdict, "NEGATIVE_TEACHER")
        self.assertEqual(report.gate_status, "FAIL")

    def test_unstable_teacher_collapse(self):
        report = SourceAdapterQualityReport(
            delta={"execution_accuracy": 0.05},
            stability={"collapse_detected": 1.0}
        )
        report.generate_verdict()
        self.assertEqual(report.verdict, "UNSTABLE_TEACHER")
        self.assertEqual(report.gate_status, "FAIL")

    def test_unstable_teacher_empty_output(self):
        report = SourceAdapterQualityReport(
            delta={"execution_accuracy": 0.05},
            stability={"empty_output_rate": 0.20} # Above 0.05 threshold
        )
        report.generate_verdict()
        self.assertEqual(report.verdict, "UNSTABLE_TEACHER")
        self.assertEqual(report.gate_status, "FAIL")
        self.assertTrue(any("empty output rate" in n for n in report.notes))

    def test_repetition_warning(self):
        report = SourceAdapterQualityReport(
            delta={"execution_accuracy": 0.10},
            stability={"repetition_rate": 0.15} # Above 0.10 threshold
        )
        report.generate_verdict()
        # Should be POSITIVE verdict but status WARNING
        self.assertEqual(report.verdict, "POSITIVE_TEACHER")
        self.assertEqual(report.gate_status, "WARNING")
        self.assertEqual(report.recommendation, "PROCEED_WITH_CAUTION")

    def test_weak_positive_teacher(self):
        # Default: positive_threshold=0.03, weak=0.00, negative=-0.01
        report = SourceAdapterQualityReport(
            delta={"execution_accuracy": 0.02} # Between 0.00 and 0.03
        )
        report.generate_verdict()
        self.assertEqual(report.verdict, "WEAK_POSITIVE_TEACHER")
        self.assertEqual(report.gate_status, "WARNING")

    def test_neutral_teacher(self):
        report = SourceAdapterQualityReport(
            delta={"execution_accuracy": -0.005} # Between -0.01 and 0.00
        )
        report.generate_verdict()
        self.assertEqual(report.verdict, "NEUTRAL_TEACHER")
        self.assertEqual(report.gate_status, "WARNING")

    def test_inconclusive_missing_metric(self):
        report = SourceAdapterQualityReport(
            delta={"wrong_metric": 0.10}
        )
        report.generate_verdict()
        self.assertEqual(report.verdict, "INCONCLUSIVE")
        self.assertEqual(report.gate_status, "WARNING")
        self.assertTrue(any("not found" in n for n in report.notes))

    def test_serialization(self):
        report = SourceAdapterQualityReport(
            base_model="test-base",
            delta={"execution_accuracy": 0.10},
            metadata={"test": "data"}
        )
        report.generate_verdict()
        
        # Markdown contains metadata
        md = report.to_markdown()
        self.assertIn("## Metadata", md)
        self.assertIn('"test": "data"', md)
        
        # JSON export
        path = "test_report.json"
        report.to_json(path)
        self.assertTrue(os.path.exists(path))
        with open(path, "r") as f:
            data = json.load(f)
            self.assertEqual(data["base_model"], "test-base")
        os.remove(path)

if __name__ == "__main__":
    unittest.main()
