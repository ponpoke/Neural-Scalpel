from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json

@dataclass
class ValidationReport:
    """
    Standardized report for the multi-stage Behavioral Alignment validation gates.
    Encapsulates results from G1 (Signal Presence) to G9 (Task Evaluation).
    """
    phase: str
    status: str  # e.g., "SUCCESS", "WARNING", "FAILURE"
    gates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_gate(self, gate_id: str, success: bool, message: str, data: Optional[Dict[str, Any]] = None):
        self.gates[gate_id] = {
            "success": success,
            "message": message,
            "data": data or {}
        }

    def to_json(self) -> str:
        return json.dumps({
            "phase": self.phase,
            "status": self.status,
            "gates": self.gates,
            "metrics": self.metrics,
            "metadata": self.metadata
        }, indent=2)

    def is_all_passed(self, gate_ids: Optional[List[str]] = None) -> bool:
        targets = gate_ids if gate_ids else self.gates.keys()
        return all(self.gates[gid]["success"] for gid in targets if gid in self.gates)
