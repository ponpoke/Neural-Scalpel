import torch
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json

@dataclass
class ActivationAdapterSolution:
    """
    Represents the solved weight delta (full-rank) for a specific module.
    This is the intermediate step before low-rank compression.
    """
    module_weights: Dict[str, torch.Tensor]  # {module_name: weight_delta_matrix}
    target_model_id: str
    reconstruction_errors: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PeftExportResult:
    """
    The final low-rank (SVD) weights and PEFT configuration.
    Ready for deployment and loading with PeftModel.
    """
    lora_state_dict: Dict[str, torch.Tensor]  # {layer_key: weight_tensor}
    config: Dict[str, Any]
    rank: int
    lora_alpha: int
    mean_error: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save_pretrained(self, save_directory: Union[str, Path]):
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        torch.save(self.lora_state_dict, save_path / "adapter_model.bin")
        
        # Save config
        with open(save_path / "adapter_config.json", "w") as f:
            json.dump(self.config, f, indent=2)
            
        # Save metadata/report
        report = {
            "rank": self.rank,
            "lora_alpha": self.lora_alpha,
            "mean_reconstruction_error": self.mean_error,
            "metadata": self.metadata
        }
        with open(save_path / "lora_export_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        print(f"PEFT Adapter saved to {save_path}")
