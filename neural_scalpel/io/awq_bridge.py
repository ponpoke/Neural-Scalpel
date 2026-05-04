import torch
import os
from typing import Dict, Optional
from safetensors.torch import save_file, load_file, safe_open
from neural_scalpel.io.base import BaseIOBridge
from neural_scalpel.io.calibration import search_optimal_awq_scales

class AWQBridge(BaseIOBridge):
    """
    I/O Bridge for AWQ-style quantized models.
    Supports "Lightweight Manifold Re-calibration" (LMR) and Incremental Writing.
    """
    def __init__(self, calibration_data: Optional[torch.Tensor] = None):
        self.calibration_data = calibration_data # (N_samples, hidden_dim)
        self.writer_path: Optional[str] = None
        self.metadata: Optional[Dict] = None

    def load_weights(self, path: str) -> Dict[str, torch.Tensor]:
        print(f"[IO] Loading AWQ-formatted safetensors from {path}")
        return load_file(path)

    def iter_layers(self, path: str):
        print(f"[IO] Streaming AWQ safetensors from {path}")
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key).clone()

    def save_weights(self, path: str, state_dict: Dict[str, torch.Tensor], metadata: Optional[Dict] = None):
        """Legacy method for small models."""
        self.open_writer(path, metadata)
        for name, tensor in state_dict.items():
            self.write_layer(name, tensor)
        self.close_writer()

    def open_writer(self, path: str, metadata: Optional[Dict] = None):
        print(f"[IO] Opening AWQ Incremental Writer at {path}")
        self.writer_path = path
        self.metadata = metadata
        self.acc_dict = {} # Accumulate for simple POC sharding

    def write_layer(self, name: str, tensor: torch.Tensor):
        if "weight" in name and tensor.dim() == 2 and self.calibration_data is not None:
            if self.calibration_data.shape[1] == tensor.shape[1]:
                print(f"  -> Re-calibrating and Packing {name}...")
                scales = search_optimal_awq_scales(tensor, self.calibration_data)
                scaled_w = tensor * scales
                packed_w, q_scales, q_zeros = self._pack_int4(scaled_w)
                self.acc_dict[f"{name}.packed"] = packed_w
                self.acc_dict[f"{name}.scales"] = q_scales
                self.acc_dict[f"{name}.zeros"] = q_zeros
                self.acc_dict[f"{name}_awq_scales"] = scales
            else:
                self.acc_dict[name] = tensor
        else:
            self.acc_dict[name] = tensor

    def close_writer(self):
        if self.writer_path:
            print(f"[IO] Finalizing AWQ save...")
            save_file(self.acc_dict, self.writer_path, metadata=self.metadata)
            print(f"[SUCCESS] AWQ surgery complete.")
            self.writer_path = None
            self.acc_dict = {}

    def _pack_int4(self, weight: torch.Tensor) -> tuple:
        """Physically packs FP16 weights into INT4."""
        device = weight.device
        out_f, in_f = weight.shape
        mn = weight.min(dim=1, keepdim=True)[0]
        mx = weight.max(dim=1, keepdim=True)[0]
        q_scales = (mx - mn) / 15.0
        q_zeros = mn
        q_w = torch.round((weight - q_zeros) / (q_scales + 1e-12)).clamp(0, 15).to(torch.int32)
        if in_f % 8 != 0:
            q_w = torch.nn.functional.pad(q_w, (0, 8 - (in_f % 8)))
        in_f_packed = q_w.shape[1] // 8
        packed_w = torch.zeros((out_f, in_f_packed), dtype=torch.int32, device=device)
        for i in range(8):
            packed_w |= q_w[:, i::8] << (4 * i)
        return packed_w, q_scales.to(torch.float16), q_zeros.to(torch.float16)
