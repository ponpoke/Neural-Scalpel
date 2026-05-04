import os
import torch
import numpy as np
from typing import Dict, Optional
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from neural_scalpel.io.base import BaseIOBridge

class GGUFBridge(BaseIOBridge):
    """
    I/O Bridge for GGUF format using the 'gguf' library.
    Handles auto-dequantization to FP16 for surgical operations
    and re-quantization for saving transformed models.
    """
    def __init__(self):
        self.writer: Optional[GGUFWriter] = None

    def load_weights(self, path: str) -> Dict[str, torch.Tensor]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"GGUF file not found at: {path}")
            
        print(f"[IO] Loading GGUF from {path}")
        reader = GGUFReader(path)
        state_dict = {}
        
        for tensor in reader.tensors:
            name = tensor.name
            qtype = tensor.tensor_type
            data = tensor.data
            
            if qtype == GGMLQuantizationType.F16:
                state_dict[name] = torch.from_numpy(data.astype(np.float16))
            elif qtype == GGMLQuantizationType.F32:
                state_dict[name] = torch.from_numpy(data.astype(np.float32))
            else:
                state_dict[name] = self._dequantize_torch(data, qtype, tensor.shape)
                
        return state_dict

    def iter_layers(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"GGUF file not found at: {path}")
            
        print(f"[IO] Streaming GGUF from {path}")
        reader = GGUFReader(path)
        
        for tensor in reader.tensors:
            name = tensor.name
            qtype = tensor.tensor_type
            data = tensor.data
            
            if qtype == GGMLQuantizationType.F16:
                yield name, torch.from_numpy(data.astype(np.float16))
            elif qtype == GGMLQuantizationType.F32:
                yield name, torch.from_numpy(data.astype(np.float32))
            else:
                yield name, self._dequantize_torch(data, qtype, tensor.shape)

    def save_weights(self, path: str, state_dict: Dict[str, torch.Tensor], metadata: Optional[Dict] = None):
        """Legacy method for small models."""
        self.open_writer(path, metadata)
        for name, tensor in state_dict.items():
            self.write_layer(name, tensor)
        self.close_writer()

    def open_writer(self, path: str, metadata: Optional[Dict] = None):
        print(f"[IO] Opening GGUF writer at {path}...")
        self.writer = GGUFWriter(path, "neural-scalpel-v1")
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, str):
                    self.writer.add_string(k, v)
                elif isinstance(v, int):
                    self.writer.add_uint32(k, v)
                elif isinstance(v, float):
                    self.writer.add_float32(k, v)

    def write_layer(self, name: str, tensor: torch.Tensor):
        if self.writer is None:
            raise RuntimeError("Writer not opened. Call open_writer() first.")
            
        if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
            print(f"[IO] Quantizing {name} to Q8_0 and writing...")
            q_data, q_type = self._quantize_q8_0(tensor)
            try:
                self.writer.add_tensor(name, q_data, raw_dtype=q_type)
            except TypeError:
                # Fallback for gguf library versions with different API
                self.writer.add_tensor(name, q_data)
        else:
            self.writer.add_tensor(name, tensor.numpy())

    def close_writer(self):
        if self.writer:
            print(f"[IO] Closing GGUF container...")
            self.writer.write_header_to_file()
            self.writer.write_kv_data_to_file()
            self.writer.write_tensors_to_file()
            self.writer.close()
            self.writer = None
            print(f"[SUCCESS] Surgery complete.")

    def _dequantize_torch(self, data: np.ndarray, qtype: GGMLQuantizationType, shape: tuple) -> torch.Tensor:
        """Vectorized dequantization using PyTorch."""
        shape = tuple(map(int, shape))
        data_u8 = torch.from_numpy(data.copy().view(np.uint8))
        
        if qtype == GGMLQuantizationType.Q8_0:
            n_blocks = data_u8.numel() // 34
            if n_blocks == 0: return torch.zeros(shape, dtype=torch.float16)
            blocks = data_u8[:n_blocks * 34].reshape(n_blocks, 34)
            deltas = blocks[:, :2].view(torch.float16).to(torch.float32).reshape(n_blocks, 1)
            qs = blocks[:, 2:].reshape(n_blocks, 32).view(torch.int8).to(torch.float32)
            return (qs * deltas).reshape(-1)[:np.prod(shape)].reshape(shape).to(torch.float16)
            
        elif qtype == GGMLQuantizationType.Q4_0:
            n_blocks = data_u8.numel() // 18
            if n_blocks == 0: return torch.zeros(shape, dtype=torch.float16)
            blocks = data_u8[:n_blocks * 18].reshape(n_blocks, 18)
            deltas = blocks[:, :2].view(torch.float16).to(torch.float32).reshape(n_blocks, 1)
            qs_raw = blocks[:, 2:].reshape(n_blocks, 16)
            q0, q1 = (qs_raw & 0x0F).to(torch.int8) - 8, (qs_raw >> 4).to(torch.int8) - 8
            qs = torch.stack([q0, q1], dim=2).reshape(n_blocks, 32).to(torch.float32)
            return (qs * deltas).reshape(-1)[:np.prod(shape)].reshape(shape).to(torch.float16)

        return torch.zeros(shape, dtype=torch.float16)

    def _quantize_q8_0(self, tensor: torch.Tensor) -> tuple:
        """Vectorized Q8_0 quantization."""
        x = tensor.to(torch.float32).flatten()
        if len(x) % 32 != 0:
            x = torch.cat([x, torch.zeros(32 - (len(x) % 32), device=x.device)])
        n_blocks = len(x) // 32
        blocks = x.view(n_blocks, 32)
        abs_max, _ = torch.max(torch.abs(blocks), dim=1, keepdim=True)
        deltas = abs_max / 127.0
        qs = torch.round(blocks / (deltas + 1e-12)).clamp(-128, 127).to(torch.int8)
        q_data = torch.cat([deltas.to(torch.float16).view(torch.uint8), qs.view(torch.uint8)], dim=1).numpy()
        return q_data, GGMLQuantizationType.Q8_0
