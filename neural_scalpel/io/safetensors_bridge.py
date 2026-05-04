import os
import torch
import json
from typing import Dict, Optional, List
from safetensors.torch import load_file, save_file, safe_open
from neural_scalpel.io.base import BaseIOBridge

class SafetensorsBridge(BaseIOBridge):
    """
    I/O Bridge for standard Hugging Face .safetensors format.
    Supports Incremental Writing via Sharding to prevent OOM.
    """
    def __init__(self):
        self.writer_path: Optional[str] = None
        self.current_shard: Dict[str, torch.Tensor] = {}
        self.shard_count = 0
        self.max_shard_size = 2 * 1024 * 1024 * 1024 # 2GB limit per shard

    def load_weights(self, path: str) -> Dict[str, torch.Tensor]:
        if os.path.isdir(path):
            file_path = os.path.join(path, "adapter_model.safetensors")
            if not os.path.exists(file_path):
                file_path = os.path.join(path, "model.safetensors")
        else:
            file_path = path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Safetensors file not found at: {file_path}")
        return load_file(file_path)

    def iter_layers(self, path: str):
        if os.path.isdir(path):
            file_path = os.path.join(path, "adapter_model.safetensors")
            if not os.path.exists(file_path):
                file_path = os.path.join(path, "model.safetensors")
        else:
            file_path = path

        print(f"[IO] Streaming safetensors from {file_path}")
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Clone to ensure we don't hold references to the mmap
                yield key, f.get_tensor(key).clone()

    def save_weights(self, path: str, state_dict: Dict[str, torch.Tensor], metadata: Optional[Dict] = None):
        """Legacy non-streaming save."""
        save_file(state_dict, path, metadata=metadata)

    def open_writer(self, path: str, metadata: Optional[Dict] = None):
        print(f"[IO] Opening Safetensors Incremental Writer at {path}")
        self.writer_path = path
        self.current_shard = {}
        self.shard_count = 0
        self.metadata = metadata

    def write_layer(self, name: str, tensor: torch.Tensor):
        if self.writer_path is None:
            raise RuntimeError("Writer not opened.")
            
        self.current_shard[name] = tensor.cpu().contiguous()
        
        # Check if shard size exceeds limit (approximate)
        current_size = sum(t.element_size() * t.nelement() for t in self.current_shard.values())
        if current_size > self.max_shard_size:
            self._flush_shard()

    def _flush_shard(self):
        if not self.current_shard:
            return
        
        shard_name = f"shard_{self.shard_count}.safetensors"
        if self.writer_path.endswith(".safetensors"):
             # If path is a file, use its directory
            dir_path = os.path.dirname(self.writer_path)
            shard_path = os.path.join(dir_path, shard_name)
        else:
            shard_path = os.path.join(self.writer_path, shard_name)
            
        print(f"[IO] Flushing shard to {shard_path}...")
        save_file(self.current_shard, shard_path, metadata=self.metadata)
        self.current_shard = {}
        self.shard_count += 1

    def close_writer(self):
        if self.shard_count == 0 and self.current_shard:
            # Single file mode for compatibility with PEFT/existing tests
            if os.path.isdir(self.writer_path):
                file_path = os.path.join(self.writer_path, "adapter_model.safetensors")
            elif self.writer_path.endswith(".safetensors"):
                file_path = self.writer_path
            else:
                # Treat as prefix/directory
                os.makedirs(self.writer_path, exist_ok=True)
                file_path = os.path.join(self.writer_path, "adapter_model.safetensors")
                
            print(f"[IO] Saving single safetensors to {file_path}")
            save_file(self.current_shard, file_path, metadata=self.metadata)
        elif self.current_shard:
            self._flush_shard()
            
        print(f"[SUCCESS] Safetensors surgery complete.")
        self.writer_path = None
        self.current_shard = {}
