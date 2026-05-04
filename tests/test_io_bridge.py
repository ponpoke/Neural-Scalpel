import unittest
import torch
import numpy as np
import os
from neural_scalpel.io.gguf_bridge import GGUFBridge
from gguf import GGUFWriter, GGMLQuantizationType

class TestIOBridge(unittest.TestCase):
    def setUp(self):
        self.temp_gguf = "test_model.gguf"
        self.bridge = GGUFBridge()

    def tearDown(self):
        if os.path.exists(self.temp_gguf):
            os.remove(self.temp_gguf)

    def test_gguf_loading_and_thawing(self):
        # 1. Create a mock GGUF file with F16 data
        writer = GGUFWriter(self.temp_gguf, "llama")
        
        # Original data (2, 4)
        original_data = np.random.randn(2, 4).astype(np.float32)
        writer.add_tensor("test_tensor_f32", original_data)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        
        # 2. Load via bridge
        state_dict = self.bridge.load_weights(self.temp_gguf)
        
        # 3. Verify
        self.assertIn("test_tensor_f32", state_dict)
        loaded_tensor = state_dict["test_tensor_f32"]
        torch.testing.assert_close(loaded_tensor, torch.from_numpy(original_data), atol=1e-5, rtol=1e-5)

    def test_q8_0_dequantization_logic(self):
        # We simulate the Q8_0 block structure manually
        delta = 0.5
        weights = np.arange(-16, 16, dtype=np.int8) 
        
        delta_bytes = np.array([delta], dtype=np.float16).view(np.uint8).tobytes()
        block_bytes = delta_bytes + weights.tobytes()
        
        data = np.frombuffer(block_bytes, dtype=np.uint8)
        
        # Run new vectorized dequantization
        dequantized = self.bridge._dequantize_torch(data, GGMLQuantizationType.Q8_0, (1, 32))
        
        expected = torch.from_numpy(weights.astype(np.float32) * delta).to(torch.float16)
        torch.testing.assert_close(dequantized.flatten(), expected, atol=1e-3, rtol=1e-3)

    def test_gguf_full_cycle(self):
        # 1. Save data to GGUF (will be quantized to Q8_0)
        # Use 32 multiple to avoid padding issues in basic test
        original_data = torch.randn(1, 64).to(torch.float16)
        state_dict = {"layer.weight": original_data}
        self.bridge.save_weights(self.temp_gguf, state_dict)
        
        # 2. Load back
        loaded_dict = self.bridge.load_weights(self.temp_gguf)
        
        # 3. Verify
        self.assertIn("layer.weight", loaded_dict)
        loaded_data = loaded_dict["layer.weight"].reshape(original_data.shape)
        error = torch.abs(original_data - loaded_data).mean()
        self.assertLess(error, 0.05) 

if __name__ == '__main__':
    unittest.main()
