import unittest
import json
import os
from scratch.generate_v211_recommendation import generate_safety_map

class TestSafetyMapGenerator(unittest.TestCase):
    def setUp(self):
        self.test_dataset_file = "scratch/test_risk_dataset.json"
        self.test_output_map = "scratch/test_safety_map.json"
        
        # Create a mock dataset for testing
        self.mock_dataset = [
            {"setting": "baseline", "alpha_map": {}, "outcome": {"accuracy": 0.24, "verdict": "BASELINE"}},
            # Attention Sweep
            {"setting": "attn_0.5", "alpha_map": {"q_proj": 0.5, "gate_proj": 0.0}, "outcome": {"accuracy": 0.24, "verdict": "SAFE"}},
            {"setting": "attn_1.0", "alpha_map": {"q_proj": 1.0, "gate_proj": 0.0}, "outcome": {"accuracy": 0.24, "verdict": "SAFE"}},
            {"setting": "attn_2.0", "alpha_map": {"q_proj": 2.0, "gate_proj": 0.0}, "outcome": {"accuracy": 0.24, "verdict": "SAFE"}},
            {"setting": "attn_3.0", "alpha_map": {"q_proj": 3.0, "gate_proj": 0.0}, "outcome": {"accuracy": 0.22, "verdict": "UNSAFE"}},
            {"setting": "attn_4.0", "alpha_map": {"q_proj": 4.0, "gate_proj": 0.0}, "outcome": {"accuracy": 0.24, "verdict": "SAFE"}},
            # MLP boundary
            {"setting": "mlp_0.06", "alpha_map": {"q_proj": 4.0, "gate_proj": 0.06}, "outcome": {"accuracy": 0.24, "verdict": "BOUNDARY"}}
        ]
        
        with open(self.test_dataset_file, "w") as f:
            json.dump(self.mock_dataset, f)

    def tearDown(self):
        if os.path.exists(self.test_dataset_file):
            os.remove(self.test_dataset_file)
        if os.path.exists(self.test_output_map):
            os.remove(self.test_output_map)

    def test_avoid_band_detection(self):
        # We need to temporarily patch the script's constants
        import scratch.generate_v211_recommendation as gen
        original_dataset = gen.DATASET_FILE
        original_output = gen.OUTPUT_MAP
        
        gen.DATASET_FILE = self.test_dataset_file
        gen.OUTPUT_MAP = self.test_output_map
        
        try:
            gen.generate_safety_map()
            
            with open(self.test_output_map, "r") as f:
                safety_map = json.load(f)
                
            # Check Attention Map
            self.assertEqual(safety_map["attention"]["safe_alphas"], [0.5, 1.0, 2.0, 4.0])
            self.assertEqual(safety_map["attention"]["unsafe_alphas"], [3.0])
            # Should have an avoid band around 3.0
            self.assertEqual(len(safety_map["attention"]["avoid_bands"]), 1)
            self.assertEqual(safety_map["attention"]["avoid_bands"][0], [2.75, 3.25])
            
            # Check MLP Policy
            self.assertEqual(safety_map["mlp"]["recommended_alpha"], 0.0)
            self.assertEqual(safety_map["mlp"]["boundary_alphas"], [0.06])
            
        finally:
            gen.DATASET_FILE = original_dataset
            gen.OUTPUT_MAP = original_output

if __name__ == "__main__":
    unittest.main()
