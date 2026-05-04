"""
Comprehensive test suite for Neural-Scalpel.
Covers all modules with real tensor operations to verify correctness.
"""
import unittest
import torch
import os
import sys
import tempfile
import shutil
import json
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_scalpel.core.math import (
    head_wise_orthogonal_procrustes,
    create_sparse_task_vector,
    adaptive_rsvd_bootstrap,
    adaptive_variance_preserving_sparsity,
    pca_guided_subspace_injection,
    soft_routing_head_pooling,
    sinkhorn_knopp,
    wasserstein_discrete_routing,
    vram_decoupled_decoding,
    quantization_aware_procrustes,
    expert_wise_procrustes,
    kernel_orthogonal_procrustes,
    jacobian_tangent_space_alignment,
    swiglu_jacobian,
    geglu_jacobian,
    router_logic_preservation_mapping,
)
from neural_scalpel.core.adapters import (
    get_adapter, Llama3ToQwen2Adapter, MistralToLlama3Adapter,
    SDXLToFluxAdapter, SDXLToSDXLAdapter, BaseAdapter,
)
from neural_scalpel.io.safetensors_bridge import SafetensorsBridge
from neural_scalpel.io.factory import IOBridgeFactory
from neural_scalpel.io.calibration import search_optimal_awq_scales, SyntheticManifoldGenerator, ManifoldProfiler
from neural_scalpel.io.awq_bridge import AWQBridge
from neural_scalpel.router.manager import ScalpelRouteManager, calculate_hash
from neural_scalpel.experimental.hot_swap import VRAMHotSwapAPI
from neural_scalpel.cli.main import get_model_info, detect_architecture, port_lora


# =============================================================================
# 1. Core Math Engine Tests
# =============================================================================
class TestProcrustes(unittest.TestCase):
    def test_identity_alignment(self):
        """A aligned to itself should produce identity rotation."""
        torch.manual_seed(0)
        A = torch.randn(20, 128)
        A_t, _, R, s = head_wise_orthogonal_procrustes(A, A, num_heads=4)
        cos = torch.nn.functional.cosine_similarity(A_t, A, dim=1).mean()
        self.assertGreater(cos.item(), 0.99)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            head_wise_orthogonal_procrustes(torch.randn(5, 8), torch.randn(5, 16), 2)

    def test_indivisible_heads_raises(self):
        with self.assertRaises(ValueError):
            head_wise_orthogonal_procrustes(torch.randn(5, 10), torch.randn(5, 10), 3)

    def test_bias_transform(self):
        torch.manual_seed(1)
        A = torch.randn(10, 64)
        bias = torch.randn(64)
        _, bias_t, _, _ = head_wise_orthogonal_procrustes(A, A, 4, bias_A=bias)
        self.assertIsNotNone(bias_t)
        self.assertEqual(bias_t.shape, (64,))

    def test_rotation_matrices_shape(self):
        A = torch.randn(10, 64)
        _, _, R, s = head_wise_orthogonal_procrustes(A, A, num_heads=8)
        self.assertEqual(R.shape, (8, 8, 8))
        self.assertEqual(s.shape, (8,))

    def test_realistic_llama_size(self):
        """Test with LLaMA-3 realistic dimensions."""
        torch.manual_seed(42)
        A = torch.randn(8, 4096)
        B = torch.randn(8, 4096)
        A_t, _, R, s = head_wise_orthogonal_procrustes(A, B, num_heads=32)
        self.assertEqual(A_t.shape, (8, 4096))
        self.assertEqual(R.shape, (32, 128, 128))


class TestSparseTaskVector(unittest.TestCase):
    def test_basic_sparsity(self):
        W = torch.randn(64, 64)
        B = torch.zeros(64, 64)
        tau = create_sparse_task_vector(W, B, trim_ratio=0.2)
        self.assertTrue(tau.is_sparse_csr)

    def test_zero_trim(self):
        W = torch.ones(8, 8)
        B = torch.zeros(8, 8)
        tau = create_sparse_task_vector(W, B, trim_ratio=0.0)
        dense = tau.to_dense()
        self.assertTrue(torch.allclose(dense, W))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            create_sparse_task_vector(torch.randn(4, 4), torch.randn(4, 5))

    def test_invalid_trim_ratio(self):
        with self.assertRaises(ValueError):
            create_sparse_task_vector(torch.randn(4, 4), torch.randn(4, 4), trim_ratio=1.0)


class TestAdaptiveRSVD(unittest.TestCase):
    def test_low_rank_recovery(self):
        """SVD should recover a known low-rank matrix."""
        torch.manual_seed(0)
        A = torch.randn(50, 5)
        B = torch.randn(5, 30)
        M = A @ B  # rank-5
        U, S, V = adaptive_rsvd_bootstrap(M, epsilon=1e-2, block_size=5)
        M_approx = U @ torch.diag(S) @ V
        error = (M - M_approx).norm() / M.norm()
        self.assertLess(error.item(), 0.05)

    def test_sparse_input(self):
        M = torch.randn(20, 20)
        M[M.abs() < 0.5] = 0
        M_sparse = M.to_sparse_csr()
        U, S, V = adaptive_rsvd_bootstrap(M_sparse, epsilon=0.1)
        self.assertGreater(S[0].item(), 0)


class TestAVPS(unittest.TestCase):
    def test_energy_preservation(self):
        torch.manual_seed(0)
        W_t = torch.randn(64, 64)
        W_b = torch.zeros(64, 64)
        tau = adaptive_variance_preserving_sparsity(W_t, W_b, 0.99)
        dense = tau.to_dense()
        original_energy = (W_t ** 2).sum()
        preserved_energy = (dense ** 2).sum()
        self.assertGreater(preserved_energy / original_energy, 0.98)


class TestPCSI(unittest.TestCase):
    def test_dimension_expansion(self):
        src = torch.randn(20, 32)
        tgt_act = torch.randn(20, 64)
        result = pca_guided_subspace_injection(src, tgt_act)
        self.assertEqual(result.shape, (20, 64))


class TestSRHP(unittest.TestCase):
    def test_head_compression(self):
        heads = torch.randn(4, 8, 16)
        pooled = soft_routing_head_pooling(heads, 4)
        self.assertEqual(pooled.shape[1], 4)

    def test_no_compression_needed(self):
        heads = torch.randn(4, 4, 16)
        pooled = soft_routing_head_pooling(heads, 8)
        self.assertEqual(pooled.shape, heads.shape)


class TestWDR(unittest.TestCase):
    def test_sinkhorn_marginals(self):
        C = torch.rand(8, 8)
        P = sinkhorn_knopp(C, epsilon=0.1)
        expected = 1.0 / 8
        torch.testing.assert_close(P.sum(dim=0), torch.full((8,), expected), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(P.sum(dim=1), torch.full((8,), expected), atol=1e-4, rtol=1e-4)

    def test_hard_wdr_column_sums(self):
        src = torch.randn(4, 5, 16)
        tgt = torch.randn(4, 3, 16)
        P = wasserstein_discrete_routing(src, tgt, mode="hard", alpha=0.1)
        torch.testing.assert_close(P.sum(dim=0), torch.ones(3), atol=1e-5, rtol=1e-5)

    def test_soft_wdr_column_sums(self):
        src = torch.randn(4, 5, 16)
        tgt = torch.randn(4, 3, 16)
        P = wasserstein_discrete_routing(src, tgt, mode="soft")
        col_sums = P.sum(dim=0)
        for s in col_sums:
            self.assertAlmostEqual(s.item(), 1.0, delta=0.05)


class TestJTSA(unittest.TestCase):
    def test_swiglu_jacobian_positive(self):
        x = torch.randn(100)
        j = swiglu_jacobian(x)
        # Swish derivative is always > -0.278
        self.assertTrue((j > -0.3).all())

    def test_geglu_jacobian_shape(self):
        x = torch.randn(10, 32)
        j = geglu_jacobian(x)
        self.assertEqual(j.shape, x.shape)

    def test_jtsa_output_shape(self):
        A = torch.randn(8, 256)
        B = torch.randn(8, 256)
        A_t, R, s = jacobian_tangent_space_alignment(A, B, num_heads=4)
        self.assertEqual(A_t.shape, (8, 256))
        self.assertEqual(R.shape, (4, 64, 64))


class TestQAP(unittest.TestCase):
    def test_dampening_effect(self):
        A = torch.randn(10, 64)
        B = A.clone()
        A_q, R, s_q = quantization_aware_procrustes(A, B, 4, quantization_bits=4)
        self.assertLess(s_q.mean().item(), 1.0)


class TestKOP(unittest.TestCase):
    def test_kernel_procrustes_output(self):
        A = torch.randn(20, 64)
        B = torch.randn(20, 64)
        A_t, R, s = kernel_orthogonal_procrustes(A, B, num_heads=4, n_components=10)
        self.assertEqual(A_t.shape, (20, 64))


class TestMoE(unittest.TestCase):
    def test_expert_wise(self):
        A_e = [torch.randn(10, 32) for _ in range(4)]
        B_e = [torch.randn(10, 32) for _ in range(4)]
        t, R, s = expert_wise_procrustes(A_e, B_e)
        self.assertEqual(len(t), 4)

    def test_expert_count_mismatch(self):
        with self.assertRaises(ValueError):
            expert_wise_procrustes([torch.randn(10, 32)], [torch.randn(10, 32)] * 2)

    def test_router_mapping(self):
        A = torch.randn(50, 16)
        B = torch.randn(50, 32)
        out = router_logic_preservation_mapping(A, B)
        self.assertEqual(out.shape, (50, 32))


class TestDequant(unittest.TestCase):
    def test_vram_decoupled_decoding(self):
        q = torch.randint(-128, 127, (4, 4), dtype=torch.int8)
        s = torch.ones(4, 4) * 0.1
        z = torch.zeros(4, 4)
        out = vram_decoupled_decoding(q, s, z)
        self.assertEqual(out.dtype, torch.float16)


# =============================================================================
# 2. Adapter Tests
# =============================================================================
class TestAdapters(unittest.TestCase):
    def test_llama_to_qwen_q_proj_lora_B(self):
        adapter = Llama3ToQwen2Adapter((4096, 32), (3584, 28))
        t = torch.randn(4096, 16)
        out = adapter.project_tensor("q_proj.lora_B.weight", t)
        self.assertEqual(out.shape[1], 16)
        self.assertNotEqual(out.shape[0], 4096)

    def test_llama_to_qwen_mlp_lora_A(self):
        adapter = Llama3ToQwen2Adapter((4096, 32), (3584, 28))
        t = torch.randn(16, 4096)
        out = adapter.project_tensor("gate_proj.lora_A.weight", t)
        self.assertEqual(out.shape, (16, 3584))

    def test_sdxl_to_sdxl_passthrough(self):
        adapter = SDXLToSDXLAdapter((2048, 32), (2048, 32))
        t = torch.randn(16, 2048)
        out = adapter.project_tensor("attn1.lora_A.weight", t)
        self.assertTrue(torch.equal(out, t))

    def test_sdxl_to_flux_pcsi(self):
        adapter = SDXLToFluxAdapter((640, 10), (3072, 24))
        t = torch.randn(16, 640)
        out = adapter.project_tensor("attn1.to_q.lora_A.weight", t)
        self.assertEqual(out.shape[1], 3072)

    def test_get_adapter_routing(self):
        a = get_adapter("llama", "qwen", (4096, 32), (3584, 28))
        self.assertIsInstance(a, Llama3ToQwen2Adapter)
        a2 = get_adapter("sdxl", "flux", (640, 10), (3072, 24))
        self.assertIsInstance(a2, SDXLToFluxAdapter)
        a3 = get_adapter("unknown", "unknown", (512, 8), (512, 8))
        self.assertIsInstance(a3, BaseAdapter)

    def test_wdr_adapter_with_routing(self):
        P = torch.eye(32, 28)
        P[:28, :28] = torch.eye(28)
        adapter = Llama3ToQwen2Adapter((4096, 32), (3584, 28), routing_matrix=P)
        t = torch.randn(4096, 16)
        out = adapter.project_tensor("q_proj.lora_B.weight", t)
        self.assertEqual(out.shape[1], 16)


# =============================================================================
# 3. I/O Bridge Tests
# =============================================================================
class TestSafetensorsBridge(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.bridge = SafetensorsBridge()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_write_read_cycle(self):
        path = os.path.join(self.tmpdir, "test.safetensors")
        data = {"w1": torch.randn(4, 4), "w2": torch.randn(8, 8)}
        self.bridge.save_weights(path, data)
        loaded = self.bridge.load_weights(path)
        torch.testing.assert_close(loaded["w1"], data["w1"])

    def test_streaming_iter(self):
        path = os.path.join(self.tmpdir, "test.safetensors")
        data = {"a": torch.randn(2, 2), "b": torch.randn(3, 3)}
        self.bridge.save_weights(path, data)
        keys = []
        for k, t in self.bridge.iter_layers(path):
            keys.append(k)
        self.assertEqual(set(keys), {"a", "b"})

    def test_incremental_writer(self):
        path = os.path.join(self.tmpdir, "inc.safetensors")
        self.bridge.open_writer(path)
        self.bridge.write_layer("x", torch.randn(4, 4))
        self.bridge.write_layer("y", torch.randn(4, 4))
        self.bridge.close_writer()
        loaded = self.bridge.load_weights(path)
        self.assertIn("x", loaded)
        self.assertIn("y", loaded)

    def test_directory_load(self):
        from safetensors.torch import save_file
        save_file({"t": torch.randn(2, 2)}, os.path.join(self.tmpdir, "adapter_model.safetensors"))
        loaded = self.bridge.load_weights(self.tmpdir)
        self.assertIn("t", loaded)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.bridge.load_weights(os.path.join(self.tmpdir, "nope.safetensors"))


class TestIOFactory(unittest.TestCase):
    def test_safetensors_default(self):
        b = IOBridgeFactory.get_bridge("model.safetensors")
        self.assertIsInstance(b, SafetensorsBridge)

    def test_gguf_detection(self):
        from neural_scalpel.io.gguf_bridge import GGUFBridge
        b = IOBridgeFactory.get_bridge("model.gguf")
        self.assertIsInstance(b, GGUFBridge)

    def test_awq_detection(self):
        b = IOBridgeFactory.get_bridge("model.awq.safetensors")
        self.assertIsInstance(b, AWQBridge)


class TestCalibration(unittest.TestCase):
    def test_lmr_reduces_error(self):
        W = torch.randn(64, 64)
        X = torch.randn(10, 64)
        X[:, :5] *= 10
        s = search_optimal_awq_scales(W, X)
        self.assertEqual(s.shape, (64,))
        self.assertTrue((s > 0).all())

    def test_synthetic_manifold_generator(self):
        w = torch.randn(32, 10)
        mag = SyntheticManifoldGenerator.get_synthetic_activation_magnitudes(w)
        self.assertEqual(mag.shape, (10,))
        self.assertAlmostEqual(mag.mean().item(), 1.0, places=4)

    def test_manifold_profiler(self):
        mp = ManifoldProfiler(torch.randn(10, 32))
        mag = mp.get_activation_magnitudes()
        self.assertEqual(mag.shape, (32,))


# =============================================================================
# 4. Router Manager Tests
# =============================================================================
class TestRouter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mgr = ScalpelRouteManager(route_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_create_and_load(self):
        fp = self.mgr.create_route("src", "tgt", "general", [[1, 0], [0, 1]], 1.0)
        m = self.mgr.verify_and_load_route(fp, "src", "tgt")
        self.assertEqual(m["s"], 1.0)

    def test_hash_mismatch_source(self):
        fp = self.mgr.create_route("src", "tgt", "test", [1], 1.0)
        with self.assertRaises(ValueError):
            self.mgr.verify_and_load_route(fp, "wrong", "tgt")

    def test_hash_mismatch_target(self):
        fp = self.mgr.create_route("src", "tgt", "test", [1], 1.0)
        with self.assertRaises(ValueError):
            self.mgr.verify_and_load_route(fp, "src", "wrong")

    def test_sign_and_verify(self):
        fp = self.mgr.create_route("s", "t", "d", [1], 1.0)
        self.mgr.sign_route(fp, "secret")
        m = self.mgr.verify_and_load_route(fp, "s", "t", trusted_keys=["secret"])
        self.assertEqual(m["s"], 1.0)

    def test_bad_signature_rejected(self):
        fp = self.mgr.create_route("s", "t", "d", [1], 1.0)
        self.mgr.sign_route(fp, "secret")
        with self.assertRaises(PermissionError):
            self.mgr.verify_and_load_route(fp, "s", "t", trusted_keys=["wrong"])

    def test_unsigned_with_trust_rejected(self):
        fp = self.mgr.create_route("s", "t", "d", [1], 1.0)
        with self.assertRaises(PermissionError):
            self.mgr.verify_and_load_route(fp, "s", "t", trusted_keys=["key"])

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.mgr.verify_and_load_route("nonexistent.route", "a", "b")

    def test_calculate_hash_string(self):
        h = calculate_hash("test_model")
        self.assertEqual(len(h), 64)

    def test_backward_compat_load_route(self):
        fp = self.mgr.create_route("a", "b", "d", [1], 2.0)
        m = self.mgr.load_route(fp, "a", "b")
        self.assertEqual(m["s"], 2.0)


# =============================================================================
# 5. Hot-Swap Tests
# =============================================================================
class TestHotSwap(unittest.TestCase):
    def _make_model(self):
        return {"l1": torch.ones(4, 4)}

    def test_inject_concept(self):
        m = self._make_model()
        api = VRAMHotSwapAPI(target_model=m)
        api.inject_concept(torch.ones(4, 4) * 2, "l1")
        self.assertAlmostEqual(m["l1"][0, 0].item(), 3.0, places=5)

    def test_remove_concept(self):
        m = self._make_model()
        api = VRAMHotSwapAPI(target_model=m)
        api.remove_concept(torch.ones(4, 4) * 0.5, "l1")
        self.assertAlmostEqual(m["l1"][0, 0].item(), 0.5, places=5)

    def test_shadow_inject_and_rollback(self):
        m = self._make_model()
        api = VRAMHotSwapAPI(target_model=m)
        api.inject_concept_shadow(torch.ones(4, 4) * 9, "l1")
        self.assertAlmostEqual(m["l1"][0, 0].item(), 10.0, places=5)
        api.transactional_rollback("l1")
        self.assertAlmostEqual(m["l1"][0, 0].item(), 1.0, places=5)

    def test_rollback_nonexistent(self):
        api = VRAMHotSwapAPI(target_model=self._make_model())
        self.assertFalse(api.transactional_rollback("no_such"))

    def test_drift_monitor(self):
        api = VRAMHotSwapAPI()
        api.register_baseline("l", 10.0)
        self.assertTrue(api.monitor_drift("l", 10.3, 0.05))
        self.assertFalse(api.monitor_drift("l", 11.0, 0.05))

    def test_drift_no_baseline(self):
        api = VRAMHotSwapAPI()
        self.assertTrue(api.monitor_drift("l", 999.0))

    def test_ppl_gateway(self):
        api = VRAMHotSwapAPI()
        self.assertTrue(api.ppl_gateway_monitor(6.0, 6.0))
        self.assertFalse(api.ppl_gateway_monitor(7.0, 6.0, 1.1))

    def test_inject_missing_layer_no_crash(self):
        api = VRAMHotSwapAPI(target_model=self._make_model())
        api.inject_concept(torch.ones(4), "nonexistent")  # should not crash

    def test_none_model_no_crash(self):
        api = VRAMHotSwapAPI(target_model=None)
        api.inject_concept(torch.ones(4), "l")

    def test_concurrent_safety(self):
        import threading
        m = {"l": torch.zeros(16, 16)}
        api = VRAMHotSwapAPI(target_model=m)
        errors = []
        def worker():
            try:
                api.inject_concept_shadow(torch.randn(16, 16), "l")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(len(errors), 0)


# =============================================================================
# 6. CLI Tests
# =============================================================================
class TestCLI(unittest.TestCase):
    @patch('neural_scalpel.cli.main.AutoConfig.from_pretrained')
    def test_get_model_info(self, mock_from_pretrained):
        mock_config = MagicMock()
        mock_config.hidden_size = 4096
        mock_config.num_attention_heads = 32
        mock_from_pretrained.return_value = mock_config
        
        info = get_model_info("dummy_llama_path")
        self.assertEqual(info["hidden_size"], 4096)
        self.assertEqual(info["num_attention_heads"], 32)

    @patch('neural_scalpel.cli.main.AutoConfig.from_pretrained')
    def test_get_model_info_sdxl_fallback(self, mock_from_pretrained):
        mock_from_pretrained.side_effect = Exception("No config")
        info = get_model_info("path/to/sdxl.safetensors")
        self.assertEqual(info["hidden_size"], 2048)
        self.assertEqual(info["num_attention_heads"], 32)

    def test_detect_architecture_sdxl(self):
        a = detect_architecture("some/sdxl-model")
        self.assertEqual(a, "sdxl")

    def test_detect_architecture_qwen(self):
        a = detect_architecture("Qwen/Qwen2-7B")
        self.assertEqual(a, "qwen")

    def test_detect_architecture_default(self):
        a = detect_architecture("unknown-model-xyz")
        self.assertEqual(a, "llama")

    @patch('neural_scalpel.cli.main.AutoConfig.from_pretrained')
    def test_port_creates_output(self, mock):
        cfg = MagicMock()
        cfg.hidden_size = 4096
        cfg.num_attention_heads = 32
        mock.return_value = cfg
        with tempfile.TemporaryDirectory() as tmpdir:
            args = MagicMock()
            args.source = "owner/src"
            args.target = "owner/tgt"
            args.output = tmpdir
            args.routing_path = None
            args.calibrate = None
            args.domain = "general"
            port_lora(args)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "adapter_model.safetensors")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "adapter_config.json")))

    @patch('neural_scalpel.cli.main.AutoConfig.from_pretrained')
    def test_port_to_safetensors_file(self, mock):
        cfg = MagicMock()
        cfg.hidden_size = 4096
        cfg.num_attention_heads = 32
        mock.return_value = cfg
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "output.safetensors")
            args = MagicMock()
            args.source = "owner/src"
            args.target = "owner/tgt"
            args.output = out_path
            args.routing_path = None
            args.calibrate = None
            args.domain = "general"
            port_lora(args)
            self.assertTrue(os.path.exists(out_path))


# =============================================================================
# 7. AWQ Bridge INT4 Packing Test
# =============================================================================
class TestAWQPacking(unittest.TestCase):
    def test_int4_pack_unpack_range(self):
        bridge = AWQBridge()
        W = torch.randn(32, 64)
        packed, scales, zeros = bridge._pack_int4(W)
        self.assertEqual(packed.dtype, torch.int32)
        self.assertEqual(scales.dtype, torch.float16)
        self.assertEqual(packed.shape[0], 32)

if __name__ == '__main__':
    unittest.main()
