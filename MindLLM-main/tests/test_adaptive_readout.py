import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import numpy as np
import torch

from src.modules.neuro_informed_attn import NeuroscienceInformedAttention, NeuroscienceInformedAttentionLayer
from src.modules.adaptive_neuro_informed_attn import AdaptiveNeuroscienceInformedAttention, AdaptiveReadoutAttention
from train_readout import audit_checkpoint, prepare_config, preflight


class FakeNifti:
    def get_fdata(self):
        return np.ones((2, 2, 2))


class AdaptiveReadoutTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def layer(self, mode="adaptive", **kwargs):
        query = torch.nn.Parameter(torch.randn(1, 8, 4))
        return AdaptiveReadoutAttention(query, mode=mode, adapter_rank=3,
                                        query_chunk_size=3, **kwargs)

    def test_initial_output_and_baseline_gradients_are_preserved(self):
        for mode in ("adaptive", "mlp_control"):
            layer = self.layer(mode)
            original = NeuroscienceInformedAttentionLayer(8, 4)
            original.query_embeddings.data.copy_(layer.query_embeddings)
            values, keys = torch.randn(2, 7, requires_grad=True), torch.randn(2, 7, 4, requires_grad=True)
            vref, kref = values.detach().clone().requires_grad_(), keys.detach().clone().requires_grad_()
            actual, expected = layer(values, keys), original(vref, kref)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(values.grad, vref.grad)
            torch.testing.assert_close(keys.grad, kref.grad)
            torch.testing.assert_close(layer.query_embeddings.grad, original.query_embeddings.grad)
            active = layer.context_projection.weight if mode == "adaptive" else layer.mlp_up.weight
            self.assertGreater(active.grad.abs().sum().item(), 0)

    def test_second_pass_matches_dense_reference_and_gradients(self):
        layer = self.layer()
        torch.nn.init.normal_(layer.context_projection.weight, std=0.2)
        reference = copy.deepcopy(layer)
        values, keys = torch.randn(2, 7, requires_grad=True), torch.randn(2, 7, 4, requires_grad=True)
        vr, kr = values.detach().clone().requires_grad_(), keys.detach().clone().requires_grad_()
        actual = layer(values, keys)
        first = NeuroscienceInformedAttentionLayer.forward(reference, vr, kr)
        c = torch.tanh(reference.context_projection(reference.context_norm(first)))
        dq = (c[:, None, :] * reference.query_factors) @ reference.key_factors
        logits = reference.query_embeddings @ kr.transpose(1, 2)
        second = ((logits + dq @ kr.transpose(1, 2)).softmax(-1) @ vr.unsqueeze(-1)).squeeze(-1)
        expected = first + reference.gate.tanh() * (second - first)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(values.grad, vr.grad)
        torch.testing.assert_close(keys.grad, kr.grad)
        for name, parameter in layer.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad,
                                       atol=2e-6, rtol=2e-4)

    def test_new_branch_learns_and_queries_are_sample_dependent(self):
        layer = self.layer()
        values, keys = torch.randn(2, 7), torch.randn(1, 7, 4).expand(2, -1, -1)
        original = layer(values, keys).detach()
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.2)
        for _ in range(3):
            optimizer.zero_grad()
            # Train ONLY the new branch to make this check independent of Q0 learning.
            layer.query_embeddings.requires_grad_(False)
            (layer(values, keys) - torch.randn(2, 8)).square().mean().backward()
            optimizer.step()
        first = NeuroscienceInformedAttentionLayer.forward(layer, values, keys)
        c = torch.tanh(layer.context_projection(layer.context_norm(first)))
        self.assertFalse(torch.allclose(c[0], c[1]))
        self.assertFalse(torch.allclose(layer(values, keys), original))
        self.assertGreater(layer.query_factors.grad.abs().sum().item(), 0)
        self.assertGreater(layer.key_factors.grad.abs().sum().item(), 0)

    def test_checkpointing_and_autocast_are_finite(self):
        for mode in ("adaptive", "mlp_control"):
            layer = self.layer(mode)
            values, keys = torch.randn(2, 7, requires_grad=True), torch.randn(2, 7, 4, requires_grad=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                result = layer(values, keys)
                baseline = NeuroscienceInformedAttentionLayer.forward(layer, values, keys)
                torch.testing.assert_close(result, baseline, rtol=0, atol=0, check_dtype=False)
                result.square().mean().backward()
            self.assertTrue(torch.isfinite(result).all())
            for parameter in layer.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        eager = copy.deepcopy(layer)
        eager.checkpoint_second_pass = False
        torch.testing.assert_close(layer(values, keys), eager(values, keys))

    def test_full_encoder_transfer_rng_and_roundtrip(self):
        kwargs = dict(hidden_dim=8, rank=4, token_dim=4, n_fmri_tokens=2, n_mlp_layers=1)
        with patch("src.modules.neuro_informed_attn.nib.load", return_value=FakeNifti()):
            for mode in ("adaptive", "mlp_control"):
                torch.manual_seed(42)
                baseline = NeuroscienceInformedAttention(**kwargs).eval()
                rng = torch.get_rng_state().clone()
                torch.manual_seed(42)
                adaptive = AdaptiveNeuroscienceInformedAttention(mode=mode, adapter_rank=3, **kwargs).eval()
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                loaded = adaptive.load_state_dict(baseline.state_dict(), strict=False)
                self.assertEqual(loaded.unexpected_keys, [])
                self.assertTrue(all(k.startswith("neuro_informed_attn.") for k in loaded.missing_keys))
                for key, tensor in baseline.state_dict().items():
                    torch.testing.assert_close(tensor, adaptive.state_dict()[key], rtol=0, atol=0)
                values, subjects = torch.randn(2, 8), ["subject_1"] * 2
                expected = baseline(values, subject=subjects)
                actual = adaptive(values, subject=subjects)
                self.assertEqual(actual.shape, (2, 2, 4))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                restored = AdaptiveNeuroscienceInformedAttention(mode=mode, adapter_rank=3, **kwargs).eval()
                restored.load_state_dict(adaptive.state_dict(), strict=True)
                torch.testing.assert_close(restored(values.reshape(2, 2, 2, 2), subject=subjects), actual)

    def test_wrong_mode_rejected_even_with_non_strict_loading(self):
        with self.assertRaisesRegex(RuntimeError, "different readout mode"):
            self.layer("mlp_control").load_state_dict(self.layer().state_dict(), strict=False)
        for bad in (0, 1, -1, float("nan")):
            with self.assertRaises(ValueError):
                self.layer(gate_init=bad)


class ReadoutConfigTest(unittest.TestCase):
    def config(self, arm, path):
        root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root / "config"), version_base=None):
            cfg = compose(config_name="readout_" + arm)
        cfg.checkpoint = str(path)
        return cfg

    def test_three_configs_and_checkpoint_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.ckpt"
            state = {"fmri_encoder.neuro_informed_attn.query_embeddings": torch.zeros(1, 8, 4),
                     "projector.weight": torch.zeros(4, 4)}
            torch.save({"state_dict": state}, path)
            for arm in ("baseline", "adaptive", "mlp_control"):
                cfg = prepare_config(self.config(arm, path))
                self.assertEqual(cfg.readout_arm, arm)
                self.assertEqual(set(cfg.trainer.logger), {"_target_", "save_dir", "name"})
                self.assertEqual(cfg.data.split_seed, 42)
                self.assertFalse(cfg.lora)
                self.assertEqual(cfg.trainer.max_epochs, 20)
                audit_checkpoint(path, arm, allow_baseline=True)
            with self.assertRaisesRegex(ValueError, "does not match"):
                audit_checkpoint(path, "adaptive", allow_baseline=False)
            state["fmri_encoder.neuro_informed_attn.cluster_ids_subject_1"] = torch.zeros(8)
            torch.save({"state_dict": state}, path)
            with self.assertRaisesRegex(ValueError, "another attention experiment"):
                audit_checkpoint(path, "adaptive", allow_baseline=True)

    def test_existing_experiments_cannot_be_overwritten_or_silently_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.ckpt"
            torch.save({"state_dict": {"fmri_encoder.neuro_informed_attn.query_embeddings": torch.zeros(1, 8, 4),
                                      "projector.weight": torch.zeros(4, 4)}}, path)
            cfg = self.config("baseline", path)
            cfg.output_dir = str(Path(directory) / "run")
            preflight(cfg)
            with self.assertRaisesRegex(ValueError, "not empty"):
                preflight(cfg)
            cfg.resume = True
            with self.assertRaisesRegex(ValueError, "manifest and last"):
                preflight(cfg)


if __name__ == "__main__":
    unittest.main()
