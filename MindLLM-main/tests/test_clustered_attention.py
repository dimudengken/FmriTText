from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.modules.clustered_neuro_informed_attn import (
    ClusteredNeuroscienceInformedAttention,
    ClusterRestrictedAttention,
)
from src.modules.neuro_informed_attn import (
    NeuroscienceInformedAttention,
    NeuroscienceInformedAttentionLayer,
)


class FakeNifti:
    def get_fdata(self):
        return np.ones((2, 2, 2))


def bank(labels, n_clusters):
    return {"version": 1, "n_clusters": n_clusters, "method": "encoding_kmeans",
            "subjects": {"subject_1": {"cluster_ids": torch.tensor(labels, dtype=torch.long),
                "flat_indices": torch.arange(8), "mask_shape": [2, 2, 2]}}}


class ClusteredAttentionTest(unittest.TestCase):
    def test_cluster_isolation_uniform_pooling_and_gradients(self):
        query = torch.nn.Parameter(torch.zeros(1, 4, 3))
        labels = torch.tensor([0, 0, 0, 1, 1])
        layer = ClusterRestrictedAttention(query, {"subject_1": labels}, 2)
        voxels = torch.tensor([[1., 2., 3., 8., 10.]], requires_grad=True)
        keys = torch.randn(1, 5, 3, requires_grad=True)
        output = layer(voxels, keys, "subject_1")
        torch.testing.assert_close(output, torch.tensor([[2., 2., 9., 9.]]))
        changed = voxels.detach().clone()
        changed[:, 3:] += 1000
        torch.testing.assert_close(layer(changed, keys, "subject_1")[:, :2], output[:, :2])
        output[:, :2].sum().backward()
        torch.testing.assert_close(voxels.grad[:, 3:], torch.zeros(1, 2))
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertGreater(query.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(keys.grad).all())

    def test_single_cluster_matches_original_attention(self):
        original = NeuroscienceInformedAttentionLayer(size=8, rank=4)
        clustered = ClusterRestrictedAttention(original.query_embeddings,
                                               {"subject_1": torch.zeros(6, dtype=torch.long)}, 1)
        voxels, keys = torch.randn(2, 6), torch.randn(2, 6, 4)
        torch.testing.assert_close(clustered(voxels, keys, "subject_1"), original(voxels, keys))

    def test_segment_softmax_matches_dense_mask_values_and_gradients(self):
        torch.manual_seed(12)
        query = torch.nn.Parameter(torch.randn(1, 6, 3))
        labels = torch.tensor([0, 1, 1, 1, 2])
        layer = ClusterRestrictedAttention(query, {"subject_1": labels}, 3)
        voxels = torch.randn(2, 5, requires_grad=True)
        keys = torch.randn(2, 5, 3, requires_grad=True)
        result = layer(voxels, keys, "subject_1")
        result.square().sum().backward()

        reference_query = query.detach().clone().requires_grad_(True)
        reference_voxels = voxels.detach().clone().requires_grad_(True)
        reference_keys = keys.detach().clone().requires_grad_(True)
        logits = reference_query @ reference_keys.transpose(1, 2)
        query_clusters = torch.arange(3).repeat_interleave(2)
        allowed = query_clusters[:, None] == labels[None, :]
        weights = logits.masked_fill(~allowed, -torch.inf).softmax(-1)
        reference = torch.einsum("bqv,bv->bq", weights, reference_voxels)
        reference.square().sum().backward()
        torch.testing.assert_close(result, reference)
        torch.testing.assert_close(query.grad, reference_query.grad)
        torch.testing.assert_close(voxels.grad, reference_voxels.grad)
        torch.testing.assert_close(keys.grad, reference_keys.grad)

    def test_cpu_bfloat16_autocast_backward_is_finite(self):
        query = torch.nn.Parameter(torch.randn(1, 8, 4))
        layer = ClusterRestrictedAttention(query, {"subject_1": torch.tensor([0, 0, 1, 1, 1])}, 2)
        voxels = torch.randn(2, 5, requires_grad=True)
        keys = torch.randn(2, 5, 4, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = layer(voxels, keys, "subject_1")
            loss = result.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(result).all())
        for value in (query, voxels, keys):
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_absent_clusters_produce_zero_and_finite_gradients(self):
        query = torch.nn.Parameter(torch.randn(1, 6, 2))
        layer = ClusterRestrictedAttention(query, {"subject_1": torch.tensor([0, 0, 2])}, 3)
        voxels = torch.randn(2, 3, requires_grad=True)
        output = layer(voxels, torch.randn(2, 3, 2), "subject_1")
        torch.testing.assert_close(output[:, 2:4], torch.zeros(2, 2))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(voxels.grad).all())
        self.assertTrue(torch.isfinite(query.grad).all())

    def test_invalid_voxel_width_and_mixed_subjects_fail_early(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "src.modules.neuro_informed_attn.nib.load", return_value=FakeNifti()):
            path = Path(directory) / "bank.pt"
            torch.save(bank([0] * 4 + [1] * 4, 2), path)
            encoder = ClusteredNeuroscienceInformedAttention(path, 2, hidden_dim=8,
                rank=4, token_dim=4, n_fmri_tokens=2, n_mlp_layers=1, pe_method="coord")
            with self.assertRaisesRegex(ValueError, "one subject"):
                encoder(torch.randn(2, 8), subject=["subject_1", "subject_2"])
            with self.assertRaisesRegex(ValueError, "voxel width"):
                encoder(torch.randn(2, 7), subject=["subject_1"] * 2)
            with self.assertRaisesRegex(ValueError, "no assignments"):
                encoder(torch.randn(2, 8), subject=["subject_2"] * 2)

    def test_baseline_checkpoint_transfer_and_full_encoder_parity(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "src.modules.neuro_informed_attn.nib.load", return_value=FakeNifti()):
            path = Path(directory) / "bank.pt"
            torch.save(bank([0] * 8, 1), path)
            kwargs = dict(hidden_dim=8, rank=4, token_dim=4, n_fmri_tokens=2, n_mlp_layers=1)
            original = NeuroscienceInformedAttention(**kwargs).eval()
            clustered = ClusteredNeuroscienceInformedAttention(path, 1, **kwargs).eval()
            result = clustered.load_state_dict(original.state_dict(), strict=False)
            self.assertEqual(result.unexpected_keys, [])
            self.assertEqual(len(result.missing_keys), 1)
            self.assertTrue(all(name.startswith("neuro_informed_attn.") for name in result.missing_keys))
            self.assertEqual(set(dict(original.named_parameters())), set(dict(clustered.named_parameters())))
            voxels = torch.randn(2, 8, requires_grad=True)
            subjects = ["subject_1"] * 2
            output = clustered(voxels, subject=subjects)
            self.assertEqual(output.shape, (2, 2, 4))
            torch.testing.assert_close(output, original(voxels, subject=subjects))
            torch.testing.assert_close(output, clustered(voxels.reshape(2, 2, 2, 2), subject=subjects))
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(voxels.grad).all())
            for parameter in clustered.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            restored = ClusteredNeuroscienceInformedAttention(path, 1, **kwargs).eval()
            restored.load_state_dict(clustered.state_dict(), strict=True)
            torch.testing.assert_close(output, restored(voxels, subject=subjects))

    def test_mask_order_and_checkpoint_bank_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "src.modules.neuro_informed_attn.nib.load", return_value=FakeNifti()):
            path = Path(directory) / "bank.pt"
            data = bank([0] * 4 + [1] * 4, 2)
            data["subjects"]["subject_1"]["flat_indices"] = torch.arange(7, -1, -1)
            torch.save(data, path)
            kwargs = dict(hidden_dim=8, rank=4, token_dim=4, n_fmri_tokens=2, n_mlp_layers=1)
            with self.assertRaisesRegex(ValueError, "voxel order/mask"):
                ClusteredNeuroscienceInformedAttention(path, 2, **kwargs)
            torch.save(bank([0] * 4 + [1] * 4, 2), path)
            first = ClusteredNeuroscienceInformedAttention(path, 2, **kwargs)
            torch.save(bank([1] * 4 + [0] * 4, 2), path)
            second = ClusteredNeuroscienceInformedAttention(path, 2, **kwargs)
            with self.assertRaisesRegex(RuntimeError, "cluster bank differs"):
                second.load_state_dict(first.state_dict(), strict=False)
            incompatible = first.state_dict()
            incompatible["visual_mask_subject_1"] = torch.zeros_like(incompatible["visual_mask_subject_1"])
            with self.assertRaisesRegex(RuntimeError, "checkpoint mask differs"):
                first.load_state_dict(incompatible, strict=False)

    def test_llm_token_shape_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                "src.modules.neuro_informed_attn.nib.load", return_value=FakeNifti()):
            path = Path(directory) / "bank.pt"
            torch.save(bank([0] * 4 + [1] * 4, 2), path)
            encoder = ClusteredNeuroscienceInformedAttention(path, 2, hidden_dim=8,
                                                             rank=4, pe_method="coord").eval()
            result = encoder(torch.randn(2, 8), subject=["subject_1"] * 2)
            self.assertEqual(result.shape, (2, 128, 1024))
            self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
