import unittest

import torch
import torch.nn.functional as F

from src.models.losses.caption_joint import (
    caption_contrastive_metrics,
    lookup_caption_targets,
    warmup_weight,
)
from src.modules.contrastive_projector import BrainContrastiveProjector


class CaptionJointTest(unittest.TestCase):
    def test_cpu_bank_lookup_and_missing_id(self):
        bank = F.normalize(torch.randn(2, 4), dim=1)
        targets, ids = lookup_caption_targets(bank, {10: 0, 20: 1}, [20, 20, 10], "cpu", "image")
        self.assertEqual(ids.tolist(), [20, 20, 10])
        self.assertEqual(targets.device.type, "cpu")
        self.assertTrue(torch.equal(targets[0], targets[1]))
        with self.assertRaisesRegex(KeyError, "coco_id=30"):
            lookup_caption_targets(bank, {10: 0, 20: 1}, [30], "cpu", "image")

    def test_weight_warmup(self):
        self.assertEqual(warmup_weight(0, 0.05, 1000), 0)
        self.assertAlmostEqual(warmup_weight(500, 0.05, 1000), 0.025)
        self.assertEqual(warmup_weight(1000, 0.05, 1000), 0.05)
        self.assertEqual(warmup_weight(1200, 0.05, 1000), 0.05)

    def test_joint_gradients_and_repeated_stimuli(self):
        torch.manual_seed(7)
        encoder = torch.nn.Linear(6, 4)
        projector = torch.nn.Linear(4, 8)
        frozen_lm = torch.nn.Linear(8, 8)
        frozen_lm.requires_grad_(False)
        head = BrainContrastiveProjector(4, 4, 8)
        voxels = torch.randn(4, 6)
        ids = torch.tensor([10, 10, 20, 20])
        targets = F.normalize(torch.randn(2, 4), dim=1)[[0, 0, 1, 1]]
        self.assertTrue(torch.equal(targets[0], targets[1]))

        brain_tokens = encoder(voxels).unsqueeze(1).expand(-1, 3, -1)
        lm_loss = F.mse_loss(frozen_lm(projector(brain_tokens)).float(), torch.zeros(4, 3, 8))
        brain_z = head(brain_tokens)
        auxiliary, top1, positive, negative = caption_contrastive_metrics(
            brain_z, targets, ids, 0.07
        )
        weight = warmup_weight(500, 0.05, 1000)
        total = lm_loss + weight * auxiliary
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.equal(total, lm_loss + weight * auxiliary))
        self.assertTrue(0 <= top1.item() <= 1)
        self.assertTrue(torch.isfinite(positive) and torch.isfinite(negative))
        total.backward()
        self.assertIsNotNone(encoder.weight.grad)
        self.assertIsNotNone(projector.weight.grad)
        self.assertIsNotNone(head.proj[0].weight.grad)
        self.assertIsNone(frozen_lm.weight.grad)


if __name__ == "__main__":
    unittest.main()
