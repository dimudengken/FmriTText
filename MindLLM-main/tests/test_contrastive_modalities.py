import unittest
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.precompute_clip_features import encode_text_bank
from scripts.align_clip_text_bank import align
from src.models.losses.contrastive import multi_positive_infonce
from src.modules.contrastive_projector import BrainContrastiveProjector
from src.utils.retrieval import retrieval_metrics_unique_gallery


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": torch.tensor([[ord(text[0])] for text in texts])}


class FakeClip:
    def get_text_features(self, input_ids):
        x = input_ids.float().squeeze(1)
        return torch.stack((x, 200 - x), dim=1)


class ContrastiveModalitiesTest(unittest.TestCase):
    def test_align_existing_banks_without_reencoding(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            image_path = directory / "image.pt"
            text_path = directory / "text.pt"
            out_path = directory / "aligned.pt"
            image = {"coco_ids": torch.tensor([5, 3]),
                     "embeddings": F.normalize(torch.randn(2, 4), dim=1),
                     "model_id": "same-clip"}
            text = {"coco_ids": torch.tensor([3, 4, 5]),
                    "embeddings": F.normalize(torch.randn(3, 4), dim=1),
                    "model_id": "same-clip", "modality": "text",
                    "caption_aggregation": "mean_then_l2"}
            torch.save(image, image_path)
            torch.save(text, text_path)
            align(str(image_path), str(text_path), str(out_path))
            aligned = torch.load(out_path, map_location="cpu")
            self.assertEqual(aligned["coco_ids"].tolist(), [5, 3])
            self.assertTrue(torch.equal(aligned["embeddings"], text["embeddings"][[2, 0]]))
            self.assertEqual(torch.load(text_path, map_location="cpu")["coco_ids"].tolist(), [3, 4, 5])

    def test_caption_mean_then_l2_and_bank_identity(self):
        ids = [1, 2]
        captions = {1: ["a", "b", "c"], 2: ["d", "e"]}
        bank = encode_text_bank(ids, captions, FakeClip(), FakeTokenizer(), 2, "cpu")
        individual = F.normalize(FakeClip().get_text_features(
            FakeTokenizer()(["a", "b", "c"])["input_ids"]
        ), dim=-1)
        expected = F.normalize(individual.mean(0), dim=0)
        self.assertTrue(torch.allclose(bank[0], expected))
        self.assertTrue(torch.allclose(bank.norm(dim=1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.isfinite(bank).all())

    def test_m2_m3_backward_and_unique_gallery(self):
        torch.manual_seed(3)
        ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        encoder = torch.nn.Linear(6, 4)
        head = BrainContrastiveProjector(4, 4, 8)
        voxels = torch.randn(8, 6)
        tokens = encoder(voxels).unsqueeze(1).expand(-1, 3, -1)
        brain_z = head(tokens)
        image_bank = F.normalize(torch.randn(4, 4), dim=1)
        text_bank = F.normalize(torch.randn(4, 4), dim=1)
        image_z, text_z = image_bank[ids], text_bank[ids]
        self.assertTrue(torch.equal(image_z[0], image_z[1]))
        self.assertTrue(torch.equal(text_z[0], text_z[1]))
        image_loss = multi_positive_infonce(brain_z, image_z, ids)
        text_loss = multi_positive_infonce(brain_z, text_z, ids)
        self.assertTrue(torch.isfinite(image_loss))
        self.assertTrue(torch.isfinite(text_loss))
        total = image_loss + text_loss
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(encoder.weight.grad)
        self.assertIsNotNone(head.proj[0].weight.grad)

        gallery = F.normalize(torch.eye(20), dim=1)
        repeated_ids = torch.arange(20).repeat_interleave(2)
        target = gallery[repeated_ids]
        matched = retrieval_metrics_unique_gallery(target, target, repeated_ids)
        shuffled = retrieval_metrics_unique_gallery(target, gallery[torch.arange(20).roll(7)][repeated_ids], repeated_ids)
        self.assertEqual(matched["r@1"], 1.0)
        self.assertEqual(shuffled["r@1"], 0.0)


if __name__ == "__main__":
    unittest.main()
