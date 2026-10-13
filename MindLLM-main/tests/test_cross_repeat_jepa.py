import unittest

import numpy as np
import torch

from src.dataset_jepa import CrossRepeatDataModule, CrossRepeatDataset, collate_cross_repeat
from src.models.losses.cross_repeat_jepa import cosine_distance, ema_update


def sample(coco_id, voxel):
    return {
        "coco_id": np.array(coco_id),
        "visual_voxels": np.array([voxel], dtype=np.float32),
    }


class CrossRepeatJEPATest(unittest.TestCase):
    def test_pairs_are_distinct_and_validation_pair_is_fixed(self):
        group = {
            "a": sample(7, 1),
            "b": sample(7, 2),
            "c": sample(7, 3),
        }
        entries = [("subject_1", ("a", "b", "c"), 7)]
        train = CrossRepeatDataset({"subject_1": group}, entries, random_pair=True)
        val = CrossRepeatDataset({"subject_1": group}, entries, random_pair=False)
        for _ in range(20):
            pair = train[0]
            self.assertFalse(torch.equal(pair["context_voxels"], pair["target_voxels"]))
        batch = collate_cross_repeat([val[0]])
        self.assertEqual(batch["context_voxels"].tolist(), [[1.0]])
        self.assertEqual(batch["target_voxels"].tolist(), [[2.0]])
        self.assertEqual(batch["coco_id"].tolist(), [7])

    def test_global_split_keeps_shared_coco_on_one_side(self):
        group_1 = {"a": sample(7, 1), "b": sample(8, 2)}
        group_2 = {"c": sample(7, 3), "d": sample(9, 4)}
        data = CrossRepeatDataModule.__new__(CrossRepeatDataModule)
        data.split_seed = 42
        data.val_fraction = 0.5
        data.test_coco_ids = set()
        data.files = {
            "subject_1": {"train": group_1},
            "subject_2": {"train": group_2},
        }
        train_1, val_1 = data._split(group_1, "subject_1")
        train_2, val_2 = data._split(group_2, "subject_2")
        self.assertEqual("a" in train_1, "c" in train_2)
        self.assertEqual("a" in val_1, "c" in val_2)
        self.assertFalse(set(data._heldout_ids) & {
            int(group[key]["coco_id"][()])
            for group, subset in ((group_1, train_1), (group_2, train_2))
            for key in subset
        })

        leaking = CrossRepeatDataModule.__new__(CrossRepeatDataModule)
        leaking.split_seed = 42
        leaking.val_fraction = 0.5
        leaking.test_coco_ids = {7}
        leaking.files = data.files
        with self.assertRaisesRegex(ValueError, "train/test stimuli overlap"):
            leaking._split(group_1, "subject_1")

    def test_cosine_loss_and_ema_target(self):
        prediction = torch.tensor([[1.0, 0.0]], requires_grad=True)
        target = torch.tensor([[0.0, 1.0]])
        self.assertAlmostEqual(cosine_distance(prediction, target).item(), 1.0)
        cosine_distance(prediction, target).backward()
        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(target.grad)

        online = torch.nn.Linear(2, 2, bias=False)
        teacher = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            online.weight.fill_(2.0)
            teacher.weight.zero_()
        teacher.requires_grad_(False)
        ema_update(teacher, online, 0.75)
        self.assertTrue(torch.allclose(teacher.weight, torch.full_like(teacher.weight, 0.5)))
        self.assertFalse(teacher.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
