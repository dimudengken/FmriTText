import copy
from pathlib import Path
import tempfile
import unittest

import h5py
import nibabel as nib
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from scripts.build_voxel_clusters import build_bank
from src.voxel_clustering import (
    audit_cluster_split,
    encoding_features,
    heldout_ids,
    training_responses,
    validation_ids,
)


def sample(coco_id, responses=None):
    result = {"coco_id": np.array(coco_id)}
    if responses is not None:
        result["visual_voxels"] = np.array(responses, dtype=np.float32)
    return result


class VoxelClusteringTest(unittest.TestCase):
    def test_split_order_matches_original_and_repeats_stay_together(self):
        group = {"a": sample(9), "b": sample(2), "c": sample(9), "d": sample(7)}
        self.assertEqual(validation_ids(group, 1, 42), {2})
        self.assertEqual(validation_ids(group, 0, 42), set())
        handles = {"subject_1": {"train": group, "test": {"e": sample(7)}}}
        self.assertEqual(heldout_ids(handles, 1, 42), {2, 7})

    def test_excluded_responses_are_not_read_and_repeats_are_averaged(self):
        group = {"a": sample(1, [1, 2]), "b": sample(1, [3, 4]),
                 "c": sample(2), "d": sample(3, [5, 6])}
        ids, responses = training_responses(group, {2}, {1: 0, 3: 1})
        self.assertEqual(ids, [1, 3])
        np.testing.assert_array_equal(responses, [[2, 3], [5, 6]])
        with self.assertRaisesRegex(ValueError, "missing training coco_id"):
            training_responses(group, {2}, {1: 0})

    def test_ridge_features_capture_function_and_handle_constant_voxels(self):
        x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
        y = np.column_stack([x[:, 0], 10 * x[:, 0] + 3, x[:, 1], np.ones(4)])
        features = encoding_features(x, y, 1.0)
        np.testing.assert_allclose(features[0], features[1], atol=1e-6)
        np.testing.assert_allclose(features[:3], [[1, 0], [1, 0], [0, 1]], atol=1e-6)
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(features[3], [0, 0])

    def test_multisubject_exclusion_is_global(self):
        handles = {
            "subject_1": {"train": {"a": sample(1, [1, 2]), "b": sample(2),
                                     "c": sample(3, [3, 4])}, "test": {"t": sample(4)}},
            "subject_2": {"train": {"a": sample(4), "b": sample(5, [5, 6]),
                                     "c": sample(6, [7, 8])}, "test": {"t": sample(2)}},
        }
        excluded = heldout_ids(handles, 0, 42)
        self.assertEqual(excluded, {2, 4})
        index = {1: 0, 3: 1, 5: 2, 6: 3}
        self.assertEqual(training_responses(handles["subject_1"]["train"], excluded, index)[0], [1, 3])
        self.assertEqual(training_responses(handles["subject_2"]["train"], excluded, index)[0], [5, 6])

    def test_audit_detects_changed_split_and_other_subject_centroid_leakage(self):
        handles = {"subject_1": {"train": {"a": sample(1), "b": sample(2),
                                          "c": sample(3)}, "test": {"t": sample(4)}}}
        bank = {"version": 1, "subjects": {"subject_1": {"fit_coco_ids": [1, 2, 3]}}}
        audit_cluster_split(bank, handles, 0, 42)
        with self.assertRaisesRegex(ValueError, "overlaps held-out"):
            audit_cluster_split(bank, handles, 1, 42)
        bank["subjects"]["subject_2"] = {"fit_coco_ids": [4]}
        with self.assertRaisesRegex(ValueError, "overlaps held-out"):
            audit_cluster_split(bank, handles, 0, 42)
        bank["subjects"].pop("subject_2")
        bank["subjects"]["subject_1"]["fit_coco_ids"] = [8]
        with self.assertRaisesRegex(ValueError, "not a subset"):
            audit_cluster_split(bank, handles, 0, 42)

    def test_full_bank_build_ignores_heldout_data_and_random_control_keeps_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roi = root / "roi" / "subj01"
            roi.mkdir(parents=True)
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), roi / "nsdgeneral.nii.gz")
            x = torch.tensor([[-2., -1.], [-2., 1.], [0., -1.], [0., 1.], [2., -1.], [2., 1.]])
            feature_bank = {"coco_ids": torch.arange(1, 7), "embeddings": x, "modality": "image"}
            with h5py.File(root / "subject_1.h5", "w") as handle:
                train = handle.create_group("train")
                test = handle.create_group("test")
                test.create_group("t").create_dataset("coco_id", data=6)
                for i in range(6):
                    item = train.create_group(str(i))
                    item.create_dataset("coco_id", data=i + 1)
                    values = np.concatenate([np.full(4, x[i, 0]), np.full(4, x[i, 1])])
                    item.create_dataset("visual_voxels", data=values)
                handles = {"subject_1": handle}
                bank = build_bank(handles, feature_bank, root / "roi", n_clusters=2)
                labels = bank["subjects"]["subject_1"]["cluster_ids"].numpy()
                self.assertEqual(len(set(labels[:4])), 1)
                self.assertEqual(len(set(labels[4:])), 1)
                self.assertNotEqual(labels[0], labels[4])
                self.assertNotIn(6, bank["subjects"]["subject_1"]["fit_coco_ids"])
                train["5"]["visual_voxels"][...] = 1e9
                changed = copy.deepcopy(feature_bank)
                changed["embeddings"][5] = 1e9
                second = build_bank(handles, changed, root / "roi", n_clusters=2)
                self.assertTrue(torch.equal(bank["centroids"], second["centroids"]))
                self.assertTrue(torch.equal(bank["subjects"]["subject_1"]["cluster_ids"],
                                            second["subjects"]["subject_1"]["cluster_ids"]))
                control = build_bank(handles, feature_bank, root / "roi", n_clusters=2,
                                     random_clusters=True)
                np.testing.assert_array_equal(np.bincount(labels),
                    np.bincount(control["subjects"]["subject_1"]["cluster_ids"].numpy()))
                torch.save(bank, root / "bank.pt")
                loaded = torch.load(root / "bank.pt", weights_only=True)
                audit_cluster_split(loaded, handles, 0, 42)

    def test_config_is_opt_in_and_output_directory_is_separate(self):
        config = str(Path(__file__).resolve().parents[1] / "config")
        with initialize_config_dir(config_dir=config, version_base=None):
            baseline = compose(config_name="default")
            clustered = compose(config_name="clustered")
        self.assertEqual(baseline.encoder._target_, "src.modules.NeuroscienceInformedAttention")
        self.assertTrue(clustered.encoder._target_.endswith("ClusteredNeuroscienceInformedAttention"))
        self.assertEqual(clustered.encoder.token_dim, baseline.encoder.token_dim)
        self.assertEqual(clustered.encoder.n_fmri_tokens, baseline.encoder.n_fmri_tokens)
        self.assertNotEqual(clustered.output_dir, baseline.output_dir)
        self.assertTrue(clustered.data.split_val)
        self.assertFalse(clustered.resume)


if __name__ == "__main__":
    unittest.main()
