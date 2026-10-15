import unittest
import numpy as np
from src.roi_cluster_transfer import transfer_labels, frozen_split


def anatomy(coords, rois, labels=None):
    value={'coords':np.asarray(coords,float),'hemisphere':np.array([0,0,1,1]),
           'rois':np.repeat(np.asarray(rois)[:,None],3,axis=1)}
    if labels is not None: value['labels']=np.array(labels)
    return value


class TransferTest(unittest.TestCase):
    def setUp(self):
        self.source=anatomy([[-2,0,0],[-1,1,0],[1,0,0],[2,1,0]],[1,1,1,1],[0,1,2,3])

    def test_parcel_position_transfer_preserves_hemisphere_and_assigns_every_voxel(self):
        target=anatomy([[-12,5,0],[-10,7,0],[12,5,0],[14,7,0]],[1,1,1,1])
        labels,diagnostics=transfer_labels([self.source],target,4,neighbors=1)
        np.testing.assert_array_equal(labels,[0,1,2,3])
        np.testing.assert_array_equal(diagnostics['mapping_level'],0)
        np.testing.assert_allclose(diagnostics['vote_confidence'],1)

    def test_missing_roi_uses_recorded_fallback_without_target_response_input(self):
        target=anatomy(self.source['coords'],[7,7,7,7])
        labels,diagnostics=transfer_labels([self.source],target,4,neighbors=1)
        np.testing.assert_array_equal(labels,[0,1,2,3])
        np.testing.assert_array_equal(diagnostics['mapping_level'],3)
        target['responses']=np.zeros(4)
        with self.assertRaises(ValueError): transfer_labels([self.source],target,4)

    def test_source_label_permutation_changes_target_correspondingly(self):
        target=anatomy(self.source['coords'],[1,1,1,1])
        self.source['labels']=np.array([3,2,1,0])
        labels,_=transfer_labels([self.source],target,4,neighbors=1)
        np.testing.assert_array_equal(labels,[3,2,1,0])

    def test_global_split_excludes_other_subject_validation_and_all_test_images(self):
        ids={'subject_2':{'train':list(range(20)),'test':[30]},
             'subject_5':{'train':list(range(10,30)),'test':[31]}}
        split,test,val=frozen_split(ids,[12,18,32],val_count=3,seed=42)
        for parts in split.values():
            self.assertFalse(set(parts['train'])&set(test))
            self.assertFalse(set(parts['train'])&set(val))
            self.assertFalse(set(parts['val'])&set(test))
        self.assertTrue({12,18,30,31,32}<=set(test))


class TargetDataBoundaryTest(unittest.TestCase):
    def test_target_data_module_never_reads_target_train_group(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from src.dataset_zero_shot import SourceOnlyDataModule

        class TestOnlyHandle:
            def __getitem__(self,key):
                if key!='test': raise AssertionError('target training group accessed')
                return {'repeat':{'coco_id':np.array(99)}}

        manifest={'sources':[2,5,7],'target':1,'sha256':'fixture',
                  'subjects':{'subject_1':{'train':[],'val':[],'test':[99]}}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json';path.write_text(json.dumps(manifest))
            with patch('h5py.File', return_value=TestOnlyHandle()):
                data=SourceOnlyDataModule(path,target_mode=True,subjects=[1],tokenizer=None)
            self.assertFalse(data.file['train']['subject_1'])
            self.assertEqual(len(data.file['test']['subject_1']),1)
