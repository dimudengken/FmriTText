import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if os.name == 'posix':
    from scripts import run_readout_campaign as campaign


@unittest.skipUnless(os.name == 'posix', 'server campaign uses POSIX process locks')
class CampaignTest(unittest.TestCase):
    def test_readout_comparison_uses_correct_arms_and_rejects_mismatched_keys(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for arm, score in [('baseline', .4), ('adaptive', .6), ('mlp_control', .5)]:
                run = root / arm
                run.mkdir()
                campaign.save(run / 'evaluation.json', {'keys': ['a','b'], 'references': [['a'],['b']]})
                campaign.save(run / 'result.json', {'epochs': 60})
                campaign.save(run / 'test_metrics.json', {'real': {'CIDEr_x100': score*100, 'BLEU4': .2, 'ROUGE_L': .4}})
                np.savez(run / 'per_image_cider.npz', real=[score,score], shuffle=[0.,0.])
            args = argparse.Namespace(output=root)
            campaign.compare(args)
            comparisons = json.loads((root / 'summary.json').read_text())['paired_comparisons']
            self.assertAlmostEqual(comparisons['adaptive_minus_baseline']['delta_CIDEr_x100'], 20)
            self.assertAlmostEqual(comparisons['adaptive_minus_mlp_control']['delta_CIDEr_x100'], 10)
            self.assertIn('mlp_control', (root / 'RESULTS.md').read_text())
            campaign.save(root / 'adaptive/evaluation.json', {'keys': ['b','a'], 'references': [['a'],['b']]})
            with self.assertRaises(ValueError):
                campaign.compare(args)

    def test_failed_dependency_does_not_launch_readout_training(self):
        class StopWatching(Exception):
            pass
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)/'queued'
            root.mkdir()
            dependency = Path(name)/'prior'
            dependency.mkdir()
            campaign.save(dependency/'status.json', {'state': 'failed'})
            campaign.save(root/'protocol.json', {'max_epochs':60, 'after':str(dependency)})
            args = argparse.Namespace(output=root, after=dependency, epochs=60, max_hours=48)
            with patch.object(campaign,'hourly'), patch.object(campaign.time,'sleep',side_effect=StopWatching), patch.object(campaign.subprocess,'Popen') as launch:
                with self.assertRaises(StopWatching):
                    campaign.campaign_locked(args,root)
            launch.assert_not_called()
            self.assertEqual(json.loads((root/'status.json').read_text())['state'],'waiting')
