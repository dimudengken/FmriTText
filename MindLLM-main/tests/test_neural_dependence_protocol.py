import unittest
from scripts.neural_dependence_protocol import shuffle_donors


class ShuffleProtocolTests(unittest.TestCase):
    def test_subject_local_bijective_no_fixed_points_reproducible(self):
        keys = [(subject, stimulus) for subject in (1, 2) for stimulus in range(30)]
        donors = shuffle_donors(keys, 42)
        self.assertEqual(donors, shuffle_donors(keys, 42))
        self.assertEqual(sorted(donors), list(range(len(keys))))
        for recipient, donor in enumerate(donors):
            self.assertEqual(keys[recipient][0], keys[donor][0])
            self.assertNotEqual(keys[recipient][1], keys[donor][1])

    def test_invalid_singleton_and_repeated_stimulus(self):
        for keys in ([(1, 10)], [(1, 10), (1, 10)]):
            with self.assertRaises(ValueError):
                shuffle_donors(keys, 42)


if __name__ == "__main__":
    unittest.main()
