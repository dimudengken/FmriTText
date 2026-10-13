"""Pure protocol helpers; no model or dataset dependencies."""
import random
from collections import defaultdict


def shuffle_donors(keys, seed):
    """A fixed, subject-local derangement of unique (subject, stimulus) keys."""
    if len(set(keys)) != len(keys):
        raise ValueError("each subject/stimulus must appear exactly once")
    groups = defaultdict(list)
    for i, (subject, _) in enumerate(keys):
        groups[subject].append(i)
    rng = random.Random(seed)
    donors = list(range(len(keys)))
    for rows in groups.values():
        if len(rows) < 2:
            raise ValueError("Shuffle needs at least two stimuli per subject")
        while True:
            shuffled = rows.copy()
            rng.shuffle(shuffled)
            if all(a != b for a, b in zip(rows, shuffled)):
                break
        for recipient, donor in zip(rows, shuffled):
            donors[recipient] = donor
    return donors
