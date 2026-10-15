"""Source-only split construction and metadata-only cross-subject label transfer."""
import numpy as np
from scipy.spatial import cKDTree


def frozen_split(source_ids, target_test_ids, val_count=900, seed=42):
    """Global image disjointness; target response arrays are not an input."""
    rng = np.random.default_rng(seed)
    test = set(map(int, target_test_ids))
    for entry in source_ids.values():
        test.update(map(int, entry['test']))
    validation = set()
    for subject in sorted(source_ids):
        candidates = np.array(sorted(set(source_ids[subject]['train']) - test), dtype=np.int64)
        if len(candidates) <= val_count:
            raise ValueError('not enough source training images')
        validation.update(map(int, rng.permutation(candidates)[:val_count]))
    result = {}
    for subject, entry in source_ids.items():
        available = set(map(int, entry['train']))
        result[subject] = {'train': sorted(available - validation - test),
                           'val': sorted(available & validation - test), 'test': []}
    return result, sorted(test), sorted(validation)


def normalized_position(coords):
    coords = np.asarray(coords, dtype=np.float64)
    return (coords - coords.mean(0)) / np.maximum(coords.std(0), 1.8)


def transfer_labels(sources, target, n_clusters, neighbors=5):
    """Each subject supplies coords/hemisphere/rois; only sources supply labels.

    Match HCP, then Kastner, then pRF, then hemisphere. Standardize coordinates
    independently within the matched parcel. Each source has equal vote mass.
    This is an approximate native-space correspondence, NOT MNI registration.
    """
    if not sources or neighbors < 1 or n_clusters < 1:
        raise ValueError('sources, neighbors and n_clusters must be positive')
    if 'labels' in target or 'responses' in target:
        raise ValueError('target may supply anatomical metadata only')
    count = len(target['coords'])
    labels = np.full(count, -1, dtype=np.int64)
    confidence = np.zeros(count, dtype=np.float32)
    levels = np.full(count, -1, dtype=np.int64)
    distances = np.zeros(count, dtype=np.float32)
    for source in sources:
        if len(source['labels']) != len(source['coords']):
            raise ValueError('source label/coordinate length mismatch')
        if np.any((source['labels'] < 0) | (source['labels'] >= n_clusters)):
            raise ValueError('source cluster label out of range')
    for level in range(4):
        pending = np.flatnonzero(labels < 0)
        if not len(pending):
            break
        groups = {}
        for index in pending:
            roi = int(target['rois'][index, level]) if level < 3 else 0
            if level < 3 and roi <= 0:
                continue
            groups.setdefault((int(target['hemisphere'][index]), roi), []).append(index)
        for (hemisphere, roi), indices in groups.items():
            # Normalize using the full anatomical parcel, not just fallback voxels.
            target_mask = target['hemisphere'] == hemisphere
            if level < 3:
                target_mask &= target['rois'][:, level] == roi
            parcel_indices = np.flatnonzero(target_mask)
            target_coords = normalized_position(target['coords'][target_mask])
            target_coords = target_coords[np.searchsorted(parcel_indices, indices)]
            votes = np.zeros((len(indices), n_clusters), dtype=np.float64)
            nearest = []
            voters = 0
            for source in sources:
                mask = source['hemisphere'] == hemisphere
                if level < 3:
                    mask &= source['rois'][:, level] == roi
                if not mask.any():
                    continue
                coords = normalized_position(source['coords'][mask])
                k = min(neighbors, len(coords))
                dist, selected = cKDTree(coords).query(target_coords, k=k)
                dist, selected = np.asarray(dist).reshape(len(indices), k), np.asarray(selected).reshape(len(indices), k)
                weights = 1 / np.maximum(dist, .05)
                weights /= weights.sum(1, keepdims=True)
                source_labels = source['labels'][mask][selected]
                np.add.at(votes, (np.arange(len(indices))[:, None], source_labels), weights)
                nearest.append(dist[:, 0])
                voters += 1
            if not voters:
                continue
            chosen = votes.argmax(1)
            labels[indices] = chosen
            confidence[indices] = votes[np.arange(len(indices)), chosen] / voters
            levels[indices] = level
            distances[indices] = np.mean(nearest, axis=0)
    if np.any(labels < 0):
        raise ValueError('some target voxels have no source hemisphere correspondence')
    return labels, {'vote_confidence': confidence, 'mapping_level': levels,
                    'nearest_standardized_distance': distances}
