import numpy as np

from skinclf.splits import image_split, images_with_seen_lesion, lesion_split, shared_lesions


def _partition_ok(split, n):
    all_rows = np.concatenate([split.train, split.val, split.test])
    assert len(all_rows) == n
    assert len(np.unique(all_rows)) == n


def test_lesion_split_keeps_each_lesion_in_one_partition(toy_meta):
    s = lesion_split(toy_meta, seed=3)
    _partition_ok(s, len(toy_meta))
    for a, b in [(s.train, s.test), (s.train, s.val), (s.val, s.test)]:
        assert shared_lesions(toy_meta, a, b) == 0
    assert images_with_seen_lesion(toy_meta, s.train, s.test) == 0


def test_lesion_split_sizes_and_strata(toy_meta):
    s = lesion_split(toy_meta, seed=0)
    n = len(toy_meta)
    assert 0.06 * n < len(s.test) < 0.14 * n
    assert 0.06 * n < len(s.val) < 0.14 * n
    # every class present in test
    assert set(toy_meta["dx"].iloc[s.test]) == set(toy_meta["dx"])


def test_image_split_leaks_lesions_on_multi_photo_data(toy_meta):
    s = image_split(toy_meta, seed=42)
    _partition_ok(s, len(toy_meta))
    assert shared_lesions(toy_meta, s.train, s.test) > 0
    assert images_with_seen_lesion(toy_meta, s.train, s.test) > 0


def test_image_split_matches_original_proportions(toy_meta):
    s = image_split(toy_meta, seed=42)
    n = len(toy_meta)
    assert len(s.train) == round(0.8 * n) or abs(len(s.train) - 0.8 * n) <= 1


def test_splits_are_deterministic(toy_meta):
    a, b = lesion_split(toy_meta, seed=7), lesion_split(toy_meta, seed=7)
    assert np.array_equal(a.test, b.test) and np.array_equal(a.val, b.val)
