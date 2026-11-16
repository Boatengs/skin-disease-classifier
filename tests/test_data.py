import pandas as pd
import pytest

from skinclf.data import load_metadata


def _write(tmp_path, rows):
    p = tmp_path / "meta.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_load_metadata_labels_and_sorts(tmp_path):
    p = _write(tmp_path, [
        {"lesion_id": "HAM_2", "image_id": "ISIC_2", "dx": "nv"},
        {"lesion_id": "HAM_1", "image_id": "ISIC_1", "dx": "mel"},
    ])
    m = load_metadata(p)
    assert list(m["image_id"]) == ["ISIC_1", "ISIC_2"]
    assert list(m["label"]) == [0, 1]


@pytest.mark.parametrize("rows,match", [
    ([{"lesion_id": "a", "image_id": "x", "dx": "zzz"}], "unexpected"),
    ([{"lesion_id": "a", "image_id": "x", "dx": "nv"}, {"lesion_id": "b", "image_id": "x", "dx": "nv"}], "duplicate"),
    ([{"lesion_id": "a", "image_id": "x", "dx": "nv"}, {"lesion_id": "a", "image_id": "y", "dx": "mel"}], "more than one"),
])
def test_load_metadata_rejects_bad_tables(tmp_path, rows, match):
    with pytest.raises(ValueError, match=match):
        load_metadata(_write(tmp_path, rows))
