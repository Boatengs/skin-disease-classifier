import numpy as np
import pandas as pd
import pytest

from skinclf import CLASSES


@pytest.fixture
def toy_meta() -> pd.DataFrame:
    """Synthetic metadata shaped like HAM10000: lesions with one to four photos each."""
    rng = np.random.default_rng(0)
    rows, img = [], 0
    weights = np.array([11, 67, 5, 3, 11, 1.5, 1.5])
    for lesion in range(1500):
        dx = CLASSES[rng.choice(len(CLASSES), p=weights / weights.sum())]
        for _ in range(rng.choice([1, 1, 1, 2, 2, 3, 4])):
            rows.append({"lesion_id": f"HAM_{lesion:07d}", "image_id": f"ISIC_{img:07d}", "dx": dx})
            img += 1
    meta = pd.DataFrame(rows)
    meta["label"] = meta["dx"].map({c: i for i, c in enumerate(CLASSES)})
    return meta
