import numpy as np
import torch

from skinclf.model import build_model
from skinclf.train import CachedImages, class_weights


def test_model_output_shape_and_trainable_blocks():
    m = build_model(pretrained=False)
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 7)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    # The original notebook reports 3,485,475 trainable of 4,337,283.
    assert trainable == 3_485_475 and total == 4_337_283


def test_class_weights_match_notebook_formula():
    labels = np.array([0] * 10 + [1] * 90 + [2] * 5 + [3] * 5 + [4] * 10 + [5] * 2 + [6] * 2)
    w = class_weights(labels).numpy()
    assert w.sum() == np.float32(7) or abs(w.sum() - 7) < 1e-5
    assert w[5] > w[0] > w[1]


def test_cached_dataset_returns_normalised_tensor():
    imgs = np.full((3, 224, 224, 3), 128, np.uint8)
    ds = CachedImages(imgs, np.array([0, 1, 2]), np.array([2, 0]), train=False)
    x, y = ds[0]
    assert x.shape == (3, 224, 224) and y == 2
    assert torch.isfinite(x).all()
