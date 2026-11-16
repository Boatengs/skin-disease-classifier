"""Download HAM10000 from Harvard Dataverse and cache resized images.

Source: Tschandl, Rosendahl and Kittler, "The HAM10000 dataset", Scientific Data
5, 180161 (2018). Harvard Dataverse doi:10.7910/DVN/DBW86T, licensed CC BY-NC 4.0.
"""
from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASSES

DATAVERSE = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/DBW86T"
WANTED = ["HAM10000_metadata.tab", "HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"]


def _file_ids() -> dict[str, int]:
    url = f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={DOI}"
    with urllib.request.urlopen(url, timeout=60) as r:
        files = json.load(r)["data"]["latestVersion"]["files"]
    return {f["dataFile"]["filename"]: f["dataFile"]["id"] for f in files}


def download(root: Path, metadata_only: bool = False) -> Path:
    """Fetch metadata and both image archives into `root`, then unzip."""
    root.mkdir(parents=True, exist_ok=True)
    ids = _file_ids()
    for name in WANTED[:1] if metadata_only else WANTED:
        target = root / name.replace(".tab", ".csv")
        if target.exists():
            continue
        url = f"{DATAVERSE}/api/access/datafile/{ids[name]}?format=original"
        print(f"downloading {name}", flush=True)
        urllib.request.urlretrieve(url, target)
    images = root / "images"
    if not metadata_only and not images.exists():
        images.mkdir()
        for part in ("HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"):
            with zipfile.ZipFile(root / part) as z:
                for member in z.namelist():
                    if member.lower().endswith(".jpg"):
                        (images / Path(member).name).write_bytes(z.read(member))
    return root


def load_metadata(path: Path) -> pd.DataFrame:
    """Read the metadata table, sorted by image id, with an integer label."""
    meta = pd.read_csv(path)
    required = {"lesion_id", "image_id", "dx"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    unknown = set(meta["dx"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"unexpected diagnosis codes: {sorted(unknown)}")
    if meta["image_id"].duplicated().any():
        raise ValueError("duplicate image ids in metadata")
    per_lesion = meta.groupby("lesion_id")["dx"].nunique()
    if (per_lesion > 1).any():
        raise ValueError("a lesion carries more than one diagnosis")
    meta = meta.sort_values("image_id").reset_index(drop=True)
    meta["label"] = meta["dx"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    return meta


def build_cache(meta: pd.DataFrame, image_dir: Path, out: Path, size: int = 224) -> np.ndarray:
    """Resize every image to size x size (as the original notebook did) and store uint8."""
    from PIL import Image

    if out.exists():
        arr = np.load(out, mmap_mode="r")
        if arr.shape == (len(meta), size, size, 3):
            return arr
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(len(meta), size, size, 3))
    for i, image_id in enumerate(meta["image_id"]):
        with Image.open(image_dir / f"{image_id}.jpg") as im:
            arr[i] = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))
        if i % 2000 == 0:
            print(f"cached {i}/{len(meta)}", flush=True)
    arr.flush()
    return np.load(out, mmap_mode="r")
