"""Download and load the FER-2013 splits used by the evaluation scripts.

The Hugging Face mirror keeps the original Kaggle usage split: Training
(28,709 faces), PublicTest (3,589) and PrivateTest (3,589). The images are
48 x 48 grayscale faces and labels use the same order as DeepFace's model.
"""

from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path

import numpy as np

PARQUET_URL = (
    "https://huggingface.co/api/datasets/Aaryan333/fer2013_train_publicTest_privateTest"
    "/parquet/default/{split}/0.parquet"
)
SPLITS = ("train", "publicTest", "privateTest")
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "fer2013"


def download(data_dir: Path = DEFAULT_DATA_DIR, attempts: int = 4) -> None:
    """Fetch any missing split, retrying because the mirror is sometimes slow."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        target = data_dir / f"{split}.parquet"
        if target.exists() and target.stat().st_size > 1_000_000:
            continue
        for attempt in range(1, attempts + 1):
            try:
                print(f"Downloading FER-2013 {split} split (attempt {attempt})…")
                with urllib.request.urlopen(PARQUET_URL.format(split=split), timeout=300) as response:
                    target.write_bytes(response.read())
                break
            except OSError as error:
                if attempt == attempts:
                    raise SystemExit(f"Could not download {split}: {error}") from error
                time.sleep(3 * attempt)


def load_split(split: str, data_dir: Path = DEFAULT_DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Return (images uint8 [n, 48, 48], labels int [n]) for one split."""
    import pyarrow.parquet as pq
    from PIL import Image

    path = data_dir / f"{split}.parquet"
    if not path.exists():
        download(data_dir)
    table = pq.read_table(path).to_pydict()
    images = np.stack(
        [np.asarray(Image.open(io.BytesIO(record["bytes"])).convert("L")) for record in table["image"]]
    ).astype(np.uint8)
    return images, np.asarray(table["label"], dtype=np.int64)
