"""Dataset, seed, split, and masking utilities shared by all questions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import fetch_olivetti_faces


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetBundle:
    data: np.ndarray
    images: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    seed: int


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a JSON object.")
    return config


def project_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the assignment directory."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def seed_from_roll_number(roll_number: object) -> int:
    """Return the integer formed by all ASCII digits in the complete roll number."""
    if roll_number is None:
        raise ValueError("The complete roll number is required to derive the assignment seed.")
    roll_text = str(roll_number).strip()
    digits = "".join(re.findall(r"[0-9]", roll_text))
    if not digits:
        raise ValueError(
            "The configured roll number contains no numeric characters; "
            "the official assignment seed cannot be derived."
        )
    return int(digits)


def derive_seed(master_seed: int, stream_name: str) -> int:
    """Derive a stable 32-bit child seed without consuming another RNG stream."""
    payload = f"{master_seed}:{stream_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def load_olivetti(data_home: str | Path) -> DatasetBundle:
    data_dir = project_path(data_home)
    data_dir.mkdir(parents=True, exist_ok=True)
    faces = fetch_olivetti_faces(
        data_home=str(data_dir),
        shuffle=False,
        download_if_missing=True,
    )
    data = np.asarray(faces.data, dtype=np.float64)
    images = np.asarray(faces.images, dtype=np.float64)
    target = np.asarray(faces.target, dtype=np.int64)
    if data.shape != (400, 4096) or images.shape != (400, 64, 64):
        raise RuntimeError(
            f"Unexpected Olivetti shape: data={data.shape}, images={images.shape}."
        )
    if not np.allclose(data.reshape(-1, 64, 64), images):
        raise RuntimeError("Flattened data and image arrays do not agree.")
    return DatasetBundle(data=data, images=images, target=target)


def create_identity_split(target: np.ndarray, seed: int) -> SplitIndices:
    """Create the required independent 6/2/2 random split within every identity."""
    target = np.asarray(target)
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []

    for class_id in np.unique(target):
        members = np.flatnonzero(target == class_id)
        if members.size != 10:
            raise ValueError(
                f"Identity {class_id!r} has {members.size} images; exactly 10 are required."
            )
        shuffled = rng.permutation(members)
        train.extend(shuffled[:6].tolist())
        validation.extend(shuffled[6:8].tolist())
        test.extend(shuffled[8:10].tolist())

    split = SplitIndices(
        train=np.asarray(train, dtype=np.int64),
        validation=np.asarray(validation, dtype=np.int64),
        test=np.asarray(test, dtype=np.int64),
        seed=seed,
    )
    validate_split(split, target)
    return split


def validate_split(split: SplitIndices, target: np.ndarray) -> None:
    target = np.asarray(target)
    arrays = {
        "train": np.asarray(split.train, dtype=np.int64),
        "validation": np.asarray(split.validation, dtype=np.int64),
        "test": np.asarray(split.test, dtype=np.int64),
    }
    expected_sizes = {"train": 240, "validation": 80, "test": 80}
    expected_per_class = {"train": 6, "validation": 2, "test": 2}

    for name, indices in arrays.items():
        if indices.ndim != 1 or indices.size != expected_sizes[name]:
            raise ValueError(
                f"{name} indices have shape {indices.shape}; expected "
                f"({expected_sizes[name]},)."
            )
        if np.any(indices < 0) or np.any(indices >= target.size):
            raise ValueError(f"{name} contains an out-of-range index.")
        labels, counts = np.unique(target[indices], return_counts=True)
        if labels.size != 40 or not np.all(counts == expected_per_class[name]):
            raise ValueError(
                f"{name} does not have {expected_per_class[name]} samples per identity."
            )

    combined = np.concatenate(list(arrays.values()))
    if combined.size != target.size or np.unique(combined).size != target.size:
        raise ValueError("Split sets are not disjoint or do not cover all samples exactly once.")


def split_digest(split: SplitIndices) -> str:
    digest = hashlib.sha256()
    for values in (split.train, split.validation, split.test):
        digest.update(np.asarray(values, dtype="<i8").tobytes())
    digest.update(str(split.seed).encode("ascii"))
    return digest.hexdigest()


def save_split(split: SplitIndices, path: str | Path) -> None:
    destination = project_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        train=split.train,
        validation=split.validation,
        test=split.test,
        seed=np.asarray(str(split.seed)),
        sha256=np.asarray(split_digest(split)),
    )


def load_split(path: str | Path, target: np.ndarray) -> SplitIndices:
    source = project_path(path)
    with np.load(source, allow_pickle=False) as saved:
        split = SplitIndices(
            train=np.asarray(saved["train"], dtype=np.int64),
            validation=np.asarray(saved["validation"], dtype=np.int64),
            test=np.asarray(saved["test"], dtype=np.int64),
            seed=int(str(saved["seed"].item())),
        )
        recorded_digest = str(saved["sha256"].item())
    validate_split(split, target)
    if split_digest(split) != recorded_digest:
        raise ValueError("Saved split digest does not match its contents.")
    return split


def get_or_create_split(
    target: np.ndarray, seed: int, path: str | Path
) -> tuple[SplitIndices, bool]:
    """Load the immutable split or create it once; return (split, was_created)."""
    split_path = project_path(path)
    if split_path.exists():
        split = load_split(split_path, target)
        if split.seed != seed:
            raise ValueError(
                "The saved split was created with a different roll-number seed. "
                "Do not overwrite it silently; verify the configured roll number first."
            )
        return split, False

    split = create_identity_split(target, seed)
    save_split(split, split_path)
    return split, True


def mask_central_block(flattened_images: np.ndarray) -> np.ndarray:
    """Zero rows/columns 24..39 in copies of flattened 64 x 64 faces."""
    values = np.asarray(flattened_images)
    if values.ndim != 2 or values.shape[1] != 4096:
        raise ValueError(f"Expected shape (n_samples, 4096), received {values.shape}.")
    images = values.reshape(-1, 64, 64).copy()
    images[:, 24:40, 24:40] = 0.0
    return images.reshape(-1, 4096)
