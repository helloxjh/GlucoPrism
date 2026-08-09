"""Leave-one-subject-out cross-validation split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LOSOSplit:
    """One LOSO-CV fold with subject-independent train/val/test partitions."""

    fold_id: int
    test_subject: str
    val_subjects: List[str]
    train_subjects: List[str]
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


def build_loso_splits(
    metadata_path: Union[str, Path],
    num_val_subjects: int = 3,
    seed: int = 42,
) -> List[LOSOSplit]:
    """
    Build 16 LOSO-CV folds from window metadata.

    For each fold:
    - 1 subject is held out as test.
    - 3 subjects are selected from the remaining subjects as validation.
    - The rest are used for training.

    Validation subjects are chosen deterministically with the provided seed.
    """
    metadata = pd.read_csv(metadata_path, dtype={"subject_id": str})
    if "subject_id" not in metadata.columns:
        raise ValueError("metadata_path must contain a subject_id column.")

    subjects = sorted(metadata["subject_id"].unique().tolist())
    if len(subjects) < num_val_subjects + 2:
        raise ValueError("Not enough subjects to build train/val/test LOSO splits.")

    splits: List[LOSOSplit] = []
    for fold_id, test_subject in enumerate(subjects, start=1):
        remaining = [subject for subject in subjects if subject != test_subject]
        rng = np.random.default_rng(seed + fold_id)
        val_subjects = sorted(rng.choice(remaining, size=num_val_subjects, replace=False).tolist())
        train_subjects = [subject for subject in remaining if subject not in set(val_subjects)]

        subject_series = metadata["subject_id"]
        train_indices = metadata.index[subject_series.isin(train_subjects)].to_numpy(dtype=np.int64)
        val_indices = metadata.index[subject_series.isin(val_subjects)].to_numpy(dtype=np.int64)
        test_indices = metadata.index[subject_series == test_subject].to_numpy(dtype=np.int64)

        splits.append(
            LOSOSplit(
                fold_id=fold_id,
                test_subject=test_subject,
                val_subjects=val_subjects,
                train_subjects=train_subjects,
                train_indices=train_indices,
                val_indices=val_indices,
                test_indices=test_indices,
            )
        )

    return splits
