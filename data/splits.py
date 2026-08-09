"""Subject-level LOSO-CV split utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class LOSOSplit:
    fold_id: int
    test_subject: str
    val_subjects: List[str]
    train_subjects: List[str]
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]


def build_loso_splits(subject_ids: Sequence[str], num_val_subjects: int = 3, seed: int = 42) -> List[LOSOSplit]:
    """Build leave-one-subject-out folds with train/val/test subject independence."""
    subjects = sorted(set(subject_ids))
    if len(subjects) < num_val_subjects + 2:
        raise ValueError("Not enough subjects for LOSO-CV.")
    splits: List[LOSOSplit] = []
    for fold_id, test_subject in enumerate(subjects, start=1):
        remaining = [subject for subject in subjects if subject != test_subject]
        rng = random.Random(seed + fold_id)
        val_subjects = sorted(rng.sample(remaining, k=num_val_subjects))
        val_set = set(val_subjects)
        train_subjects = [subject for subject in remaining if subject not in val_set]
        train_set = set(train_subjects)
        train_indices = [idx for idx, subject in enumerate(subject_ids) if subject in train_set]
        val_indices = [idx for idx, subject in enumerate(subject_ids) if subject in val_set]
        test_indices = [idx for idx, subject in enumerate(subject_ids) if subject == test_subject]
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
