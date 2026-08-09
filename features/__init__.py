"""Feature engineering package for GlucoPrism."""

from .physio_features import aggregate_signal_mean, compute_acc_l2, expected_physio_order
from .segmentation import split_by_time_gaps
from .windowing import make_sliding_windows

__all__ = [
    "aggregate_signal_mean",
    "compute_acc_l2",
    "expected_physio_order",
    "make_sliding_windows",
    "split_by_time_gaps",
]
