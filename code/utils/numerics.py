"""Protect undefined numerical cases without perturbing normal denominators."""
import warnings
import numpy as np


def require_finite(array, label):
    if not isinstance(array, np.ndarray) or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f'{label} must be a nonempty finite NumPy array')


def minmax_normalize(array, label='Array'):
    require_finite(array, label)
    lo, hi = array.min(), array.max()
    if hi > lo:
        # Exact original expression for every positive range; no epsilon is added.
        return (array - lo) / (hi - lo)
    warnings.warn(f'{label} is constant; its normalized contrast is defined as zero', RuntimeWarning)
    return np.zeros_like(array, dtype=np.result_type(array.dtype, np.float32))


def percentile_stretch(array, low, high, label='Percentile stretch'):
    require_finite(array, label)
    low, high = np.asarray(low), np.asarray(high)
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high < low):
        raise ValueError(f'{label}: invalid percentile bounds')
    span = high - low
    if np.all(span > 0):
        return (array - low) / span
    warnings.warn(f'{label}: zero percentile span; affected channels keep their input values before the original clipping/LUT', RuntimeWarning)
    result = np.array(array, dtype=np.result_type(array.dtype, low.dtype, high.dtype, np.float32), copy=True)
    return np.divide(array - low, span, out=result, where=span > 0)


def relative_change(new, old):
    require_finite(new, 'ROP new transmission')
    require_finite(old, 'ROP previous transmission')
    numerator = np.linalg.norm(new - old, ord='fro')
    denominator = np.linalg.norm(old, ord='fro')
    if denominator > 0:
        return numerator / denominator
    return 0.0 if numerator == 0 else float('inf')
