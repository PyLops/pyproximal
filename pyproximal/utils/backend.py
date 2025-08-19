__all__ = [
    "cp_dtype",
]

import numpy as np
from pylops.utils import deps

if deps.cupy_enabled:
    import cupy as cp


cp_dtype = np.ndarray if not deps.cupy_enabled else cp.ndarray
