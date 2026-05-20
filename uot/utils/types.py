from __future__ import annotations

import jax
import numpy as np
from typing import TypeAlias, Literal

ArrayLike: TypeAlias = jax.Array | np.ndarray
MeasureMode: TypeAlias = Literal["grid", "point_cloud", "auto"]
ShareMode: TypeAlias = Literal["same", "union", "intersection", "first"]
Backend: TypeAlias = Literal["auto", "jax", "numpy"]
