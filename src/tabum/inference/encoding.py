"""DataFrame -> float32 matrix encoding for tabUM inputs.

Strings/categoricals become integer codes (non-ordinal is fine — the model
treats them as unordered), datetimes become int64 nanoseconds, missing values
stay NaN (handled natively downstream). This is encoding, not cleaning: no
imputation, no scaling, no outlier handling — the model does that internally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_numeric(df: pd.DataFrame) -> np.ndarray:
    out = np.empty(df.shape, dtype=np.float32)
    for j, c in enumerate(df.columns):
        col = df[c]
        if isinstance(col.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(
            col.dtype
        ) or pd.api.types.is_string_dtype(col.dtype):
            codes = pd.factorize(col)[0].astype(np.float32)
            codes[codes < 0] = np.nan  # factorize marks missing as -1
            out[:, j] = codes
        elif pd.api.types.is_datetime64_any_dtype(col.dtype):
            vals = col.astype("int64").to_numpy(dtype=np.float64)
            vals[col.isna().to_numpy()] = np.nan
            out[:, j] = (vals / 1e9).astype(np.float32)  # seconds: keep fp32 range sane
        elif pd.api.types.is_timedelta64_dtype(col.dtype):
            out[:, j] = (col.astype("int64").to_numpy(dtype=np.float64) / 1e9).astype(
                np.float32
            )
        else:
            out[:, j] = col.to_numpy(dtype=np.float32, na_value=np.nan)
    return out
