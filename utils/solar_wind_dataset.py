# ------------------------------------------------------------------------------------------------------------------------
# FANBEATS: Frequency and Attention-augmented Neural Basis Expansion Analysis for interpretable Time Series forecasting
#
# Copyright (c) 2026 Danish Abbas
#
# This file is part of the FANBEATS project.
# Licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License (CC BY-NC 4.0).
#
# See the LICENSE file in the root directory for full details.
# ------------------------------------------------------------------------------------------------------------------------

# utils/solar_wind_dataset.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


def _read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(p)


def _read_speed_array(path: str, column: str = "speed") -> np.ndarray:
    df = _read_csv(path)

    if column in df.columns:
        arr = df[column].values
    elif df.shape[1] == 1:
        arr = df.iloc[:, 0].values
    else:
        raise KeyError(
            f"Column '{column}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    return np.asarray(arr, dtype=np.float32).flatten()


def _read_datetime_series(
    path: str,
    datetime_col: Optional[str] = "date_time",
    fallback_start: Optional[str] = None,
    freq: str = "h",
) -> pd.DatetimeIndex:

    df = _read_csv(path)

    if datetime_col is not None and datetime_col in df.columns:
        dt = pd.to_datetime(df[datetime_col])
        dt = dt.sort_values().reset_index(drop=True)
        return pd.DatetimeIndex(dt)

    if fallback_start is None:
        raise ValueError(
            "No datetime column found/provided and no fallback_start supplied."
        )

    return pd.date_range(
        start=pd.Timestamp(fallback_start),
        periods=len(df),
        freq=freq,
    )


def load_train_val_test(
    train_path: str,
    val_path: str,
    test_path: str,
    column: str = "speed",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load train, validation, and test 1D arrays."""
    train = _read_speed_array(train_path, column=column)
    val = _read_speed_array(val_path, column=column)
    test = _read_speed_array(test_path, column=column)
    return train, val, test


def load_train_val_test_datetimes(
    train_path: str,
    val_path: str,
    test_path: str,
    datetime_col: Optional[str] = "date_time",
    train_start: Optional[str] = None,
    val_start: Optional[str] = None,
    test_start: Optional[str] = None,
    freq: str = "h",
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:

    train_dt = _read_datetime_series(train_path, datetime_col, train_start, freq=freq)
    val_dt = _read_datetime_series(val_path, datetime_col, val_start, freq=freq)
    test_dt = _read_datetime_series(test_path, datetime_col, test_start, freq=freq)
    return train_dt, val_dt, test_dt


def to_timeseries_list(arr: np.ndarray) -> List[np.ndarray]:
    if arr.ndim != 1:
        raise ValueError("Input array must be 1D.")
    return [np.asarray(arr, dtype=np.float32)]


def fit_standard_scaler(arr: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(arr, dtype=np.float32)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))
    if sigma <= 0:
        sigma = 1.0
    return mu, sigma


def apply_standard_scaler(arr: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return ((arr - mu) / sigma).astype(np.float32)


def inverse_standard_scaler(arr: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return (arr * sigma + mu).astype(np.float32)


def sliding_windows(
    series: np.ndarray,
    insample: int,
    outsample: int,
    step: int = 1,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:

    series = np.asarray(series, dtype=np.float32).flatten()

    if insample <= 0 or outsample <= 0:
        raise ValueError("insample and outsample must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")
    if len(series) < insample + outsample:
        raise ValueError(
            f"Series length {len(series)} is too short for "
            f"insample={insample}, outsample={outsample}."
        )

    windows: List[Tuple[np.ndarray, np.ndarray, int]] = []
    last_start = len(series) - insample - outsample

    for start in range(0, last_start + 1, step):
        x = series[start : start + insample]
        y = series[start + insample : start + insample + outsample]
        windows.append((x.astype(np.float32), y.astype(np.float32), start))

    return windows


def batch_windows(
    windows: Sequence[Tuple[np.ndarray, np.ndarray, int]],
    batch_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        x_batch = np.stack([w[0] for w in chunk], axis=0).astype(np.float32)
        y_batch = np.stack([w[1] for w in chunk], axis=0).astype(np.float32)
        start_idx_batch = np.asarray([w[2] for w in chunk], dtype=np.int64)
        x_mask = np.ones_like(x_batch, dtype=np.float32)
        yield x_batch, x_mask, y_batch, start_idx_batch


def evaluate_model_on_windows(
    model,
    windows: Sequence[Tuple[np.ndarray, np.ndarray, int]],
    to_tensor_fn,
    inverse_transform_fn,
    mu: float,
    sigma: float,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    import torch as t

    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    starts: List[np.ndarray] = []

    model.eval()
    with t.no_grad():
        for x_batch, x_mask, y_batch, start_idx_batch in batch_windows(windows, batch_size=batch_size):
            x_t = to_tensor_fn(x_batch)
            x_mask_t = to_tensor_fn(x_mask)
            forecast_t = model(x_t, x_mask_t)
            forecast_np = forecast_t.detach().cpu().numpy().astype(np.float32)

            preds.append(inverse_transform_fn(forecast_np, mu, sigma))
            trues.append(inverse_transform_fn(y_batch, mu, sigma))
            starts.append(start_idx_batch)

    return (
        np.concatenate(preds, axis=0),
        np.concatenate(trues, axis=0),
        np.concatenate(starts, axis=0),
    )


def horizon_target_predictions_dataframe(
    preds: np.ndarray,
    trues: np.ndarray,
    start_indices: np.ndarray,
    split_name: str,
    datetime_index: pd.DatetimeIndex,
    insample: int,
    horizon: int,
) -> pd.DataFrame:

    preds = np.asarray(preds)
    trues = np.asarray(trues)
    start_indices = np.asarray(start_indices)

    if preds.ndim != 2 or trues.ndim != 2:
        raise ValueError("preds and trues must be 2D arrays [num_windows, horizon].")
    if preds.shape != trues.shape:
        raise ValueError("preds and trues must have the same shape.")
    if preds.shape[1] != horizon:
        raise ValueError(
            f"Expected preds.shape[1] == horizon ({horizon}), got {preds.shape[1]}"
        )

    rows = []
    final_step_idx = horizon - 1

    for window_idx in range(preds.shape[0]):
        start_idx = int(start_indices[window_idx])
        forecast_origin_idx = start_idx + insample - 1
        target_idx = start_idx + insample + final_step_idx

        if target_idx >= len(datetime_index):
            raise IndexError(
                f"Computed target_idx={target_idx} exceeds datetime index length={len(datetime_index)}"
            )

        rows.append(
            {
                "split": split_name,
                "window_idx": window_idx,
                "forecast_origin_idx": forecast_origin_idx,
                "horizon_step": horizon,
                "date_time": datetime_index[target_idx],
                "y_true": float(trues[window_idx, final_step_idx]),
                "y_pred": float(preds[window_idx, final_step_idx]),
            }
        )

    return pd.DataFrame(rows).sort_values("date_time").reset_index(drop=True)


def compute_metrics_from_predictions_df(df: pd.DataFrame) -> Dict[str, float]:
    y_true = df["y_true"].to_numpy(dtype=np.float64)
    y_pred = df["y_pred"].to_numpy(dtype=np.float64)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        cc = float(pearsonr(y_true, y_pred)[0]) 
    else:
        cc = 0.0  # safe fallback

    return {
        "RMSE": rmse,
        "MAE": mae,
        "CC": cc,
    }


def metrics_dataframe(metrics: Dict[str, float], split_name: str, experiment_tag: str) -> pd.DataFrame:
    row = {"split": split_name, "experiment_tag": experiment_tag}
    row.update(metrics)
    return pd.DataFrame([row])


def save_dataframe_csv(path: str, df: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)