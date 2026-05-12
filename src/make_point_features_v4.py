from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# These wells exist in raw/train/ with full TVT but are also the Kaggle test wells.
# Excluding them from the train CSV prevents data leakage.
TEST_WELL_IDS = {"000d7d20", "00bbac68", "00e12e8b"}

FEATURE_COLUMNS = [
    "row_index",
    "n_rows",
    "row_from_ps",
    "frac_from_ps",
    "MD",
    "md_from_ps",
    "Z",
    "x_from_ps",
    "y_from_ps",
    "z_from_ps",
    "xy_dist_from_ps",
    "gr",
    "gr_was_missing",
    "gr_from_ps",
    "gr_roll_mean_11",
    "gr_roll_mean_51",
    "gr_roll_std_51",
    "gr_delta_1",
    "gr_delta_10",
    "last_tvt_input",
    "first_tvt_input",
    "tvt_input_range",
    "tvt_slope_last_25",
    "tvt_slope_last_100",
    "baseline_tvt",
    "typewell_tvt_min",
    "typewell_tvt_max",
    "typewell_gr_at_baseline_tvt",
    "gr_minus_typewell_gr_at_baseline",
    # Replaced by window-based matching in v4:
    "nearest_typewell_tvt_by_gr_window",
    "nearest_typewell_window_mse",
    "typewell_gr_at_window_match",
    "gr_minus_typewell_gr_at_window_match",
    # New v4: GR lag/lead values (5 rows back and forward)
    "gr_lag_1",
    "gr_lag_2",
    "gr_lag_3",
    "gr_lag_4",
    "gr_lag_5",
    "gr_lead_1",
    "gr_lead_2",
    "gr_lead_3",
    "gr_lead_4",
    "gr_lead_5",
    # New v4: local window stats (gr_lag_2 .. gr_lead_2, i.e. 5-point window centered)
    "gr_local_mean_5",
    "gr_local_std_5",
    # New v4: GR deviation from regional background (not from PS reference point)
    "gr_anomaly",
    # New v4: disagreement between the two TVT estimators
    "typewell_tvt_vs_baseline",
]


def interpolate_with_extrapolation(x: np.ndarray, y: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    x_known = x[valid]
    y_known = y[valid]

    if len(x_known) == 0:
        return np.full_like(x_query, np.nan, dtype=float)
    if len(x_known) == 1:
        return np.full_like(x_query, y_known[0], dtype=float)

    order = np.argsort(x_known)
    x_known = x_known[order]
    y_known = y_known[order]
    pred = np.interp(x_query, x_known, y_known)

    left = x_query < x_known[0]
    if left.any():
        slope = (y_known[1] - y_known[0]) / (x_known[1] - x_known[0])
        pred[left] = y_known[0] + slope * (x_query[left] - x_known[0])

    right = x_query > x_known[-1]
    if right.any():
        slope = (y_known[-1] - y_known[-2]) / (x_known[-1] - x_known[-2])
        pred[right] = y_known[-1] + slope * (x_query[right] - x_known[-1])

    return pred


def slope_last(values: pd.Series, x: pd.Series, n: int) -> float:
    known = values.notna()
    if known.sum() < 2:
        return 0.0
    y = values.loc[known].tail(n).to_numpy(dtype=float)
    x_tail = x.loc[known].tail(n).to_numpy(dtype=float)
    if len(y) < 2 or np.isclose(x_tail[-1], x_tail[0]):
        return 0.0
    return float((y[-1] - y[0]) / (x_tail[-1] - x_tail[0]))


def nearest_typewell_by_gr_window(
    typewell: pd.DataFrame,
    gr_windows: np.ndarray,  # shape (n_rows, 5): [lag_2, lag_1, gr, lead_1, lead_2]
    batch_size: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Template matching: find the typewell position whose 5-point GR window
    best matches the query window, using mean squared error as distance.
    This disambiguates repeated GR values by requiring the local sequence to match.
    Returns matched TVT, MSE of best match, and the central GR of the matched window.
    """
    type_gr = typewell["GR"].to_numpy(dtype=float)
    type_tvt = typewell["TVT"].to_numpy(dtype=float)

    valid_type = np.isfinite(type_gr)
    type_gr = type_gr[valid_type]
    type_tvt = type_tvt[valid_type]
    m = len(type_gr)

    half = 2  # window half-width → 5-point window
    # Build typewell sliding windows: shape (m - 2*half, 5)
    tw_centers = np.arange(half, m - half)
    tw_windows = np.stack([type_gr[j - half: j + half + 1] for j in tw_centers])
    tw_tvt_centers = type_tvt[tw_centers]
    tw_gr_centers = type_gr[tw_centers]  # central GR of each typewell window

    n = len(gr_windows)
    matched_tvt = np.empty(n, dtype=float)
    matched_mse = np.empty(n, dtype=float)
    matched_gr = np.empty(n, dtype=float)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = gr_windows[start:end]  # (batch, 5)
        if not np.all(np.isfinite(batch)):
            # Fall back row by row for the few NaN cases
            for k in range(end - start):
                w = batch[k]
                if not np.all(np.isfinite(w)) or m == 0:
                    matched_tvt[start + k] = np.nan
                    matched_mse[start + k] = np.nan
                    matched_gr[start + k] = np.nan
                else:
                    mse = np.mean((tw_windows - w) ** 2, axis=1)
                    best = int(np.argmin(mse))
                    matched_tvt[start + k] = tw_tvt_centers[best]
                    matched_mse[start + k] = float(mse[best])
                    matched_gr[start + k] = tw_gr_centers[best]
        else:
            # Vectorised batch: (batch, n_tw, 5)
            diff = batch[:, None, :] - tw_windows[None, :, :]
            mse = (diff ** 2).mean(axis=-1)  # (batch, n_tw)
            best = mse.argmin(axis=1)  # (batch,)
            matched_tvt[start:end] = tw_tvt_centers[best]
            matched_mse[start:end] = mse[np.arange(end - start), best]
            matched_gr[start:end] = tw_gr_centers[best]

    return matched_tvt, matched_mse, matched_gr


def build_well_features(horizontal_path: Path, typewell_path: Path, split: str) -> pd.DataFrame:
    well_id = horizontal_path.name.split("__")[0]
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)

    row_index = np.arange(len(horizontal))
    tvt_input_missing = horizontal["TVT_input"].isna()
    ps_row = int(tvt_input_missing.idxmax()) if tvt_input_missing.any() else len(horizontal)
    ps_row_safe = min(ps_row, len(horizontal) - 1)

    gr_raw = horizontal["GR"]
    gr = gr_raw.interpolate(limit_direction="both")
    md = horizontal["MD"]
    baseline_tvt = interpolate_with_extrapolation(
        md.to_numpy(dtype=float),
        horizontal["TVT_input"].to_numpy(dtype=float),
        md.to_numpy(dtype=float),
    )

    typewell_sorted = typewell.sort_values("TVT")
    typewell_gr_at_baseline = interpolate_with_extrapolation(
        typewell_sorted["TVT"].to_numpy(dtype=float),
        typewell_sorted["GR"].to_numpy(dtype=float),
        baseline_tvt,
    )

    known_tvt = horizontal["TVT_input"].dropna()
    first_tvt = float(known_tvt.iloc[0]) if len(known_tvt) else np.nan
    last_tvt = float(known_tvt.iloc[-1]) if len(known_tvt) else np.nan

    # GR lags: fill borders with the nearest edge value so there are no NaNs
    gr_first = float(gr.iloc[0])
    gr_last = float(gr.iloc[-1])
    gr_lag_1 = gr.shift(1).fillna(gr_first)
    gr_lag_2 = gr.shift(2).fillna(gr_first)
    gr_lag_3 = gr.shift(3).fillna(gr_first)
    gr_lag_4 = gr.shift(4).fillna(gr_first)
    gr_lag_5 = gr.shift(5).fillna(gr_first)

    # GR leads: fill borders with the last edge value
    gr_lead_1 = gr.shift(-1).fillna(gr_last)
    gr_lead_2 = gr.shift(-2).fillna(gr_last)
    gr_lead_3 = gr.shift(-3).fillna(gr_last)
    gr_lead_4 = gr.shift(-4).fillna(gr_last)
    gr_lead_5 = gr.shift(-5).fillna(gr_last)

    # Local 5-point window: [lag_2, lag_1, current, lead_1, lead_2]
    gr_local_window = pd.concat(
        [gr_lag_2, gr_lag_1, gr, gr_lead_1, gr_lead_2], axis=1
    )
    gr_local_mean_5 = gr_local_window.mean(axis=1)
    gr_local_std_5 = gr_local_window.std(axis=1).fillna(0.0)

    # GR anomaly: deviation from the 51-row rolling background
    gr_anomaly = gr - gr.rolling(51, center=True, min_periods=1).mean()

    # Window-based template matching against typewell: 5-point GR sequence match
    gr_window_matrix = gr_local_window.to_numpy(dtype=float)  # (n_rows, 5)
    nearest_typewell_tvt_window, nearest_typewell_window_mse, typewell_gr_at_window_match = (
        nearest_typewell_by_gr_window(typewell, gr_window_matrix)
    )

    # Discrepancy between the two TVT estimators: typewell window-match vs linear extrapolation
    typewell_tvt_vs_baseline = nearest_typewell_tvt_window - baseline_tvt

    # GR difference: actual GR vs the central GR of the matched typewell window
    gr_minus_typewell_gr_at_window_match = gr.to_numpy(dtype=float) - typewell_gr_at_window_match

    features = pd.DataFrame(
        {
            "split": split,
            "well_id": well_id,
            "row_index": row_index,
            "n_rows": len(horizontal),
            "row_from_ps": row_index - ps_row,
            "frac_from_ps": (row_index - ps_row) / max(len(horizontal) - ps_row, 1),
            "MD": md,
            "md_from_ps": md - float(md.iloc[ps_row_safe]),
            "Z": horizontal["Z"],
            "x_from_ps": horizontal["X"] - float(horizontal["X"].iloc[ps_row_safe]),
            "y_from_ps": horizontal["Y"] - float(horizontal["Y"].iloc[ps_row_safe]),
            "z_from_ps": horizontal["Z"] - float(horizontal["Z"].iloc[ps_row_safe]),
            "gr": gr,
            "gr_was_missing": gr_raw.isna().astype(int),
            "gr_from_ps": gr - float(gr.iloc[ps_row_safe]),
            "gr_roll_mean_11": gr.rolling(11, center=True, min_periods=1).mean(),
            "gr_roll_mean_51": gr.rolling(51, center=True, min_periods=1).mean(),
            "gr_roll_std_51": gr.rolling(51, center=True, min_periods=2).std().fillna(0.0),
            "gr_delta_1": gr.diff(1).fillna(0.0),
            "gr_delta_10": gr.diff(10).fillna(0.0),
            "last_tvt_input": last_tvt,
            "first_tvt_input": first_tvt,
            "tvt_input_range": last_tvt - first_tvt,
            "tvt_slope_last_25": slope_last(horizontal["TVT_input"], md, 25),
            "tvt_slope_last_100": slope_last(horizontal["TVT_input"], md, 100),
            "baseline_tvt": baseline_tvt,
            "typewell_tvt_min": typewell["TVT"].min(),
            "typewell_tvt_max": typewell["TVT"].max(),
            "typewell_gr_at_baseline_tvt": typewell_gr_at_baseline,
            "gr_minus_typewell_gr_at_baseline": gr.to_numpy(dtype=float) - typewell_gr_at_baseline,
            "nearest_typewell_tvt_by_gr_window": nearest_typewell_tvt_window,
            "nearest_typewell_window_mse": nearest_typewell_window_mse,
            "typewell_gr_at_window_match": typewell_gr_at_window_match,
            "gr_minus_typewell_gr_at_window_match": gr_minus_typewell_gr_at_window_match,
            "gr_lag_1": gr_lag_1,
            "gr_lag_2": gr_lag_2,
            "gr_lag_3": gr_lag_3,
            "gr_lag_4": gr_lag_4,
            "gr_lag_5": gr_lag_5,
            "gr_lead_1": gr_lead_1,
            "gr_lead_2": gr_lead_2,
            "gr_lead_3": gr_lead_3,
            "gr_lead_4": gr_lead_4,
            "gr_lead_5": gr_lead_5,
            "gr_local_mean_5": gr_local_mean_5,
            "gr_local_std_5": gr_local_std_5,
            "gr_anomaly": gr_anomaly,
            "typewell_tvt_vs_baseline": typewell_tvt_vs_baseline,
        }
    )
    features["xy_dist_from_ps"] = np.sqrt(features["x_from_ps"] ** 2 + features["y_from_ps"] ** 2)
    features["is_prediction_row"] = tvt_input_missing.astype(int)

    if "TVT" in horizontal.columns:
        features["target_tvt"] = horizontal["TVT"]

    return features


def build_split(raw_dir: Path, split: str, exclude_wells: set[str] | None = None) -> pd.DataFrame:
    frames = []
    for horizontal_path in sorted((raw_dir / split).glob("*__horizontal_well.csv")):
        well_id = horizontal_path.name.split("__")[0]
        if exclude_wells and well_id in exclude_wells:
            continue
        typewell_path = raw_dir / split / f"{well_id}__typewell.csv"
        frames.append(build_well_features(horizontal_path, typewell_path, split))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_v4"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = build_split(args.raw_dir, "train", exclude_wells=TEST_WELL_IDS)
    test = build_split(args.raw_dir, "test")

    train_prediction_rows = train[train["is_prediction_row"] == 1].copy()
    test_prediction_rows = test[test["is_prediction_row"] == 1].copy()

    train_prediction_rows.to_csv(args.output_dir / "train_point_features.csv", index=False)
    test_prediction_rows.to_csv(args.output_dir / "test_point_features.csv", index=False)

    print(f"Train point rows: {len(train_prediction_rows):,}")
    print(f"Test point rows: {len(test_prediction_rows):,}")
    print(f"Feature columns: {len(FEATURE_COLUMNS)} (33 base + 14 new)")
    print(f"Wrote {args.output_dir / 'train_point_features.csv'}")
    print(f"Wrote {args.output_dir / 'test_point_features.csv'}")


if __name__ == "__main__":
    main()
