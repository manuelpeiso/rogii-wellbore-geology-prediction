from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "row_index",
    "n_rows",
    "row_from_ps",
    "frac_from_ps",
    "MD",
    "md_from_ps",
    "X",
    "Y",
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
    "nearest_typewell_tvt_by_gr",
    "nearest_typewell_gr_diff",
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


def nearest_typewell_by_gr(typewell: pd.DataFrame, gr_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    type_gr = typewell["GR"].to_numpy(dtype=float)
    type_tvt = typewell["TVT"].to_numpy(dtype=float)
    nearest_tvt = np.empty(len(gr_values), dtype=float)
    nearest_diff = np.empty(len(gr_values), dtype=float)

    valid_type = np.isfinite(type_gr)
    type_gr = type_gr[valid_type]
    type_tvt = type_tvt[valid_type]

    for i, gr in enumerate(gr_values):
        if not np.isfinite(gr) or len(type_gr) == 0:
            nearest_tvt[i] = np.nan
            nearest_diff[i] = np.nan
            continue
        idx = int(np.argmin(np.abs(type_gr - gr)))
        nearest_tvt[i] = type_tvt[idx]
        nearest_diff[i] = abs(type_gr[idx] - gr)

    return nearest_tvt, nearest_diff


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
    nearest_typewell_tvt, nearest_typewell_gr_diff = nearest_typewell_by_gr(
        typewell, gr.to_numpy(dtype=float)
    )

    known_tvt = horizontal["TVT_input"].dropna()
    first_tvt = float(known_tvt.iloc[0]) if len(known_tvt) else np.nan
    last_tvt = float(known_tvt.iloc[-1]) if len(known_tvt) else np.nan

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
            "X": horizontal["X"],
            "Y": horizontal["Y"],
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
            "nearest_typewell_tvt_by_gr": nearest_typewell_tvt,
            "nearest_typewell_gr_diff": nearest_typewell_gr_diff,
        }
    )
    features["xy_dist_from_ps"] = np.sqrt(features["x_from_ps"] ** 2 + features["y_from_ps"] ** 2)
    features["is_prediction_row"] = tvt_input_missing.astype(int)

    if "TVT" in horizontal.columns:
        features["target_tvt"] = horizontal["TVT"]

    return features


def build_split(raw_dir: Path, split: str) -> pd.DataFrame:
    frames = []
    for horizontal_path in sorted((raw_dir / split).glob("*__horizontal_well.csv")):
        well_id = horizontal_path.name.split("__")[0]
        typewell_path = raw_dir / split / f"{well_id}__typewell.csv"
        frames.append(build_well_features(horizontal_path, typewell_path, split))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = build_split(args.raw_dir, "train")
    test = build_split(args.raw_dir, "test")

    train_prediction_rows = train[train["is_prediction_row"] == 1].copy()
    test_prediction_rows = test[test["is_prediction_row"] == 1].copy()

    train_prediction_rows.to_csv(args.output_dir / "train_point_features.csv", index=False)
    test_prediction_rows.to_csv(args.output_dir / "test_point_features.csv", index=False)

    print(f"Train point rows: {len(train_prediction_rows):,}")
    print(f"Test point rows: {len(test_prediction_rows):,}")
    print(f"Feature columns: {len(FEATURE_COLUMNS)}")
    print(f"Wrote {args.output_dir / 'train_point_features.csv'}")
    print(f"Wrote {args.output_dir / 'test_point_features.csv'}")


if __name__ == "__main__":
    main()
