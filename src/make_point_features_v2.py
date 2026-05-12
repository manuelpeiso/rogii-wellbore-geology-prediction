from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from make_point_features import FEATURE_COLUMNS as BASE_FEATURE_COLUMNS
from make_point_features import build_split as build_split_v1


EXTRA_FEATURE_COLUMNS = [
    "baseline_delta_from_last",
    "baseline_slope_from_ps",
    "md_norm",
    "z_delta_1",
    "z_delta_10",
    "gr_roll_mean_101",
    "gr_roll_std_101",
    "gr_ewm_mean_20",
    "gr_gradient_md",
    "gr_residual_roll_mean_51",
    "nearest_typewell_tvt_minus_baseline",
    "baseline_to_typewell_min",
    "baseline_to_typewell_max",
]

FEATURE_COLUMNS_V2 = BASE_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS


def add_v2_features(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, well in df.groupby("well_id", sort=False):
        well = well.sort_values("row_index").copy()
        md_span = max(float(well["MD"].max() - well["MD"].min()), 1.0)
        md_from_ps = well["md_from_ps"].replace(0, np.nan)
        gr_residual = well["gr_minus_typewell_gr_at_baseline"]

        well["baseline_delta_from_last"] = well["baseline_tvt"] - well["last_tvt_input"]
        well["baseline_slope_from_ps"] = (well["baseline_tvt"] - well["last_tvt_input"]) / md_from_ps
        well["baseline_slope_from_ps"] = well["baseline_slope_from_ps"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        well["md_norm"] = (well["MD"] - well["MD"].min()) / md_span
        well["z_delta_1"] = well["Z"].diff(1).fillna(0.0)
        well["z_delta_10"] = well["Z"].diff(10).fillna(0.0)
        well["gr_roll_mean_101"] = well["gr"].rolling(101, center=True, min_periods=1).mean()
        well["gr_roll_std_101"] = well["gr"].rolling(101, center=True, min_periods=2).std().fillna(0.0)
        well["gr_ewm_mean_20"] = well["gr"].ewm(span=20, adjust=False).mean()
        well["gr_gradient_md"] = well["gr"].diff().div(well["MD"].diff()).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        well["gr_residual_roll_mean_51"] = gr_residual.rolling(51, center=True, min_periods=1).mean()
        well["nearest_typewell_tvt_minus_baseline"] = well["nearest_typewell_tvt_by_gr"] - well["baseline_tvt"]
        well["baseline_to_typewell_min"] = well["baseline_tvt"] - well["typewell_tvt_min"]
        well["baseline_to_typewell_max"] = well["typewell_tvt_max"] - well["baseline_tvt"]
        frames.append(well)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_v2"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = add_v2_features(build_split_v1(args.raw_dir, "train"))
    test = add_v2_features(build_split_v1(args.raw_dir, "test"))

    train_prediction_rows = train[train["is_prediction_row"] == 1].copy()
    test_prediction_rows = test[test["is_prediction_row"] == 1].copy()

    train_prediction_rows.to_csv(args.output_dir / "train_point_features_v2.csv", index=False)
    test_prediction_rows.to_csv(args.output_dir / "test_point_features_v2.csv", index=False)

    print(f"Train point rows: {len(train_prediction_rows):,}")
    print(f"Test point rows: {len(test_prediction_rows):,}")
    print(f"Feature columns V2: {len(FEATURE_COLUMNS_V2)}")
    print(f"Wrote {args.output_dir / 'train_point_features_v2.csv'}")
    print(f"Wrote {args.output_dir / 'test_point_features_v2.csv'}")


if __name__ == "__main__":
    main()

