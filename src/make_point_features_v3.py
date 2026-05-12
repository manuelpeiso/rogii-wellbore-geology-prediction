from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from make_point_features import FEATURE_COLUMNS as BASE_FEATURE_COLUMNS
from make_point_features import build_split as build_split_v1


WINDOWS = [21, 101]

EXTRA_FEATURE_COLUMNS = [
    "tw_anchor_shift_21",
    "tw_match_tvt_21",
    "tw_match_error_21",
    "tw_match_minus_baseline_21",
    "tw_match_tvt_corrected_21",
    "tw_anchor_shift_101",
    "tw_match_tvt_101",
    "tw_match_error_101",
    "tw_match_minus_baseline_101",
    "tw_match_tvt_corrected_101",
]

FEATURE_COLUMNS_V3 = BASE_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS


def well_id(path: Path) -> str:
    return path.name.split("__")[0]


def local_gr_descriptors(gr: pd.Series, depth: pd.Series, window: int) -> pd.DataFrame:
    half = max(window // 2, 1)
    depth_delta = depth.shift(-half) - depth.shift(half)
    slope = (gr.shift(-half) - gr.shift(half)).div(depth_delta)
    return pd.DataFrame(
        {
            "gr_mean": gr.rolling(window, center=True, min_periods=1).mean(),
            "gr_std": gr.rolling(window, center=True, min_periods=2).std().fillna(0.0),
            "gr_slope": slope.replace([np.inf, -np.inf], 0.0).fillna(0.0),
            "gr_value": gr,
        }
    )


def nearest_typewell_pattern(
    horizontal_gr: pd.Series,
    horizontal_md: pd.Series,
    typewell: pd.DataFrame,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    typewell_sorted = typewell.sort_values("TVT").reset_index(drop=True)
    type_gr = typewell_sorted["GR"].interpolate(limit_direction="both")

    horizontal_desc = local_gr_descriptors(horizontal_gr, horizontal_md, window)
    type_desc = local_gr_descriptors(type_gr, typewell_sorted["TVT"], window)

    center = type_desc.mean(axis=0)
    scale = type_desc.std(axis=0).replace(0.0, 1.0)
    type_x = ((type_desc - center) / scale).to_numpy(dtype=float)
    horizontal_x = ((horizontal_desc - center) / scale).to_numpy(dtype=float)

    matcher = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
    matcher.fit(type_x)
    distances, indices = matcher.kneighbors(horizontal_x, return_distance=True)

    match_tvt = typewell_sorted["TVT"].to_numpy(dtype=float)[indices[:, 0]]
    match_error = distances[:, 0]
    return match_tvt, match_error


def add_v3_features_for_split(base: pd.DataFrame, raw_dir: Path, split: str) -> pd.DataFrame:
    frames = []
    typewell_cache = {
        well_id(path): pd.read_csv(path)
        for path in sorted((raw_dir / split).glob("*__typewell.csv"))
    }

    for wid, well in base.groupby("well_id", sort=False):
        well = well.sort_values("row_index").copy()
        typewell = typewell_cache[wid]
        horizontal_gr = well["gr"].interpolate(limit_direction="both")
        horizontal_md = well["MD"]

        known = well["is_prediction_row"] == 0
        for window in WINDOWS:
            match_tvt, match_error = nearest_typewell_pattern(
                horizontal_gr=horizontal_gr,
                horizontal_md=horizontal_md,
                typewell=typewell,
                window=window,
            )
            # In known rows baseline_tvt follows TVT_input, and V1 features do not keep raw TVT_input.
            raw_shift = well.loc[known, "baseline_tvt"].to_numpy(dtype=float) - match_tvt[known.to_numpy()]
            anchor_shift = float(np.nanmedian(raw_shift)) if len(raw_shift) else 0.0

            well[f"tw_anchor_shift_{window}"] = anchor_shift
            well[f"tw_match_tvt_{window}"] = match_tvt
            well[f"tw_match_error_{window}"] = match_error
            well[f"tw_match_minus_baseline_{window}"] = match_tvt - well["baseline_tvt"].to_numpy(dtype=float)
            well[f"tw_match_tvt_corrected_{window}"] = match_tvt + anchor_shift

        frames.append(well)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_v3"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = add_v3_features_for_split(build_split_v1(args.raw_dir, "train"), args.raw_dir, "train")
    test = add_v3_features_for_split(build_split_v1(args.raw_dir, "test"), args.raw_dir, "test")

    train_prediction_rows = train[train["is_prediction_row"] == 1].copy()
    test_prediction_rows = test[test["is_prediction_row"] == 1].copy()

    train_prediction_rows.to_csv(args.output_dir / "train_point_features_v3.csv", index=False)
    test_prediction_rows.to_csv(args.output_dir / "test_point_features_v3.csv", index=False)

    print(f"Train point rows: {len(train_prediction_rows):,}")
    print(f"Test point rows: {len(test_prediction_rows):,}")
    print(f"Feature columns V3: {len(FEATURE_COLUMNS_V3)}")
    print(f"Wrote {args.output_dir / 'train_point_features_v3.csv'}")
    print(f"Wrote {args.output_dir / 'test_point_features_v3.csv'}")


if __name__ == "__main__":
    main()
