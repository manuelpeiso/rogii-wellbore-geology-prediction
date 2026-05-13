from __future__ import annotations

"""v5 feature engineering.

Changes vs v4
─────────────
* GR lags/leads extended from 5 → 15 rows (20 extra features)
* Larger local GR window stats: 11-point and 21-point (4 extra features)
* gr_delta_5  (1 extra feature)
* Neighbor-well features: for each well, 3 nearest wells (by PS X/Y position)
  contribute dist, last_tvt_input and tvt_range → 9 extra features
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Wells with full TVT in raw/train/ but used as Kaggle test wells → exclude from train.
TEST_WELL_IDS = {"000d7d20", "00bbac68", "00e12e8b"}

N_NEIGHBORS = 3  # nearest wells to include as context


FEATURE_COLUMNS = [
    # ── same 47 features as v4 (baseline for convergence experiments) ─────────
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
    "gr_delta_lead_1",
    "gr_delta_lead_10",
    "X",
    "Y",
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
    "nearest_typewell_tvt_by_gr_window",
    "nearest_typewell_window_mse",
    "typewell_gr_at_window_match",
    "gr_minus_typewell_gr_at_window_match",
    "typewell_tvt_vs_baseline",
    "nearest_typewell_tvt_by_gr_smooth",
    "nearest_typewell_smooth_mse",
    "typewell_tvt_vs_baseline_smooth",
    "nearest_typewell_tvt_by_gr_delta",
    "nearest_typewell_delta_mse",
    "typewell_tvt_vs_baseline_delta",
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
    "gr_local_mean_5",
    "gr_local_std_5",
    "gr_anomaly",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_with_extrapolation(
    x: np.ndarray, y: np.ndarray, x_query: np.ndarray
) -> np.ndarray:
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
    gr_windows: np.ndarray,   # shape (n_rows, 2*half+1)
    half: int = 2,
    batch_size: int = 200,
    delta_windows: np.ndarray | None = None,  # shape (n_rows, 2*half+1), query-side GR deltas
    delta_weight: float = 4.0,               # scale factor so deltas contribute ~equally to GR
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Template matching: find the typewell row whose (2*half+1)-point GR window
    best matches the query window (MSE distance).  Returns (matched_tvt, best_mse,
    central_gr_at_match).

    If delta_windows is provided, the MSE is computed over a combined vector
    [GR_window, delta_window * delta_weight], so the match also requires similar
    instantaneous rate of change at each position in the window.
    """
    type_gr = typewell["GR"].to_numpy(dtype=float)
    type_tvt = typewell["TVT"].to_numpy(dtype=float)

    valid_type = np.isfinite(type_gr)
    type_gr = type_gr[valid_type]
    type_tvt = type_tvt[valid_type]
    m = len(type_gr)
    tw_centers = np.arange(half, m - half)
    tw_windows = np.stack([type_gr[j - half: j + half + 1] for j in tw_centers])
    tw_tvt_centers = type_tvt[tw_centers]
    tw_gr_centers = type_gr[tw_centers]

    # If delta windows provided, build typewell delta windows and concatenate
    if delta_windows is not None:
        type_delta = np.diff(type_gr, prepend=type_gr[0])
        tw_delta_windows = np.stack(
            [type_delta[j - half: j + half + 1] for j in tw_centers]
        ) * delta_weight
        tw_match_windows = np.concatenate([tw_windows, tw_delta_windows], axis=1)
    else:
        tw_match_windows = tw_windows

    n = len(gr_windows)
    matched_tvt = np.empty(n, dtype=float)
    matched_mse = np.empty(n, dtype=float)
    matched_gr = np.empty(n, dtype=float)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_gr = gr_windows[start:end]
        if delta_windows is not None:
            batch = np.concatenate(
                [batch_gr, delta_windows[start:end] * delta_weight], axis=1
            )
        else:
            batch = batch_gr
        if not np.all(np.isfinite(batch)):
            for k in range(end - start):
                w = batch[k]
                if not np.all(np.isfinite(w)) or m == 0:
                    matched_tvt[start + k] = np.nan
                    matched_mse[start + k] = np.nan
                    matched_gr[start + k] = np.nan
                else:
                    mse = np.mean((tw_match_windows - w) ** 2, axis=1)
                    best = int(np.argmin(mse))
                    matched_tvt[start + k] = tw_tvt_centers[best]
                    matched_mse[start + k] = float(mse[best])
                    matched_gr[start + k] = tw_gr_centers[best]
        else:
            diff = batch[:, None, :] - tw_match_windows[None, :, :]
            mse = (diff ** 2).mean(axis=-1)
            best = mse.argmin(axis=1)
            matched_tvt[start:end] = tw_tvt_centers[best]
            matched_mse[start:end] = mse[np.arange(end - start), best]
            matched_gr[start:end] = tw_gr_centers[best]

    return matched_tvt, matched_mse, matched_gr


# ─────────────────────────────────────────────────────────────────────────────
# Neighbor-well index
# ─────────────────────────────────────────────────────────────────────────────

def collect_well_positions(raw_dir: Path) -> dict[str, dict]:
    """Scan every horizontal_well CSV in train/ and test/ to collect the PS
    position (X, Y at the pickup-spot row) plus TVT boundary stats.
    Used to build the nearest-neighbor index.
    """
    well_info: dict[str, dict] = {}
    for split in ("train", "test"):
        split_dir = raw_dir / split
        if not split_dir.exists():
            continue
        for hp in sorted(split_dir.glob("*__horizontal_well.csv")):
            well_id = hp.name.split("__")[0]
            df = pd.read_csv(hp, usecols=["X", "Y", "TVT_input"])
            tvt_missing = df["TVT_input"].isna()
            ps_row = int(tvt_missing.idxmax()) if tvt_missing.any() else (len(df) - 1)
            known_tvt = df["TVT_input"].dropna()
            first_tvt = float(known_tvt.iloc[0]) if len(known_tvt) else np.nan
            last_tvt = float(known_tvt.iloc[-1]) if len(known_tvt) else np.nan
            tvt_range = (last_tvt - first_tvt) if (np.isfinite(first_tvt) and np.isfinite(last_tvt)) else 0.0
            well_info[well_id] = {
                "ps_x": float(df["X"].iloc[ps_row]),
                "ps_y": float(df["Y"].iloc[ps_row]),
                "last_tvt": last_tvt,
                "tvt_range": tvt_range,
            }
    return well_info


def build_neighbor_lookup(
    well_info: dict[str, dict], k: int = N_NEIGHBORS
) -> dict[str, dict]:
    """For every well return the K nearest neighbours (by PS X/Y).
    Self is excluded from the neighbour list.
    """
    well_ids = list(well_info.keys())
    positions = np.array([[well_info[w]["ps_x"], well_info[w]["ps_y"]] for w in well_ids])
    tree = cKDTree(positions)

    neighbor_lookup: dict[str, dict] = {}
    for i, wid in enumerate(well_ids):
        dists, idxs = tree.query(positions[i], k=k + 1)  # +1 to exclude self
        nn_dists: list[float] = []
        nn_last_tvts: list[float] = []
        nn_tvt_ranges: list[float] = []
        for d, idx in zip(dists, idxs):
            nid = well_ids[idx]
            if nid == wid:
                continue
            nn_dists.append(float(d))
            nn_last_tvts.append(well_info[nid]["last_tvt"])
            nn_tvt_ranges.append(well_info[nid]["tvt_range"])
            if len(nn_dists) == k:
                break
        # Pad in the unlikely event fewer than k neighbours are found
        while len(nn_dists) < k:
            nn_dists.append(np.nan)
            nn_last_tvts.append(np.nan)
            nn_tvt_ranges.append(np.nan)
        neighbor_lookup[wid] = {
            "dists": nn_dists,
            "last_tvts": nn_last_tvts,
            "tvt_ranges": nn_tvt_ranges,
        }
    return neighbor_lookup


# ─────────────────────────────────────────────────────────────────────────────
# Per-well feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_well_features(
    horizontal_path: Path,
    typewell_path: Path,
    split: str,
    neighbor_lookup: dict[str, dict] | None = None,
) -> pd.DataFrame:
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

    # ── GR lags 1-15 ─────────────────────────────────────────────────────────
    gr_first = float(gr.iloc[0])
    gr_last = float(gr.iloc[-1])
    gr_lags = {f"gr_lag_{i}": gr.shift(i).fillna(gr_first) for i in range(1, 16)}

    # ── GR leads 1-15 ────────────────────────────────────────────────────────
    gr_leads = {f"gr_lead_{i}": gr.shift(-i).fillna(gr_last) for i in range(1, 16)}

    # ── Local window stats (5, 11, 21) ────────────────────────────────────────
    gr_win5 = pd.concat(
        [gr_lags["gr_lag_2"], gr_lags["gr_lag_1"], gr, gr_leads["gr_lead_1"], gr_leads["gr_lead_2"]],
        axis=1,
    )
    gr_local_mean_5 = gr_win5.mean(axis=1)
    gr_local_std_5 = gr_win5.std(axis=1).fillna(0.0)

    gr_win11 = pd.concat(
        [gr_lags[f"gr_lag_{i}"] for i in range(5, 0, -1)]
        + [gr]
        + [gr_leads[f"gr_lead_{i}"] for i in range(1, 6)],
        axis=1,
    )
    gr_local_mean_11 = gr_win11.mean(axis=1)
    gr_local_std_11 = gr_win11.std(axis=1).fillna(0.0)

    gr_win21 = pd.concat(
        [gr_lags[f"gr_lag_{i}"] for i in range(10, 0, -1)]
        + [gr]
        + [gr_leads[f"gr_lead_{i}"] for i in range(1, 11)],
        axis=1,
    )
    gr_local_mean_21 = gr_win21.mean(axis=1)
    gr_local_std_21 = gr_win21.std(axis=1).fillna(0.0)

    # ── GR anomaly ────────────────────────────────────────────────────────────
    gr_anomaly = gr - gr.rolling(51, center=True, min_periods=1).mean()

    # ── Forward GR deltas ────────────────────────────────────────────
    gr_delta_lead_1 = gr_leads["gr_lead_1"] - gr
    gr_delta_lead_10 = gr_leads["gr_lead_10"] - gr

    # ── GR delta window (5-point, centered) — used for delta-aware matching ──
    gr_delta_1_series = gr.diff(1).fillna(0.0)
    gr_delta_win5 = pd.concat(
        [
            gr_delta_1_series.shift(2).fillna(0.0),
            gr_delta_1_series.shift(1).fillna(0.0),
            gr_delta_1_series,
            gr_delta_1_series.shift(-1).fillna(0.0),
            gr_delta_1_series.shift(-2).fillna(0.0),
        ],
        axis=1,
    ).to_numpy(dtype=float)

    # ── Typewell template matching: 5-point window (same as v4) ─────────────
    gr_window_matrix = gr_win5.to_numpy(dtype=float)
    nearest_typewell_tvt_window, nearest_typewell_window_mse, typewell_gr_at_window_match = (
        nearest_typewell_by_gr_window(typewell, gr_window_matrix, half=2)
    )

    typewell_tvt_vs_baseline = nearest_typewell_tvt_window - baseline_tvt
    gr_minus_typewell_gr_at_window_match = gr.to_numpy(dtype=float) - typewell_gr_at_window_match

    # ── Typewell template matching: 5-point window on smoothed GR (gr_roll_mean_11) ──
    gr_smooth = gr.rolling(11, center=True, min_periods=1).mean()
    gr_smooth_first = float(gr_smooth.iloc[0])
    gr_smooth_last = float(gr_smooth.iloc[-1])
    gr_smooth_win5 = pd.concat(
        [
            gr_smooth.shift(2).fillna(gr_smooth_first),
            gr_smooth.shift(1).fillna(gr_smooth_first),
            gr_smooth,
            gr_smooth.shift(-1).fillna(gr_smooth_last),
            gr_smooth.shift(-2).fillna(gr_smooth_last),
        ],
        axis=1,
    )
    nearest_typewell_tvt_smooth, nearest_typewell_smooth_mse, _ = nearest_typewell_by_gr_window(
        typewell, gr_smooth_win5.to_numpy(dtype=float), half=2
    )
    typewell_tvt_vs_baseline_smooth = nearest_typewell_tvt_smooth - baseline_tvt

    # ── Typewell template matching: 5-point window GR + delta (delta-aware) ──
    nearest_typewell_tvt_delta, nearest_typewell_delta_mse, _ = nearest_typewell_by_gr_window(
        typewell, gr_window_matrix, half=2, delta_windows=gr_delta_win5, delta_weight=4.0
    )
    typewell_tvt_vs_baseline_delta = nearest_typewell_tvt_delta - baseline_tvt

    # ── Typewell template matching: 11-point window (NEW v5) ─────────────────
    nearest_typewell_tvt_11, nearest_typewell_mse_11, _ = nearest_typewell_by_gr_window(
        typewell, gr_win11.to_numpy(dtype=float), half=5
    )
    typewell_tvt_vs_baseline_11 = nearest_typewell_tvt_11 - baseline_tvt

    # ── Typewell template matching: 21-point window (NEW v5) ─────────────────
    nearest_typewell_tvt_21, nearest_typewell_mse_21, _ = nearest_typewell_by_gr_window(
        typewell, gr_win21.to_numpy(dtype=float), half=10
    )
    typewell_tvt_vs_baseline_21 = nearest_typewell_tvt_21 - baseline_tvt

    # ── Neighbor-well features ────────────────────────────────────────────────
    if neighbor_lookup and well_id in neighbor_lookup:
        nn = neighbor_lookup[well_id]
        nn1_dist = nn["dists"][0]
        nn1_last_tvt = nn["last_tvts"][0]
        nn1_tvt_range = nn["tvt_ranges"][0]
        nn2_dist = nn["dists"][1]
        nn2_last_tvt = nn["last_tvts"][1]
        nn2_tvt_range = nn["tvt_ranges"][1]
        nn3_dist = nn["dists"][2]
        nn3_last_tvt = nn["last_tvts"][2]
        nn3_tvt_range = nn["tvt_ranges"][2]
    else:
        nn1_dist = nn1_last_tvt = nn1_tvt_range = np.nan
        nn2_dist = nn2_last_tvt = nn2_tvt_range = np.nan
        nn3_dist = nn3_last_tvt = nn3_tvt_range = np.nan

    # ── Assemble DataFrame ────────────────────────────────────────────────────
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
            "gr_delta_5": gr.diff(5).fillna(0.0),
            "gr_delta_10": gr.diff(10).fillna(0.0),
            "gr_delta_lead_1": gr_delta_lead_1,
            "gr_delta_lead_10": gr_delta_lead_10,
            "X": horizontal["X"],
            "Y": horizontal["Y"],
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
            "typewell_tvt_vs_baseline": typewell_tvt_vs_baseline,
            "nearest_typewell_tvt_by_gr_smooth": nearest_typewell_tvt_smooth,
            "nearest_typewell_smooth_mse": nearest_typewell_smooth_mse,
            "typewell_tvt_vs_baseline_smooth": typewell_tvt_vs_baseline_smooth,
            "nearest_typewell_tvt_by_gr_delta": nearest_typewell_tvt_delta,
            "nearest_typewell_delta_mse": nearest_typewell_delta_mse,
            "typewell_tvt_vs_baseline_delta": typewell_tvt_vs_baseline_delta,
            "nearest_typewell_tvt_by_gr_window_11": nearest_typewell_tvt_11,
            "nearest_typewell_window_mse_11": nearest_typewell_mse_11,
            "typewell_tvt_vs_baseline_11": typewell_tvt_vs_baseline_11,
            "nearest_typewell_tvt_by_gr_window_21": nearest_typewell_tvt_21,
            "nearest_typewell_window_mse_21": nearest_typewell_mse_21,
            "typewell_tvt_vs_baseline_21": typewell_tvt_vs_baseline_21,
            **{k: v for k, v in gr_lags.items()},
            **{k: v for k, v in gr_leads.items()},
            "gr_local_mean_5": gr_local_mean_5,
            "gr_local_std_5": gr_local_std_5,
            "gr_local_mean_11": gr_local_mean_11,
            "gr_local_std_11": gr_local_std_11,
            "gr_local_mean_21": gr_local_mean_21,
            "gr_local_std_21": gr_local_std_21,
            "gr_anomaly": gr_anomaly,
            "nn1_dist": nn1_dist,
            "nn1_last_tvt": nn1_last_tvt,
            "nn1_tvt_range": nn1_tvt_range,
            "nn2_dist": nn2_dist,
            "nn2_last_tvt": nn2_last_tvt,
            "nn2_tvt_range": nn2_tvt_range,
            "nn3_dist": nn3_dist,
            "nn3_last_tvt": nn3_last_tvt,
            "nn3_tvt_range": nn3_tvt_range,
        }
    )
    features["xy_dist_from_ps"] = np.sqrt(features["x_from_ps"] ** 2 + features["y_from_ps"] ** 2)
    features["is_prediction_row"] = tvt_input_missing.astype(int)

    if "TVT" in horizontal.columns:
        features["target_tvt"] = horizontal["TVT"]

    return features


def build_split(
    raw_dir: Path,
    split: str,
    neighbor_lookup: dict[str, dict] | None = None,
    exclude_wells: set[str] | None = None,
) -> pd.DataFrame:
    frames = []
    for horizontal_path in sorted((raw_dir / split).glob("*__horizontal_well.csv")):
        well_id = horizontal_path.name.split("__")[0]
        if exclude_wells and well_id in exclude_wells:
            continue
        typewell_path = raw_dir / split / f"{well_id}__typewell.csv"
        frames.append(
            build_well_features(horizontal_path, typewell_path, split, neighbor_lookup)
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_v5"))
    args = parser.parse_args()

    print("Building neighbour-well index …")
    well_info = collect_well_positions(args.raw_dir)
    neighbor_lookup = build_neighbor_lookup(well_info)
    print(f"  {len(well_info)} wells indexed")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Building train split …")
    train = build_split(args.raw_dir, "train", neighbor_lookup, exclude_wells=TEST_WELL_IDS)

    print("Building test split …")
    test = build_split(args.raw_dir, "test", neighbor_lookup)

    train_pred = train[train["is_prediction_row"] == 1].copy()
    test_pred = test[test["is_prediction_row"] == 1].copy()

    train_pred.to_csv(args.output_dir / "train_point_features.csv", index=False)
    test_pred.to_csv(args.output_dir / "test_point_features.csv", index=False)

    print(f"Train prediction rows : {len(train_pred):,}")
    print(f"Test  prediction rows : {len(test_pred):,}")
    print(f"Feature columns       : {len(FEATURE_COLUMNS)}")
    print(f"Wrote to {args.output_dir}")


if __name__ == "__main__":
    main()
