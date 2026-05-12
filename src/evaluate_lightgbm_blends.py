from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from make_point_features import FEATURE_COLUMNS
from train_model import META_COLUMNS
from train_model import TARGET_COLUMN
from train_model import choose_validation_wells


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_cap_list(value: str) -> list[float | None]:
    caps: list[float | None] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        caps.append(None if item in {"none", "nan", "null"} else float(item))
    return caps


def blended_prediction(
    baseline_pred: np.ndarray,
    model_pred: np.ndarray,
    alpha: float,
    cap: float | None,
) -> np.ndarray:
    correction = alpha * (model_pred - baseline_pred)
    if cap is not None:
        correction = np.clip(correction, -cap, cap)
    return baseline_pred + correction


def sample_well_groups(n_wells: int, group_size: int, n_groups: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    groups = np.empty((n_groups, group_size), dtype=int)
    for i in range(n_groups):
        groups[i] = rng.choice(n_wells, size=group_size, replace=False)
    return groups


def well_error_parts(valid: pd.DataFrame, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    well_codes = valid["well_code"].to_numpy(dtype=int)
    n_wells = int(well_codes.max()) + 1
    y_true = valid[TARGET_COLUMN].to_numpy(dtype=float)
    se = (y_true - pred) ** 2
    well_sse = np.bincount(well_codes, weights=se, minlength=n_wells)
    well_counts = np.bincount(well_codes, minlength=n_wells)
    well_rmse = np.sqrt(well_sse / well_counts)
    return well_sse, well_counts, well_rmse


def random_group_rmse_from_parts(
    well_sse: np.ndarray,
    well_counts: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    group_sse = well_sse[groups].sum(axis=1)
    group_counts = well_counts[groups].sum(axis=1)
    return np.sqrt(group_sse / group_counts)


def summarize_candidate(
    valid: pd.DataFrame,
    pred: np.ndarray,
    alpha: float,
    cap: float | None,
    groups: np.ndarray,
) -> dict[str, float | str]:
    y_true = valid[TARGET_COLUMN].to_numpy(dtype=float)
    well_sse, well_counts, well_scores = well_error_parts(valid, pred)
    group_scores = random_group_rmse_from_parts(well_sse, well_counts, groups)

    return {
        "alpha": alpha,
        "cap": "none" if cap is None else cap,
        "global_rmse": rmse(y_true, pred),
        "well_rmse_mean": float(np.mean(well_scores)),
        "well_rmse_median": float(np.median(well_scores)),
        "well_rmse_p90": float(np.quantile(well_scores, 0.90)),
        "well_rmse_p95": float(np.quantile(well_scores, 0.95)),
        "well_rmse_max": float(np.max(well_scores)),
        "group3_rmse_mean": float(group_scores.mean()),
        "group3_rmse_p90": float(np.quantile(group_scores, 0.90)),
        "group3_rmse_p95": float(np.quantile(group_scores, 0.95)),
        "group3_rmse_p99": float(np.quantile(group_scores, 0.99)),
        "group3_rmse_max": float(group_scores.max()),
    }


def write_submission(
    model: object,
    test_features_path: Path,
    sample_submission_path: Path,
    output_path: Path,
    alpha: float,
    cap: float | None,
) -> None:
    test = pd.read_csv(test_features_path, usecols=META_COLUMNS + FEATURE_COLUMNS)
    baseline_pred = test["baseline_tvt"].to_numpy(dtype=float)
    model_pred = model.predict(test[FEATURE_COLUMNS])
    pred = blended_prediction(baseline_pred, model_pred, alpha, cap)

    prediction = pd.DataFrame(
        {
            "id": test["well_id"].astype(str) + "_" + test["row_index"].astype(str),
            "tvt": pred,
        }
    )
    sample = pd.read_csv(sample_submission_path)
    submission = sample[["id"]].merge(prediction, on="id", how="left")
    if submission["tvt"].isna().any():
        raise ValueError(f"Submission has {int(submission['tvt'].isna().sum())} missing predictions.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("models/lightgbm_3000k.joblib"))
    parser.add_argument("--train-features", type=Path, default=Path("data/processed/train_point_features.csv"))
    parser.add_argument("--test-features", type=Path, default=Path("data/processed/test_point_features.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/lightgbm_blend_results.csv"))
    parser.add_argument("--submission-dir", type=Path, default=Path("submissions"))
    parser.add_argument("--alphas", default="0,0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1.0")
    parser.add_argument("--caps", default="none,20,30,40,50,75,100")
    parser.add_argument("--valid-size", type=float, default=0.12)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-groups", type=int, default=5000)
    parser.add_argument("--write-top", type=int, default=5)
    args = parser.parse_args()

    bundle = joblib.load(args.model_path)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle

    valid_wells = choose_validation_wells(args.train_features, args.valid_size, args.random_state)
    usecols = META_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    train = pd.read_csv(args.train_features, usecols=usecols)
    valid = train.loc[train["well_id"].isin(valid_wells)].copy()
    valid["well_code"] = pd.Categorical(valid["well_id"]).codes
    groups = sample_well_groups(
        n_wells=valid["well_code"].nunique(),
        group_size=3,
        n_groups=args.n_groups,
        random_state=args.random_state,
    )

    baseline_pred = valid["baseline_tvt"].to_numpy(dtype=float)
    model_pred = model.predict(valid[FEATURE_COLUMNS])

    rows = []
    for alpha in parse_float_list(args.alphas):
        for cap in parse_cap_list(args.caps):
            pred = blended_prediction(baseline_pred, model_pred, alpha, cap)
            rows.append(
                summarize_candidate(
                    valid=valid,
                    pred=pred,
                    alpha=alpha,
                    cap=cap,
                    groups=groups,
                )
            )

    results = pd.DataFrame(rows).sort_values(
        ["group3_rmse_p95", "well_rmse_p95", "global_rmse"],
        ascending=True,
    )
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.metrics_output, index=False)

    sample_submission_path = args.raw_dir / "sample_submission.csv"
    for _, row in results.head(args.write_top).iterrows():
        cap = None if row["cap"] == "none" else float(row["cap"])
        cap_label = "none" if cap is None else str(int(cap))
        alpha_label = str(row["alpha"]).replace(".", "p")
        output_path = args.submission_dir / f"lightgbm_blend_a{alpha_label}_cap{cap_label}.csv"
        write_submission(
            model=model,
            test_features_path=args.test_features,
            sample_submission_path=sample_submission_path,
            output_path=output_path,
            alpha=float(row["alpha"]),
            cap=cap,
        )

    print(f"Wrote metrics to {args.metrics_output}")
    print("Top candidates:")
    display_cols = [
        "alpha",
        "cap",
        "global_rmse",
        "well_rmse_p95",
        "well_rmse_max",
        "group3_rmse_p95",
        "group3_rmse_p99",
        "group3_rmse_max",
    ]
    print(results[display_cols].head(args.write_top).to_string(index=False))


if __name__ == "__main__":
    main()
