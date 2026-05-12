from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from make_point_features_v3 import FEATURE_COLUMNS_V3
from train_model import META_COLUMNS
from train_model import TARGET_COLUMN
from train_model import append_metrics
from train_model import choose_validation_wells
from train_model import default_output_path
from train_model import require_package
from train_model import rmse


MODEL_TYPES_V3 = ["lightgbm_v3", "catboost_v3"]


def load_train_data_v3(
    train_path: Path,
    valid_wells: set[str],
    max_train_rows: int,
    chunksize: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    usecols = META_COLUMNS + FEATURE_COLUMNS_V3 + [TARGET_COLUMN]
    train_parts = []
    valid_parts = []
    train_rows = 0

    for chunk in pd.read_csv(train_path, usecols=usecols, chunksize=chunksize):
        is_valid = chunk["well_id"].isin(valid_wells)
        valid_parts.append(chunk.loc[is_valid])

        train_chunk = chunk.loc[~is_valid]
        remaining = max_train_rows - train_rows
        if remaining > 0 and len(train_chunk) > 0:
            if len(train_chunk) > remaining:
                selected = rng.choice(train_chunk.index.to_numpy(), size=remaining, replace=False)
                train_chunk = train_chunk.loc[selected]
            train_parts.append(train_chunk)
            train_rows += len(train_chunk)

    return pd.concat(train_parts, ignore_index=True), pd.concat(valid_parts, ignore_index=True)


def build_model_v3(model_type: str, random_state: int) -> Any:
    if model_type == "lightgbm_v3":
        lgb = require_package("lightgbm")
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=1400,
            learning_rate=0.025,
            num_leaves=95,
            max_depth=-1,
            min_child_samples=60,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.08,
            reg_lambda=0.12,
            n_jobs=-1,
            random_state=random_state,
            verbosity=1,
        )

    if model_type == "catboost_v3":
        catboost = require_package("catboost")
        return catboost.CatBoostRegressor(
            iterations=1200,
            learning_rate=0.025,
            depth=8,
            l2_leaf_reg=4.0,
            random_strength=0.8,
            loss_function="RMSE",
            random_seed=random_state,
            allow_writing_files=False,
            verbose=100,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_model_v3(model: Any, valid: pd.DataFrame) -> dict[str, float]:
    y_true = valid[TARGET_COLUMN].to_numpy(dtype=float)
    model_pred = model.predict(valid[FEATURE_COLUMNS_V3])
    baseline_pred = valid["baseline_tvt"].to_numpy(dtype=float)
    return {
        "model_rmse": rmse(y_true, model_pred),
        "baseline_rmse": rmse(y_true, baseline_pred),
        "n_valid_rows": float(len(valid)),
        "n_valid_wells": float(valid["well_id"].nunique()),
    }


def write_submission_v3(
    model: Any,
    test_features_path: Path,
    sample_submission_path: Path,
    output_path: Path,
) -> None:
    test = pd.read_csv(test_features_path, usecols=META_COLUMNS + FEATURE_COLUMNS_V3)
    pred = model.predict(test[FEATURE_COLUMNS_V3])
    prediction = pd.DataFrame(
        {
            "id": test["well_id"].astype(str) + "_" + test["row_index"].astype(str),
            "tvt": pred,
        }
    )

    sample = pd.read_csv(sample_submission_path)
    submission = sample[["id"]].merge(prediction, on="id", how="left")
    if submission["tvt"].isna().any():
        missing = int(submission["tvt"].isna().sum())
        raise ValueError(f"Submission has {missing} missing predictions.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=MODEL_TYPES_V3, default="lightgbm_v3")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_v3"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument("--submission-output", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/model_results_v3.csv"))
    parser.add_argument("--max-train-rows", type=int, default=3_000_000)
    parser.add_argument("--valid-size", type=float, default=0.12)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_path = args.processed_dir / "train_point_features_v3.csv"
    test_path = args.processed_dir / "test_point_features_v3.csv"
    sample_path = args.raw_dir / "sample_submission.csv"
    model_output = args.model_output or default_output_path(Path("models"), args.model_type, args.max_train_rows, "joblib")
    submission_output = args.submission_output or default_output_path(
        Path("submissions"), args.model_type, args.max_train_rows, "csv"
    )

    valid_wells = choose_validation_wells(train_path, args.valid_size, args.random_state)
    print(f"Model type: {args.model_type}")
    print(f"Validation wells: {len(valid_wells)}")

    train, valid = load_train_data_v3(
        train_path=train_path,
        valid_wells=valid_wells,
        max_train_rows=args.max_train_rows,
        chunksize=args.chunksize,
        random_state=args.random_state,
    )
    print(f"Train rows used: {len(train):,}")
    print(f"Validation rows: {len(valid):,}")

    model = build_model_v3(args.model_type, args.random_state)
    model.fit(train[FEATURE_COLUMNS_V3], train[TARGET_COLUMN])

    metrics = evaluate_model_v3(model, valid)
    metrics["n_train_rows"] = float(len(train))
    metrics["feature_set"] = "v3"
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "model_type": args.model_type, "features": FEATURE_COLUMNS_V3, "metrics": metrics},
        model_output,
    )
    print(f"Wrote model to {model_output}")

    write_submission_v3(model, test_path, sample_path, submission_output)
    print(f"Wrote submission to {submission_output}")

    append_metrics(args.metrics_output, args.model_type, args.max_train_rows, metrics)
    print(f"Wrote metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()

