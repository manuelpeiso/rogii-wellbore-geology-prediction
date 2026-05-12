from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from make_point_features_v2 import FEATURE_COLUMNS_V2
from train_model import META_COLUMNS
from train_model import TARGET_COLUMN
from train_model import append_metrics
from train_model import choose_validation_wells
from train_model import default_output_path
from train_model import rmse
from train_model import require_package


MODEL_TYPES_V2 = [
    "random_forest_v2",
    "extra_trees_v2",
    "hist_gradient_boosting_v2",
    "lightgbm_v2",
    "xgboost_v2",
    "catboost_v2",
]


def load_train_data_v2(
    train_path: Path,
    valid_wells: set[str],
    max_train_rows: int,
    chunksize: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import numpy as np

    rng = np.random.default_rng(random_state)
    usecols = META_COLUMNS + FEATURE_COLUMNS_V2 + [TARGET_COLUMN]
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


def build_model_v2(model_type: str, random_state: int) -> Any:
    if model_type == "random_forest_v2":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=120,
            max_depth=20,
            min_samples_leaf=4,
            max_features=0.75,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )

    if model_type == "extra_trees_v2":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=160,
            max_depth=24,
            min_samples_leaf=4,
            max_features=0.75,
            bootstrap=False,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )

    if model_type == "hist_gradient_boosting_v2":
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            learning_rate=0.035,
            max_iter=450,
            max_leaf_nodes=63,
            min_samples_leaf=40,
            l2_regularization=0.05,
            random_state=random_state,
            verbose=1,
        )

    if model_type == "lightgbm_v2":
        lgb = require_package("lightgbm")
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=1800,
            learning_rate=0.02,
            num_leaves=127,
            max_depth=-1,
            min_child_samples=80,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.10,
            reg_lambda=0.20,
            n_jobs=-1,
            random_state=random_state,
            verbosity=1,
        )

    if model_type == "xgboost_v2":
        xgb = require_package("xgboost")
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1400,
            learning_rate=0.02,
            max_depth=9,
            min_child_weight=12,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.10,
            reg_lambda=1.50,
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
            verbosity=1,
        )

    if model_type == "catboost_v2":
        catboost = require_package("catboost")
        return catboost.CatBoostRegressor(
            iterations=1400,
            learning_rate=0.02,
            depth=8,
            l2_leaf_reg=5.0,
            random_strength=1.0,
            loss_function="RMSE",
            random_seed=random_state,
            allow_writing_files=False,
            verbose=100,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def train_model_v2(train: pd.DataFrame, model_type: str, random_state: int) -> Any:
    model = build_model_v2(model_type, random_state)
    model.fit(train[FEATURE_COLUMNS_V2], train[TARGET_COLUMN])
    return model


def evaluate_model_v2(model: Any, valid: pd.DataFrame) -> dict[str, float]:
    y_true = valid[TARGET_COLUMN].to_numpy(dtype=float)
    model_pred = model.predict(valid[FEATURE_COLUMNS_V2])
    baseline_pred = valid["baseline_tvt"].to_numpy(dtype=float)
    return {
        "model_rmse": rmse(y_true, model_pred),
        "baseline_rmse": rmse(y_true, baseline_pred),
        "n_valid_rows": float(len(valid)),
        "n_valid_wells": float(valid["well_id"].nunique()),
    }


def write_submission_v2(
    model: Any,
    test_features_path: Path,
    sample_submission_path: Path,
    output_path: Path,
) -> None:
    test = pd.read_csv(test_features_path, usecols=META_COLUMNS + FEATURE_COLUMNS_V2)
    pred = model.predict(test[FEATURE_COLUMNS_V2])
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
    parser.add_argument("--model-type", choices=MODEL_TYPES_V2, default="lightgbm_v2")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_v2"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument("--submission-output", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=Path("reports/model_results_v2.csv"))
    parser.add_argument("--max-train-rows", type=int, default=3_000_000)
    parser.add_argument("--valid-size", type=float, default=0.12)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_path = args.processed_dir / "train_point_features_v2.csv"
    test_path = args.processed_dir / "test_point_features_v2.csv"
    sample_path = args.raw_dir / "sample_submission.csv"
    model_output = args.model_output or default_output_path(Path("models"), args.model_type, args.max_train_rows, "joblib")
    submission_output = args.submission_output or default_output_path(
        Path("submissions"), args.model_type, args.max_train_rows, "csv"
    )

    valid_wells = choose_validation_wells(train_path, args.valid_size, args.random_state)
    print(f"Model type: {args.model_type}")
    print(f"Validation wells: {len(valid_wells)}")

    train, valid = load_train_data_v2(
        train_path=train_path,
        valid_wells=valid_wells,
        max_train_rows=args.max_train_rows,
        chunksize=args.chunksize,
        random_state=args.random_state,
    )
    print(f"Train rows used: {len(train):,}")
    print(f"Validation rows: {len(valid):,}")

    model = train_model_v2(train, args.model_type, args.random_state)
    metrics = evaluate_model_v2(model, valid)
    metrics["n_train_rows"] = float(len(train))
    metrics["feature_set"] = "v2"
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "model_type": args.model_type, "features": FEATURE_COLUMNS_V2, "metrics": metrics},
        model_output,
    )
    print(f"Wrote model to {model_output}")

    write_submission_v2(model, test_path, sample_path, submission_output)
    print(f"Wrote submission to {submission_output}")

    append_metrics(args.metrics_output, args.model_type, args.max_train_rows, metrics)
    print(f"Wrote metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
