from __future__ import annotations

"""v7 training script.

Changes vs v5
─────────────
* Uses make_point_features_v7 (v5 base + pre-PS/typewell similarity features)
* LightGBM: 3 000 estimators, lr=0.02, num_leaves=95, early stopping on validation
* CatBoost: early stopping on validation
* Default processed dir: data/processed_v7
* Metrics written to reports/model_results_v7.csv
"""

import argparse
import importlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from make_point_features_v7 import FEATURE_COLUMNS

# Wells that appear in raw/train/ with full TVT but are Kaggle test wells.
TEST_WELL_IDS = {"000d7d20", "00bbac68", "00e12e8b"}

META_COLUMNS = ["well_id", "row_index"]
TARGET_COLUMN = "target_tvt"
MODEL_TYPES = [
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
    "catboost",
]


def require_package(package_name: str, install_name: str | None = None) -> Any:
    try:
        return importlib.import_module(package_name)
    except ImportError as exc:
        name = install_name or package_name
        raise ImportError(
            f"Model requires optional package '{package_name}'. "
            f"Install it with: python -m pip install {name}"
        ) from exc


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def choose_validation_wells(
    train_path: Path, valid_size: float, random_state: int
) -> set[str]:
    wells = pd.read_csv(train_path, usecols=["well_id"])["well_id"]
    unique_wells = pd.Series(
        [w for w in wells.unique() if w not in TEST_WELL_IDS]
    )
    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=random_state)
    dummy = np.zeros(len(unique_wells))
    _, valid_idx = next(splitter.split(dummy, groups=unique_wells.to_numpy()))
    return set(unique_wells.iloc[valid_idx])


def load_train_data(
    train_path: Path,
    valid_wells: set[str],
    max_train_rows: int,
    chunksize: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    usecols = META_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    train_rows = 0

    for chunk in pd.read_csv(train_path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[~chunk["well_id"].isin(TEST_WELL_IDS)]

        is_valid = chunk["well_id"].isin(valid_wells)
        valid_parts.append(chunk.loc[is_valid])

        train_chunk = chunk.loc[~is_valid]
        remaining = max_train_rows - train_rows
        if remaining > 0 and len(train_chunk) > 0:
            if len(train_chunk) > remaining:
                selected = rng.choice(
                    train_chunk.index.to_numpy(), size=remaining, replace=False
                )
                train_chunk = train_chunk.loc[selected]
            train_parts.append(train_chunk)
            train_rows += len(train_chunk)

    train = pd.concat(train_parts, ignore_index=True)
    valid = pd.concat(valid_parts, ignore_index=True)
    return train, valid


def build_model(
    model_type: str,
    random_state: int,
    lgb_learning_rate: float,
    lgb_num_leaves: int,
    lgb_min_child_samples: int,
) -> Any:
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=5,
            max_features=0.7,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )

    if model_type == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=120,
            max_depth=22,
            min_samples_leaf=3,
            max_features=0.8,
            bootstrap=False,
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )

    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=0.01,
            random_state=random_state,
            verbose=1,
        )

    if model_type == "lightgbm":
        lgb = require_package("lightgbm")
        return lgb.LGBMRegressor(
            objective="regression",
            n_estimators=16000,         # early stopping finds best round
            learning_rate=lgb_learning_rate,
            num_leaves=lgb_num_leaves,
            max_depth=-1,
            min_child_samples=lgb_min_child_samples,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.05,
            n_jobs=-1,
            random_state=random_state,
            verbosity=1,
        )

    if model_type == "xgboost":
        xgb = require_package("xgboost")
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=900,
            learning_rate=0.03,
            max_depth=8,
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
            verbosity=1,
        )

    if model_type == "catboost":
        catboost = require_package("catboost")
        return catboost.CatBoostRegressor(
            iterations=1500,            # ↑ from 900  (early stopping will find best)
            learning_rate=0.02,
            depth=8,
            loss_function="RMSE",
            eval_metric="RMSE",
            early_stopping_rounds=200,
            random_seed=random_state,
            allow_writing_files=False,
            verbose=100,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def train_model(
    train: pd.DataFrame,
    model_type: str,
    random_state: int,
    valid: pd.DataFrame | None = None,
    lgb_learning_rate: float = 0.0275,
    lgb_num_leaves: int = 63,
    lgb_min_child_samples: int = 30,
) -> Any:
    """Fit the model.  For LightGBM and CatBoost the validation split is used
    for early stopping when provided."""
    model = build_model(
        model_type,
        random_state,
        lgb_learning_rate,
        lgb_num_leaves,
        lgb_min_child_samples,
    )

    if model_type == "lightgbm" and valid is not None:
        lgb = require_package("lightgbm")
        model.fit(
            train[FEATURE_COLUMNS],
            train[TARGET_COLUMN],
            eval_set=[(valid[FEATURE_COLUMNS], valid[TARGET_COLUMN])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=500, verbose=True),
                lgb.log_evaluation(period=100),
            ],
        )
    elif model_type == "catboost" and valid is not None:
        model.fit(
            train[FEATURE_COLUMNS],
            train[TARGET_COLUMN],
            eval_set=(valid[FEATURE_COLUMNS], valid[TARGET_COLUMN]),
        )
    else:
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

    return model


def evaluate_model(model: Any, valid: pd.DataFrame) -> dict[str, float]:
    y_true = valid[TARGET_COLUMN].to_numpy(dtype=float)
    model_pred = model.predict(valid[FEATURE_COLUMNS])
    baseline_pred = valid["baseline_tvt"].to_numpy(dtype=float)
    return {
        "model_rmse": rmse(y_true, model_pred),
        "baseline_rmse": rmse(y_true, baseline_pred),
        "n_valid_rows": float(len(valid)),
        "n_valid_wells": float(valid["well_id"].nunique()),
    }


def write_submission(
    model: Any,
    test_features_path: Path,
    sample_submission_path: Path,
    output_path: Path,
) -> None:
    test = pd.read_csv(test_features_path, usecols=META_COLUMNS + FEATURE_COLUMNS)
    pred = model.predict(test[FEATURE_COLUMNS])
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


def append_metrics(
    metrics_path: Path,
    model_type: str,
    max_train_rows: int,
    metrics: dict[str, float],
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"model_type": model_type, "max_train_rows": max_train_rows, **metrics}
    new_row = pd.DataFrame([row])
    if metrics_path.exists():
        previous = pd.read_csv(metrics_path)
        results = pd.concat([previous, new_row], ignore_index=True)
    else:
        results = new_row
    results.to_csv(metrics_path, index=False)


def default_output_path(
    base_dir: Path, model_type: str, max_train_rows: int, suffix: str
) -> Path:
    rows_label = (
        f"{max_train_rows // 1000}k" if max_train_rows % 1000 == 0 else str(max_train_rows)
    )
    return base_dir / f"{model_type}_v7_{rows_label}.{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="lightgbm")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_v7"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument("--submission-output", type=Path, default=None)
    parser.add_argument(
        "--metrics-output", type=Path, default=Path("reports/model_results_v7.csv")
    )
    parser.add_argument("--max-train-rows", type=int, default=3_000_000)
    parser.add_argument("--valid-size", type=float, default=0.20)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--lgb-learning-rate", type=float, default=0.0275)
    parser.add_argument("--lgb-num-leaves", type=int, default=63)
    parser.add_argument("--lgb-min-child-samples", type=int, default=30)
    args = parser.parse_args()

    train_path = args.processed_dir / "train_point_features.csv"
    test_path = args.processed_dir / "test_point_features.csv"
    sample_path = args.raw_dir / "sample_submission.csv"
    model_output = args.model_output or default_output_path(
        Path("models"), args.model_type, args.max_train_rows, "joblib"
    )
    submission_output = args.submission_output or default_output_path(
        Path("submissions"), args.model_type, args.max_train_rows, "csv"
    )

    valid_wells = choose_validation_wells(train_path, args.valid_size, args.random_state)
    print(f"Model type          : {args.model_type}")
    print(f"Validation wells    : {len(valid_wells)}")
    print(f"Test wells excluded : {sorted(TEST_WELL_IDS)}")
    if args.model_type == "lightgbm":
        print(
            "LGB params          : "
            f"lr={args.lgb_learning_rate}, "
            f"num_leaves={args.lgb_num_leaves}, "
            f"min_child_samples={args.lgb_min_child_samples}"
        )

    train, valid = load_train_data(
        train_path=train_path,
        valid_wells=valid_wells,
        max_train_rows=args.max_train_rows,
        chunksize=args.chunksize,
        random_state=args.random_state,
    )
    print(f"Train rows used  : {len(train):,}")
    print(f"Validation rows  : {len(valid):,}")

    model = train_model(
        train,
        args.model_type,
        args.random_state,
        valid=valid,
        lgb_learning_rate=args.lgb_learning_rate,
        lgb_num_leaves=args.lgb_num_leaves,
        lgb_min_child_samples=args.lgb_min_child_samples,
    )
    metrics = evaluate_model(model, valid)
    metrics["n_train_rows"] = float(len(train))
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_type": args.model_type,
            "features": FEATURE_COLUMNS,
            "metrics": metrics,
        },
        model_output,
    )
    print(f"Wrote model to {model_output}")

    write_submission(model, test_path, sample_path, submission_output)
    print(f"Wrote submission to {submission_output}")

    append_metrics(args.metrics_output, args.model_type, args.max_train_rows, metrics)
    print(f"Wrote metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
