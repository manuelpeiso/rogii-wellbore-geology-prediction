"""Stacking with meta-learner using V7 features.

Trains 4 base models (LightGBM, XGBoost, Random Forest, Extra Trees) on V7 features,
then trains a meta-learner (Ridge) to combine their predictions optimally.
Output goes to submissions/lightgbm_v7_stacking_*.csv
and reports/model_results_v7_stacking.csv
"""

import argparse
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from make_point_features_v7 import FEATURE_COLUMNS

# Wells that appear in raw/train/ with full TVT but are Kaggle test wells.
TEST_WELL_IDS = {"000d7d20", "00bbac68", "00e12e8b"}

META_COLUMNS = ["well_id", "row_index"]
TARGET_COLUMN = "target_tvt"


def require_package(package_name: str, install_name: Optional[str] = None) -> Any:
    import importlib
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


def build_lightgbm_model(
    random_state: int,
    lgb_learning_rate: float,
    lgb_num_leaves: int,
    lgb_min_child_samples: int,
) -> Any:
    lgb = require_package("lightgbm")
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=16000,
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


def build_xgboost_model(random_state: int) -> Any:
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


def build_random_forest_model(random_state: int) -> Any:
    return RandomForestRegressor(
        n_estimators=80,
        max_depth=18,
        min_samples_leaf=5,
        max_features=0.7,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )


def build_extra_trees_model(random_state: int) -> Any:
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


def train_model(
    model: Any,
    train: pd.DataFrame,
    model_type: str,
    valid: Optional[pd.DataFrame] = None,
) -> Any:
    """Fit the model with optional early stopping for LightGBM."""
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
    else:
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    return model


def generate_meta_features(
    models: dict[str, Any],
    data: pd.DataFrame,
) -> np.ndarray:
    """Generate meta-features by stacking predictions from base models."""
    meta_features = []
    for model_type, model in models.items():
        print(f"  Generating meta-features with {model_type}...")
        pred = model.predict(data[FEATURE_COLUMNS])
        meta_features.append(pred)
    return np.column_stack(meta_features)


def evaluate_stacking(
    models: dict[str, Any],
    meta_learner: Ridge,
    valid: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate stacking by using meta-learner on base predictions."""
    y_true = valid[TARGET_COLUMN].to_numpy(dtype=float)
    
    # Generate meta-features from base models
    meta_features = generate_meta_features(models, valid)
    
    # Use meta-learner to combine
    stacking_pred = meta_learner.predict(meta_features)
    baseline_pred = valid["baseline_tvt"].to_numpy(dtype=float)
    
    return {
        "stacking_rmse": rmse(y_true, stacking_pred),
        "baseline_rmse": rmse(y_true, baseline_pred),
        "n_valid_rows": float(len(valid)),
        "n_valid_wells": float(valid["well_id"].nunique()),
    }


def write_stacking_submission(
    models: dict[str, Any],
    meta_learner: Ridge,
    test_features_path: Path,
    sample_submission_path: Path,
    output_path: Path,
) -> None:
    """Generate submission using stacking."""
    test = pd.read_csv(test_features_path, usecols=META_COLUMNS + FEATURE_COLUMNS)
    
    # Generate meta-features from base models
    meta_features = generate_meta_features(models, test)
    
    # Use meta-learner to combine
    stacking_pred = meta_learner.predict(meta_features)
    
    prediction = pd.DataFrame(
        {
            "id": test["well_id"].astype(str) + "_" + test["row_index"].astype(str),
            "tvt": stacking_pred,
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
    max_train_rows: int,
    metrics: dict[str, float],
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"model_type": "stacking_v7", "max_train_rows": max_train_rows, **metrics}
    new_row = pd.DataFrame([row])
    if metrics_path.exists():
        previous = pd.read_csv(metrics_path)
        results = pd.concat([previous, new_row], ignore_index=True)
    else:
        results = new_row
    results.to_csv(metrics_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Stacking with meta-learner on V7 features.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_v7"),
                        help="Directory containing processed features")
    parser.add_argument("--max-train-rows", type=int, default=3000000,
                        help="Max rows to use for training")
    parser.add_argument("--valid-size", type=float, default=0.2,
                        help="Proportion of wells to use for validation")
    parser.add_argument("--meta-valid-size", type=float, default=0.5,
                        help="Proportion of validation wells for meta-learner training")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random state for reproducibility")
    parser.add_argument("--lgb-learning-rate", type=float, default=0.0275,
                        help="LightGBM learning rate")
    parser.add_argument("--lgb-num-leaves", type=int, default=63,
                        help="LightGBM num_leaves")
    parser.add_argument("--lgb-min-child-samples", type=int, default=30,
                        help="LightGBM min_child_samples")
    args = parser.parse_args()

    processed_dir = args.processed_dir
    train_path = processed_dir / "train_point_features.csv"
    test_path = processed_dir / "test_point_features.csv"
    sample_submission_path = Path("data/raw/sample_submission.csv")

    print(f"Loading training data from {train_path}...")
    valid_wells = choose_validation_wells(train_path, args.valid_size, args.random_state)
    train, valid = load_train_data(
        train_path,
        valid_wells,
        args.max_train_rows,
        chunksize=100000,
        random_state=args.random_state,
    )
    print(f"  Train: {len(train)} rows")
    print(f"  Valid: {len(valid)} rows from {len(valid_wells)} wells")

    # Split validation into meta-train and meta-test
    rng = np.random.default_rng(args.random_state)
    meta_train_idx = rng.choice(len(valid), size=int(len(valid) * args.meta_valid_size), replace=False)
    meta_test_idx = np.setdiff1d(np.arange(len(valid)), meta_train_idx)
    
    valid_meta_train = valid.iloc[meta_train_idx].reset_index(drop=True)
    valid_meta_test = valid.iloc[meta_test_idx].reset_index(drop=True)
    
    print(f"  Meta-train: {len(valid_meta_train)} rows")
    print(f"  Meta-test: {len(valid_meta_test)} rows")

    print("\nTraining base models...")
    print("  LightGBM...")
    lgb_model = train_model(
        build_lightgbm_model(
            args.random_state,
            args.lgb_learning_rate,
            args.lgb_num_leaves,
            args.lgb_min_child_samples,
        ),
        train,
        "lightgbm",
        valid_meta_train,
    )

    print("  XGBoost...")
    xgb_model = train_model(
        build_xgboost_model(args.random_state),
        train,
        "xgboost",
    )

    print("  Random Forest...")
    rf_model = train_model(
        build_random_forest_model(args.random_state),
        train,
        "random_forest",
    )

    print("  Extra Trees...")
    et_model = train_model(
        build_extra_trees_model(args.random_state),
        train,
        "extra_trees",
    )

    base_models = {
        "lightgbm": lgb_model,
        "xgboost": xgb_model,
        "random_forest": rf_model,
        "extra_trees": et_model,
    }

    print("\nGenerating meta-features for meta-learner training...")
    meta_train_features = generate_meta_features(base_models, valid_meta_train)
    meta_train_target = valid_meta_train[TARGET_COLUMN].to_numpy(dtype=float)

    print("\nTraining meta-learner (Ridge)...")
    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(meta_train_features, meta_train_target)
    print(f"  Meta-learner weights: {meta_learner.coef_}")

    print("\nEvaluating stacking on meta-test set...")
    metrics = evaluate_stacking(base_models, meta_learner, valid_meta_test)
    print(f"Stacking metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nGenerating submission...")
    submission_path = Path("submissions") / f"lightgbm_v7_stacking_{args.max_train_rows // 1000}k.csv"
    write_stacking_submission(base_models, meta_learner, test_path, sample_submission_path, submission_path)
    print(f"Wrote submission to {submission_path}")

    print("\nAppending metrics...")
    metrics_path = Path("reports/model_results_v7_stacking.csv")
    append_metrics(metrics_path, args.max_train_rows, metrics)
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
