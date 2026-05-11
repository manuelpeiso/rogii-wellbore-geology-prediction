from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def interpolate_with_extrapolation(x: np.ndarray, y: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """Linear interpolation with endpoint extrapolation for one well."""
    valid = np.isfinite(x) & np.isfinite(y)
    x_known = x[valid]
    y_known = y[valid]

    if len(x_known) == 0:
        raise ValueError("Cannot interpolate a well without known TVT_input values.")
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


def predict_horizontal_well(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    pred = interpolate_with_extrapolation(
        df["MD"].to_numpy(dtype=float),
        df["TVT_input"].to_numpy(dtype=float),
        df["MD"].to_numpy(dtype=float),
    )
    return pd.Series(pred, index=df.index, name="tvt")


def validate_train(raw_dir: Path) -> float:
    errors = []
    for path in sorted((raw_dir / "train").glob("*__horizontal_well.csv")):
        df = pd.read_csv(path)
        hidden = df["TVT_input"].isna()
        if not hidden.any():
            continue
        pred = predict_horizontal_well(path)
        diff = pred.loc[hidden].to_numpy() - df.loc[hidden, "TVT"].to_numpy()
        errors.append(diff)

    if not errors:
        return float("nan")

    all_errors = np.concatenate(errors)
    return float(np.sqrt(np.mean(all_errors**2)))


def build_submission(raw_dir: Path, output: Path) -> pd.DataFrame:
    sample = pd.read_csv(raw_dir / "sample_submission.csv")
    predictions_by_well = {
        path.name.split("__")[0]: predict_horizontal_well(path)
        for path in sorted((raw_dir / "test").glob("*__horizontal_well.csv"))
    }

    tvt = []
    for row_id in sample["id"]:
        well_id, row_index = row_id.rsplit("_", 1)
        tvt.append(predictions_by_well[well_id].iloc[int(row_index)])

    submission = pd.DataFrame({"id": sample["id"], "tvt": tvt})
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("submissions/baseline_interpolate.csv"))
    args = parser.parse_args()

    rmse = validate_train(args.raw_dir)
    submission = build_submission(args.raw_dir, args.output)

    print(f"Validation RMSE on hidden train TVT_input rows: {rmse:.4f}")
    print(f"Wrote {len(submission):,} predictions to {args.output}")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
