#!/usr/bin/env python3
"""Error analysis for V7 model: load model, predict on validation set, analyze errors."""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import joblib
import sys

# Import FEATURE_COLUMNS from v7 feature engineering
sys.path.insert(0, 'src')
from make_point_features_v7 import FEATURE_COLUMNS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed_v7", help="Path to processed features")
    parser.add_argument("--model-path", default="models/lightgbm_v7_3000k.joblib", help="Path to trained model")
    parser.add_argument("--output", default="reports/error_analysis_v7_detailed.csv", help="Output file")
    parser.add_argument("--sample", type=int, default=1000000, help="Sample size for analysis (default 1M)")
    args = parser.parse_args()

    # Load features
    print(f"Loading train features from {args.processed_dir}...")
    try:
        X = pd.read_csv(Path(args.processed_dir) / "train_point_features.csv")
    except FileNotFoundError:
        print(f"❌ Feature file not found: {args.processed_dir}/train_point_features.csv")
        sys.exit(1)
    
    if 'target_tvt' not in X.columns:
        print("❌ 'target_tvt' target column not found in features")
        print(f"Available columns: {list(X.columns)}")
        sys.exit(1)
    
    # Extract target and metadata before sampling
    y_true = X['target_tvt'].values
    well_id = X['well_id'].values if 'well_id' in X.columns else np.arange(len(X))
    row_from_ps = X['row_from_ps'].values if 'row_from_ps' in X.columns else np.zeros(len(X))
    
    print(f"Loaded {len(X)} total samples")
    print(f"Target TVT range: {y_true.min():.2f} to {y_true.max():.2f}")
    print(f"Using {len(FEATURE_COLUMNS)} features from V7")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    try:
        model_dict = joblib.load(args.model_path)
        if isinstance(model_dict, dict) and 'model' in model_dict:
            model = model_dict['model']
        else:
            model = model_dict
    except FileNotFoundError:
        print(f"❌ Model not found: {args.model_path}")
        sys.exit(1)
    
    # Prepare feature matrix with correct columns
    X_features = X[FEATURE_COLUMNS].values
    
    # Sample for analysis
    n_pred = min(args.sample, len(X))
    sample_idx = np.random.choice(len(X), n_pred, replace=False)
    
    print(f"Making predictions on {n_pred} random samples...")
    X_sample = X_features[sample_idx]
    y_pred = model.predict(X_sample)
    y_true_sample = y_true[sample_idx]
    well_id_sample = well_id[sample_idx]
    row_from_ps_sample = row_from_ps[sample_idx]
    
    # Calculate errors
    residual = y_pred - y_true_sample
    abs_error = np.abs(residual)
    
    # Build results dataframe
    results_df = pd.DataFrame({
        'well_id': well_id_sample,
        'row_from_ps': row_from_ps_sample,
        'actual_tvt': y_true_sample,
        'predicted_tvt': y_pred,
        'residual': residual,
        'abs_error': abs_error,
    })
    
    print("\n" + "="*70)
    print("=== GLOBAL ERROR METRICS ===")
    print("="*70)
    mae = abs_error.mean()
    rmse = np.sqrt((abs_error**2).mean())
    print(f"MAE (Mean Absolute Error):         {mae:.4f}")
    print(f"RMSE (Root Mean Squared Error):    {rmse:.4f}")
    print(f"Median Absolute Error:             {np.median(abs_error):.4f}")
    print(f"95th percentile error:             {np.percentile(abs_error, 95):.4f}")
    print(f"Max error:                         {abs_error.max():.4f}")
    print(f"Min error:                         {abs_error.min():.4f}")
    
    # Error by well
    print("\n" + "="*70)
    print("=== TOP 10 WELLS WITH HIGHEST AVG ERROR ===")
    print("="*70)
    well_stats = results_df.groupby('well_id').agg({
        'abs_error': ['count', 'mean', 'std', 'max', 'min']
    }).sort_values(('abs_error', 'mean'), ascending=False).head(10)
    well_stats.columns = ['n_samples', 'mean_error', 'std_error', 'max_error', 'min_error']
    print(well_stats)
    
    # Error by depth
    results_df['depth_bucket'] = pd.cut(
        results_df['row_from_ps'], 
        bins=[0, 100, 500, 1000, 2000, 5000, 100000],
        labels=['0-100', '100-500', '500-1k', '1k-2k', '2k-5k', '5k+']
    )
    print("\n" + "="*70)
    print("=== ERROR BY DEPTH FROM PS ===")
    print("="*70)
    depth_stats = results_df.groupby('depth_bucket', observed=True).agg({
        'abs_error': ['count', 'mean', 'std', 'max']
    })
    depth_stats.columns = ['n_samples', 'mean_error', 'std_error', 'max_error']
    print(depth_stats)
    
    # Error by TVT magnitude
    results_df['tvt_bucket'] = pd.qcut(
        results_df['actual_tvt'].rank(method='first'), 
        q=5, 
        labels=['Bottom 20%', '20-40%', '40-60%', '60-80%', 'Top 20%'],
        duplicates='drop'
    )
    print("\n" + "="*70)
    print("=== ERROR BY ACTUAL TVT VALUE (PERCENTILES) ===")
    print("="*70)
    tvt_stats = results_df.groupby('tvt_bucket', observed=True).agg({
        'abs_error': ['count', 'mean', 'std', 'max'],
        'actual_tvt': ['min', 'max']
    })
    tvt_stats.columns = ['n_samples', 'mean_error', 'std_error', 'max_error', 'tvt_min', 'tvt_max']
    print(tvt_stats)
    
    # Error distribution
    print("\n" + "="*70)
    print("=== ERROR DISTRIBUTION ===")
    print("="*70)
    thresholds = [0.5, 1, 2, 5, 10, 20, 50]
    for t in thresholds:
        count = (abs_error < t).sum()
        pct = count / len(abs_error) * 100
        print(f"  Errors < {t:5.1f}: {count:8d} ({pct:6.2f}%)")
    
    # Worst predictions
    print("\n" + "="*70)
    print("=== TOP 20 WORST PREDICTIONS ===")
    print("="*70)
    worst = results_df.nlargest(20, 'abs_error')[
        ['well_id', 'row_from_ps', 'actual_tvt', 'predicted_tvt', 'residual', 'abs_error']
    ].reset_index(drop=True)
    for idx, row in worst.iterrows():
        print(f"{idx+1:2d}. Well {row['well_id'][:8]}: depth={row['row_from_ps']:6.0f}  "
              f"actual={row['actual_tvt']:8.2f}  pred={row['predicted_tvt']:8.2f}  "
              f"error={row['abs_error']:8.2f}")
    
    # Best predictions
    print("\n" + "="*70)
    print("=== TOP 20 BEST PREDICTIONS (smallest error) ===")
    print("="*70)
    best = results_df.nsmallest(20, 'abs_error')[
        ['well_id', 'row_from_ps', 'actual_tvt', 'predicted_tvt', 'residual', 'abs_error']
    ].reset_index(drop=True)
    for idx, row in best.iterrows():
        print(f"{idx+1:2d}. Well {row['well_id'][:8]}: depth={row['row_from_ps']:6.0f}  "
              f"actual={row['actual_tvt']:8.2f}  pred={row['predicted_tvt']:8.2f}  "
              f"error={row['abs_error']:8.2f}")
    
    # Save detailed results
    print(f"\n💾 Saving detailed results to {args.output}...")
    results_df.to_csv(args.output, index=False)
    print(f"✅ Done!")

if __name__ == "__main__":
    main()
