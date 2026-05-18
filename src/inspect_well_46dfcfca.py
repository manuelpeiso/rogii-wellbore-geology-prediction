import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def run_analysis():
    well_id = '46dfcfca'
    data_path = 'data/processed_v7/train_point_features.csv'
    df = pd.read_csv(data_path)
    
    # Filter well and non-null target
    well_df = df[df['well_id'] == well_id].dropna(subset=['target_tvt']).sort_values('row_index')
    global_df = df.dropna(subset=['target_tvt'])
    
    # 1) Statistics
    print(f"--- Statistics ---")
    print(f"Well {well_id} count: {len(well_df)}")
    print(f"Well {well_id} target_tvt: min={well_df['target_tvt'].min():.2f}, max={well_df['target_tvt'].max():.2f}, std={well_df['target_tvt'].std():.4f}")
    print(f"Global target_tvt: min={global_df['target_tvt'].min():.2f}, max={global_df['target_tvt'].max():.2f}, std={global_df['target_tvt'].std():.4f}")

    # 2) Correlations
    features = ['row_index', 'row_from_ps', 'MD', 'md_from_ps', 'baseline_tvt', 'gr', 'gr_roll_mean_11', 'nearest_typewell_tvt_by_gr_window', 'typewell_tvt_vs_baseline']
    # Check if features exist
    existing_features = [f for f in features if f in well_df.columns]
    corrs = well_df[existing_features + ['target_tvt']].corr()['target_tvt'].sort_values(ascending=False)
    print(f"\n--- Correlations with target_tvt ---")
    print(corrs)

    # 3) Predictability Test
    split_idx = int(0.7 * len(well_df))
    train_df = well_df.iloc[:split_idx]
    test_df = well_df.iloc[split_idx:]
    
    def get_rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    # Baseline: Mean
    y_baseline = np.full(len(test_df), train_df['target_tvt'].mean())
    rmse_base = get_rmse(test_df['target_tvt'], y_baseline)
    
    # Model A: LR with [row_from_ps, md_from_ps]
    feats_a = [f for f in ['row_from_ps', 'md_from_ps'] if f in well_df.columns]
    X_train_a, y_train = train_df[feats_a].fillna(0), train_df['target_tvt']
    X_test_a, y_test = test_df[feats_a].fillna(0), test_df['target_tvt']
    lr_a = LinearRegression().fit(X_train_a, y_train)
    rmse_a = get_rmse(y_test, lr_a.predict(X_test_a))
    
    # Model B: LR with all 9 features
    X_train_b = train_df[existing_features].fillna(0)
    X_test_b = test_df[existing_features].fillna(0)
    lr_b = LinearRegression().fit(X_train_b, y_train)
    rmse_b = get_rmse(y_test, lr_b.predict(X_test_b))
    
    # Model C: RF with 9 features
    rf_c = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train_b, y_train)
    rmse_c = get_rmse(y_test, rf_c.predict(X_test_b))
    
    print(f"\n--- Test RMSE ---")
    print(f"Baseline (Mean): {rmse_base:.4f}")
    print(f"Model A (LR dist): {rmse_a:.4f}")
    print(f"Model B (LR all):  {rmse_b:.4f}")
    print(f"Model C (RF all):  {rmse_c:.4f}")

    # 4) Regime shift check: check if train mean is very different from test mean
    diff_mean = abs(train_df['target_tvt'].mean() - test_df['target_tvt'].mean())
    print(f"\n--- Regime/Pattern Analysis ---")
    print(f"Train/Test mean diff: {diff_mean:.4f}")
    if rmse_c < rmse_base * 0.5:
        print("Conclusion: Stable predictable pattern found.")
    else:
        print("Conclusion: High noise or regime shift likely (high RMSE vs baseline).")

if __name__ == "__main__":
    run_analysis()
