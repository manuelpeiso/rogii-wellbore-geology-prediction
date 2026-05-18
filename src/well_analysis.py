import pandas as pd
import numpy as np

def analyze_well():
    print("Loading data...")
    df = pd.read_csv('data/processed_v7/train_point_features.csv')
    
    # Correct column names: 'gr', 'target_tvt', 'row_from_ps'
    well_id = '46dfcfca'
    well_df = df[df['well_id'] == well_id]
    
    print(f"\n--- Analysis for Well {well_id} ---")
    print(f"Number of rows: {len(well_df)}")
    
    if len(well_df) > 0:
        if 'target_tvt' in well_df.columns:
            print(f"target_tvt range: {well_df['target_tvt'].min():.2f} to {well_df['target_tvt'].max():.2f}")
        if 'gr' in well_df.columns:
            print(f"gr range: {well_df['gr'].min():.2f} to {well_df['gr'].max():.2f}")
        if 'row_from_ps' in well_df.columns:
            print(f"row_from_ps range: {well_df['row_from_ps'].min():.2f} to {well_df['row_from_ps'].max():.2f}")
        
        nan_counts = well_df.isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        if not cols_with_nan.empty:
            print("\nFeatures with NaN in this well:")
            print(cols_with_nan)
        else:
            print("\nNo NaNs found in this well's features.")
            
    print("\n--- Overall Dataset Statistics ---")
    print(f"Total rows: {len(df)}")
    if 'target_tvt' in df.columns:
        print(f"Overall target_tvt range: {df['target_tvt'].min():.2f} to {df['target_tvt'].max():.2f}")
    if 'gr' in df.columns:
        print(f"Overall gr range: {df['gr'].min():.2f} to {df['gr'].max():.2f}")
    
    # Comparison
    print("\n--- Comparison ---")
    if 'gr' in well_df.columns:
        avg_gr_well = well_df['gr'].mean()
        avg_gr_all = df['gr'].mean()
        print(f"Mean gr (Well): {avg_gr_well:.2f} vs Mean gr (All): {avg_gr_all:.2f}")

if __name__ == "__main__":
    analyze_well()
