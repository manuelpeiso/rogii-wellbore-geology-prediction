import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

def analyze_predictions():
    model_path = 'models/lightgbm_v7_3000k.joblib'
    data_path = 'data/processed_v7/train_point_features.csv'
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Loading model from {model_path}...")
    model_data = joblib.load(model_path)
    
    # Check if loaded object is a dictionary or the model itself
    if isinstance(model_data, dict):
        print("Model loaded as dictionary. Keys:", model_data.keys())
        model = model_data.get('model')
        if model is None:
            # Try to find common keys for models in dicts
            for k in model_data:
                if hasattr(model_data[k], 'predict'):
                    model = model_data[k]
                    break
    else:
        model = model_data

    if model is None or not hasattr(model, 'predict'):
        print("Could not find model with 'predict' attribute.")
        return

    # Identify feature columns
    drop_cols = ['split', 'well_id', 'row_index', 'target_tvt', 'is_prediction_row']
    feature_cols = [c for c in df.columns if c in model.feature_name_] # Use model's expected features if available
    
    print(f"Using {len(feature_cols)} features...")

    # Filter for target well
    well_id = '46dfcfca'
    well_df = df[df['well_id'] == well_id].dropna(subset=['target_tvt'])
    other_df = df[df['well_id'] != well_id].dropna(subset=['target_tvt']).sample(n=min(10000, len(df)-len(well_df)), random_state=42)
    
    def evaluate(eval_df, label):
        X = eval_df[feature_cols]
        y_true = eval_df['target_tvt']
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        print(f"\n--- {label} ---")
        print(f"Rows: {len(eval_df)}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        return rmse

    rmse_well = evaluate(well_df, f"Well {well_id}")
    rmse_others = evaluate(other_df, "Other Wells (Sample)")
    
    print(f"\nComparison: Well {well_id} RMSE is {rmse_well/rmse_others:.4f}x the sample average.")

if __name__ == "__main__":
    analyze_predictions()
