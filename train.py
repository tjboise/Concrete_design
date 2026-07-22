"""
Run this script once to train and save the prediction models.
Features: 11 raw material inputs + 6 derived engineering ratios (w/b, b/a, SCM%, etc.)
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# Raw input features (what the user controls)
RAW_FEATURES = ['PC', 'FA', 'SC', 'SF', 'FAGG', 'CAGG', 'WATER', 'AEA', 'WR_HR', 'WR', 'ACC']

# Derived engineering ratios (computed from raw inputs; critical predictors)
DERIVED_FEATURES = ['TOTAL_BINDER', 'w/b', 'b/a', 'SCM%', 'CAGG%', 'FAGG%']

ALL_FEATURES = RAW_FEATURES + DERIVED_FEATURES
TARGETS = ['7day', '28day', '56day']

DATA_PATH = 'Super_Cleaned_Concrete_Data - backup.csv'
MODELS_DIR = 'models'


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Re-compute derived features from raw columns to ensure consistency."""
    d = df.copy()
    d['TOTAL_BINDER'] = d['PC'] + d['FA'] + d['SC'] + d['SF']
    d['w/b']  = d['WATER'] / d['TOTAL_BINDER'].replace(0, np.nan)
    d['b/a']  = d['TOTAL_BINDER'] / (d['FAGG'] + d['CAGG']).replace(0, np.nan)
    d['SCM%'] = (d['FA'] + d['SC'] + d['SF']) / d['TOTAL_BINDER'].replace(0, np.nan)
    d['CAGG%'] = d['CAGG'] / (d['FAGG'] + d['CAGG']).replace(0, np.nan)
    d['FAGG%'] = d['FAGG'] / (d['FAGG'] + d['CAGG']).replace(0, np.nan)
    return d


def train(data_path: str = DATA_PATH, models_dir: str = MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    df = compute_derived(df)

    X = df[ALL_FEATURES].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    print(f"Scaler saved.  Dataset: {len(df)} rows  |  Features: {len(ALL_FEATURES)}")
    print(f"  Raw: {RAW_FEATURES}")
    print(f"  Derived: {DERIVED_FEATURES}\n")

    for target in TARGETS:
        mask = df[target].notna().values
        X_t = X_scaled[mask]
        y_t = df.loc[mask, target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X_t, y_t, test_size=0.2, random_state=42
        )

        model = GradientBoostingRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            min_samples_leaf=3,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"  {target:5s}  R²={r2:.3f}  RMSE={rmse:.2f} MPa  "
              f"(train={len(y_train)}, test={len(y_test)})")

        joblib.dump(model, os.path.join(models_dir, f'model_{target}.pkl'))

    print(f"\nAll models saved to: {models_dir}/")


if __name__ == '__main__':
    train()
