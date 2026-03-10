"""
train_models.py
---------------
Trains six regression models on the processed energy consumption dataset,
evaluates them on the held-out test set, and saves all artefacts to disk.

Run from the project root:
    python app/train_models.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ── Feature set ───────────────────────────────────────────────────────────────
FEATURES = [
    'log_population', 'log_gdp', 'log_electricity_generation',
    'coal_production', 'oil_production', 'gas_production',
    'renewables_share_elec', 'fossil_share_elec',
    'solar_share_elec', 'wind_share_elec', 'hydro_share_elec',
    'year_norm'
]
TARGET = 'log_primary_energy_consumption'  # train on log scale, invert for evaluation


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val   = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    # Combine train + val for final model fitting (after tuning on val)
    train_full = pd.concat([train, val], ignore_index=True)

    X_train = train[FEATURES].values
    y_train = train[TARGET].values
    X_val   = val[FEATURES].values
    y_val   = val[TARGET].values
    X_test  = test[FEATURES].values
    y_test_log = test[TARGET].values
    y_test_raw = test['primary_energy_consumption'].values  # already in TWh

    X_full = train_full[FEATURES].values
    y_full = train_full[TARGET].values

    return X_train, y_train, X_val, y_val, X_test, y_test_log, y_test_raw, X_full, y_full, test


def evaluate(model, X, y_log, y_raw):
    """Returns metrics on original TWh scale."""
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    y_true = y_raw

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    # MAPE — skip zeros to avoid divide-by-zero
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return {'RMSE': round(rmse, 2), 'MAE': round(mae, 2),
            'R2': round(r2, 4), 'MAPE': round(mape, 2)}


def main():
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test_log, y_test_raw, X_full, y_full, test_df = load_data()

    # ── Define models ────────────────────────────────────────────────────────
    # Each model is wrapped in a Pipeline so scaling is handled consistently.
    # Tree-based models don't need scaling but it doesn't hurt.

    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=10.0))
        ]),

        'SVR': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=10.0, epsilon=0.1))
        ]),

        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=200, max_depth=12,
                min_samples_split=4, random_state=42, n_jobs=-1
            ))
        ]),

        'GradientBoosting': Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05,
                max_depth=5, subsample=0.8, random_state=42
            ))
        ]),

        'GPR': Pipeline([
            ('scaler', StandardScaler()),
            ('model', GaussianProcessRegressor(
                kernel=Matern(nu=1.5) + WhiteKernel(),
                normalize_y=True, n_restarts_optimizer=3, random_state=42
            ))
        ]),

        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation='relu', max_iter=500,
                learning_rate_init=0.001, early_stopping=True,
                random_state=42
            ))
        ])
    }

    # ── Train and evaluate ───────────────────────────────────────────────────
    results = {}
    feature_importance = {}

    for name, pipeline in models.items():
        print(f"\nTraining {name}...")

        # GPR is slow on large datasets — subsample for training
        if name == 'GPR':
            idx = np.random.RandomState(42).choice(len(X_full), size=min(2000, len(X_full)), replace=False)
            pipeline.fit(X_full[idx], y_full[idx])
        else:
            pipeline.fit(X_full, y_full)

        metrics = evaluate(pipeline, X_test, y_test_log, y_test_raw)
        results[name] = metrics
        print(f"  RMSE: {metrics['RMSE']:,.1f} | MAE: {metrics['MAE']:,.1f} | R²: {metrics['R2']:.4f} | MAPE: {metrics['MAPE']:.1f}%")

        # Save model
        joblib.dump(pipeline, os.path.join(MODELS_DIR, f'{name.lower()}.pkl'))

        # Feature importance (where available)
        inner_model = pipeline.named_steps['model']
        if hasattr(inner_model, 'feature_importances_'):
            feature_importance[name] = dict(zip(FEATURES, inner_model.feature_importances_.round(4).tolist()))
        elif hasattr(inner_model, 'coef_'):
            feature_importance[name] = dict(zip(FEATURES, np.abs(inner_model.coef_).round(4).tolist()))

    # ── Save artefacts ───────────────────────────────────────────────────────
    with open(os.path.join(MODELS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(MODELS_DIR, 'feature_importance.json'), 'w') as f:
        json.dump(feature_importance, f, indent=2)

    # Save feature list and country list for the web app
    meta = {
        'features': FEATURES,
        'countries': sorted(test_df['country'].unique().tolist()),
        'year_range': [int(test_df['year'].min()), int(test_df['year'].max())]
    }
    with open(os.path.join(MODELS_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n\nAll models saved to models/")
    print("\nTest set results:")
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())

    # Save predictions for comparison plots
    best_model_name = min(results, key=lambda k: results[k]['RMSE'])
    best_model = joblib.load(os.path.join(MODELS_DIR, f'{best_model_name.lower()}.pkl'))
    preds = np.expm1(best_model.predict(X_test))
    pred_df = test_df[['country', 'year', 'primary_energy_consumption']].copy()
    pred_df['predicted'] = preds.round(2)
    pred_df.to_csv(os.path.join(STATIC_DIR, 'predictions.csv'), index=False)
    print(f"\nBest model: {best_model_name}")


if __name__ == '__main__':
    main()
