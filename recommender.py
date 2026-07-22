from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import differential_evolution, NonlinearConstraint
import joblib
import os

RAW_FEATURES = ['PC', 'FA', 'SC', 'SF', 'FAGG', 'CAGG', 'WATER', 'AEA', 'WR_HR', 'WR', 'ACC']
DERIVED_FEATURES = ['TOTAL_BINDER', 'w/b', 'b/a', 'SCM%', 'CAGG%', 'FAGG%']
ALL_FEATURES = RAW_FEATURES + DERIVED_FEATURES
TARGETS = ['7day', '28day', '56day']

# Dataset-derived bounds for optimizer
RAW_BOUNDS = {
    'PC':    (150, 850),
    'FA':    (0,   273),
    'SC':    (0,   560),
    'SF':    (0,    68),
    'FAGG':  (800, 1800),
    'CAGG':  (700, 2300),
    'WATER': (150,  360),
    'AEA':   (0,    30),
    'WR_HR': (0,   127),
    'WR':    (0,   100),
    'ACC':   (0,   768),
}


def compute_derived_vector(x_raw: np.ndarray) -> np.ndarray:
    """Compute derived engineering ratios from an 11-element raw feature vector."""
    PC, FA, SC, SF, FAGG, CAGG, WATER = x_raw[:7]
    total_binder = PC + FA + SC + SF
    total_agg    = FAGG + CAGG

    wb   = WATER / total_binder if total_binder > 0 else 0.0
    ba   = total_binder / total_agg if total_agg > 0 else 0.0
    scm  = (FA + SC + SF) / total_binder if total_binder > 0 else 0.0
    cpct = CAGG / total_agg if total_agg > 0 else 0.0
    fpct = FAGG / total_agg if total_agg > 0 else 0.0

    return np.concatenate([x_raw, [total_binder, wb, ba, scm, cpct, fpct]])


class ConcreteRecommender:
    def __init__(self, df: pd.DataFrame, scaler, models: dict):
        self.df = df.copy()
        self.scaler = scaler
        self.models = models  # {'7day': model, '28day': model, '56day': model}

        # Build KNN index on scaled ALL_FEATURES space
        X = self.df[ALL_FEATURES].fillna(0).values
        X_scaled = self.scaler.transform(X)
        self.knn = NearestNeighbors(n_neighbors=min(30, len(df)), metric='euclidean')
        self.knn.fit(X_scaled)
        self.X_scaled = X_scaled

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _predict_raw(self, x_raw: np.ndarray, target: str) -> float:
        """Predict from an 11-element raw feature vector."""
        x_full = compute_derived_vector(x_raw)
        x_scaled = self.scaler.transform(x_full.reshape(1, -1))
        return float(self.models[target].predict(x_scaled)[0])

    def predict_all(self, mix: dict) -> dict:
        """Predict all three strengths from a mix-design dict keyed by RAW_FEATURES."""
        x_raw = np.array([mix.get(f, 0.0) for f in RAW_FEATURES])
        return {t: self._predict_raw(x_raw, t) for t in TARGETS}

    # ------------------------------------------------------------------
    # Strategy 1 – Historical similar mixes
    # ------------------------------------------------------------------

    def recommend_historical(
        self,
        target_28day: float,
        use_fa: bool = True,
        use_sc: bool = True,
        use_sf: bool = False,
        max_wb: float = 0.55,
        max_binder: float = 700,
        target_7day: float | None = None,
        n_results: int = 5,
    ) -> pd.DataFrame:
        """Return top-N historical mixes satisfying all constraints."""
        df = self.df.copy()

        df = df[df['28day'] >= target_28day]
        if target_7day is not None:
            df = df[df['7day'] >= target_7day]
        if not use_fa:
            df = df[df['FA'] == 0]
        if not use_sc:
            df = df[df['SC'] == 0]
        if not use_sf:
            df = df[df['SF'] == 0]
        df = df[df['w/b'] <= max_wb]
        df = df[df['TOTAL_BINDER'] <= max_binder]

        if df.empty:
            return pd.DataFrame()

        # Sort by minimum over-design (closest to target without going under)
        df = df.copy()
        df['_margin'] = df['28day'] - target_28day
        df = df.sort_values('_margin').head(n_results).drop(columns='_margin')

        keep = RAW_FEATURES + ['TOTAL_BINDER', 'w/b', 'SCM%', '7day', '28day', '56day']
        return df[keep].reset_index(drop=True).round(2)

    # ------------------------------------------------------------------
    # Strategy 2 – Optimized new mix (differential evolution)
    # ------------------------------------------------------------------

    def recommend_optimized(
        self,
        target_28day: float,
        use_fa: bool = True,
        use_sc: bool = True,
        use_sf: bool = False,
        max_wb: float = 0.55,
        target_7day: float | None = None,
        min_binder: float = 300,
        max_binder: float = 700,
        maxiter: int = 200,
        seed: int = 42,
    ) -> dict:
        """Use differential evolution to find an optimal new mix design."""

        eps = 0.01
        bounds = []
        for f in RAW_FEATURES:
            lo, hi = RAW_BOUNDS[f]
            if f == 'FA' and not use_fa:
                bounds.append((0, eps))
            elif f == 'SC' and not use_sc:
                bounds.append((0, eps))
            elif f == 'SF' and not use_sf:
                bounds.append((0, eps))
            else:
                bounds.append((lo, hi))

        def _total_binder(x):
            return x[0] + x[1] + x[2] + x[3]

        def _wb(x):
            tb = _total_binder(x)
            return x[6] / tb if tb > 1 else 99.0

        # Objective: minimize Portland cement content (cost & CO₂ proxy)
        def objective(x):
            return x[0]

        constraints = [
            NonlinearConstraint(lambda x: self._predict_raw(np.array(x), '28day'),
                                target_28day, np.inf),
            NonlinearConstraint(_wb, 0.28, max_wb),
            NonlinearConstraint(_total_binder, min_binder, max_binder),
        ]
        if target_7day is not None:
            constraints.append(
                NonlinearConstraint(lambda x: self._predict_raw(np.array(x), '7day'),
                                    target_7day, np.inf)
            )

        result = differential_evolution(
            objective,
            bounds,
            constraints=constraints,
            seed=seed,
            maxiter=maxiter,
            popsize=12,
            tol=1.0,
            mutation=(0.5, 1.0),
            recombination=0.7,
            workers=1,
            polish=False,  # tree model gradients are discontinuous; skip L-BFGS polish
        )

        x = result.x
        mix = {f: float(x[i]) for i, f in enumerate(RAW_FEATURES)}
        total_binder = sum(mix[f] for f in ['PC', 'FA', 'SC', 'SF'])
        wb   = mix['WATER'] / total_binder if total_binder > 0 else None
        scm  = (mix['FA'] + mix['SC'] + mix['SF']) / total_binder if total_binder > 0 else 0
        predicted = self.predict_all(mix)

        return {
            'success': result.success or predicted['28day'] >= target_28day * 0.98,
            'mix': mix,
            'predicted': predicted,
            'total_binder': total_binder,
            'wb_ratio': wb,
            'scm_pct': scm,
            'message': result.message,
        }
