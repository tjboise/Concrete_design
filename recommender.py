from __future__ import annotations

__version__ = "2.0.0"  # min_7d / min_56d constraints

import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize

# ---------------------------------------------------------------------------
# Feature layout (SF excluded from optimization)
# ---------------------------------------------------------------------------
RAW_FEATURES = ['PC', 'FA', 'SC', 'FAGG', 'CAGG', 'WATER', 'AEA', 'WR_HR', 'WR', 'ACC']

BASE_DERIVED = [
    'TOTAL_BINDER', 'w/b', 'b/a', 'SCM%', 'CAGG%', 'FAGG%',
    'PC%', 'FA%', 'SC%', 'AEA_pct', 'WR_HR_pct', 'WR_pct', 'ACC_pct',
]

FEATURES_7D  = RAW_FEATURES + BASE_DERIVED                         # 23 features
FEATURES_28D = RAW_FEATURES + BASE_DERIVED + ['7day']             # 24 features
FEATURES_56D = RAW_FEATURES + BASE_DERIVED + ['7day', '28day']    # 25 features

TARGETS = ['7day', '28day', '56day']

# Layer 1 — raw ingredient bounds (kg/m³)
BOUNDS_L1 = {
    'PC':    (97.3,   504.3),
    'FA':    (0.0,    162.0),
    'SC':    (0.0,    332.2),
    'FAGG':  (473.5,  1067.9),
    'CAGG':  (400.5,  1364.6),
    'WATER': (90.8,   214.8),
    'AEA':   (0.0,    1.5),
    'WR_HR': (0.0,    4.7),
    'WR':    (0.0,    7.8),
    'ACC':   (0.0,    28.5),
}

# GWP emission factors (kg CO₂-eq / kg material)
GWP_COEF = {'PC': 1.048, 'FA': 0.328, 'SC': 0.264,
             'CAGG': 0.0037, 'FAGG': 0.0026}

# Material densities for volume calculation (kg/m³)
_RHO = {'PC': 3150, 'FA': 2400, 'SC': 2900,
        'FAGG': 2630, 'CAGG': 2710, 'WATER': 1000, 'AEA': 1000}

_FA_IDX  = RAW_FEATURES.index('FA')
_SC_IDX  = RAW_FEATURES.index('SC')
_AEA_IDX = RAW_FEATURES.index('AEA')


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def compute_derived_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with all derived columns added / refreshed."""
    df = df.copy()
    binder = df['PC'] + df['FA'] + df['SC']
    agg    = df['FAGG'] + df['CAGG']
    sb = binder.replace(0, np.nan)
    sa = agg.replace(0, np.nan)

    df['TOTAL_BINDER'] = binder
    df['w/b']          = df['WATER'] / sb
    df['b/a']          = binder / sa
    df['SCM%']         = (df['FA'] + df['SC']) / sb
    df['CAGG%']        = df['CAGG'] / sa
    df['FAGG%']        = df['FAGG'] / sa
    df['PC%']          = df['PC']    / sb
    df['FA%']          = df['FA']    / sb
    df['SC%']          = df['SC']    / sb
    df['AEA_pct']      = df['AEA']   / sb
    df['WR_HR_pct']    = df['WR_HR'] / sb
    df['WR_pct']       = df['WR']    / sb
    df['ACC_pct']      = df['ACC']   / sb
    return df


def _derived_batch(X: np.ndarray) -> np.ndarray:
    """
    X shape: (n, 10) — RAW_FEATURES order.
    Returns (n, 23) = raw + 13 derived columns.
    """
    PC=X[:,0]; FA=X[:,1]; SC=X[:,2]
    FAGG=X[:,3]; CAGG=X[:,4]; WATER=X[:,5]
    AEA=X[:,6]; WR_HR=X[:,7]; WR=X[:,8]; ACC=X[:,9]

    binder = PC + FA + SC
    agg    = FAGG + CAGG
    sb = np.maximum(binder, 1e-9)
    sa = np.maximum(agg,    1e-9)

    derived = np.column_stack([
        binder,         WATER/sb,      binder/sa,
        (FA+SC)/sb,     CAGG/sa,       FAGG/sa,
        PC/sb,          FA/sb,         SC/sb,
        AEA/sb,         WR_HR/sb,      WR/sb,         ACC/sb,
    ])
    return np.hstack([X, derived])


def compute_gwp(mix: dict) -> float:
    return sum(mix.get(k, 0.0) * c for k, c in GWP_COEF.items())


def _gwp_batch(X: np.ndarray) -> np.ndarray:
    return (X[:,0]*1.048 + X[:,1]*0.328 + X[:,2]*0.264
            + X[:,4]*0.0037 + X[:,3]*0.0026)
    # indices: PC=0, FA=1, SC=2, FAGG=3, CAGG=4


def _vfinal_batch(X: np.ndarray) -> np.ndarray:
    PC=X[:,0]; FA=X[:,1]; SC=X[:,2]
    FAGG=X[:,3]; CAGG=X[:,4]; WATER=X[:,5]; AEA=X[:,6]
    vm = PC/3150 + FA/2400 + SC/2900 + FAGG/2630 + CAGG/2710 + WATER/1000 + AEA/1000
    return vm + np.where(AEA > 0.001, 0.07, 0.03)


# ---------------------------------------------------------------------------
# NSGA-II problem definition
# ---------------------------------------------------------------------------

class _ConcreteProblem(Problem):
    def __init__(self, rec: 'ConcreteRecommender',
                 use_fa: bool, use_sc: bool,
                 min_7d:  float | None, min_28d: float | None,
                 min_56d: float | None, max_gwp:  float | None):
        self.rec     = rec
        self.min_7d  = min_7d
        self.min_28d = min_28d
        self.min_56d = min_56d if (min_56d is not None and rec.model_56d is not None) else None
        self.max_gwp = max_gwp

        xl = np.array([BOUNDS_L1[f][0] for f in RAW_FEATURES], dtype=float)
        xu = np.array([BOUNDS_L1[f][1] for f in RAW_FEATURES], dtype=float)
        if not use_fa:
            xl[_FA_IDX] = xu[_FA_IDX] = 0.0
        if not use_sc:
            xl[_SC_IDX] = xu[_SC_IDX] = 0.0

        n_constr = 30
        if min_7d  is not None: n_constr += 1
        if min_28d is not None: n_constr += 1
        if self.min_56d is not None: n_constr += 1
        if max_gwp is not None: n_constr += 1

        super().__init__(n_var=10, n_obj=2, n_ieq_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        PC=X[:,0]; FA=X[:,1]; SC=X[:,2]
        FAGG=X[:,3]; CAGG=X[:,4]; WATER=X[:,5]
        AEA=X[:,6]; WR_HR=X[:,7]; WR=X[:,8]; ACC=X[:,9]

        binder = PC + FA + SC
        agg    = FAGG + CAGG
        sb = np.maximum(binder, 1e-9)
        sa = np.maximum(agg,    1e-9)

        wb       = WATER  / sb
        ba       = binder / sa
        scm_pct  = (FA+SC) / sb
        cagg_pct = CAGG / sa
        fagg_pct = FAGG / sa
        pc_pct   = PC   / sb
        fa_pct   = FA   / sb
        sc_pct   = SC   / sb
        aea_pct  = AEA   / sb
        wrhr_pct = WR_HR / sb
        wr_pct   = WR    / sb
        acc_pct  = ACC   / sb

        gwp    = _gwp_batch(X)
        feat23 = _derived_batch(X)
        pred_7 = self.rec._predict_7d_batch(X)
        feat24 = np.hstack([feat23, pred_7.reshape(-1, 1)])
        pred_28 = self.rec.model_28d.predict(feat24)

        out["F"] = np.column_stack([gwp, -pred_28])

        vagg   = FAGG/2630 + CAGG/2710
        vfinal = _vfinal_batch(X)

        G = np.column_stack([
            # Layer 2 — ratio constraints
            0.235-wb,   wb-0.714,
            0.105-ba,   ba-0.488,
            -scm_pct,   scm_pct-0.765,
            0.315-cagg_pct, cagg_pct-0.721,
            0.279-fagg_pct, fagg_pct-0.685,
            0.235-pc_pct,   pc_pct-1.0,
            -fa_pct,    fa_pct-0.375,
            -sc_pct,    sc_pct-0.717,
            -aea_pct,   aea_pct-0.003,
            -wrhr_pct,  wrhr_pct-0.012,
            -wr_pct,    wr_pct-0.020,
            -acc_pct,   acc_pct-0.061,
            # Layer 3 — physics constraints
            207.7-binder, binder-590.3,
            0.553-vagg,   vagg-0.768,
            0.950-vfinal, vfinal-1.050,
        ])

        if self.min_7d is not None:
            G = np.column_stack([G, self.min_7d - pred_7])
        if self.min_28d is not None:
            G = np.column_stack([G, self.min_28d - pred_28])
        if self.min_56d is not None:
            pred_56 = self.rec.model_56d.predict(
                np.hstack([feat24, pred_28.reshape(-1, 1)])
            )
            G = np.column_stack([G, self.min_56d - pred_56])
        if self.max_gwp is not None:
            G = np.column_stack([G, gwp - self.max_gwp])

        out["G"] = G


# ---------------------------------------------------------------------------
# Main recommender class
# ---------------------------------------------------------------------------

class ConcreteRecommender:
    def __init__(self, df: pd.DataFrame,
                 model_7d:  CatBoostRegressor,
                 model_28d: CatBoostRegressor,
                 model_56d: CatBoostRegressor | None = None):
        self.df       = df.copy()
        self.model_7d  = model_7d
        self.model_28d = model_28d
        self.model_56d = model_56d

    # ------------------------------------------------------------------
    # Internal batch prediction helpers
    # ------------------------------------------------------------------

    def _predict_7d_batch(self, X: np.ndarray) -> np.ndarray:
        """X: (n, 10) raw features → (n,) 7-day predictions."""
        return self.model_7d.predict(_derived_batch(X))

    def _predict_28d_batch(self, X: np.ndarray) -> np.ndarray:
        feat23 = _derived_batch(X)
        pred7  = self._predict_7d_batch(X).reshape(-1, 1)
        return self.model_28d.predict(np.hstack([feat23, pred7]))

    # ------------------------------------------------------------------
    # Public prediction
    # ------------------------------------------------------------------

    def predict_all(self, mix: dict) -> dict:
        """Return {'7day': float, '28day': float, '56day': float|None}."""
        x   = np.array([[mix.get(f, 0.0) for f in RAW_FEATURES]])
        p7  = float(self._predict_7d_batch(x)[0])
        f23 = _derived_batch(x)
        p28 = float(self.model_28d.predict(np.hstack([f23, [[p7]]]))[0])
        p56 = None
        if self.model_56d is not None:
            p56 = float(self.model_56d.predict(np.hstack([f23, [[p7]], [[p28]]]))[0])
        return {'7day': p7, '28day': p28, '56day': p56}

    # ------------------------------------------------------------------
    # NSGA-II Pareto search
    # ------------------------------------------------------------------

    def run_nsga2(
        self,
        use_fa:  bool = True,
        use_sc:  bool = True,
        min_7d:  float | None = None,
        min_28d: float | None = None,
        min_56d: float | None = None,
        max_gwp: float | None = None,
        pop_size: int = 100,
        n_gen:    int = 100,
        seed:     int = 42,
    ) -> list[dict]:
        """
        Run NSGA-II minimizing GWP and maximizing 28-day strength.
        Returns Pareto front as a list of solution dicts sorted by GWP.
        """
        problem = _ConcreteProblem(self, use_fa, use_sc, min_7d, min_28d, min_56d, max_gwp)

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = pymoo_minimize(
                problem, algorithm,
                ('n_gen', n_gen),
                seed=seed, verbose=False,
            )

        if res.X is None:
            return []

        solutions = []
        for i in range(len(res.X)):
            x   = res.X[i]
            mix = {f: float(x[j]) for j, f in enumerate(RAW_FEATURES)}
            tb  = mix['PC'] + mix['FA'] + mix['SC']
            wb  = mix['WATER'] / max(tb, 1e-9)
            solutions.append({
                'mix':         mix,
                'gwp':         float(res.F[i, 0]),
                'strength_28d': float(-res.F[i, 1]),
                'predicted':   self.predict_all(mix),
                'total_binder': tb,
                'wb_ratio':    wb,
                'scm_pct':     (mix['FA'] + mix['SC']) / max(tb, 1e-9),
            })

        solutions.sort(key=lambda s: s['gwp'])
        return solutions

    # ------------------------------------------------------------------
    # Historical similar mixes
    # ------------------------------------------------------------------

    def recommend_historical(
        self,
        min_28d:  float | None = 30.0,
        max_gwp:  float | None = 500.0,
        use_fa:   bool  = True,
        use_sc:   bool  = True,
        n_results: int  = 5,
        exclude_sf: bool = True,
    ) -> pd.DataFrame:
        df = self.df.copy()
        if min_28d is not None:
            df = df[df['28day'] >= min_28d]
        if max_gwp is not None:
            df = df[df['GWP'] <= max_gwp]
        if not use_fa:
            df = df[df['FA'] == 0]
        if not use_sc:
            df = df[df['SC'] == 0]
        if exclude_sf and 'SF' in df.columns:
            df = df[df['SF'] == 0]

        if df.empty:
            return pd.DataFrame()

        df = df.sort_values(['GWP', '28day'], ascending=[True, False])
        keep = RAW_FEATURES + ['TOTAL_BINDER', 'w/b', 'SCM%', 'GWP', '7day', '28day', '56day']
        keep = [c for c in keep if c in df.columns]
        result = df[keep].head(n_results).reset_index(drop=True)
        return result.apply(pd.to_numeric, errors='coerce').round(3)
