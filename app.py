from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
from catboost import CatBoostRegressor

from recommender import (
    ConcreteRecommender, compute_derived_df, compute_gwp,
    RAW_FEATURES, FEATURES_7D, FEATURES_28D, FEATURES_56D,
    BOUNDS_L1, GWP_COEF,
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Concrete Mix Design Advisor — NJDOT",
    page_icon="🏗️",
    layout="wide",
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'Concrete_Data_SI_clean.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

NJDOT_BLUE = '#003087'
GREEN      = '#10B981'
AMBER      = '#F59E0B'
RED        = '#EF4444'

MATERIAL_LABELS = {
    'PC':    'Portland Cement',
    'FA':    'Fly Ash',
    'SC':    'Slag Cement',
    'FAGG':  'Fine Aggregate',
    'CAGG':  'Coarse Aggregate',
    'WATER': 'Water',
    'AEA':   'Air-Entraining Agent',
    'WR_HR': 'High-Range WR',
    'WR':    'Water Reducer',
    'ACC':   'Accelerator',
}
ADMIX = {'AEA', 'WR_HR', 'WR', 'ACC'}

UNIT_S  = {'Metric': 1.0,      'Imperial': 145.038}
UNIT_M  = {'Metric': 1.0,      'Imperial': 1.68556}
UNIT_SL = {'Metric': 'MPa',    'Imperial': 'psi'}
UNIT_ML = {'Metric': 'kg/m³',  'Imperial': 'lb/yd³'}


# ---------------------------------------------------------------------------
# Model loading / auto-training (CatBoost Chain)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_or_train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df_raw = pd.read_csv(DATA_PATH)

    num_cols = RAW_FEATURES + ['7day', '28day', '56day']
    for c in num_cols:
        if c in df_raw.columns:
            df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')

    df = compute_derived_df(df_raw)

    p7  = os.path.join(MODELS_DIR, 'model_7d.cbm')
    p28 = os.path.join(MODELS_DIR, 'model_28d.cbm')
    p56 = os.path.join(MODELS_DIR, 'model_56d.cbm')

    loaded = False
    model_7d  = CatBoostRegressor()
    model_28d = CatBoostRegressor()
    model_56d = CatBoostRegressor()
    try:
        model_7d.load_model(p7)
        model_28d.load_model(p28)
        model_56d.load_model(p56)
        loaded = True
    except Exception:
        pass

    if not loaded:
        cb_params = dict(
            iterations=500, learning_rate=0.05, depth=6,
            loss_function='RMSE', verbose=False, random_seed=42,
        )

        # Chain model 1 — 7-day
        mask7 = df['7day'].notna()
        X7 = df.loc[mask7, FEATURES_7D].fillna(0).values
        y7 = df.loc[mask7, '7day'].values
        model_7d = CatBoostRegressor(**cb_params)
        model_7d.fit(X7, y7)

        # Chain model 2 — 28-day (uses actual 7day as chain input)
        mask28 = df['28day'].notna() & df['7day'].notna()
        X28_base = df.loc[mask28, FEATURES_28D[:-1]].fillna(0).values  # all but '7day'
        chain_7  = df.loc[mask28, '7day'].values.reshape(-1, 1)
        X28 = np.hstack([X28_base, chain_7])
        y28 = df.loc[mask28, '28day'].values
        model_28d = CatBoostRegressor(**cb_params)
        model_28d.fit(X28, y28)

        # Chain model 3 — 56-day
        mask56 = df['56day'].notna() & df['7day'].notna() & df['28day'].notna()
        X56_base = df.loc[mask56, FEATURES_56D[:-2]].fillna(0).values
        chain_56 = np.hstack([
            df.loc[mask56, '7day'].values.reshape(-1, 1),
            df.loc[mask56, '28day'].values.reshape(-1, 1),
        ])
        X56 = np.hstack([X56_base, chain_56])
        y56 = df.loc[mask56, '56day'].values
        model_56d = CatBoostRegressor(**cb_params)
        model_56d.fit(X56, y56)

        try:
            model_7d.save_model(p7)
            model_28d.save_model(p28)
            model_56d.save_model(p56)
        except Exception:
            pass

    rec = ConcreteRecommender(df, model_7d, model_28d, model_56d)
    return df, rec


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def pareto_chart(solutions: list[dict], selected_idx: int,
                 us: float, sl: str) -> go.Figure:
    gwps = [s['gwp'] for s in solutions]
    strs = [s['strength_28d'] * us for s in solutions]

    fig = go.Figure()

    # All Pareto solutions
    fig.add_trace(go.Scatter(
        x=gwps, y=strs,
        mode='markers+lines',
        name='Pareto Front',
        marker=dict(size=10, color=NJDOT_BLUE, opacity=0.7),
        line=dict(color=NJDOT_BLUE, width=1, dash='dot'),
        hovertemplate='GWP: %{x:.1f} kg CO₂/m³<br>28-Day: %{y:.1f} ' + sl + '<extra></extra>',
    ))

    # Selected solution
    fig.add_trace(go.Scatter(
        x=[gwps[selected_idx]], y=[strs[selected_idx]],
        mode='markers',
        name='Selected',
        marker=dict(size=18, color=RED, symbol='star'),
        hovertemplate='Selected<br>GWP: %{x:.1f}<br>Strength: %{y:.1f} ' + sl + '<extra></extra>',
    ))

    # Ideal point annotation
    fig.add_annotation(
        x=min(gwps), y=max(strs),
        text='Ideal\n(unachievable)',
        showarrow=True, arrowhead=2,
        ax=30, ay=-30,
        font=dict(size=10, color='gray'),
        arrowcolor='gray',
    )

    fig.update_layout(
        xaxis_title='GWP (kg CO₂-eq / m³)  ←  Lower is better',
        yaxis_title=f'28-Day Strength ({sl})  ↑  Higher is better',
        height=380, margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(orientation='h', y=-0.22),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def gauge_chart(predicted: dict, target_28: float, us: float, sl: str) -> go.Figure:
    max_val = 120 * us
    specs   = [[{'type': 'indicator'}] * 3]
    fig     = make_subplots(rows=1, cols=3, specs=specs)

    for col_i, (key, lbl, thr) in enumerate([
        ('7day',  '7-Day',  None),
        ('28day', '28-Day', target_28 * us if target_28 else None),
        ('56day', '56-Day', None),
    ], start=1):
        val = (predicted.get(key) or 0) * us
        gauge_cfg = {
            'axis': {'range': [0, max_val]},
            'bar':  {'color': NJDOT_BLUE},
            'steps': [
                {'range': [0, max_val*.4],  'color': '#FEE2E2'},
                {'range': [max_val*.4, max_val*.65], 'color': '#FEF9C3'},
                {'range': [max_val*.65, max_val], 'color': '#DCFCE7'},
            ],
        }
        if thr:
            gauge_cfg['threshold'] = {
                'line': {'color': RED, 'width': 3},
                'thickness': 0.75, 'value': thr,
            }
        fig.add_trace(go.Indicator(
            mode='gauge+number',
            value=val,
            title={'text': f'<b>{lbl}</b><br><span style="font-size:0.75em;color:gray">{sl}</span>'},
            number={'font': {'size': 22}, 'valueformat': '.1f'},
            gauge=gauge_cfg,
        ), row=1, col=col_i)

    fig.update_layout(
        height=240, margin=dict(t=50, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def gwp_breakdown_chart(mix: dict) -> go.Figure:
    labels, vals, colors = [], [], []
    palette = [NJDOT_BLUE, '#60A5FA', '#34D399', '#FCA5A5', '#D1D5DB', '#F59E0B']
    for i, (mat, coef) in enumerate(GWP_COEF.items()):
        v = mix.get(mat, 0) * coef
        if v > 0.5:
            labels.append(MATERIAL_LABELS.get(mat, mat))
            vals.append(round(v, 1))
            colors.append(palette[i % len(palette)])
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation='h',
        marker_color=colors,
        text=[f'{v:.1f}' for v in vals],
        textposition='auto',
    ))
    fig.update_layout(
        height=220, xaxis_title='kg CO₂-eq / m³',
        margin=dict(t=10, b=10, l=0, r=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def mix_pie_chart(mix: dict) -> go.Figure:
    label_map = {k: MATERIAL_LABELS[k] for k in ['PC','FA','SC','FAGG','CAGG','WATER']}
    lbls, vals = [], []
    for k, lbl in label_map.items():
        v = mix.get(k, 0)
        if v > 1:
            lbls.append(lbl); vals.append(round(v, 1))
    palette = [NJDOT_BLUE,'#1D6FA4','#34D399','#6EE7B7','#D1D5DB','#F59E0B']
    fig = go.Figure(go.Pie(
        labels=lbls, values=vals, hole=0.35,
        marker=dict(colors=palette), textinfo='label+percent',
    ))
    fig.update_layout(
        height=260, showlegend=False,
        margin=dict(t=10, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def hist_material_chart(df_hist: pd.DataFrame, um: float, ml: str) -> go.Figure:
    show = ['PC','FA','SC','FAGG','CAGG','WATER']
    palette = [NJDOT_BLUE,'#1D6FA4','#34D399','#6EE7B7','#D1D5DB','#F59E0B']
    fig = go.Figure()
    for i, col in enumerate(show):
        if col not in df_hist.columns: continue
        fig.add_trace(go.Bar(
            name=MATERIAL_LABELS[col],
            x=[f'Mix {j+1}' for j in range(len(df_hist))],
            y=(df_hist[col] * um).tolist(),
            marker_color=palette[i],
        ))
    fig.update_layout(
        barmode='group', height=300,
        yaxis_title=f'Quantity ({ml})',
        legend=dict(orientation='h', y=-0.35),
        margin=dict(t=10, b=10, l=0, r=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def pareto_csv(solutions: list[dict]) -> str:
    rows = []
    for i, s in enumerate(solutions):
        row = {'Solution': i+1, 'GWP_kg_CO2_m3': round(s['gwp'], 2),
               'Pred_28d_MPa': round(s['strength_28d'], 2),
               'Total_Cementitious_kg_m3': round(s['total_binder'], 1),
               'wb_ratio': round(s['wb_ratio'], 3),
               'SCM_pct': round(s['scm_pct']*100, 1)}
        for f in RAW_FEATURES:
            row[f'{f}_kg_m3'] = round(s['mix'].get(f, 0), 2)
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False)


def hist_csv(df_hist: pd.DataFrame) -> str:
    return df_hist.to_csv(index=False)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    # Inject CSS to tighten sidebar padding
    st.markdown("""
<style>
/* Reduce sidebar top/bottom padding */
[data-testid="stSidebarUserContent"] {
    padding-top: 1rem;
    padding-bottom: 0.5rem;
}
/* Tighten spacing between every widget block */
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > div {
    gap: 0.25rem;
}
/* Shrink slider label + widget gap */
[data-testid="stSidebarUserContent"] .stSlider {
    padding-top: 0;
    padding-bottom: 0.1rem;
}
/* Shrink checkbox rows */
[data-testid="stSidebarUserContent"] .stCheckbox {
    margin-bottom: 0;
}
/* Shrink radio button row */
[data-testid="stSidebarUserContent"] .stRadio {
    margin-bottom: 0.25rem;
}
/* Tighten markdown paragraphs in sidebar */
[data-testid="stSidebarUserContent"] .stMarkdown p {
    margin-bottom: 0.15rem;
}
/* Shrink horizontal rule */
[data-testid="stSidebarUserContent"] hr {
    margin: 0.4rem 0;
}
/* Circular Run button */
[data-testid="stSidebarUserContent"] .stButton button {
    aspect-ratio: 1 !important;
    border-radius: 50% !important;
    padding: 0 !important;
    font-size: 1rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

    st.markdown(
        f"<h2 style='color:{NJDOT_BLUE};margin-bottom:0'>🏗️ Concrete Mix Design Advisor</h2>",
        unsafe_allow_html=True,
    )
    st.caption("**NJDOT — New Jersey Department of Transportation**")
    st.divider()

    with st.spinner("Loading models… (first run trains CatBoost Chain, ~30–60 s)"):
        df, rec = load_or_train()

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("**Unit System**")
        unit_sys = st.radio("Unit System", ["Metric", "Imperial"],
                            horizontal=True, label_visibility="collapsed")
        us = UNIT_S[unit_sys]; um = UNIT_M[unit_sys]
        sl = UNIT_SL[unit_sys]; ml = UNIT_ML[unit_sys]

        st.markdown("---")
        st.markdown("**Strength Constraints**")

        en_7d = st.checkbox("Set min 7-Day strength", value=False)
        min_7d_mpa = None
        if en_7d:
            if unit_sys == 'Metric':
                min_7d_mpa = float(st.slider("Min 7-Day (MPa)", 5, 50, 20, step=1))
            else:
                min_7d_mpa = st.slider("Min 7-Day (psi)", 700, 7000, 2900, step=100) / 145.038

        en_28d = st.checkbox("Set min 28-Day strength", value=True)
        min_28d_mpa = None
        if en_28d:
            if unit_sys == 'Metric':
                min_28d_mpa = float(st.slider("Min 28-Day (MPa)", 20, 80, 30, step=1))
            else:
                min_28d_mpa = st.slider("Min 28-Day (psi)", 2900, 11600, 4350, step=100) / 145.038

        en_56d = st.checkbox("Set min 56-Day strength", value=False)
        min_56d_mpa = None
        if en_56d:
            if unit_sys == 'Metric':
                min_56d_mpa = float(st.slider("Min 56-Day (MPa)", 20, 90, 40, step=1))
            else:
                min_56d_mpa = st.slider("Min 56-Day (psi)", 2900, 13000, 5800, step=100) / 145.038

        st.markdown("**GWP Constraint**")
        max_gwp_val = st.slider("Max GWP (kg CO₂-eq/m³)", 150, 550, 500, step=10)

        st.markdown("**Available SCMs**")
        scm_c1, scm_c2 = st.columns(2)
        use_fa = scm_c1.checkbox("Fly Ash", value=True)
        use_sc = scm_c2.checkbox("Slag Cement", value=True)

        st.markdown("---")
        st.markdown("**Historical Mixes to Show**")
        n_hist = st.slider("n_hist", 3, 10, 3, label_visibility="collapsed")

        st.markdown("---")
        _, _mid, _ = st.columns([3, 2, 3])
        with _mid:
            run_all = st.button("▶ Run", type="primary", use_container_width=True)

    # -----------------------------------------------------------------------
    # Run computations → session state
    # -----------------------------------------------------------------------
    if run_all:
        with st.spinner("Running…"):
            try:
                solutions = rec.run_nsga2(
                    use_fa=use_fa, use_sc=use_sc,
                    min_7d=min_7d_mpa, min_28d=min_28d_mpa,
                    min_56d=min_56d_mpa, max_gwp=max_gwp_val,
                    pop_size=100, n_gen=100,
                )
            except Exception as _e:
                st.error(f"NSGA-II error ({type(_e).__name__}): {_e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
        st.session_state['pareto_solutions'] = solutions
        st.session_state['pareto_params']    = {
            'min_28d_mpa': min_28d_mpa, 'max_gwp': max_gwp_val,
        }
        st.session_state.pop('adj_predicted', None)

        df_hist = rec.recommend_historical(
            min_28d=min_28d_mpa, max_gwp=max_gwp_val,
            use_fa=use_fa, use_sc=use_sc,
            n_results=n_hist,
        )
        st.session_state['hist_result']    = df_hist
        st.session_state['hist_min_28d']   = min_28d_mpa

    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------
    tab1, tab2 = st.tabs(["🌿 Pareto Front", "📚 Historical Similar Mixes"])

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 1 — Pareto Front
    # ═══════════════════════════════════════════════════════════════════════
    with tab1:
        if 'pareto_solutions' not in st.session_state:
            st.info(
                "Set parameters in the sidebar and click **🚀 Run Pareto Search** to "
                "generate Pareto-optimal mixes that trade off GWP vs compressive strength."
            )
            with st.expander("ℹ️ About the optimization"):
                st.markdown(f"""
**Objectives** (conflicting):
- 🟢 **Minimize GWP** = PC × 1.048 + FA × 0.328 + SC × 0.264 + CAGG × 0.0037 + FAGG × 0.0026
- 🔵 **Maximize 28-Day Strength** — predicted by CatBoost-Chain surrogate

**Algorithm**: NSGA-II (Non-dominated Sorting Genetic Algorithm II)

**Three constraint layers**:
1. Raw ingredient bounds (kg/m³)
2. Derived ratio bounds (w/cm, b/a, SCM%, aggregate fractions)
3. Physics constraints (total volume, aggregate volume fraction)

**Pareto front**: the set of solutions where no objective can be improved without worsening the other.
The dataset covers {len(df)} mixes with GWP ranging **{df['GWP'].min():.0f}–{df['GWP'].max():.0f} kg CO₂/m³**
and 28-day strength **{df['28day'].min():.1f}–{df['28day'].max():.1f} MPa**.
                """)
        else:
            solutions = st.session_state['pareto_solutions']

            if not solutions:
                st.warning("No feasible solutions found. Try relaxing the GWP limit or strength target.")
            else:
                n_sol = len(solutions)
                params = st.session_state['pareto_params']

                # Pareto chart + Mix Composition side by side
                pc1, pc2 = st.columns([3, 2])
                with pc1:
                    st.markdown("**Select Your Preferred Solution**")
                    pref = st.slider(
                        "Preference weight",
                        min_value=0, max_value=100, value=50, step=1,
                        format="%d",
                        help="0 = maximize strength (ignore GWP)   |   100 = minimize GWP (ignore strength)",
                    )
                    _cl, _cr = st.columns(2)
                    _cl.caption("← Maximize Strength")
                    _cr.markdown("<div style='text-align:right'>Minimize GWP →</div>",
                                 unsafe_allow_html=True)

                    # Score each Pareto solution
                    gwps = np.array([s['gwp'] for s in solutions])
                    strs = np.array([s['strength_28d'] for s in solutions])
                    g_min, g_rng = gwps.min(), max(gwps.max() - gwps.min(), 1e-9)
                    s_min, s_rng = strs.min(), max(strs.max() - strs.min(), 1e-9)
                    w = pref / 100
                    scores = (1 - w) * (strs - s_min) / s_rng - w * (gwps - g_min) / g_rng
                    sel = int(np.argmax(scores))

                    st.plotly_chart(pareto_chart(solutions, sel, us, sl),
                                    use_container_width=True)

                sol = solutions[sel]
                mix = sol['mix']

                with pc2:
                    st.markdown("**Mix Composition**")
                    st.plotly_chart(mix_pie_chart(mix), use_container_width=True)

                st.divider()

                # Clear adjusted state when solution selection changes
                if st.session_state.get('_prev_sel') != sel:
                    st.session_state.pop('adj_predicted', None)
                    st.session_state.pop('adj_gwp', None)
                st.session_state['_prev_sel'] = sel

                st.header(f"Selected Mix — Solution #{sel+1}")
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.markdown("**Mix Proportions** — edit Adjusted column & recalculate")
                    mix_df_edit = pd.DataFrame([
                        {
                            'Material': MATERIAL_LABELS.get(k, k),
                            f'Original ({ml})': round(float(mix.get(k, 0)) * um, 2),
                            f'Adjusted ({ml})': round(float(mix.get(k, 0)) * um, 2),
                        }
                        for k in RAW_FEATURES
                    ])
                    edited_df = st.data_editor(
                        mix_df_edit,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            'Material': st.column_config.TextColumn(disabled=True),
                            f'Original ({ml})': st.column_config.NumberColumn(disabled=True),
                            f'Adjusted ({ml})': st.column_config.NumberColumn(
                                min_value=0.0, step=1.0),
                        },
                        key=f"mix_editor_{sel}",
                    )
                    current_mix = {
                        feat: float(edited_df.iloc[i][f'Adjusted ({ml})']) / um
                        for i, feat in enumerate(RAW_FEATURES)
                    }
                    if st.button("🔄 Recalculate", key="recalc_btn"):
                        st.session_state['adj_predicted'] = rec.predict_all(current_mix)
                        st.session_state['adj_gwp']       = compute_gwp(current_mix)

                with col_b:
                    st.markdown("**Predicted Strength & GWP**")
                    orig_pred = sol['predicted']
                    adj_pred  = st.session_state.get('adj_predicted')
                    adj_gwp   = st.session_state.get('adj_gwp')

                    comp_rows = []
                    for _lbl, _key in [
                        ('7-Day Strength',  '7day'),
                        ('28-Day Strength', '28day'),
                        ('56-Day Strength', '56day'),
                    ]:
                        comp_rows.append({
                            'Metric': _lbl,
                            f'Original (NSGA-II)': f"{(orig_pred.get(_key) or 0)*us:.1f} {sl}",
                            'Adjusted': f"{(adj_pred.get(_key) or 0)*us:.1f} {sl}" if adj_pred else "—",
                        })
                    comp_rows.append({
                        'Metric': 'GWP',
                        f'Original (NSGA-II)': f"{sol['gwp']:.1f} kg CO₂/m³",
                        'Adjusted': f"{adj_gwp:.1f} kg CO₂/m³" if adj_gwp is not None else "—",
                    })
                    st.dataframe(
                        pd.DataFrame(comp_rows),
                        hide_index=True,
                        use_container_width=True,
                    )

                st.download_button(
                    "📥 Download Full Pareto Front (CSV)",
                    data=pareto_csv(solutions),
                    file_name="njdot_pareto_front.csv",
                    mime="text/csv",
                )

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 2 — Historical Similar Mixes
    # ═══════════════════════════════════════════════════════════════════════
    with tab2:
        if 'hist_result' not in st.session_state:
            st.info(
                "Set parameters in the sidebar and click **🔍 Find Historical Mixes** "
                "to retrieve similar mixes from the field dataset."
            )
            with st.expander("📊 Dataset Overview"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Records",  len(df))
                c2.metric("28-Day Range",   f"{df['28day'].min():.1f}–{df['28day'].max():.1f} MPa")
                c3.metric("GWP Range",      f"{df['GWP'].min():.0f}–{df['GWP'].max():.0f} kg CO₂/m³")
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(
                    x=df['GWP'].dropna(), name='GWP', nbinsx=30,
                    marker_color=GREEN, opacity=0.75,
                ))
                fig_h.add_trace(go.Histogram(
                    x=df['28day'].dropna() * (145.038 if unit_sys=='Imperial' else 1),
                    name=f'28-Day ({sl})', nbinsx=30,
                    marker_color=NJDOT_BLUE, opacity=0.75,
                ))
                fig_h.update_layout(
                    barmode='overlay', height=250,
                    legend=dict(orientation='h'),
                    margin=dict(t=10, b=10),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_h, use_container_width=True)
        else:
            df_hist   = st.session_state['hist_result']
            t28_mpa   = st.session_state['hist_min_28d']

            st.subheader("Historical Similar Mixes (from field data)")

            if df_hist.empty:
                st.warning(
                    "No mixes satisfy all filters. "
                    "Try raising GWP limit, lowering strength target, or relaxing w/cm."
                )
            else:
                st.info(
                    f"Found **{len(df_hist)} mixes**. Sorted by lowest GWP first."
                )

                df_disp = df_hist.copy()
                for col in [f for f in RAW_FEATURES if f not in ADMIX]:
                    if col in df_disp.columns:
                        df_disp[col] = (df_disp[col] * um).round(1)
                if 'TOTAL_BINDER' in df_disp.columns:
                    df_disp['TOTAL_BINDER'] = (df_disp['TOTAL_BINDER'] * um).round(1)
                for col in ['7day','28day','56day']:
                    if col in df_disp.columns:
                        df_disp[col] = (df_disp[col] * us).round(1)

                rename = {
                    'PC':'Cement','FA':'Fly Ash','SC':'Slag','FAGG':'Fine Agg.',
                    'CAGG':'Coarse Agg.','WATER':'Water','AEA':'AEA',
                    'WR_HR':'HRWR','WR':'WR','ACC':'Acc.',
                    'TOTAL_BINDER':'Total Cem.','w/b':'w/cm','SCM%':'SCM%',
                    'GWP':'GWP','7day':'7-Day','28day':'28-Day','56day':'56-Day',
                }
                df_disp = df_disp.rename(columns=rename)
                df_disp.index = [f"Mix {i+1}" for i in range(len(df_disp))]

                st.dataframe(
                    df_disp,
                    column_config={
                        '28-Day': st.column_config.ProgressColumn(
                            f"28-Day ({sl})",
                            min_value=0,
                            max_value=float(df['28day'].max() * us),
                            format=f"%.1f",
                        ),
                        'GWP': st.column_config.ProgressColumn(
                            "GWP (kg CO₂/m³)",
                            min_value=0,
                            max_value=float(df['GWP'].max()),
                            format="%.0f",
                        ),
                    },
                    use_container_width=True,
                )

                st.download_button(
                    "📥 Download CSV",
                    data=hist_csv(df_hist),
                    file_name="njdot_historical_mixes.csv",
                    mime="text/csv",
                )



if __name__ == '__main__':
    main()
