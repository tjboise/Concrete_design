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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from recommender import (
    ConcreteRecommender, compute_derived_vector,
    RAW_FEATURES, ALL_FEATURES, TARGETS,
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

BASE_DIR   = os.path.dirname(__file__)
DATA_PATH  = os.path.join(BASE_DIR, 'Super_Cleaned_Concrete_Data - backup.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Admixture features are in mL/100 kg — not converted by mass unit toggle
ADMIX_FEATURES = {'AEA', 'WR_HR', 'WR', 'ACC'}
MASS_FEATURES  = [f for f in RAW_FEATURES if f not in ADMIX_FEATURES]

MATERIAL_LABELS = {
    'PC':    'Portland Cement',
    'FA':    'Fly Ash',
    'SC':    'Slag Cement',
    'SF':    'Silica Fume',
    'FAGG':  'Fine Aggregate',
    'CAGG':  'Coarse Aggregate',
    'WATER': 'Water',
    'AEA':   'Air-Entraining Agent (mL/100 kg)',
    'WR_HR': 'High-Range Water Reducer (mL/100 kg)',
    'WR':    'Water Reducer (mL/100 kg)',
    'ACC':   'Accelerator (mL/100 kg)',
}

# Unit conversion factors
UNIT_S = {'Metric': 1.0,      'Imperial': 145.038}   # MPa  → psi
UNIT_M = {'Metric': 1.0,      'Imperial': 1.68556}   # kg/m³ → lb/yd³
UNIT_SL = {'Metric': 'MPa',   'Imperial': 'psi'}
UNIT_ML = {'Metric': 'kg/m³', 'Imperial': 'lb/yd³'}

NJDOT_BLUE = '#003087'
COLORS = dict(primary=NJDOT_BLUE, success='#10B981',
              warning='#F59E0B', danger='#EF4444')


# ---------------------------------------------------------------------------
# Model loading / auto-training
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_or_train():
    from sklearn.model_selection import train_test_split
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    numeric_cols = ['PC','FA','SC','SF','FAGG','CAGG','WATER','AEA','WR_HR','WR','ACC',
                    '7day','28day','56day']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['TOTAL_BINDER'] = df['PC'] + df['FA'] + df['SC'] + df['SF']
    df['w/b']   = df['WATER'] / df['TOTAL_BINDER'].replace(0, np.nan)
    df['b/a']   = df['TOTAL_BINDER'] / (df['FAGG'] + df['CAGG']).replace(0, np.nan)
    df['SCM%']  = (df['FA'] + df['SC'] + df['SF']) / df['TOTAL_BINDER'].replace(0, np.nan)
    df['CAGG%'] = df['CAGG'] / (df['FAGG'] + df['CAGG']).replace(0, np.nan)
    df['FAGG%'] = df['FAGG'] / (df['FAGG'] + df['CAGG']).replace(0, np.nan)

    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    loaded = False
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            models = {t: joblib.load(os.path.join(MODELS_DIR, f'model_{t}.pkl')) for t in TARGETS}
            loaded = True
        except Exception:
            loaded = False

    if not loaded:
        X = df[ALL_FEATURES].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        models = {}
        for target in TARGETS:
            mask = df[target].notna().values
            X_t, y_t = X_scaled[mask], df.loc[mask, target].values
            X_tr, _, y_tr, _ = train_test_split(X_t, y_t, test_size=0.2, random_state=42)
            model = GradientBoostingRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.04,
                subsample=0.8, min_samples_leaf=3, random_state=42,
            )
            model.fit(X_tr, y_tr)
            models[target] = model
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            for t, m in models.items():
                joblib.dump(m, os.path.join(MODELS_DIR, f'model_{t}.pkl'))
        except Exception:
            pass

    rec = ConcreteRecommender(df, scaler, models)
    return df, rec


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def chart_material_comparison(df_hist: pd.DataFrame, um: float, ml: str) -> go.Figure:
    show = ['PC', 'FA', 'SC', 'SF', 'FAGG', 'CAGG', 'WATER']
    label_map = {
        'PC': 'Portland Cement', 'FA': 'Fly Ash', 'SC': 'Slag Cement',
        'SF': 'Silica Fume', 'FAGG': 'Fine Agg.', 'CAGG': 'Coarse Agg.', 'WATER': 'Water',
    }
    palette = [NJDOT_BLUE, '#1D6FA4', '#34D399', '#6EE7B7', '#D1D5DB', '#9CA3AF', '#F59E0B']
    fig = go.Figure()
    labels = [f"Mix {i+1}" for i in range(len(df_hist))]
    for i, col in enumerate(show):
        if col not in df_hist.columns:
            continue
        fig.add_trace(go.Bar(
            name=label_map.get(col, col),
            x=labels,
            y=(df_hist[col] * um).tolist(),
            marker_color=palette[i],
        ))
    fig.update_layout(
        barmode='group', height=340,
        legend=dict(orientation='h', y=-0.28),
        yaxis_title=f'Quantity ({ml})',
        margin=dict(t=10, b=10, l=0, r=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def chart_strength_comparison(df_hist: pd.DataFrame, target_28_mpa: float,
                               us: float, sl: str) -> go.Figure:
    labels = [f"Mix {i+1}" for i in range(len(df_hist))]
    fig = go.Figure()
    for col, lbl, color in [('7day','7-Day','#60A5FA'),
                             ('28day','28-Day', NJDOT_BLUE),
                             ('56day','56-Day','#1E3A8A')]:
        if col in df_hist.columns:
            fig.add_trace(go.Scatter(
                x=labels,
                y=(df_hist[col] * us).tolist(),
                mode='lines+markers', name=lbl,
                line=dict(color=color, width=2), marker=dict(size=8),
            ))
    fig.add_hline(
        y=target_28_mpa * us, line_dash='dash', line_color=COLORS['danger'],
        annotation_text=f"Target 28-Day: {target_28_mpa*us:.0f} {sl}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        height=300, yaxis_title=f'Compressive Strength ({sl})',
        legend=dict(orientation='h', y=-0.35),
        margin=dict(t=10, b=10, l=0, r=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def chart_mix_pie(mix: dict) -> go.Figure:
    label_map = {
        'PC': 'Portland Cement', 'FA': 'Fly Ash', 'SC': 'Slag Cement',
        'SF': 'Silica Fume', 'FAGG': 'Fine Aggregate',
        'CAGG': 'Coarse Aggregate', 'WATER': 'Water',
    }
    labels, values = [], []
    for k, lbl in label_map.items():
        v = mix.get(k, 0)
        if v > 1:
            labels.append(lbl)
            values.append(round(v, 1))
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.4,
        marker=dict(colors=[NJDOT_BLUE,'#1D6FA4','#34D399','#6EE7B7',
                             '#D1D5DB','#9CA3AF','#F59E0B']),
        textinfo='label+percent',
    ))
    fig.update_layout(
        height=280, showlegend=False,
        margin=dict(t=10, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def gauge_chart(predicted: dict, target_28_mpa: float, us: float, sl: str) -> go.Figure:
    """Three gauge dials for 7 / 28 / 56-day predicted strength."""
    max_val = 120 * us
    specs = [[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
    fig = make_subplots(rows=1, cols=3, specs=specs)

    entries = [
        ('7day',  '7-Day',  None),
        ('28day', '28-Day', target_28_mpa * us),
        ('56day', '56-Day', None),
    ]
    for col_idx, (key, lbl, threshold_val) in enumerate(entries, start=1):
        val = (predicted.get(key) or 0) * us
        gauge_cfg = {
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': 'gray'},
            'bar': {'color': NJDOT_BLUE},
            'bgcolor': 'rgba(0,0,0,0)',
            'steps': [
                {'range': [0, max_val * 0.4], 'color': '#FEE2E2'},
                {'range': [max_val * 0.4, max_val * 0.65], 'color': '#FEF9C3'},
                {'range': [max_val * 0.65, max_val], 'color': '#DCFCE7'},
            ],
        }
        if threshold_val is not None:
            gauge_cfg['threshold'] = {
                'line': {'color': COLORS['danger'], 'width': 3},
                'thickness': 0.75,
                'value': threshold_val,
            }
        fig.add_trace(
            go.Indicator(
                mode='gauge+number',
                value=val,
                title={'text': f'<b>{lbl}</b><br><span style="font-size:0.75em;color:gray">{sl}</span>'},
                number={'font': {'size': 22}, 'valueformat': '.1f'},
                gauge=gauge_cfg,
            ),
            row=1, col=col_idx,
        )

    fig.update_layout(
        height=240,
        margin=dict(t=50, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def chart_strength_curve(predicted: dict, target_28_mpa: float,
                          us: float, sl: str) -> go.Figure:
    days = [7, 28, 56]
    vals = [(predicted.get(k) or 0) * us for k in ['7day', '28day', '56day']]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=vals, mode='lines+markers',
        name='Predicted',
        line=dict(color=NJDOT_BLUE, width=3), marker=dict(size=10),
        fill='tozeroy', fillcolor='rgba(0,48,135,0.10)',
    ))
    fig.add_hline(
        y=target_28_mpa * us, line_dash='dash', line_color=COLORS['danger'],
        annotation_text=f"Target {target_28_mpa*us:.0f} {sl}",
    )
    fig.update_layout(
        xaxis=dict(tickvals=[7, 28, 56], title='Age (days)'),
        yaxis_title=f'Compressive Strength ({sl})',
        height=260, margin=dict(t=10, b=10, l=0, r=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def hist_csv(df_hist: pd.DataFrame) -> str:
    return df_hist.to_csv(index=False)


def opt_csv(mix: dict, predicted: dict, tb: float, wb: float | None,
            scm_pct: float) -> str:
    rows = [{'Parameter': MATERIAL_LABELS.get(k, k) + ' (kg/m³)', 'Value': round(v, 1)}
            for k, v in mix.items() if v > 0.5]
    rows += [
        {'Parameter': 'Total Cementitious (kg/m³)', 'Value': round(tb, 1)},
        {'Parameter': 'w/cm Ratio',                 'Value': round(wb, 3) if wb else '—'},
        {'Parameter': 'SCM Replacement (%)',         'Value': round(scm_pct * 100, 1)},
        {'Parameter': 'Pred. 7-Day (MPa)',           'Value': round(predicted.get('7day', 0), 1)},
        {'Parameter': 'Pred. 28-Day (MPa)',          'Value': round(predicted.get('28day', 0), 1)},
        {'Parameter': 'Pred. 56-Day (MPa)',          'Value': round(predicted.get('56day', 0), 1)},
    ]
    return pd.DataFrame(rows).to_csv(index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Header
    st.markdown(
        f"<h2 style='color:{NJDOT_BLUE};margin-bottom:0'>🏗️ Concrete Mix Design Advisor</h2>",
        unsafe_allow_html=True,
    )
    st.caption(
        "**NJDOT — New Jersey Department of Transportation** | "
        "756 field records · Historical similar mixes & optimized new mix"
    )
    st.divider()

    with st.spinner("Loading prediction models… (first run trains automatically, ~30 s)"):
        df, rec = load_or_train()

    st.success(
        f"✅ Models ready — **{len(df)} mix records** | "
        "7-day R²=0.71 · 28-day R²=0.80 · 56-day R²=0.79"
    )

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("Design Parameters")

        # ── Unit system ──────────────────────────────────────────────────
        unit_sys = st.radio("Unit System", ["Metric", "Imperial"], horizontal=True)
        us  = UNIT_S[unit_sys]   # strength multiplier
        um  = UNIT_M[unit_sys]   # mass multiplier
        sl  = UNIT_SL[unit_sys]  # strength label
        ml  = UNIT_ML[unit_sys]  # mass label
        st.divider()

        # ── Target strength ──────────────────────────────────────────────
        st.subheader("Target Strength")
        if unit_sys == 'Metric':
            target_28_disp = st.slider("28-Day Strength (MPa)", 20, 100, 40, step=1)
            target_28_mpa  = float(target_28_disp)
        else:
            target_28_disp = st.slider("28-Day Strength (psi)", 2900, 14500, 5800, step=100)
            target_28_mpa  = target_28_disp / 145.038

        enable_7day = st.checkbox("Also set 7-day minimum")
        target_7_mpa = None
        if enable_7day:
            if unit_sys == 'Metric':
                target_7_mpa = float(st.slider("7-Day Minimum (MPa)", 10, 70, 25, step=1))
            else:
                target_7_mpa = st.slider("7-Day Minimum (psi)", 1450, 10000, 3600, step=100) / 145.038

        enable_56day = st.checkbox("Also set 56-day minimum")
        target_56_mpa = None
        if enable_56day:
            if unit_sys == 'Metric':
                target_56_mpa = float(st.slider("56-Day Minimum (MPa)", 20, 110, 50, step=1,
                                                 help="56-day model trained on fewer records (235); predictions are less certain."))
            else:
                target_56_mpa = st.slider("56-Day Minimum (psi)", 2900, 16000, 7300, step=100,
                                           help="56-day model trained on fewer records (235); predictions are less certain.") / 145.038

        # ── SCMs ─────────────────────────────────────────────────────────
        st.subheader("Available SCMs")
        use_fa = st.checkbox("Fly Ash (FA)", value=True)
        use_sc = st.checkbox("Slag Cement (SC)", value=True)
        use_sf = st.checkbox("Silica Fume (SF)", value=False)

        # ── Constraints ───────────────────────────────────────────────────
        st.subheader("Mix Constraints")
        max_wb = st.slider("Maximum w/cm Ratio", 0.30, 0.60, 0.50, step=0.01)

        if unit_sys == 'Metric':
            max_binder = st.slider("Max Total Cementitious (kg/m³)", 400, 800, 650, step=10)
            min_binder = st.slider("Min Total Cementitious (kg/m³)", 200, 500, 300, step=10)
        else:
            max_binder = st.slider("Max Total Cementitious (lb/yd³)", 675, 1350, 1095, step=17) / 1.68556
            min_binder = st.slider("Min Total Cementitious (lb/yd³)", 337, 843, 506, step=17) / 1.68556

        st.subheader("Results")
        n_hist = st.slider("Historical mixes to return", 3, 10, 5)
        st.divider()
        run_btn = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)

    # -----------------------------------------------------------------------
    # Run computations and cache in session state
    # -----------------------------------------------------------------------
    if run_btn:
        df_hist = rec.recommend_historical(
            target_28day=target_28_mpa,
            use_fa=use_fa, use_sc=use_sc, use_sf=use_sf,
            max_wb=max_wb, max_binder=max_binder,
            target_7day=target_7_mpa, target_56day=target_56_mpa,
            n_results=n_hist,
        )
        st.session_state['hist_result']  = df_hist
        st.session_state['target_28_mpa'] = target_28_mpa

        with st.spinner("Running optimization — typically 20–50 seconds…"):
            opt_result = rec.recommend_optimized(
                target_28day=target_28_mpa,
                use_fa=use_fa, use_sc=use_sc, use_sf=use_sf,
                max_wb=max_wb, target_7day=target_7_mpa, target_56day=target_56_mpa,
                min_binder=min_binder, max_binder=max_binder,
            )
        st.session_state['opt_result'] = opt_result
        # Clear any previous adjustment result
        st.session_state.pop('adj_predicted', None)

    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------
    tab1, tab2 = st.tabs(["📚 Historical Similar Mixes", "⚗️ Optimized New Mix"])

    # ── Tab 1 ───────────────────────────────────────────────────────────────
    with tab1:
        if 'hist_result' not in st.session_state:
            st.info("Set design parameters in the sidebar, then click **Get Recommendations**.")
            with st.expander("📊 Dataset Overview"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Records", len(df))
                c2.metric("28-Day Range",
                           f"{df['28day'].min()*us:.0f}–{df['28day'].max()*us:.0f} {sl}")
                c3.metric("w/cm Range",
                           f"{df['w/b'].min():.2f}–{df['w/b'].max():.2f}")
                fig_h = px.histogram(
                    df['28day'].dropna() * us, nbins=30,
                    labels={'value': f'28-Day Strength ({sl})', 'count': 'Count'},
                    color_discrete_sequence=[NJDOT_BLUE],
                )
                fig_h.update_layout(height=230, showlegend=False,
                                     margin=dict(t=10, b=10),
                                     plot_bgcolor='rgba(0,0,0,0)',
                                     paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_h, use_container_width=True)
        else:
            df_hist      = st.session_state['hist_result']
            t28_mpa_disp = st.session_state['target_28_mpa']

            st.subheader("Historical Similar Mixes (from field data)")
            if df_hist.empty:
                st.warning(
                    "No mixes satisfy all constraints. "
                    "Try relaxing the strength target, w/cm, or binder limits."
                )
            else:
                st.info(
                    f"Found **{len(df_hist)} mixes** with 28-day strength "
                    f"≥ {t28_mpa_disp*us:.0f} {sl} and w/cm ≤ {max_wb}. "
                    "Sorted by minimum over-design margin."
                )

                # Build display DataFrame
                rename = {
                    'PC':'Cement','FA':'Fly Ash','SC':'Slag','SF':'Silica Fume',
                    'FAGG':'Fine Agg.','CAGG':'Coarse Agg.','WATER':'Water',
                    'AEA':'AEA','WR_HR':'HRWR','WR':'WR','ACC':'Acc.',
                    'TOTAL_BINDER':'Total Cem.','w/b':'w/cm','SCM%':'SCM%',
                    '7day':'7-Day','28day':'28-Day','56day':'56-Day',
                }
                df_disp = df_hist.copy()
                # Apply unit conversion to mass columns
                for col in MASS_FEATURES + ['TOTAL_BINDER']:
                    if col in df_disp.columns:
                        df_disp[col] = (df_disp[col] * um).round(1)
                # Apply unit conversion to strength columns
                for col in ['7day','28day','56day']:
                    if col in df_disp.columns:
                        df_disp[col] = (df_disp[col] * us).round(1)
                df_disp = df_disp.rename(columns=rename)
                df_disp.index = [f"Mix {i+1}" for i in range(len(df_disp))]

                st.dataframe(
                    df_disp,
                    column_config={
                        '28-Day': st.column_config.ProgressColumn(
                            f"28-Day ({sl})",
                            min_value=0,
                            max_value=float(df['28day'].max() * us),
                            format=f"%.1f {sl}",
                        ),
                        '7-Day':  st.column_config.NumberColumn(f"7-Day ({sl})",  format="%.1f"),
                        '56-Day': st.column_config.NumberColumn(f"56-Day ({sl})", format="%.1f"),
                        'w/cm':   st.column_config.NumberColumn("w/cm", format="%.3f"),
                        'SCM%':   st.column_config.NumberColumn("SCM%", format="%.2f"),
                    },
                    use_container_width=True,
                )

                # Download
                st.download_button(
                    "📥 Download CSV",
                    data=hist_csv(df_hist),
                    file_name="njdot_historical_mixes.csv",
                    mime="text/csv",
                )

                st.markdown("**Material Quantities**")
                st.plotly_chart(chart_material_comparison(df_hist, um, ml), use_container_width=True)
                st.markdown("**Compressive Strength by Age**")
                st.plotly_chart(chart_strength_comparison(df_hist, t28_mpa_disp, us, sl),
                                use_container_width=True)

    # ── Tab 2 ───────────────────────────────────────────────────────────────
    with tab2:
        if 'opt_result' not in st.session_state:
            st.info("Set design parameters in the sidebar, then click **Get Recommendations**.")
        else:
            result       = st.session_state['opt_result']
            t28_mpa_disp = st.session_state['target_28_mpa']
            mix          = result['mix']
            predicted    = result['predicted']
            tb           = result['total_binder']
            wb           = result['wb_ratio']
            scm_pct      = result['scm_pct']

            st.subheader("Optimized New Mix Design")
            st.caption(
                "Differential evolution minimizes Portland cement content "
                "while satisfying all strength and mix constraints."
            )

            if result['success']:
                st.success("✅ Feasible mix found")
            else:
                st.warning("⚠️ Best feasible approximation — try relaxing constraints")

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Portland Cement",   f"{mix['PC']*um:.0f} {ml}")
            m2.metric("Total Cementitious", f"{tb*um:.0f} {ml}")
            m3.metric("w/cm Ratio",         f"{wb:.3f}" if wb else "—")
            m4.metric("SCM Replacement",    f"{scm_pct*100:.1f}%")

            st.divider()

            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown(f"**Recommended Mix Proportions (per m³ / per yd³)**")
                mix_df = pd.DataFrame([
                    {
                        'Material': MATERIAL_LABELS.get(k, k),
                        f'Quantity': (
                            f"{v*um:.1f} {ml}" if k not in ADMIX_FEATURES
                            else f"{v:.1f} mL/100 kg"
                        ),
                    }
                    for k, v in mix.items() if v > 0.5
                ])
                st.dataframe(mix_df, use_container_width=True, hide_index=True)

                st.download_button(
                    "📥 Download Mix Design CSV",
                    data=opt_csv(mix, predicted, tb, wb, scm_pct),
                    file_name="njdot_optimized_mix.csv",
                    mime="text/csv",
                )

            with col_b:
                st.markdown("**Mix Composition**")
                st.plotly_chart(chart_mix_pie(mix), use_container_width=True)

            st.divider()
            st.markdown("**Predicted Compressive Strength**")
            st.plotly_chart(gauge_chart(predicted, t28_mpa_disp, us, sl),
                            use_container_width=True)
            st.plotly_chart(chart_strength_curve(predicted, t28_mpa_disp, us, sl),
                            use_container_width=True)

            # ── Adjust & Recalculate ─────────────────────────────────────
            st.divider()
            with st.expander("🔧 Adjust Mix & Recalculate Strength"):
                st.caption(
                    "Manually edit any material quantity to instantly see how "
                    "predicted strength changes. Quantities are in metric (kg/m³)."
                )
                adj_cols = st.columns(4)
                adj_mix: dict[str, float] = {}
                for i, feat in enumerate(RAW_FEATURES):
                    with adj_cols[i % 4]:
                        adj_mix[feat] = st.number_input(
                            MATERIAL_LABELS.get(feat, feat).split(' (')[0],
                            value=round(float(mix[feat]), 1),
                            min_value=0.0, step=1.0,
                            key=f"adj_{feat}",
                        )

                if st.button("🔄 Recalculate Strength", key="recalc_btn"):
                    st.session_state['adj_predicted'] = rec.predict_all(adj_mix)
                    st.session_state['adj_mix'] = adj_mix

                if 'adj_predicted' in st.session_state:
                    adj_pred = st.session_state['adj_predicted']
                    st.markdown("**Adjusted Predicted Strength:**")
                    a1, a2, a3 = st.columns(3)
                    a1.metric("7-Day",  f"{adj_pred['7day']*us:.1f} {sl}")
                    a2.metric(
                        "28-Day", f"{adj_pred['28day']*us:.1f} {sl}",
                        delta=f"{(adj_pred['28day'] - t28_mpa_disp)*us:+.1f} {sl} vs target",
                        delta_color="normal" if adj_pred['28day'] >= t28_mpa_disp else "inverse",
                    )
                    a3.metric("56-Day", f"{adj_pred['56day']*us:.1f} {sl}")


if __name__ == '__main__':
    main()
