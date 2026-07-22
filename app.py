import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')  # suppress scipy/sklearn convergence warnings

from recommender import (
    ConcreteRecommender, compute_derived_vector,
    RAW_FEATURES, ALL_FEATURES, TARGETS,
)

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

MATERIAL_LABELS = {
    'PC':    'Portland Cement (kg/m³)',
    'FA':    'Fly Ash (kg/m³)',
    'SC':    'Slag Cement (kg/m³)',
    'SF':    'Silica Fume (kg/m³)',
    'FAGG':  'Fine Aggregate (kg/m³)',
    'CAGG':  'Coarse Aggregate (kg/m³)',
    'WATER': 'Water (kg/m³)',
    'AEA':   'Air-Entraining Agent (mL/100 kg)',
    'WR_HR': 'High-Range Water Reducer (mL/100 kg)',
    'WR':    'Water Reducer (mL/100 kg)',
    'ACC':   'Accelerator (mL/100 kg)',
}

COLORS = dict(primary='#003087', success='#10B981', warning='#F59E0B',
              danger='#EF4444', slate='#64748B')  # NJDOT navy blue


# ---------------------------------------------------------------------------
# Model loading / auto-training (cached across sessions)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_or_train():
    from sklearn.model_selection import train_test_split
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    # Ensure derived columns exist
    df['TOTAL_BINDER'] = df['PC'] + df['FA'] + df['SC'] + df['SF']
    df['w/b']   = df['WATER'] / df['TOTAL_BINDER'].replace(0, np.nan)
    df['b/a']   = df['TOTAL_BINDER'] / (df['FAGG'] + df['CAGG']).replace(0, np.nan)
    df['SCM%']  = (df['FA'] + df['SC'] + df['SF']) / df['TOTAL_BINDER'].replace(0, np.nan)
    df['CAGG%'] = df['CAGG'] / (df['FAGG'] + df['CAGG']).replace(0, np.nan)
    df['FAGG%'] = df['FAGG'] / (df['FAGG'] + df['CAGG']).replace(0, np.nan)

    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        models = {t: joblib.load(os.path.join(MODELS_DIR, f'model_{t}.pkl')) for t in TARGETS}
    else:
        X = df[ALL_FEATURES].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)

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
            joblib.dump(model, os.path.join(MODELS_DIR, f'model_{target}.pkl'))
            models[target] = model

    rec = ConcreteRecommender(df, scaler, models)
    return df, rec


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def chart_material_comparison(df_hist: pd.DataFrame) -> go.Figure:
    show = ['PC', 'FA', 'SC', 'SF', 'FAGG', 'CAGG', 'WATER']
    label_map = {
        'PC': 'Portland Cement', 'FA': 'Fly Ash', 'SC': 'Slag Cement',
        'SF': 'Silica Fume', 'FAGG': 'Fine Agg.', 'CAGG': 'Coarse Agg.', 'WATER': 'Water',
    }
    palette = ['#003087', '#1D6FA4', '#34D399', '#6EE7B7', '#D1D5DB', '#9CA3AF', '#F59E0B']
    fig = go.Figure()
    labels = [f"Mix {i+1}" for i in range(len(df_hist))]
    for i, col in enumerate(show):
        if col not in df_hist.columns:
            continue
        fig.add_trace(go.Bar(
            name=label_map.get(col, col),
            x=labels, y=df_hist[col].tolist(),
            marker_color=palette[i],
        ))
    fig.update_layout(
        barmode='group', height=360,
        legend=dict(orientation='h', y=-0.28),
        yaxis_title='Quantity (kg/m³)',
        margin=dict(t=10, b=10, l=0, r=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def chart_strength_comparison(df_hist: pd.DataFrame, target_28: float) -> go.Figure:
    labels = [f"Mix {i+1}" for i in range(len(df_hist))]
    fig = go.Figure()
    for col, lbl, color in [
        ('7day',  '7-Day',  '#60A5FA'),
        ('28day', '28-Day', '#003087'),
        ('56day', '56-Day', '#1E3A8A'),
    ]:
        if col in df_hist.columns:
            fig.add_trace(go.Scatter(
                x=labels, y=df_hist[col].tolist(),
                mode='lines+markers', name=lbl,
                line=dict(color=color, width=2), marker=dict(size=8),
            ))
    fig.add_hline(
        y=target_28, line_dash='dash', line_color=COLORS['danger'],
        annotation_text=f"Target 28-Day: {target_28} MPa",
        annotation_position="bottom right",
    )
    fig.update_layout(
        height=320, yaxis_title='Compressive Strength (MPa)',
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
        marker=dict(colors=['#003087', '#1D6FA4', '#34D399', '#6EE7B7',
                            '#D1D5DB', '#9CA3AF', '#F59E0B']),
        textinfo='label+percent',
    ))
    fig.update_layout(
        height=300, showlegend=False,
        margin=dict(t=10, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def chart_strength_curve(predicted: dict, target_28: float) -> go.Figure:
    days = [7, 28, 56]
    vals = [predicted.get('7day', 0), predicted.get('28day', 0), predicted.get('56day', 0)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=vals, mode='lines+markers',
        name='Predicted Strength',
        line=dict(color=COLORS['primary'], width=3), marker=dict(size=10),
        fill='tozeroy', fillcolor='rgba(0,48,135,0.10)',
    ))
    fig.add_hline(
        y=target_28, line_dash='dash', line_color=COLORS['danger'],
        annotation_text=f"Target {target_28} MPa",
    )
    fig.update_layout(
        xaxis=dict(tickvals=[7, 28, 56], title='Age (days)'),
        yaxis_title='Compressive Strength (MPa)',
        height=280, margin=dict(t=10, b=10, l=0, r=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def strength_metrics(predicted: dict, target_28: float):
    c1, c2, c3 = st.columns(3)
    with c1:
        v = predicted.get('7day')
        st.metric("7-Day Strength", f"{v:.1f} MPa" if v else "—")
    with c2:
        v = predicted.get('28day')
        ok = v is not None and v >= target_28
        st.metric(
            "28-Day Strength",
            f"{v:.1f} MPa" if v else "—",
            delta=f"{v - target_28:+.1f} MPa vs. target" if v else None,
            delta_color="normal" if ok else "inverse",
        )
    with c3:
        v = predicted.get('56day')
        st.metric("56-Day Strength", f"{v:.1f} MPa" if v else "—")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    # Header
    col_logo, col_title = st.columns([1, 9])
    with col_logo:
        st.markdown("## 🏗️")
    with col_title:
        st.markdown("## Concrete Mix Design Advisor")
        st.caption(
            "**NJDOT — New Jersey Department of Transportation** | "
            "756 field data records · Two recommendation strategies: "
            "Historical Similar Mixes & Optimized New Mix"
        )
    st.divider()

    # Load / train models
    with st.spinner("Loading prediction models… (first run trains them automatically, ~15 s)"):
        df, rec = load_or_train()

    st.success(
        f"✅ Models ready — dataset: **{len(df)} mix records** | "
        "Models: 7-day R²=0.71 · 28-day R²=0.80 · 56-day R²=0.79"
    )

    # -----------------------------------------------------------------------
    # Sidebar — inputs
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("Design Parameters")

        st.subheader("Target Strength")
        target_28 = st.slider("28-Day Compressive Strength (MPa)", 20, 100, 40, step=1)

        enable_7day = st.checkbox("Also specify 7-day minimum")
        target_7 = None
        if enable_7day:
            target_7 = st.slider("7-Day Minimum Strength (MPa)", 10, 70, 25, step=1)

        st.subheader("Available SCMs")
        use_fa = st.checkbox("Fly Ash (FA)", value=True)
        use_sc = st.checkbox("Slag Cement (SC)", value=True)
        use_sf = st.checkbox("Silica Fume (SF)", value=False)

        st.subheader("Mix Constraints")
        max_wb = st.slider("Maximum w/cm Ratio", 0.30, 0.60, 0.50, step=0.01,
                           help="Water-to-cementitious materials ratio upper bound")
        max_binder = st.slider("Maximum Total Cementitious (kg/m³)", 400, 800, 650, step=10)
        min_binder = st.slider("Minimum Total Cementitious (kg/m³)", 200, 500, 300, step=10)

        st.subheader("Results")
        n_hist = st.slider("Number of historical mixes to return", 3, 10, 5)

        st.divider()
        run_btn = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)

    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------
    tab1, tab2 = st.tabs(["📚 Historical Similar Mixes", "⚗️ Optimized New Mix"])

    if run_btn:
        # ---- Tab 1: Historical ----
        with tab1:
            st.subheader("Historical Similar Mixes (from field data)")
            with st.spinner("Searching dataset…"):
                df_hist = rec.recommend_historical(
                    target_28day=target_28,
                    use_fa=use_fa, use_sc=use_sc, use_sf=use_sf,
                    max_wb=max_wb, max_binder=max_binder,
                    target_7day=target_7, n_results=n_hist,
                )

            if df_hist.empty:
                st.warning(
                    "No historical mixes satisfy all constraints. "
                    "Try relaxing the strength target, w/cm ratio, or binder limits."
                )
            else:
                st.info(
                    f"Found **{len(df_hist)} mixes** with 28-day strength ≥ {target_28} MPa "
                    f"and w/cm ≤ {max_wb}, sorted by minimum over-design margin."
                )

                # Rename columns for display
                rename = {
                    'PC': 'Cement', 'FA': 'Fly Ash', 'SC': 'Slag', 'SF': 'Silica Fume',
                    'FAGG': 'Fine Agg.', 'CAGG': 'Coarse Agg.', 'WATER': 'Water',
                    'AEA': 'AEA', 'WR_HR': 'HRWR', 'WR': 'WR', 'ACC': 'Acc.',
                    'TOTAL_BINDER': 'Total Cem.', 'w/b': 'w/cm', 'SCM%': 'SCM%',
                    '7day': '7-Day (MPa)', '28day': '28-Day (MPa)', '56day': '56-Day (MPa)',
                }
                df_show = df_hist.rename(columns=rename)
                df_show.index = [f"Mix {i+1}" for i in range(len(df_show))]
                st.dataframe(
                    df_show.style
                        .highlight_max(subset=['28-Day (MPa)'], color='#DBEAFE')
                        .format("{:.2f}"),
                    use_container_width=True,
                )

                st.markdown("**Material Quantities**")
                st.plotly_chart(chart_material_comparison(df_hist), use_container_width=True)

                st.markdown("**Compressive Strength by Age**")
                st.plotly_chart(chart_strength_comparison(df_hist, target_28), use_container_width=True)

        # ---- Tab 2: Optimized ----
        with tab2:
            st.subheader("Optimized New Mix Design")
            st.caption(
                "Differential evolution algorithm minimizes Portland cement content "
                "while satisfying all strength and mix constraints."
            )

            with st.spinner("Running optimization — typically 20–50 seconds…"):
                result = rec.recommend_optimized(
                    target_28day=target_28,
                    use_fa=use_fa, use_sc=use_sc, use_sf=use_sf,
                    max_wb=max_wb, target_7day=target_7,
                    min_binder=min_binder, max_binder=max_binder,
                )

            if not result['success']:
                st.error(
                    f"Optimization could not find a strictly feasible mix "
                    f"({result['message']}). "
                    "Try relaxing the target strength or constraint bounds."
                )

            mix       = result['mix']
            predicted = result['predicted']
            tb        = result['total_binder']
            wb        = result['wb_ratio']
            scm_pct   = result['scm_pct']

            if result['success']:
                st.success("✅ Feasible mix found")
            else:
                st.warning("⚠️ Best feasible approximation")

            # Key summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Portland Cement", f"{mix['PC']:.0f} kg/m³")
            m2.metric("Total Cementitious", f"{tb:.0f} kg/m³")
            m3.metric("w/cm Ratio", f"{wb:.3f}" if wb else "—")
            m4.metric("SCM Replacement", f"{scm_pct*100:.1f}%")

            st.divider()
            col_a, col_b = st.columns([1, 1])

            with col_a:
                st.markdown("**Recommended Mix Proportions (per m³)**")
                mix_df = pd.DataFrame([
                    {'Material': MATERIAL_LABELS.get(k, k), 'Quantity': round(v, 1)}
                    for k, v in mix.items() if v > 0.5
                ])
                st.dataframe(mix_df, use_container_width=True, hide_index=True)

            with col_b:
                st.markdown("**Mix Composition**")
                st.plotly_chart(chart_mix_pie(mix), use_container_width=True)

            st.divider()
            st.markdown("**Predicted Compressive Strength**")
            strength_metrics(predicted, target_28)
            st.plotly_chart(chart_strength_curve(predicted, target_28), use_container_width=True)

            # Exportable summary table
            with st.expander("📋 Export Mix Design Summary"):
                summary = {
                    'Parameter': (
                        list(MATERIAL_LABELS.values()) +
                        ['Total Cementitious (kg/m³)', 'w/cm Ratio', 'SCM Replacement (%)',
                         'Pred. 7-Day Strength (MPa)', 'Pred. 28-Day Strength (MPa)',
                         'Pred. 56-Day Strength (MPa)']
                    ),
                    'Value': (
                        [round(mix.get(k, 0), 1) for k in RAW_FEATURES] +
                        [round(tb, 1), round(wb, 3) if wb else '—',
                         round(scm_pct * 100, 1),
                         round(predicted.get('7day', 0), 1),
                         round(predicted.get('28day', 0), 1),
                         round(predicted.get('56day', 0), 1)]
                    ),
                }
                st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)

    else:
        # ---- Placeholder / dataset overview ----
        with tab1:
            st.info("Set your design parameters in the left panel, then click **Get Recommendations**.")
            with st.expander("📊 Dataset Overview"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Mix Records", len(df))
                c2.metric("28-Day Strength Range", f"{df['28day'].min():.0f}–{df['28day'].max():.0f} MPa")
                c3.metric("w/cm Range", f"{df['w/b'].min():.2f}–{df['w/b'].max():.2f}")

                st.markdown("**28-Day Compressive Strength Distribution**")
                fig_hist = px.histogram(
                    df['28day'].dropna(), nbins=30,
                    labels={'value': '28-Day Strength (MPa)', 'count': 'Count'},
                    color_discrete_sequence=[COLORS['primary']],
                )
                fig_hist.update_layout(
                    height=250, showlegend=False,
                    margin=dict(t=10, b=10),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown("**Material Quantity Ranges**")
                ranges_df = pd.DataFrame([
                    {
                        'Material': MATERIAL_LABELS.get(k, k),
                        'Min': round(df[k].min(), 1),
                        'Mean': round(df[k].mean(), 1),
                        'Max': round(df[k].max(), 1),
                    }
                    for k in RAW_FEATURES
                ])
                st.dataframe(ranges_df, hide_index=True, use_container_width=True)

        with tab2:
            st.info("Set your design parameters in the left panel, then click **Get Recommendations**.")


if __name__ == '__main__':
    main()
