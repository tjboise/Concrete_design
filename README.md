# Concrete Mix Design Advisor — NJDOT

A data-driven web application that recommends concrete mix designs based on 756 field records collected for the New Jersey Department of Transportation (NJDOT).

**🚀 Live App: [https://YOUR-APP-URL.streamlit.app](https://YOUR-APP-URL.streamlit.app)**

---

## Overview

The app provides two recommendation strategies:

| Strategy | Description |
|----------|-------------|
| **Historical Similar Mixes** | Retrieves the closest matching mixes from the field dataset that satisfy the target strength and constraints |
| **Optimized New Mix** | Uses a differential evolution algorithm to generate a new mix design that minimizes Portland cement content while meeting all strength and mix requirements |

---

## Prediction Models

Three independent gradient boosting regression models predict compressive strength at different ages:

| Target | R² | RMSE |
|--------|----|------|
| 7-Day Strength | 0.71 | 5.00 MPa |
| 28-Day Strength | **0.80** | 5.24 MPa |
| 56-Day Strength | 0.79 | 6.01 MPa |

Models are trained on **17 features**: 11 raw material inputs + 6 derived engineering ratios (w/cm, b/a, SCM%, etc.).

---

## Input Parameters

| Parameter | Description |
|-----------|-------------|
| Target 28-Day Strength (MPa) | Required compressive strength |
| 7-Day Minimum (optional) | Early-strength requirement |
| Available SCMs | Fly Ash, Slag Cement, Silica Fume |
| Maximum w/cm Ratio | Water-to-cementitious materials ratio upper bound |
| Total Cementitious Range | Min/max binder content (kg/m³) |

---

## Dataset

- **756 mix records** from NJDOT field projects
- Materials: Portland Cement, Fly Ash, Slag Cement, Silica Fume, Fine/Coarse Aggregate, Water, and admixtures
- Strength measurements at 7, 28, and 56 days

---

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain the prediction models:

```bash
python train.py
```

---

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io)
- **ML Models**: scikit-learn GradientBoostingRegressor
- **Optimization**: scipy differential evolution
- **Visualization**: Plotly
