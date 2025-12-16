from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .config import NUMERIC_COLUMNS, DIMENSION_COLUMNS


def detect_anomalies(
    weekly_df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
    contamination: float = 0.1,
) -> pd.DataFrame:
    """
    Fit an IsolationForest per group (if group_cols given) and flag anomalous weeks.

    Defensive version:
    - Ignores metric columns that don't exist.
    - Ignores grouping columns that don't exist.
    - Returns safe defaults if there isn't enough usable data.
    """
    df = weekly_df.copy()

    if df.empty:
        df["anomaly_score"] = np.nan
        df["is_anomaly"] = False
        return df

    # Keep only existing metric cols
    metric_cols = [c for c in metric_cols if c in df.columns]

    # Keep only existing group cols
    group_cols = [c for c in group_cols if c in df.columns]

    # If no metrics are available, cannot score anomalies
    if not metric_cols:
        df["anomaly_score"] = np.nan
        df["is_anomaly"] = False
        return df

    def _fit_iforest(sub: pd.DataFrame) -> pd.DataFrame:
        if sub.empty:
            out = sub.copy()
            out["anomaly_score"] = np.nan
            out["is_anomaly"] = False
            return out

        # Use requested metrics + Week if present
        cols = list(metric_cols)
        if "Week" in sub.columns and "Week" not in cols:
            cols.append("Week")

        # Guard against any missing cols (belt + braces)
        cols = [c for c in cols if c in sub.columns]
        if not cols:
            out = sub.copy()
            out["anomaly_score"] = np.nan
            out["is_anomaly"] = False
            return out

        X = sub[cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Simple standardisation so large metrics do not dominate
        std = X.std(ddof=0).replace(0, 1.0)
        X = (X - X.mean()) / std

        # If we have too few rows, IsolationForest can be unstable
        if len(X) < 3:
            out = sub.copy()
            out["anomaly_score"] = np.nan
            out["is_anomaly"] = False
            return out

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        )
        model.fit(X)

        scores = -model.score_samples(X)  # higher = more anomalous
        preds = model.predict(X)

        out = sub.copy()
        out["anomaly_score"] = scores
        out["is_anomaly"] = preds == -1
        return out

    if not group_cols:
        return _fit_iforest(df)

    parts = []
    for _, sub in df.groupby(group_cols, dropna=False):
        parts.append(_fit_iforest(sub))

    return pd.concat(parts, ignore_index=True)


def feature_importance_engagement(
    df: pd.DataFrame,
    target_col: str = "Engaged Visits",
) -> Dict[str, float]:
    """
    RandomForest based feature importance for engagement.

    Uses NUMERIC_COLUMNS and DIMENSION_COLUMNS from config.
    Defensive version:
    - Only uses columns present in df.
    - Cleans numerics and fills categoricals.
    - Returns {} if target or usable features are missing.
    """
    df_model = df.copy()

    if target_col not in df_model.columns:
        return {}

    df_model = df_model.dropna(subset=[target_col])
    if df_model.empty:
        return {}

    # Columns driven by config, but only those present
    numeric_cols = [c for c in NUMERIC_COLUMNS if c in df_model.columns and c != target_col]
    cat_cols = [c for c in DIMENSION_COLUMNS if c in df_model.columns]

    # If no usable features
    if not numeric_cols and not cat_cols:
        return {}

    # Target
    y = pd.to_numeric(df_model[target_col], errors="coerce")
    df_model = df_model.loc[~y.isna()].copy()
    y = y.loc[~y.isna()]
    if df_model.empty:
        return {}

    # Clean numeric
    for col in numeric_cols:
        df_model[col] = (
            pd.to_numeric(df_model[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    # Clean categoricals
    for col in cat_cols:
        s = df_model[col].astype("category")
        if "Unknown" not in s.cat.categories:
            s = s.cat.add_categories(["Unknown"])
        df_model[col] = s.fillna("Unknown")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X = df_model[numeric_cols + cat_cols]
    pipe.fit(X, y)

    rf = pipe.named_steps["model"]
    importances = rf.feature_importances_

    # Build a flat list of feature names
    ohe = pipe.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols)) if cat_cols else []
    all_features = numeric_cols + cat_feature_names

    # Safety in case of any shape mismatch
    if len(importances) != len(all_features):
        return {}

    imp_series = pd.Series(importances, index=all_features).sort_values(ascending=False)

    # Roll up importance to original dimension names
    summary: Dict[str, float] = {}
    for f, score in imp_series.items():
        if f in numeric_cols:
            key = f
        else:
            key = f.split("_", maxsplit=1)[0]
        summary[key] = summary.get(key, 0.0) + float(score)

    total = sum(summary.values()) or 1.0
    summary = {k: v / total for k, v in summary.items()}

    return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))
