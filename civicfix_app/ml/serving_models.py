"""Build CivicFix's serving models from the project dataset in Python.

The fitted estimators stay in memory for the life of the app process. This keeps
the project source-first and avoids opaque serialized model files.
"""
from __future__ import annotations

import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from civicfix_app.ml.features import build_features


def _feature_pipeline() -> ColumnTransformer:
    return ColumnTransformer([
        ("text", TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True), "descriptor"),
        ("categories", OneHotEncoder(handle_unknown="ignore"), ["borough", "complaint_type"]),
    ])


def build_serving_models(frame: pd.DataFrame) -> dict:
    """Fit agency routing and resolution-time pipelines from the supplied data."""
    if frame.empty:
        return {}

    agency_rows = frame[frame.agency.notna() & frame.agency.astype(str).str.strip().ne("")].copy()
    agency_counts = agency_rows.agency.value_counts()
    agency_rows = agency_rows[agency_rows.agency.isin(agency_counts[agency_counts >= 12].index)]
    if agency_rows.agency.nunique() < 2:
        return {}

    agency_model = Pipeline([
        ("preprocessor", _feature_pipeline()),
        ("classifier", LogisticRegression(max_iter=500, C=1.0, class_weight="balanced",
                                           random_state=42)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agency_model.fit(build_features(agency_rows), agency_rows.agency.astype(str))

    resolved = frame.dropna(subset=["resolution_hours"]).copy()
    resolved = resolved[(resolved.resolution_hours > 0) &
                        (resolved.resolution_hours <= 24 * 365 * 2)]
    if resolved.empty:
        return {}
    upper = resolved.resolution_hours.quantile(.99)
    resolved = resolved[resolved.resolution_hours <= upper]
    resolution_model = Pipeline([
        ("preprocessor", _feature_pipeline()),
        ("regressor", Ridge(alpha=1.0)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resolution_model.fit(build_features(resolved), resolved.resolution_hours.astype(float))

    return {"agency": agency_model, "resolution": resolution_model}
