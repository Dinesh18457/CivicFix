"""Canonical request feature contract shared by training and online serving."""
import pandas as pd

FEATURE_COLUMNS = ("descriptor", "borough", "complaint_type")


def build_features(records):
    """Normalize raw 311 rows or API dictionaries to the shared model schema."""
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    for column in FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = "Unknown" if column != "descriptor" else ""
        frame[column] = frame[column].fillna("" if column == "descriptor" else "Unknown").astype(str).str.strip()
    frame["borough"] = frame["borough"].str.upper()
    frame["complaint_type"] = frame["complaint_type"].replace("", "Unknown")
    return frame.loc[:, list(FEATURE_COLUMNS)]


def feature_manifest(frame):
    features = build_features(frame)
    return {
        "schema_version": 1,
        "columns": list(FEATURE_COLUMNS),
        "normalization": {"descriptor": "strip whitespace; empty string for missing", "borough": "strip and uppercase", "complaint_type": "strip; Unknown for missing"},
        "training_rows": int(len(features)),
        "borough_values": sorted(features.borough.unique().tolist()),
        "complaint_type_values": sorted(features.complaint_type.unique().tolist()),
        "text_feature": "descriptor",
        "categorical_features": ["borough", "complaint_type"],
    }
