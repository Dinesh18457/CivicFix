"""Reproducible CivicFix model training, evaluation and unsupervised analysis.

Run from the repository root with: python train_models.py
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from civicfix_app.ml.features import build_features, feature_manifest

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None
try:
    from umap import UMAP
except ImportError:
    UMAP = None
try:
    from skopt import BayesSearchCV
except ImportError:
    BayesSearchCV = None

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "civicfix_app", "data", "nyc_311_data.csv")
MODEL_DIR = os.path.join(ROOT, "models")
SEED = 42
TRAIN_SAMPLE_LIMIT = int(os.environ.get("CIVICFIX_TRAIN_SAMPLE_LIMIT", "30000"))


def load_data():
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"NYC 311 dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    required = {"created_date", "closed_date", "agency", "borough", "complaint_type", "descriptor"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    for col in ("created_date", "closed_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["borough"] = df.borough.fillna("Unknown").astype(str).str.strip()
    df["descriptor"] = df.descriptor.fillna("").astype(str).str.strip()
    df["complaint_type"] = df.complaint_type.fillna("Unknown").astype(str).str.strip()
    df["resolution_hours"] = (df.closed_date - df.created_date).dt.total_seconds() / 3600
    df.loc[(df.resolution_hours <= 0) | (df.resolution_hours > 24 * 365 * 2), "resolution_hours"] = np.nan
    return df


def feature_pipeline():
    return ColumnTransformer([
        ("text", TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=2, sublinear_tf=True), "descriptor"),
        ("categories", OneHotEncoder(handle_unknown="ignore"), ["borough", "complaint_type"]),
    ])


def model_metrics(y_true, pred, probabilities, labels):
    result = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
    }
    try:
        result["roc_auc_ovr"] = float(roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average="weighted"))
        result["pr_auc_weighted"] = float(np.mean([
            average_precision_score((np.asarray(y_true) == label).astype(int), probabilities[:, i])
            for i, label in enumerate(labels) if np.sum(np.asarray(y_true) == label) > 0
        ]))
    except (ValueError, IndexError):
        result["roc_auc_ovr"] = None
        result["pr_auc_weighted"] = None
    confidence = np.max(probabilities, axis=1)
    correct = (np.asarray(y_true) == np.asarray(pred)).astype(float)
    gaps = []
    for edge in np.linspace(0, 1, 11)[:-1]:
        upper = edge + .1
        mask = (confidence >= edge) & ((confidence < upper) if upper < 1 else (confidence <= upper))
        if mask.any():
            gaps.append(float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())))
    result["expected_calibration_error"] = float(sum(gaps))
    return result


def run_unsupervised(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    sample = df.sample(min(4000, len(df)), random_state=SEED)
    text = TfidfVectorizer(max_features=120, min_df=2, stop_words="english").fit_transform(sample.descriptor)
    borough = OneHotEncoder(handle_unknown="ignore").fit_transform(sample[["borough"]])
    matrix = hstack([text, borough]).tocsr()
    embedding = TruncatedSVD(n_components=min(20, matrix.shape[1] - 1), random_state=SEED).fit_transform(matrix)
    pca = PCA(n_components=2, random_state=SEED).fit_transform(embedding)
    clusters = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(embedding)
    dbscan = DBSCAN(eps=0.8, min_samples=8).fit_predict(embedding[:2000])
    hierarchy = AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(embedding[:2000])
    iso = IsolationForest(contamination="auto", random_state=SEED, n_estimators=100).fit_predict(embedding)
    result = {
        "sample_size": int(len(sample)),
        "kmeans": {"algorithm": "KMeans", "clusters": int(len(set(clusters))),
                   "silhouette": float(silhouette_score(embedding, clusters, sample_size=min(1500, len(clusters)), random_state=SEED))},
        "dbscan": {"algorithm": "DBSCAN", "sample_size": int(len(dbscan)),
                   "clusters_excluding_noise": int(len(set(dbscan)) - (1 if -1 in dbscan else 0)),
                   "noise_points": int(np.sum(dbscan == -1))},
        "hierarchical": {"algorithm": "Hierarchical clustering (Ward)", "sample_size": int(len(hierarchy)),
                         "clusters": int(len(set(hierarchy)))},
        "dimensionality_reduction": {"algorithm": "TruncatedSVD + PCA", "components": 2,
                                     "explained_variance_ratio": [float(x) for x in PCA(n_components=2, random_state=SEED).fit(embedding).explained_variance_ratio_]},
        "anomaly_detection": {"algorithm": "Isolation Forest", "anomalies": int(np.sum(iso == -1)),
                              "sample_size": int(len(iso))},
    }
    if len(sample) >= 100:
        tsne = TSNE(n_components=2, perplexity=min(30, (len(sample) - 1) // 3), init="pca", learning_rate="auto", random_state=SEED)
        result["dimensionality_reduction"]["tsne_sample_size"] = 1000
        tsne.fit_transform(embedding[:min(1000, len(embedding))])
    if UMAP is not None:
        UMAP(n_components=2, n_neighbors=20, min_dist=.1, random_state=SEED).fit_transform(embedding)
        result["dimensionality_reduction"]["umap_sample_size"] = int(len(embedding))
    return result


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()
    # Classification uses all labeled records; a deterministic cap keeps local training practical.
    labeled_all = df[df.agency.notna() & df.agency.astype(str).str.len().gt(0)].copy()
    agency_counts = labeled_all.agency.value_counts()
    excluded_rare_agencies = agency_counts[agency_counts < 12].index.astype(str).tolist()
    labeled = labeled_all[labeled_all.agency.isin(agency_counts[agency_counts >= 12].index)].copy()
    if len(labeled) > TRAIN_SAMPLE_LIMIT:
        pieces = []
        for _, group in labeled.groupby("agency"):
            size = min(len(group), max(12, round(TRAIN_SAMPLE_LIMIT * len(group) / len(labeled))))
            pieces.append(group.sample(size, random_state=SEED))
        labeled = pd.concat(pieces, ignore_index=True)
    X = build_features(labeled)
    y = labeled.agency.astype(str)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=.125,
                                                      random_state=SEED, stratify=y_trainval)
    prep = feature_pipeline()
    xtr = prep.fit_transform(X_train)
    xte = prep.transform(X_test)
    variants = {
        "logistic_regression": LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=SEED),
        "decision_tree": DecisionTreeClassifier(min_samples_leaf=5, class_weight="balanced", random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=80, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "extra_trees": ExtraTreesClassifier(n_estimators=80, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=SEED),
    }
    # A bounded dense projection gives gradient boosting a memory-safe numeric representation.
    svd_components = min(80, max(2, xtr.shape[1] - 1))
    svd = TruncatedSVD(n_components=svd_components, random_state=SEED)
    xtr_dense = svd.fit_transform(xtr)
    xte_dense = svd.transform(xte)
    boosted = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=20, l2_regularization=1.0, random_state=SEED)
    scores = {}
    for name, estimator in variants.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(xtr, y_train)
        labels = estimator.classes_
        pred = estimator.predict(xte)
        probs = estimator.predict_proba(xte)
        scores[name] = model_metrics(y_test, pred, probs, labels)
    # SVD + boosting is evaluated as an additional ensemble family.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boosted.fit(xtr_dense, y_train)
    scores["gradient_boosting"] = model_metrics(y_test, boosted.predict(xte_dense), boosted.predict_proba(xte_dense), boosted.classes_)
    label_encoder = LabelEncoder().fit(y_train)
    encoded_y = label_encoder.transform(y_train)
    optional_classifiers = {}
    for name, constructor in (("xgboost", XGBClassifier), ("lightgbm", LGBMClassifier)):
        if constructor is None:
            continue
        options = {"n_estimators": 80, "max_depth": 5, "learning_rate": .08,
                   "subsample": .85, "random_state": SEED, "n_jobs": -1}
        if name == "xgboost":
            options.update({"objective": "multi:softprob", "num_class": len(label_encoder.classes_),
                            "eval_metric": "mlogloss", "verbosity": 0})
        else:
            options.update({"verbosity": -1, "class_weight": "balanced"})
        estimator = constructor(**options)
        estimator.fit(xtr_dense, encoded_y)
        decoded = label_encoder.inverse_transform(estimator.predict(xte_dense).astype(int))
        scores[name] = model_metrics(y_test, decoded, estimator.predict_proba(xte_dense), label_encoder.classes_)
        optional_classifiers[name] = estimator

    # Training-only cross-validation for the primary linear classifier.
    cv_n = min(5, int(y_train.value_counts().min()))
    cv = None
    search_result = None
    if cv_n >= 2:
        search = GridSearchCV(Pipeline([("preprocessor", feature_pipeline()),
                                        ("classifier", LogisticRegression(max_iter=350, class_weight="balanced", random_state=SEED))]),
                              {"classifier__C": [.3, 1.0, 3.0]}, cv=StratifiedKFold(n_splits=min(3, cv_n), shuffle=True, random_state=SEED),
                              scoring="f1_weighted", n_jobs=-1)
        search.fit(X_train, y_train)
        search_result = {"best_parameters": search.best_params_, "best_cv_f1_weighted": float(search.best_score_),
                         "candidates": int(len(search.cv_results_["params"]))}
        cv = np.asarray(search.cv_results_["mean_test_score"])
        search_result["cv_std"] = float(search.cv_results_["std_test_score"][search.best_index_])
        search_sample = min(12000, len(X_train))
        search_x = X_train.sample(search_sample, random_state=SEED)
        search_y = y_train.loc[search_x.index]
        randomized = RandomizedSearchCV(
            Pipeline([("preprocessor", feature_pipeline()),
                      ("classifier", LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED))]),
            {"classifier__C": [.1, .3, 1.0, 3.0, 10.0]}, n_iter=3,
            cv=StratifiedKFold(n_splits=min(3, cv_n), shuffle=True, random_state=SEED),
            scoring="f1_weighted", random_state=SEED, n_jobs=-1)
        randomized.fit(search_x, search_y)
        search_result["randomized_search"] = {"best_parameters": randomized.best_params_,
            "best_cv_f1_weighted": float(randomized.best_score_), "sample_size": int(search_sample)}
        if BayesSearchCV is not None:
            bayesian = BayesSearchCV(
                Pipeline([("preprocessor", feature_pipeline()),
                          ("classifier", LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED))]),
                {"classifier__C": (.1, 10.0, "log-uniform")}, n_iter=5,
                cv=StratifiedKFold(n_splits=min(3, cv_n), shuffle=True, random_state=SEED),
                scoring="f1_weighted", random_state=SEED, n_jobs=-1)
            bayesian.fit(search_x, search_y)
            search_result["bayesian_search"] = {"best_parameters": bayesian.best_params_,
                "best_cv_f1_weighted": float(bayesian.best_score_), "sample_size": int(search_sample)}

    # Fit deployable pipelines in-process; app serving builds these from Python source.
    best_c = search_result["best_parameters"]["classifier__C"] if search_result else 1.0
    # Use the dedicated validation split to compare the searched regularization strengths.
    val_choices = []
    if search_result:
        for candidate_c in [.3, 1.0, 3.0]:
            candidate = Pipeline([("preprocessor", feature_pipeline()),
                                  ("classifier", LogisticRegression(max_iter=500, C=candidate_c,
                                                                    class_weight="balanced", random_state=SEED))])
            candidate.fit(X_train, y_train)
            val_choices.append((float(f1_score(y_val, candidate.predict(X_val), average="weighted")), candidate_c))
        best_c = max(val_choices)[1]
        search_result["validation_best_c"] = best_c
        search_result["validation_best_f1_weighted"] = max(val_choices)[0]
    agency_pipeline = Pipeline([("preprocessor", feature_pipeline()),
                                ("classifier", LogisticRegression(max_iter=500, C=best_c, class_weight="balanced", random_state=SEED))])
    agency_pipeline.fit(X_trainval, y_trainval)
    serving_agency_metrics = model_metrics(y_test, agency_pipeline.predict(X_test),
                                          agency_pipeline.predict_proba(X_test),
                                          agency_pipeline.named_steps["classifier"].classes_)

    closed = df.dropna(subset=["resolution_hours", "created_date"]).copy()
    closed = closed[closed.resolution_hours <= closed.resolution_hours.quantile(.99)]
    if len(closed) > TRAIN_SAMPLE_LIMIT:
        closed = closed.sample(TRAIN_SAMPLE_LIMIT, random_state=SEED)
    rx = build_features(closed)
    ry = closed.resolution_hours.astype(float)
    rx_trainval, rx_test, ry_trainval, ry_test = train_test_split(rx, ry, test_size=.2, random_state=SEED)
    rx_train, rx_val, ry_train, ry_val = train_test_split(rx_trainval, ry_trainval, test_size=.125, random_state=SEED)
    regression_models = {
        "linear_regression": LinearRegression(), "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=.001, max_iter=2000), "elastic_net": ElasticNet(alpha=.001, l1_ratio=.5, max_iter=2000),
    }
    # Shared sparse features; regularized linear estimators scale well to high-dimensional TF-IDF.
    rprep = feature_pipeline()
    rxtr = rprep.fit_transform(rx_train)
    rxte = rprep.transform(rx_test)
    reg_scores = {}
    for name, estimator in regression_models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(rxtr, ry_train)
        pred = np.maximum(0, estimator.predict(rxte))
        reg_scores[name] = {"mae_hours": float(mean_absolute_error(ry_test, pred)),
                            "rmse_hours": float(np.sqrt(mean_squared_error(ry_test, pred))),
                            "r2": float(r2_score(ry_test, pred))}
    # Tree regressors on a compact SVD feature space.
    rsvd = TruncatedSVD(n_components=min(80, max(2, rxtr.shape[1] - 1)), random_state=SEED)
    rxtr_dense, rxte_dense = rsvd.fit_transform(rxtr), rsvd.transform(rxte)
    for name, estimator in (("decision_tree", DecisionTreeRegressor(min_samples_leaf=5, random_state=SEED)),
                            ("random_forest", RandomForestRegressor(n_estimators=80, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
                            ("gradient_boosting", GradientBoostingRegressor(n_estimators=80, max_depth=2, random_state=SEED))):
        estimator.fit(rxtr_dense, ry_train)
        pred = np.maximum(0, estimator.predict(rxte_dense))
        reg_scores[name] = {"mae_hours": float(mean_absolute_error(ry_test, pred)),
                            "rmse_hours": float(np.sqrt(mean_squared_error(ry_test, pred))),
                            "r2": float(r2_score(ry_test, pred))}
    for name, constructor in (("xgboost", XGBRegressor), ("lightgbm", LGBMRegressor)):
        if constructor is None:
            continue
        options = {"n_estimators": 80, "max_depth": 5, "learning_rate": .07,
                   "subsample": .85, "random_state": SEED, "n_jobs": -1}
        if name == "lightgbm":
            options["verbosity"] = -1
        estimator = constructor(**options)
        estimator.fit(rxtr_dense, ry_train)
        pred = np.maximum(0, estimator.predict(rxte_dense))
        reg_scores[name] = {"mae_hours": float(mean_absolute_error(ry_test, pred)),
                            "rmse_hours": float(np.sqrt(mean_squared_error(ry_test, pred))),
                            "r2": float(r2_score(ry_test, pred))}
    resolution_pipeline = Pipeline([("preprocessor", feature_pipeline()),
                                    ("regressor", Ridge(alpha=1.0))])
    resolution_pipeline.fit(rx_trainval, ry_trainval)
    rlinear_pred = np.maximum(0, resolution_pipeline.predict(rx_test))
    resolution_metrics = {"mae_hours": float(mean_absolute_error(ry_test, rlinear_pred)),
                          "rmse_hours": float(np.sqrt(mean_squared_error(ry_test, rlinear_pred))),
                          "r2": float(r2_score(ry_test, rlinear_pred))}
    # Keep the full comparison visible while marking the model used for serving.
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "random_seed": SEED,
        "dataset": {"records": int(len(df)), "classification_train": int(len(X_train)), "classification_validation": int(len(X_val)),
                    "classification_test": int(len(X_test)), "resolution_train": int(len(rx_train)),
                    "resolution_validation": int(len(rx_val)), "resolution_test": int(len(rx_test)),
                    "resolved_records_used": int(len(closed)),
                    "excluded_rare_agencies": excluded_rare_agencies},
        "agency_classifier": serving_agency_metrics, "classification_comparison": scores,
        "classification_cross_validation_f1_weighted": {"mean": float(search_result["best_cv_f1_weighted"]),
            "std": float(search_result["cv_std"]), "folds": int(min(3, cv_n))} if cv is not None else None,
        "hyperparameter_search": search_result,
        "resolution_regressor": resolution_metrics, "regression_comparison": reg_scores,
        "serving_models": {"agency": "Logistic regression", "resolution": "Ridge regression"},
        "optional_libraries": {"xgboost": XGBClassifier is not None, "lightgbm": LGBMClassifier is not None,
                               "umap": UMAP is not None, "bayesian_search": BayesSearchCV is not None},
        "evaluation_protocol": "Reproducible holdout split; stratified train-only CV; resolution target excludes open and invalid cases; 99th percentile cap.",
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    analysis = run_unsupervised(labeled)
    with open(os.path.join(MODEL_DIR, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2)
    with open(os.path.join(MODEL_DIR, "feature_schema.json"), "w", encoding="utf-8") as fh:
        json.dump(feature_manifest(labeled), fh, indent=2)
    print(f"Training and evaluation complete. Holdout metrics: {os.path.join(MODEL_DIR, 'metrics.json')}")
    print(f"Unsupervised analysis: {os.path.join(MODEL_DIR, 'analysis.json')}")


if __name__ == "__main__":
    train()
