"""CivicFix: an explainable NYC 311 analytics and ML operations workspace."""
from __future__ import annotations

import json
import base64
import os
import time
import uuid
from io import BytesIO
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from civicfix_app.ml.features import build_features
from civicfix_app.ml.serving_models import build_serving_models

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "civicfix_app", "data", "nyc_311_data.csv")
MODEL_DIR = os.path.join(ROOT, "models")
LOG_PATH = os.path.join(ROOT, "logs", "prediction_log.jsonl")
FEEDBACK_PATH = os.path.join(ROOT, "logs", "prediction_feedback.jsonl")
app = Flask(__name__, template_folder="civicfix_app/templates", static_folder="civicfix_app/static")
app.secret_key = os.environ.get("CIVICFIX_SECRET_KEY", "civicfix-local-session-key")


@lru_cache(maxsize=1)
def dataset():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    frame = pd.read_csv(DATA_PATH, low_memory=False)
    for col in ("borough", "agency", "complaint_type", "descriptor", "status"):
        if col in frame:
            frame[col] = frame[col].fillna("Unknown").astype(str).str.strip()
    for col in ("created_date", "closed_date"):
        if col in frame:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    if {"created_date", "closed_date"}.issubset(frame.columns):
        frame["resolution_hours"] = (frame.closed_date - frame.created_date).dt.total_seconds() / 3600
        frame.loc[frame.resolution_hours <= 0, "resolution_hours"] = None
    return frame


@lru_cache(maxsize=1)
def models():
    """Fit serving pipelines from the included dataset and cache them in memory."""
    return build_serving_models(dataset())


def metrics():
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    return {}


@lru_cache(maxsize=1)
def unsupervised_metrics():
    path = os.path.join(MODEL_DIR, "analysis.json")
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    return {}


def summary():
    df = dataset()
    if df.empty:
        return {"records": 0, "boroughs": 0, "agencies": 0, "open": 0, "closed": 0,
                "avg_hours": None, "borough_counts": [], "complaint_counts": [], "agency_counts": [],
                "top_complaints": [], "recent": [], "data_available": False}
    borough = df.borough.value_counts().head(6)
    complaints = df.complaint_type.value_counts().head(8)
    agencies = df.agency.value_counts().head(8)
    status = df.status.str.lower()
    open_count = int(status.isin(["open", "in progress", "started"]).sum())
    closed_count = int(df.closed_date.notna().sum())
    recent_cols = [c for c in ("unique_key", "created_date", "complaint_type", "borough", "agency", "status") if c in df]
    recent = df.sort_values("created_date", ascending=False)[recent_cols].head(7).fillna("").to_dict("records")
    for row in recent:
        if isinstance(row.get("created_date"), (datetime, pd.Timestamp)):
            row["created_date"] = row["created_date"].strftime("%b %d, %Y")
    return {"records": len(df), "boroughs": int(df.borough.nunique()), "agencies": int(df.agency.nunique()),
            "open": open_count, "closed": closed_count,
            "avg_hours": round(float(df.resolution_hours.dropna().median()), 1) if "resolution_hours" in df and df.resolution_hours.notna().any() else None,
            "borough_counts": [{"label": k.title(), "value": int(v)} for k, v in borough.items()],
            "complaint_counts": [{"label": k, "value": int(v)} for k, v in complaints.items()],
            "agency_counts": [{"label": k, "value": int(v)} for k, v in agencies.items()],
            "top_complaints": [{"label": k, "value": int(v)} for k, v in complaints.head(5).items()],
            "recent": recent, "data_available": True}


def render(page, **kwargs):
    active_nav = kwargs.pop("active_nav", page)
    return render_template("index.html", active_page=page, stats=summary(), eval_metrics=metrics(),
                           active_nav=active_nav, history=session.get("history", []), **kwargs)


def figure_data_url(fig, pyplot, dpi=145):
    """Encode a Matplotlib figure for the page without creating a binary asset file."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    pyplot.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@lru_cache(maxsize=1)
def build_regression_charts():
    """Render held-out Ridge diagnostics and model score comparisons for the UI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.model_selection import train_test_split

    report = metrics()
    comparison = report.get("regression_comparison", {})
    required = {"resolution_regressor", "regression_comparison"}
    if not required.issubset(report) or not comparison or "resolution" not in models():
        return []

    frame = dataset().copy()
    frame = frame.dropna(subset=["created_date", "closed_date", "resolution_hours"])
    frame = frame[(frame.resolution_hours > 0) & (frame.resolution_hours <= 24 * 365 * 2)]
    if frame.empty:
        return []
    frame = frame[frame.resolution_hours <= frame.resolution_hours.quantile(.99)]
    seed = int(report.get("random_seed", 42))
    if len(frame) > 30000:
        frame = frame.sample(30000, random_state=seed)
    y = frame.resolution_hours.astype(float)
    x = build_features(frame)
    _, x_test, _, y_test = train_test_split(x, y, test_size=.2, random_state=seed)
    actual = y_test.to_numpy()
    predicted = np.maximum(0, np.asarray(models()["resolution"].predict(x_test), dtype=float))
    residual = actual - predicted
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(len(actual), size=min(1800, len(actual)), replace=False))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#dce5dd", "axes.labelcolor": "#53645a",
                         "xtick.color": "#7b887f", "ytick.color": "#7b887f",
                         "text.color": "#33443a", "figure.facecolor": "white",
                         "axes.facecolor": "white"})

    chart_sources = []

    def save_plot(title, note, alt):
        plt.tight_layout(pad=1.5)
        chart_sources.append({"title": title, "note": note, "alt": alt,
                              "src": figure_data_url(plt.gcf(), plt)})

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(actual[sample], predicted[sample], s=13, alpha=.38, color="#397a56", edgecolors="none")
    upper = float(np.quantile(np.concatenate([actual, predicted]), .98))
    ax.plot([0, upper], [0, upper], linestyle="--", color="#d28a4d", linewidth=1.5, label="Ideal estimate")
    ax.set(xlim=(0, upper), ylim=(0, upper), xlabel="Actual resolution time (hours)",
           ylabel="Predicted resolution time (hours)", title="Held-out cases · Ridge serving model")
    ax.legend(frameon=False, loc="upper left")
    save_plot("Actual vs predicted", "Ridge · held-out cases",
              "Scatter plot comparing actual and predicted resolution time with an ideal prediction line")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(predicted[sample], residual[sample], s=13, alpha=.35, color="#397a56", edgecolors="none")
    ax.axhline(0, color="#d28a4d", linewidth=1.5, linestyle="--")
    low, high = np.quantile(predicted, [.01, .99])
    span = max(high - low, 1)
    ax.set(xlim=(max(0, low - .05 * span), high + .05 * span),
           xlabel="Predicted resolution time (hours)", ylabel="Residual (actual − predicted)",
           title="Residuals vs predicted · held-out sample")
    save_plot("Residuals vs predicted", "Look for uneven error patterns",
              "Residual scatter plot for held-out resolution-time predictions")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(residual, bins=34, color="#4d8c65", edgecolor="white", linewidth=.6)
    ax.axvline(0, color="#d28a4d", linewidth=1.5, linestyle="--")
    ax.set(xlabel="Residual (hours)", ylabel="Held-out requests", title="Prediction error distribution")
    save_plot("Prediction error distribution", "Actual minus predicted · hours",
              "Histogram showing held-out prediction errors in hours")

    labels = [(name.replace("_", " ").title(), values) for name, values in comparison.items()]
    for key, title, color, note, alt in (
        ("mae_hours", "MAE by model", "#397a56", "Lower is better · hours", "Model comparison by mean absolute error"),
        ("rmse_hours", "RMSE by model", "#668dc1", "Lower is better · hours", "Model comparison by root mean squared error"),
        ("r2", "R² by model", "#d28a4d", "Higher is better", "Model comparison by explained variance R squared"),
    ):
        values = [(name, metrics_for_model.get(key)) for name, metrics_for_model in labels
                  if metrics_for_model.get(key) is not None]
        if not values:
            continue
        values.sort(key=lambda item: item[1], reverse=(key == "r2"))
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        names = [item[0] for item in values]
        numbers = [item[1] for item in values]
        bars = ax.barh(names, numbers, color=color, height=.62)
        if key != "r2":
            ax.invert_yaxis()
            ax.set_xlim(0, max(numbers) * 1.2)
        for bar, number in zip(bars, numbers):
            label = f"{number:.2f} h" if key != "r2" else f"{number:.3f}"
            ax.text(bar.get_width() + (max(numbers) * .015), bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=8, color="#53645a")
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlabel("Hours · lower is better" if key != "r2" else "R² · higher is better")
        save_plot(title, note, alt)

    return chart_sources


@lru_cache(maxsize=1)
def build_eda_report():
    """Create an NYC 311 data profile and offline visualizations for the EDA page."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    df = dataset().copy()
    if df.empty:
        return {}
    valid_hours = df.resolution_hours.dropna()
    valid_hours = valid_hours[(valid_hours > 0) & (valid_hours <= 24 * 365 * 2)]
    target_p99 = float(valid_hours.quantile(.99)) if not valid_hours.empty else None
    display_hours = valid_hours[valid_hours <= target_p99] if target_p99 is not None else valid_hours

    numeric_fields = [
        ("resolution_hours", "Resolution time (hours)", valid_hours),
        ("incident_zip", "Incident ZIP", pd.to_numeric(df.get("incident_zip"), errors="coerce").dropna()),
        ("council_district", "Council district", pd.to_numeric(df.get("council_district"), errors="coerce").dropna()),
        ("police_precinct", "Police precinct", pd.to_numeric(df.get("police_precinct"), errors="coerce").dropna()),
        ("latitude", "Latitude", pd.to_numeric(df.get("latitude"), errors="coerce").dropna()),
        ("longitude", "Longitude", pd.to_numeric(df.get("longitude"), errors="coerce").dropna()),
    ]
    descriptive = []
    outliers = []
    outlier_fields = {"resolution_hours", "latitude", "longitude"}
    for key, label, values in numeric_fields:
        if values.empty:
            continue
        descriptive.append({"name": label, "count": f"{len(values):,}",
                            "mean": f"{values.mean():,.2f}", "median": f"{values.median():,.2f}",
                            "std": f"{values.std():,.2f}", "min": f"{values.min():,.2f}",
                            "max": f"{values.max():,.2f}",
                            "missing": f"{len(df) - len(values):,}"})
        if key in outlier_fields:
            q1, q3 = values.quantile([.25, .75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            count = int(((values < lower) | (values > upper)).sum())
            outliers.append({"name": label, "q1": f"{q1:,.2f}", "q3": f"{q3:,.2f}",
                             "lower": f"{lower:,.2f}", "upper": f"{upper:,.2f}",
                             "count": f"{count:,}", "share": f"{count / len(values) * 100:.1f}%"})

    category_specs = [("borough", "Borough"), ("complaint_type", "Complaint type"),
                      ("agency", "Responding agency"), ("status", "Request status"),
                      ("open_data_channel_type", "Intake channel")]
    categories = []
    for field, label in category_specs:
        if field not in df:
            continue
        counts = df[field].fillna("Unknown").astype(str).str.strip().replace("", "Unknown").value_counts()
        categories.append({"name": label, "total": f"{counts.sum():,}", "unique": f"{counts.size:,}",
                           "items": [{"label": str(name), "count": f"{int(count):,}",
                                      "share": f"{count / len(df) * 100:.1f}%"}
                                     for name, count in counts.head(6).items()]})

    status_counts = df.status.fillna("Unknown").astype(str).str.strip().value_counts()
    closed_status = int(status_counts[status_counts.index.str.casefold() == "closed"].sum())
    missing_locations = int(df[[col for col in ("latitude", "longitude") if col in df]].isna().any(axis=1).sum())
    complaint_counts = df.complaint_type.fillna("Unknown").value_counts()
    borough_counts = df.borough.fillna("Unknown").value_counts()
    agency_counts = df.agency.fillna("Unknown").value_counts()
    findings = []
    if not complaint_counts.empty:
        top = complaint_counts.index[0]
        findings.append(f"{top} is the most common complaint type ({int(complaint_counts.iloc[0]):,} requests, {complaint_counts.iloc[0] / len(df) * 100:.1f}% of this extract).")
    if not borough_counts.empty and not agency_counts.empty:
        findings.append(f"{borough_counts.index[0].title()} contributes the largest borough share; {agency_counts.index[0]} is the most frequent responding agency.")
    findings.append(f"{closed_status:,} records are marked Closed; {len(valid_hours):,} have a valid positive resolution duration for regression analysis.")
    findings.append(f"Coordinates are missing for {missing_locations:,} requests; geographic plots include only records with valid latitude and longitude.")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#dce5dd", "axes.labelcolor": "#53645a",
                         "xtick.color": "#7b887f", "ytick.color": "#7b887f",
                         "text.color": "#33443a", "figure.facecolor": "white",
                         "axes.facecolor": "white"})

    def save(fig):
        fig.tight_layout(pad=1.35)
        return figure_data_url(fig, plt)

    colors = ["#397a56", "#6d9b75", "#88a897", "#d28a4d", "#668dc1", "#766a9e", "#a8beaa", "#bd735e"]

    chart_sources = {}

    def bar_chart(series, title, xlabel, color="#397a56", limit=10):
        values = series.head(limit).sort_values()
        fig, ax = plt.subplots(figsize=(7.1, 4.0))
        ax.barh([str(i).title() for i in values.index], values.values, color=color, height=.62)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#edf0ec", linewidth=.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        chart_sources[title] = save(fig)

    bar_chart(df.borough.fillna("Unspecified").value_counts(), "Requests by borough", "Requests", limit=8)
    bar_chart(df.complaint_type.fillna("Unknown").value_counts(), "Most reported concerns", "Requests", color="#668e74", limit=10)
    bar_chart(df.agency.fillna("Unknown").value_counts(), "Agency workload", "Requests", color="#668dc1", limit=10)

    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    status_values = status_counts[status_counts / status_counts.sum() >= .02]
    other_statuses = int(status_counts[status_counts / status_counts.sum() < .02].sum())
    if other_statuses:
        status_values.loc["Other"] = other_statuses
    wedges, _, _ = ax.pie(status_values.values, labels=None, startangle=90, counterclock=False,
                          colors=colors[:len(status_values)], wedgeprops={"width": .34, "edgecolor": "white"},
                          autopct=lambda pct: f"{pct:.0f}%" if pct >= 5 else "",
                          pctdistance=.81, textprops={"fontsize": 9, "color": "white", "weight": "bold"},
                          radius=.72)
    legend_labels = [f"{label}  ·  {int(value):,}" for label, value in status_values.items()]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(.94, .5),
              frameon=False, fontsize=8, labelspacing=1.15, handlelength=1.1)
    ax.set_aspect("equal")
    chart_sources["Request status mix"] = save(fig)

    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    if not display_hours.empty:
        ax.hist(display_hours, bins=36, color="#6d9b75", edgecolor="white", linewidth=.6)
        ax.axvline(display_hours.median(), color="#d28a4d", linestyle="--", linewidth=1.5,
                   label=f"Median {display_hours.median():.1f} h")
        ax.legend(frameon=False)
    ax.set(xlabel="Hours to close · shown through 99th percentile", ylabel="Requests")
    ax.grid(axis="y", color="#edf0ec", linewidth=.8)
    ax.set_axisbelow(True)
    chart_sources["Resolution-time distribution"] = save(fig)

    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    box_data, box_labels = [], []
    for borough in ("BROOKLYN", "QUEENS", "BRONX", "MANHATTAN", "STATEN ISLAND"):
        group = df.loc[df.borough.str.upper() == borough, "resolution_hours"].dropna()
        group = group[(group > 0) & (group <= target_p99)] if target_p99 is not None else group.iloc[0:0]
        if not group.empty:
            box_data.append(group.to_numpy())
            box_labels.append(borough.title())
    if box_data:
        boxes = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, showfliers=False,
                           medianprops={"color": "#d28a4d", "linewidth": 1.4})
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(.7)
    ax.set(ylabel="Hours · capped for readability")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", color="#edf0ec", linewidth=.8)
    ax.set_axisbelow(True)
    chart_sources["Resolution time by borough"] = save(fig)

    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    locations = df.dropna(subset=["latitude", "longitude"])
    if len(locations) > 3500:
        locations = locations.sample(3500, random_state=42)
    for idx, (borough, group) in enumerate(locations.groupby("borough", dropna=False)):
        ax.scatter(group.longitude, group.latitude, s=8, alpha=.38, color=colors[idx % len(colors)],
                   label=str(borough).title(), edgecolors="none")
    ax.set(xlabel="Longitude", ylabel="Latitude")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    ax.grid(color="#edf0ec", linewidth=.7)
    chart_sources["Request locations"] = save(fig)

    chart_cards = [
        {"title": "Requests by borough", "note": "Geographic distribution", "src": chart_sources["Requests by borough"]},
        {"title": "Most reported concerns", "note": "Top complaint types", "src": chart_sources["Most reported concerns"]},
        {"title": "Agency workload", "note": "Requests by responding team", "src": chart_sources["Agency workload"]},
        {"title": "Request status mix", "note": "Open and closed records", "src": chart_sources["Request status mix"]},
        {"title": "Resolution-time distribution", "note": "Valid closed-request durations", "src": chart_sources["Resolution-time distribution"]},
        {"title": "Resolution time by borough", "note": "Median and spread at a glance", "src": chart_sources["Resolution time by borough"]},
        {"title": "Request locations", "note": "Valid coordinates, sampled for clarity", "src": chart_sources["Request locations"]},
    ]
    return {"records": f"{len(df):,}", "fields": f"{df.shape[1]:,}",
            "agencies": f"{df.agency.nunique():,}", "closed": f"{closed_status:,}",
            "valid_targets": f"{len(valid_hours):,}",
            "target_median": f"{valid_hours.median():.1f}" if not valid_hours.empty else "—",
            "target_p95": f"{valid_hours.quantile(.95):.1f}" if not valid_hours.empty else "—",
            "descriptive": descriptive, "outliers": outliers, "categories": categories,
            "findings": findings, "chart_cards": chart_cards,
            "target_note": f"Training and charts exclude non-positive durations and cap displayed durations near the 99th percentile ({target_p99:.1f} hours) for readability." if target_p99 else "No valid resolution targets were found."}


def score_request(borough, complaint, descriptor):
    if not models().get("agency") or not models().get("resolution"):
        raise RuntimeError("Prediction models could not be built. Check the included dataset and its agency and resolution fields.")
    started = time.perf_counter()
    full_payload = build_features([{"borough": borough, "descriptor": descriptor,
                                    "complaint_type": complaint}])
    agency_model, resolution_model = models()["agency"], models()["resolution"]
    agency_features = list(getattr(agency_model, "feature_names_in_", full_payload.columns))
    resolution_features = list(getattr(resolution_model, "feature_names_in_", full_payload.columns))
    agency_payload = full_payload.reindex(columns=agency_features)
    resolution_payload = full_payload.reindex(columns=resolution_features)
    # Older checked-in artifacts only use borough + descriptor. Include the selected
    # category in the text input until the richer retrained artifact is available.
    if "complaint_type" not in agency_features:
        agency_payload["descriptor"] = f"{complaint}. {descriptor}"
    if "complaint_type" not in resolution_features:
        resolution_payload["descriptor"] = f"{complaint}. {descriptor}"
    agency = str(agency_model.predict(agency_payload)[0])
    hours = max(0, float(resolution_model.predict(resolution_payload)[0]))
    confidence = None
    if hasattr(agency_model, "predict_proba"):
        confidence = round(float(max(agency_model.predict_proba(agency_payload)[0])) * 100, 1)
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    source_file = os.path.join(ROOT, "civicfix_app", "ml", "serving_models.py")
    model_version = int(os.path.getmtime(source_file)) if os.path.exists(source_file) else 0
    result = {"agency": agency, "hours": round(hours, 1), "days": round(hours / 24, 1),
              "latency": elapsed, "confidence": confidence, "borough": borough,
              "complaint": complaint, "descriptor": descriptor,
              "model_version": model_version, "feature_schema": 1,
              "dataset_records": int(len(dataset())),
              "evaluation_run": metrics().get("generated_at"),
              "id": uuid.uuid4().hex[:12]}
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({**result, "timestamp": datetime.now(timezone.utc).isoformat()}) + "\n")
    return result


@app.route("/")
def dashboard():
    return render("dashboard")


@app.route("/data")
def data_page():
    df = dataset()
    query = request.args.get("q", "").strip().lower()
    borough = request.args.get("borough", "")
    if not df.empty:
        view = df
        if borough:
            view = view[view.borough == borough]
        if query:
            mask = view[["complaint_type", "descriptor", "agency", "borough"]].astype(str).apply(
                lambda col: col.str.lower().str.contains(query, regex=False)).any(axis=1)
            view = view[mask]
        columns = [c for c in ("unique_key", "created_date", "complaint_type", "descriptor", "borough", "agency", "status") if c in view]
        rows = view.sort_values("created_date", ascending=False)[columns].head(30).fillna("").to_dict("records")
        for row in rows:
            if isinstance(row.get("created_date"), (datetime, pd.Timestamp)):
                row["created_date"] = row["created_date"].strftime("%b %d, %Y")
    else:
        rows = []
    return render("data", rows=rows, query=query, selected_borough=borough,
                  boroughs=sorted(df.borough.unique()) if not df.empty else [])


@app.route("/explore")
def explore():
    return render("explore", eda=build_eda_report(), active_nav="outcomes")


@app.route("/models")
def model_lab():
    return render("models", analysis=unsupervised_metrics())


@app.route("/ensemble")
def ensemble_page():
    return render("models", analysis=unsupervised_metrics(), active_nav="outcomes")


@app.route("/unsupervised")
def unsupervised_page():
    return render("models", analysis=unsupervised_metrics(), active_nav="outcomes")


@app.route("/evaluation")
def evaluation_page():
    return render("models", analysis=unsupervised_metrics(), active_nav="outcomes")


@app.route("/regression")
def regression_page():
    return render("regression", chart_files=build_regression_charts(), active_nav="outcomes")


@app.route("/lifecycle")
def lifecycle():
    return render("lifecycle")


@app.route("/operations")
def operations():
    log = []
    all_traces = []
    if os.path.isfile(LOG_PATH):
        try:
            with open(LOG_PATH, encoding="utf-8") as fh:
                all_traces = [json.loads(line) for line in fh if line.strip()][-2000:]
                log = all_traces[-12:][::-1]
        except (OSError, ValueError):
            pass
    feedback = []
    if os.path.isfile(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, encoding="utf-8") as fh:
                feedback = [json.loads(line) for line in fh if line.strip()][-2000:]
        except (OSError, ValueError):
            pass
    df = dataset()
    drift = {"sample_size": len(all_traces), "borough_shift": None, "complaint_shift": None,
             "borough_label": "Insufficient traffic", "complaint_label": "Insufficient traffic"}
    if len(all_traces) >= 20 and not df.empty:
        observed = pd.DataFrame(all_traces)
        for field, label_key in (("borough", "borough_shift"), ("complaint_type", "complaint_shift")):
            if field in observed:
                baseline = df[field].fillna("Unknown").astype(str).value_counts(normalize=True)
                current = observed[field].fillna("Unknown").astype(str).value_counts(normalize=True)
                categories = set(baseline.index) | set(current.index)
                shift = max((abs(float(baseline.get(key, 0)) - float(current.get(key, 0))) for key in categories), default=0)
                drift[label_key] = round(shift * 100, 1)
                drift["borough_label" if field == "borough" else "complaint_label"] = "Review shift" if shift >= .15 else "Within baseline"
    accuracy = (sum(item["predicted_agency"] == item["actual_agency"] for item in feedback) / len(feedback) * 100) if feedback else None
    resolved = [item for item in feedback if item.get("actual_hours") is not None]
    mae = (sum(abs(item["predicted_hours"] - item["actual_hours"]) for item in resolved) / len(resolved)) if resolved else None
    return render("operations", prediction_log=log, loaded_models=sorted(models()), drift=drift,
                  feedback_count=len(feedback), feedback_accuracy=accuracy, feedback_mae=mae,
                  feedback_error=session.pop("feedback_error", None),
                  feedback_notice=session.pop("feedback_notice", None))


def record_feedback(payload):
    trace_id = str(payload.get("trace_id", "")).strip()
    actual_agency = str(payload.get("actual_agency", "")).strip()
    raw_hours = payload.get("actual_hours")
    try:
        actual_hours = float(raw_hours) if raw_hours not in (None, "") else None
    except (TypeError, ValueError):
        return {"error": "Resolution hours must be a number."}, 400
    if not trace_id or not actual_agency or len(actual_agency) > 80 or (actual_hours is not None and (actual_hours < 0 or actual_hours > 24 * 365 * 2)):
        return {"error": "Provide a trace ID, actual agency, and valid non-negative resolution hours."}, 400
    if not os.path.isfile(LOG_PATH):
        return {"error": "No prediction traces were found."}, 404
    try:
        with open(LOG_PATH, encoding="utf-8") as fh:
            matches = [json.loads(line) for line in fh if line.strip()]
    except (OSError, ValueError):
        return {"error": "Prediction trace log could not be read."}, 500
    trace = next((item for item in reversed(matches) if str(item.get("id")) == trace_id), None)
    if trace is None:
        return {"error": "That trace ID was not found."}, 404
    if os.path.isfile(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, encoding="utf-8") as fh:
                if any(str(json.loads(line).get("trace_id")) == trace_id for line in fh if line.strip()):
                    return {"error": "That trace already has a recorded outcome."}, 409
        except (OSError, ValueError):
            return {"error": "Feedback log could not be read."}, 500
    entry = {"trace_id": trace_id, "actual_agency": actual_agency,
             "predicted_agency": trace["agency"], "predicted_hours": float(trace["hours"]),
             "actual_hours": actual_hours, "timestamp": datetime.now(timezone.utc).isoformat()}
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry, 201


@app.post("/feedback")
def feedback_form():
    result, status = record_feedback(request.form)
    if status != 201:
        session["feedback_error"] = result["error"]
    else:
        session["feedback_notice"] = "Outcome saved. Monitoring metrics have been updated."
    return redirect(url_for("operations"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    error = None
    df = dataset()
    complaint_types = sorted(df.complaint_type.value_counts().head(100).index.tolist()) if not df.empty else []
    boroughs = sorted(df.borough.unique()) if not df.empty else ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]
    if request.method == "POST":
        borough = request.form.get("borough", "").strip().upper()
        complaint = request.form.get("complaint_type", "").strip()
        descriptor = request.form.get("descriptor", "").strip()[:500]
        if not complaint or not descriptor or borough not in boroughs:
            error = "Choose a borough and complaint type, and add a short description."
        else:
            try:
                result = score_request(borough, complaint, descriptor)
                history = session.get("history", [])
                history.insert(0, result)
                session["history"] = history[:12]
            except Exception as exc:
                app.logger.exception("Prediction failed")
                error = f"The request could not be scored: {type(exc).__name__}. Check that the saved model matches the input features."
    return render("predict", result=result, error=error, complaint_types=complaint_types, boroughs=boroughs)


@app.get("/api/summary")
def api_summary():
    return jsonify(summary())


@app.post("/api/predict")
def api_predict():
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    borough = str(body.get("borough", "")).strip().upper()
    complaint = str(body.get("complaint_type", "Other")).strip()[:120]
    descriptor = str(body.get("descriptor", "")).strip()[:500]
    valid_boroughs = set(dataset().borough.unique()) if not dataset().empty else {"BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"}
    if borough not in valid_boroughs or not descriptor:
        return jsonify({"error": "Provide a valid borough and a non-empty descriptor (max 500 characters)."}), 400
    try:
        return jsonify(score_request(borough, complaint, descriptor))
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503


@app.post("/api/feedback")
def api_feedback():
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    result, status = record_feedback(body)
    return jsonify(result), status


@app.get("/health")
def health():
    return jsonify({"status": "ok" if dataset().shape[0] and models() else "degraded",
                    "dataset_records": int(len(dataset())), "models_loaded": sorted(models()),
                    "timestamp": datetime.now(timezone.utc).isoformat()}), (200 if dataset().shape[0] and models() else 503)


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", 5000)))
