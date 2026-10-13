# CivicFix

CivicFix is a civic service analytics and machine-learning project built around the NYC 311 service-request dataset included in `civicfix_app/data`.

## Run the application

1. Install the dependencies in `requirements.txt`.
2. From this folder, run `python main.py`.
3. Open `http://127.0.0.1:5000`.

The app reads the dataset directly and builds its serving models from the Python source in `civicfix_app/ml/serving_models.py`. Fitted models are kept in memory for the current app process; no serialized model files are required. The first prediction may take a moment while both pipelines are trained. The first dashboard visit may take a moment while the 15 MB dataset is loaded.

## Train and evaluate models

Run `python train_models.py` from this folder. The workflow cleans the NYC fields, excludes invalid or open cases from resolution-time training, and creates reproducible training, validation, and test partitions. It compares regularized linear models and tree ensembles, reports weighted classification metrics and regression errors, runs training-only cross-validation and a small logistic-regression hyperparameter search, and records confidence calibration error.

Evaluation reports are written to `models/`:

- `metrics.json` contains holdout metrics and comparison results.
- `analysis.json` contains clustering, dimensionality-reduction, and anomaly-detection summaries.
- `feature_schema.json` records the canonical feature contract shared by training and serving.

Model implementations and serving configuration stay in readable Python files. Training does not export opaque model binaries.

Unsupervised analysis uses a deterministic sample so density clustering and t-SNE remain feasible on a local machine. The optional chart snapshots can be regenerated with `python -m civicfix_app.ml.eda`.
Optional packages `xgboost`, `lightgbm`, `umap-learn`, and `scikit-optimize` enable the additional CO3 model families, UMAP projection, and Bayesian search; the core workflow remains available without them.

## CivicFix screens

- **Overview:** live dataset totals, borough distribution, common concerns, and recent requests.
- **Service data:** searchable records with borough filtering.
- **Explore patterns:** borough, agency, and complaint distributions.
- **Model lab:** classification/regression metrics, model comparisons, validation, and unsupervised results.
- **Route a request:** agency recommendation, resolution estimate, model confidence, and inference latency.
- **Lifecycle trace:** data-to-deployment walkthrough aligned with the course outcomes.
- **Monitoring:** persisted local request traces and feature-drift review signals.

## Local serving endpoints

- `GET /health` — dataset and loaded-model readiness.
- `GET /api/summary` — live dataset summary as JSON.
- `POST /api/predict` — JSON inference. Send `{"borough":"BROOKLYN","complaint_type":"Street Condition","descriptor":"Deep pothole near the intersection"}`.
- `POST /api/feedback` — connect delayed real outcomes to a trace: `{"trace_id":"…","actual_agency":"DOT","actual_hours":36}`. The monitor reports observed routing accuracy and resolution MAE.

Prediction traces and delayed outcomes are stored locally in `logs/prediction_log.jsonl` and `logs/prediction_feedback.jsonl`. Set `CIVICFIX_SECRET_KEY` before exposing the app beyond a local development environment.
