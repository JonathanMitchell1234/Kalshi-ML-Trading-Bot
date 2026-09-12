"""MLflow experiment tracking for BitBot training.

Shared store with the mlflow-mcp tools: sqlite:////Users/Jonathan/mlflow.db
(explicit URI so local runs are visible everywhere). All logging is
best-effort: training must NEVER fail because tracking is down.
Experiments: bitbot-crypto | bitbot-weather. Tags carry model version,
feature-set hash (for feat-change <-> perf correlation), and data window.
"""
import hashlib
import json
import os

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:////Users/Jonathan/mlflow.db")


def feat_hash(feats) -> str:
    return hashlib.sha256(",".join(sorted(feats)).encode()).hexdigest()[:12]


def log_run(experiment: str, run_name: str, params: dict | None = None,
            metrics: dict | None = None, tags: dict | None = None,
            artifact_texts: dict | None = None) -> str | None:
    """Log one run. Returns run_id or None (never raises)."""
    try:
        import mlflow
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(experiment)
        flat_params = {k: (json.dumps(v) if isinstance(v, (dict, list)) else str(v))[:6000]
                       for k, v in (params or {}).items()}
        flat_metrics = {k: float(v) for k, v in (metrics or {}).items()
                        if isinstance(v, (int, float)) and v == v}
        with mlflow.start_run(run_name=run_name) as run:
            if flat_params:
                mlflow.log_params(flat_params)
            if flat_metrics:
                mlflow.log_metrics(flat_metrics)
            if tags:
                mlflow.set_tags({k: str(v)[:500] for k, v in tags.items()})
            for name, text in (artifact_texts or {}).items():
                mlflow.log_text(str(text)[:50000], name)
            return run.info.run_id
    except Exception as e:
        print(f"mlflow log skipped ({e})")
        return None
