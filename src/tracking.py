import hashlib
import subprocess
from pathlib import Path

import mlflow

TRACKING_URI = "sqlite:///mlflow.db"
EVAL_PATH = Path("eval/questions.jsonl")


def eval_hash():
    """Fingerprint of the eval set: runs are only comparable if this matches."""
    return hashlib.sha256(EVAL_PATH.read_bytes()).hexdigest()[:8]


def git_commit():
    """Short commit hash, with '-dirty' if code in src/ has uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"],
                               capture_output=True, text=True).stdout.strip()
        return (commit or "unknown") + ("-dirty" if dirty else "")
    except FileNotFoundError:
        return "unknown"


def log_run(experiment, params, metrics, details):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"eval_hash": eval_hash(), "git_commit": git_commit()})
        mlflow.log_dict(details, "per_question.json")