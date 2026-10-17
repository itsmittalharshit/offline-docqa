import json
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from generate import LLM_MODEL
from tracking import TRACKING_URI, eval_hash

MODEL_NAME = "offline-docqa-pipeline"


class PipelineConfig(mlflow.pyfunc.PythonModel):
    """Registry entry for a RAG pipeline configuration; predict() returns the config."""

    def load_context(self, context):
        with open(context.artifacts["config"], encoding="utf-8") as f:
            self.config = json.load(f)

    def predict(self, context, model_input, params=None):
        return [self.config]


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    current = eval_hash()
    runs = mlflow.search_runs(experiment_names=["retrieval"],
                              filter_string=f"tags.eval_hash = '{current}'",
                              order_by=["metrics.mrr DESC"], max_results=1)
    if runs.empty:
        raise SystemExit(f"No retrieval runs on the current eval set ({current}). "
                         f"Run evaluate.py with --mlflow first.")

    best = runs.iloc[0]
    config = {key.removeprefix("params."): value
              for key, value in best.items() if key.startswith("params.")}
    config.update({"llm": LLM_MODEL, "prompt": "v3",
                   "source_run_id": best["run_id"], "eval_hash": current,
                   "mrr": float(best["metrics.mrr"])})
    print("Champion candidate:", json.dumps(config, indent=2))

    mlflow.set_experiment("registry")
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name="register-champion"):
        path = Path(tmp) / "pipeline_config.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        info = mlflow.pyfunc.log_model(name="pipeline", python_model=PipelineConfig(),
                                       artifacts={"config": str(path)},
                                       registered_model_name=MODEL_NAME)

    version = info.registered_model_version
    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)
    print(f"Registered {MODEL_NAME} version {version} with alias 'champion'")


if __name__ == "__main__":
    main()