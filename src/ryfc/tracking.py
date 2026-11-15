"""Thin MLflow wrapper. Falls back to printing when MLflow is not installed (serving image)."""

from contextlib import contextmanager

from ryfc.config import WORK


@contextmanager
def run(experiment: str, params: dict):
    try:
        import mlflow
    except ImportError:
        yield lambda metrics, step=None: None
        return
    WORK.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (WORK / "mlflow.db").resolve().as_posix())
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(WORK / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        mlflow.log_params({k: str(v) for k, v in params.items()})
        yield lambda metrics, step=None: mlflow.log_metrics(metrics, step=step)


def log_artifact(path):
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_artifact(str(path))
    except ImportError:
        pass
