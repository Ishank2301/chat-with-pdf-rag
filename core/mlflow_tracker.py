"""MLflow tracking helpers."""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import mlflow

from .config import settings

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Small wrapper around MLflow run tracking."""

    def __init__(self):
        """Initialize MLflow when tracking is enabled."""
        self.enabled = settings.enable_mlflow
        if not self.enabled:
            return

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        logger.info("MLflow tracking enabled at %s", settings.mlflow_tracking_uri)

    @contextmanager
    def start_run(
        self,
        run_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[None]:
        """
        Start an MLflow run when tracking is enabled.

        Args:
            run_name: Name of the tracked operation.
            params: Parameters to log at run start.
        """
        if not self.enabled:
            yield
            return

        with mlflow.start_run(run_name=run_name):
            self.log_params(params or {})
            mlflow.set_tag("llm_provider", settings.llm_provider)
            mlflow.set_tag("llm_model", settings.llm_model)
            yield

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log scalar parameters."""
        if not self.enabled:
            return

        safe_params = {
            key: value
            for key, value in params.items()
            if isinstance(value, (str, int, float, bool))
        }
        if safe_params:
            mlflow.log_params(safe_params)

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log numeric metrics."""
        if not self.enabled:
            return

        safe_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        if safe_metrics:
            mlflow.log_metrics(safe_metrics)
