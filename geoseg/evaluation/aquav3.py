from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from geoseg.runtime import ResolvedExperiment


@dataclass(frozen=True)
class EvaluationRequest:
    checkpoint: Path | None = None
    split: str = "test"
    threshold: float = 0.5
    out_dir: Path | None = None
    num_workers: int | None = None
    save_predictions: bool = False
    save_probabilities: bool = False
    probability_dtype: str = "float16"
    separate_background_batches: bool = False
    max_samples: int = 0
    measure_efficiency: bool = False
    strict: bool = False

    def __post_init__(self) -> None:
        if self.split not in {"val", "test"}:
            raise ValueError(f"Unsupported evaluation split: {self.split}")
        if self.split == "test" and not math.isclose(float(self.threshold), 0.5, abs_tol=1e-12):
            raise ValueError("The test threshold is fixed at 0.5; tune thresholds on validation data only.")
        if self.probability_dtype not in {"float16", "float32"}:
            raise ValueError(f"Unsupported probability dtype: {self.probability_dtype}")
        if self.max_samples < 0:
            raise ValueError("max_samples must be non-negative")
        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


@dataclass(frozen=True)
class EvaluationSummary:
    payload: Mapping[str, Any]

    @property
    def result_row(self) -> Mapping[str, Any]:
        return self.payload["result_row"]

    def to_dict(self) -> dict[str, Any]:
        from geoseg.metrics.deadhesion import json_ready

        return json_ready(dict(self.payload))


def evaluate_checkpoint(experiment: ResolvedExperiment, request: EvaluationRequest) -> EvaluationSummary:
    if experiment.components is None:
        raise ValueError("evaluate_checkpoint requires load_experiment(..., build=True)")

    from scripts import evaluate_aquav3_checkpoint as legacy

    config = experiment.config
    legacy.override_loader_workers(config, request.num_workers)
    checkpoint = legacy.resolve_checkpoint_path(config, request.checkpoint)
    out_dir = request.out_dir or Path(config.weights_path) / f"eval_{request.split}"
    experiment.write_resolved_config(out_dir / "resolved_config.json")
    payload = legacy.evaluate(
        config=config,
        checkpoint_path=checkpoint,
        split=request.split,
        threshold=request.threshold,
        out_dir=out_dir,
        save_predictions=request.save_predictions,
        max_samples=request.max_samples,
        measure=request.measure_efficiency,
        strict=request.strict,
        save_probabilities=request.save_probabilities,
        probability_dtype=request.probability_dtype,
        separate_background=request.separate_background_batches,
    )
    return EvaluationSummary(payload=payload)
