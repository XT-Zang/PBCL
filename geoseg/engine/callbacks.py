from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from time import perf_counter

from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from geoseg.runtime import ResolvedExperiment
from geoseg.runtime.environment import write_environment_report


def _validate_checkpoint_start_epoch(start_epoch: int) -> int:
    start_epoch = int(start_epoch)
    if start_epoch <= 0:
        raise ValueError("checkpoint start_epoch must be positive")
    return start_epoch


class MultiMetricBestCheckpoint(Callback):
    """Keep one validation-best checkpoint per configured role."""

    _ROLES = ("region", "structure", "overall")

    def __init__(
        self,
        *,
        directory: str | Path,
        filename: str,
        monitors: dict[str, str],
        plain_filenames: bool = False,
        save_weights_only: bool = True,
        start_epoch: int = 1,
    ) -> None:
        super().__init__()
        roles = tuple(monitors)
        expected_order = tuple(role for role in self._ROLES if role in monitors)
        if not roles or roles != expected_order:
            raise ValueError(
                f"checkpoint monitors must be a non-empty ordered subset of {self._ROLES}"
            )
        if any(not str(monitors[role]).strip() for role in roles):
            raise ValueError("checkpoint monitor names must be non-empty")
        self.directory = Path(directory)
        self.filename = str(filename)
        self.plain_filenames = plain_filenames
        self.monitors = dict(monitors)
        self.roles = roles
        self.save_weights_only = bool(save_weights_only)
        self.start_epoch = _validate_checkpoint_start_epoch(start_epoch)
        self.write_seconds = 0.0
        self.best_scores = {role: -math.inf for role in self.roles}
        self.best_epochs = {role: -1 for role in self.roles}

    def checkpoint_path(self, role: str) -> Path:
        if role not in self.monitors:
            raise KeyError(role)
        prefix = "" if self.plain_filenames else f"{self.filename}-"
        return self.directory / f"{prefix}best_{role}.ckpt"

    @staticmethod
    def _as_finite_float(value: Any, monitor: str) -> float:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        score = float(value)
        if not math.isfinite(score):
            raise RuntimeError(f"{monitor} must be finite, got {score}")
        return score

    def _validation_scores(self, trainer: Any) -> dict[str, float]:
        metrics = trainer.callback_metrics
        scores: dict[str, float] = {}
        for role, monitor in self.monitors.items():
            if monitor not in metrics:
                raise RuntimeError(f"{monitor} was not logged during validation")
            scores[role] = self._as_finite_float(metrics[monitor], monitor)
        return scores

    @staticmethod
    def _link_or_copy(source: Path, target: Path) -> None:
        target.unlink(missing_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    def _save_improvements(self, trainer: Any, roles: list[str]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.filename}-epoch{int(trainer.current_epoch):03d}-",
            suffix=".ckpt.tmp",
            dir=self.directory,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink()
        try:
            trainer.save_checkpoint(str(temporary), weights_only=self.save_weights_only)
            first = self.checkpoint_path(roles[0])
            os.replace(temporary, first)
            for role in roles[1:]:
                self._link_or_copy(first, self.checkpoint_path(role))
        finally:
            temporary.unlink(missing_ok=True)

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        self.write_seconds = 0.0
        if bool(getattr(trainer, "sanity_checking", False)):
            return
        if int(trainer.current_epoch) + 1 < self.start_epoch:
            return
        scores = self._validation_scores(trainer)
        improved = [role for role in self.roles if scores[role] > self.best_scores[role]]
        if not improved:
            return
        if bool(getattr(trainer, "is_global_zero", True)):
            start = perf_counter()
            self._save_improvements(trainer, improved)
            self.write_seconds = perf_counter() - start
        epoch = int(trainer.current_epoch)
        for role in improved:
            self.best_scores[role] = scores[role]
            self.best_epochs[role] = epoch

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_scores": dict(self.best_scores),
            "best_epochs": dict(self.best_epochs),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.best_scores = {
            role: float(state_dict.get("best_scores", {}).get(role, -math.inf))
            for role in self.roles
        }
        self.best_epochs = {
            role: int(state_dict.get("best_epochs", {}).get(role, -1))
            for role in self.roles
        }


class PilotStateCheckpoint(Callback):
    """Persist one full-state epoch-8 checkpoint for exact pilot continuation."""

    def __init__(self, directory: str | Path, *, pilot_epochs: int = 8) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.pilot_epochs = int(pilot_epochs)
        if self.pilot_epochs <= 0:
            raise ValueError("pilot_epochs must be positive")

    @property
    def path(self) -> Path:
        return self.directory / "_state" / f"pilot_epoch_{self.pilot_epochs:03d}.ckpt"

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        if bool(getattr(trainer, "sanity_checking", False)):
            return
        if int(trainer.current_epoch) + 1 != self.pilot_epochs:
            return
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".ckpt.tmp")
        temporary.unlink(missing_ok=True)
        try:
            trainer.save_checkpoint(str(temporary), weights_only=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


class EpochStateCheckpoint(Callback):
    """Atomically keep the latest full training state for crash recovery.

    The checkpoint lives below ``_state`` so it is not confused with the three
    validation-selected role checkpoints.  A previous valid state remains in
    place until the next state has been written completely.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        start_epoch: int = 1,
        every_n_epochs: int = 1,
        plain_filename: bool = False,
    ) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.plain_filename = plain_filename
        self.write_seconds = 0.0
        self.start_epoch = _validate_checkpoint_start_epoch(start_epoch)
        self.every_n_epochs = int(every_n_epochs)
        if self.every_n_epochs <= 0:
            raise ValueError("every_n_epochs must be positive")

    @property
    def path(self) -> Path:
        return self.directory / "last.ckpt" if self.plain_filename else self.directory / "_state" / "resume_last.ckpt"

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        self.write_seconds = 0.0
        if bool(getattr(trainer, "sanity_checking", False)):
            return
        completed_epochs = int(trainer.current_epoch) + 1
        if completed_epochs < self.start_epoch:
            return
        if completed_epochs % self.every_n_epochs != 0:
            return
        if not bool(getattr(trainer, "is_global_zero", True)):
            return
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".ckpt.tmp")
        temporary.unlink(missing_ok=True)
        start = perf_counter()
        try:
            trainer.save_checkpoint(str(temporary), weights_only=False)
            os.replace(temporary, path)
            self.write_seconds = perf_counter() - start
        finally:
            temporary.unlink(missing_ok=True)


class ComparisonEpochTiming(Callback):
    """Record the completed train, validation, and checkpoint times."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory) / "epoch_timing"

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self.train_start = perf_counter()

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self.validation_start = perf_counter()

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        writes = {
            type(callback).__name__: callback.write_seconds for callback in trainer.callbacks
            if isinstance(callback, (EpochStateCheckpoint, MultiMetricBestCheckpoint))
        }
        epoch = trainer.current_epoch + 1
        record = {
            "epoch": epoch, "train_seconds": self.validation_start - self.train_start,
            "validation_seconds": perf_counter() - self.validation_start - sum(writes.values()),
            "checkpoint_write_seconds": writes,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"epoch_{epoch:03d}.json").write_text(json.dumps(record, indent=2) + "\n")


class FiniteTrainingLoss(Callback):
    """Stop a run before nonfinite gradients or checkpoint updates propagate."""

    def on_before_backward(self, trainer: Any, pl_module: Any, loss: Any) -> None:
        if not bool(loss.detach().isfinite().all()):
            raise RuntimeError(f"Nonfinite training loss at epoch {trainer.current_epoch + 1}, "
                               f"optimizer step {trainer.global_step}")

    def on_before_optimizer_step(self, trainer: Any, pl_module: Any, optimizer: Any) -> None:
        import torch

        parameters = (param for group in optimizer.param_groups for param in group["params"])
        try:
            torch.nn.utils.clip_grad_norm_(parameters, math.inf, error_if_nonfinite=True)
        except RuntimeError as error:
            raise RuntimeError(f"Gradient check failed at epoch {trainer.current_epoch + 1}, "
                               f"optimizer step {trainer.global_step}: {error}") from error


class FinalStateCheckpoint(Callback):
    """Save a complete state once the configured training schedule finishes."""

    def __init__(self, directory: str | Path) -> None:
        super().__init__()
        self.path = Path(directory) / "last.ckpt"

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        if trainer.current_epoch < trainer.max_epochs or not trainer.is_global_zero:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".ckpt.tmp")
        try:
            trainer.save_checkpoint(str(temporary), weights_only=False)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


class StartEpochModelCheckpoint(ModelCheckpoint):
    """A Lightning checkpoint that ignores epochs before a 1-based threshold."""

    def __init__(self, *args: Any, start_epoch: int = 1, **kwargs: Any) -> None:
        self.start_epoch = _validate_checkpoint_start_epoch(start_epoch)
        super().__init__(*args, **kwargs)

    def _should_skip_saving_checkpoint(self, trainer: Any) -> bool:
        return (
            int(trainer.current_epoch) + 1 < self.start_epoch
            or super()._should_skip_saving_checkpoint(trainer)
        )


@dataclass(frozen=True)
class CheckpointCallbackFactory:
    save_top_k: int
    monitor: str
    mode: str
    directory: str
    filename: str
    every_n_epochs: int
    start_epoch: int

    @classmethod
    def from_config(cls, config: Any) -> "CheckpointCallbackFactory":
        return cls(
            save_top_k=int(config.save_top_k),
            monitor=str(config.monitor),
            mode=str(config.monitor_mode),
            directory=str(config.weights_path),
            filename=str(config.weights_name),
            every_n_epochs=int(getattr(config, "checkpoint_every_n_epochs", 5)),
            start_epoch=int(getattr(config, "checkpoint_start_epoch", 10)),
        )

    def build(self) -> list[ModelCheckpoint]:
        callbacks = [
            StartEpochModelCheckpoint(
                save_top_k=self.save_top_k,
                monitor=self.monitor,
                save_last=False,
                mode=self.mode,
                dirpath=self.directory,
                filename=self.filename,
                start_epoch=self.start_epoch,
            )
        ]
        if self.every_n_epochs > 0:
            callbacks.append(
                StartEpochModelCheckpoint(
                    save_top_k=-1,
                    monitor=None,
                    save_last=False,
                    every_n_epochs=self.every_n_epochs,
                    dirpath=self.directory,
                    filename=f"{self.filename}-epoch{{epoch:03d}}",
                    auto_insert_metric_name=False,
                    start_epoch=self.start_epoch,
                )
            )
        return callbacks


class TrainingManifestCallback(Callback):
    """Persist pure configuration and environment provenance at run start."""

    def __init__(self, experiment: ResolvedExperiment) -> None:
        super().__init__()
        self.experiment = experiment

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        output_dir = self.experiment.spec.artifacts["weights_path"]
        self.experiment.write_resolved_config()
        write_environment_report(Path(output_dir) / "environment.json")
