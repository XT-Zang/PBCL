from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from geoseg.losses.binary_seg import loss_batch_kwargs
from geoseg.metrics.binary import resize_logits_to_targets
from geoseg.models.aquav3_outputs import get_main_logits
from geoseg.models.ssepnet.dinov3 import build_dinov3_vitl16_sat493m

from .task import SegmentationTask


class DistilledSegmentationTask(SegmentationTask):
    """AquaDateset task with an unregistered frozen flat-DINO feature teacher."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        spec = dict(getattr(config, "distillation_spec", {}))
        if not spec.get("enabled", False):
            raise ValueError("DistilledSegmentationTask requires enabled distillation")
        checkpoint = Path(str(getattr(config, "distillation_teacher_checkpoint")))
        teacher = build_dinov3_vitl16_sat493m(
            in_chans=4,
            pretrained=True,
            checkpoint_path=checkpoint,
        )
        teacher.requires_grad_(False)
        teacher.eval()
        # Keep the teacher outside nn.Module registration so it is absent from student
        # checkpoints and the prebuilt student optimizer.
        object.__setattr__(self, "_frozen_teacher", teacher)
        self.distillation_spec = spec
        self.teacher_taps = tuple(int(value) for value in spec["teacher_taps"])
        self.feature_loss_weights = tuple(
            float(value) for value in spec["feature_loss_weights"]
        )
        self.teacher_microbatch = int(spec.get("teacher_microbatch", 3))
        if self.teacher_microbatch <= 0:
            raise ValueError("teacher_microbatch must be positive")
        self._student_blocks_frozen: bool | None = None

    @property
    def frozen_teacher(self):
        return object.__getattribute__(self, "_frozen_teacher")

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if any(key.startswith("_frozen_teacher.") for key in state):
            raise RuntimeError("frozen teacher leaked into the student checkpoint")
        return state

    def on_fit_start(self) -> None:
        self.frozen_teacher.to(self.device)
        self.frozen_teacher.eval()

    def on_fit_end(self) -> None:
        try:
            self.frozen_teacher.to("cpu")
        finally:
            super().on_fit_end()

    def train(self, mode: bool = True):
        result = super().train(mode)
        self.frozen_teacher.eval()
        return result

    def _set_student_block_trainability(self, trainable: bool) -> None:
        if self._student_blocks_frozen is (not trainable):
            return
        callback = getattr(self.net, "student_transformer_parameters", None)
        if callback is None:
            raise AttributeError("distilled model has no student_transformer_parameters")
        for parameter in callback():
            parameter.requires_grad_(bool(trainable))
        self._student_blocks_frozen = not bool(trainable)

    def on_train_epoch_start(self) -> None:
        freeze_epochs = int(
            self.distillation_spec.get("freeze_student_blocks_epochs", 3)
        )
        self._set_student_block_trainability(int(self.current_epoch) >= freeze_epochs)

    def _distillation_lambda(self) -> float:
        epoch = int(self.current_epoch)
        start = float(self.distillation_spec.get("lambda_start", 1.0))
        end = float(self.distillation_spec.get("lambda_end", 0.1))
        decay_end = int(self.distillation_spec.get("lambda_decay_end_epoch", 19))
        freeze_epochs = int(
            self.distillation_spec.get("freeze_student_blocks_epochs", 3)
        )
        if epoch < freeze_epochs:
            return start
        if epoch >= decay_end:
            return end
        progress = (epoch - freeze_epochs) / max(1, decay_end - freeze_epochs)
        return start + progress * (end - start)

    def _teacher_features(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs: list[list[torch.Tensor]] = [list() for _ in self.teacher_taps]
        with torch.no_grad(), torch.autocast(
            device_type=image.device.type,
            dtype=torch.bfloat16,
            enabled=image.device.type == "cuda",
        ):
            for start in range(0, int(image.shape[0]), self.teacher_microbatch):
                batch = image[start : start + self.teacher_microbatch]
                values = self.frozen_teacher.model.get_intermediate_layers(
                    batch,
                    n=self.teacher_taps,
                    reshape=True,
                    return_class_token=False,
                    return_extra_tokens=False,
                    norm=True,
                )
                for bucket, value in zip(outputs, values, strict=True):
                    bucket.append(value.detach())
        return tuple(torch.cat(bucket, dim=0) for bucket in outputs)

    @staticmethod
    def _align_spatial(
        student: torch.Tensor, teacher: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        student_area = int(student.shape[-2] * student.shape[-1])
        teacher_area = int(teacher.shape[-2] * teacher.shape[-1])
        if student_area > teacher_area:
            student = F.adaptive_avg_pool2d(student, teacher.shape[-2:])
        elif teacher_area > student_area:
            teacher = F.adaptive_avg_pool2d(teacher, student.shape[-2:])
        return student, teacher

    @staticmethod
    def _normalized_features(features: torch.Tensor) -> torch.Tensor:
        channels = int(features.shape[1])
        return F.layer_norm(
            features.permute(0, 2, 3, 1),
            (channels,),
        ).permute(0, 3, 1, 2)

    def _feature_distillation_loss(
        self,
        student_features: tuple[torch.Tensor, ...],
        teacher_features: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        indices = self.net.distillation_channel_indices()
        if not (
            len(student_features)
            == len(teacher_features)
            == len(indices)
            == len(self.feature_loss_weights)
            == 4
        ):
            raise RuntimeError("distillation requires four aligned feature stages")
        cosine_weight = float(
            self.distillation_spec.get("feature_cosine_weight", 0.75)
        )
        smooth_weight = float(
            self.distillation_spec.get("feature_smooth_l1_weight", 0.25)
        )
        losses: list[torch.Tensor] = []
        for student, teacher, channel_indices in zip(
            student_features, teacher_features, indices, strict=True
        ):
            teacher = teacher.index_select(
                1, channel_indices.to(device=teacher.device)
            )
            student, teacher = self._align_spatial(student, teacher)
            student = self._normalized_features(student.float())
            teacher = self._normalized_features(teacher.float())
            cosine = 1.0 - F.cosine_similarity(
                student.flatten(2), teacher.flatten(2), dim=1
            ).mean()
            smooth = F.smooth_l1_loss(student, teacher)
            losses.append(cosine_weight * cosine + smooth_weight * smooth)
        total = sum(
            weight * loss
            for weight, loss in zip(
                self.feature_loss_weights, losses, strict=True
            )
        )
        return total, tuple(losses)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int):
        del batch_idx
        image, mask = batch["img"], batch["gt_semantic_seg"]
        prediction, student_features = self.net.forward_distill_features(image)
        teacher_features = self._teacher_features(image)
        loss_kwargs = (
            loss_batch_kwargs(batch, image.device)
            if bool(getattr(self.loss, "requires_instance_supervision", False))
            else {}
        )
        segmentation_loss = self.loss(prediction, mask, **loss_kwargs)
        feature_loss, stage_losses = self._feature_distillation_loss(
            student_features, teacher_features
        )
        distillation_lambda = self._distillation_lambda()
        budget_loss, entropy_loss = self.net.routing_regularization()
        budget_weight = float(
            self.distillation_spec.get("routing_budget_weight", 0.02)
        )
        entropy_weight = (
            float(self.distillation_spec.get("routing_entropy_weight", 0.005))
            if int(self.current_epoch)
            <= int(self.distillation_spec.get("routing_entropy_end_epoch", 19))
            else 0.0
        )
        loss = (
            segmentation_loss
            + distillation_lambda * feature_loss
            + budget_weight * budget_loss
            + entropy_weight * entropy_loss
        )
        self.log("train_segmentation_loss", segmentation_loss, on_step=False, on_epoch=True)
        self.log("train_distill_loss", feature_loss, on_step=False, on_epoch=True)
        self.log("train_distill_lambda", float(distillation_lambda), on_step=False, on_epoch=True)
        self.log("train_routing_budget_loss", budget_loss, on_step=False, on_epoch=True)
        self.log("train_routing_entropy_loss", entropy_loss, on_step=False, on_epoch=True)
        for index, stage_loss in enumerate(stage_losses):
            self.log(
                f"train_distill_stage{index + 1}",
                stage_loss,
                on_step=False,
                on_epoch=True,
            )

        logits = resize_logits_to_targets(
            get_main_logits(prediction).detach(), mask.detach()
        )
        probabilities = torch.sigmoid(logits)
        self.metrics_train_binary.update(
            logits, mask.detach(), probabilities=probabilities
        )
        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        gate_stats = self.net.head_gate_statistics()
        for gate_name, values in gate_stats.items():
            if values:
                self.log(
                    f"val_{gate_name}_mean",
                    float(values[0]),
                    on_step=False,
                    on_epoch=True,
                )
        routing = self.net.routing_statistics()
        active = [
            value
            for stage in routing
            for value in stage.get("active_fraction", ())
        ]
        if active:
            self.log(
                "val_routing_active_fraction",
                float(sum(active) / len(active)),
                on_step=False,
                on_epoch=True,
            )
            compute_reduction = self.net.routing_compute_reduction()
            if compute_reduction is not None:
                self.log(
                    "val_routing_compute_reduction",
                    float(compute_reduction),
                    on_step=False,
                    on_epoch=True,
                )
        super().on_validation_epoch_end()


__all__ = ["DistilledSegmentationTask"]
