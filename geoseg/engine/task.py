from __future__ import annotations

import json
import random
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pytorch_lightning as pl
import torch

from geoseg.losses.binary_seg import loss_batch_kwargs
from geoseg.metrics.aquadataset import AQUADATASET_OVERALL_SCORE_WEIGHTS, aquadataset_overall_score
from geoseg.metrics.binary import binary_valid_mask, resize_logits_to_targets
from geoseg.metrics.deadhesion import CachedDeadhesionEvaluator, json_ready as deadhesion_json_ready
from geoseg.models.aquav3_outputs import get_main_logits
from .artifacts import ValidationArtifactWriter, build_binary_validation_cpu_buffer
from .metrics import BinaryMetricAccumulator, MulticlassMetricAccumulator


class SegmentationTask(pl.LightningModule):
    """Shared Lightning task that preserves the legacy GeoSeg training contract."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss
        self.binary_segmentation = bool(getattr(config, "binary_segmentation", False))
        self.instance_segmentation = bool(getattr(config, "instance_segmentation", False))
        self.metric_threshold = float(getattr(config, "metric_threshold", 0.5))
        self.instance_score_threshold = float(
            getattr(config, "instance_score_threshold", 0.5)
        )
        self.instance_mask_threshold = float(
            getattr(config, "instance_mask_threshold", 0.5)
        )
        self.instance_num_queries = int(getattr(config, "instance_num_queries", 100))
        self.validation_artifacts = ValidationArtifactWriter(config)
        if self.binary_segmentation:
            self.metrics_train_binary = BinaryMetricAccumulator(self.metric_threshold)
            self.metrics_val_binary = BinaryMetricAccumulator(self.metric_threshold)
            self.deadhesion_val_evaluator = self._make_deadhesion_evaluator()
            if self.instance_segmentation:
                from torchmetrics.detection import MeanAveragePrecision

                self.metrics_val_instance = MeanAveragePrecision(iou_type="segm")
        else:
            self.metrics_train = MulticlassMetricAccumulator(config.num_classes)
            self.metrics_val = MulticlassMetricAccumulator(config.num_classes)
            self.deadhesion_val_evaluator = None

    def forward(self, inputs: torch.Tensor):
        return self.net(inputs)

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        if not isinstance(batch, Mapping):
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        canonical = dict(batch)
        had_image_alias = "image" in canonical and "img" in canonical
        had_mask_alias = "mask" in canonical and "gt_semantic_seg" in canonical
        if had_image_alias:
            canonical.pop("image")
        if had_mask_alias:
            canonical.pop("mask")
        moved = super().transfer_batch_to_device(canonical, device, dataloader_idx)
        if bool(getattr(self.config, "channels_last", False)) and isinstance(moved.get("img"), torch.Tensor):
            moved["img"] = moved["img"].contiguous(memory_format=torch.channels_last)
        if had_image_alias:
            moved["image"] = moved["img"]
        if had_mask_alias:
            moved["mask"] = moved["gt_semantic_seg"]
        return moved

    def _make_deadhesion_evaluator(self) -> CachedDeadhesionEvaluator | None:
        if not bool(getattr(self.config, "deadhesion_validation", False)):
            return None
        data_root = getattr(self.config, "data_root", None)
        if not data_root:
            raise ValueError("deadhesion_validation requires config.data_root")
        legacy_module = sys.modules.get("train_supervision")
        evaluator_cls = getattr(legacy_module, "CachedDeadhesionEvaluator", CachedDeadhesionEvaluator)
        return evaluator_cls(
            data_root,
            split=str(getattr(self.config, "deadhesion_split", "val")),
            instances_coco=getattr(self.config, "deadhesion_instances_coco", None),
            boundary_width=int(getattr(self.config, "deadhesion_boundary_width", 2)),
            boundary_tolerance=int(getattr(self.config, "deadhesion_boundary_tolerance", 2)),
            min_overlap_pixels=int(getattr(self.config, "deadhesion_min_overlap_pixels", 8)),
            min_overlap_ratio=float(getattr(self.config, "deadhesion_min_overlap_ratio", 0.05)),
        )

    @staticmethod
    def _batch_tile_ids(batch: Mapping[str, Any], batch_size: int) -> list[str]:
        tile_ids = batch.get("tile_id", batch.get("img_id"))
        if isinstance(tile_ids, (list, tuple)) and len(tile_ids) == batch_size:
            return [str(tile_id) for tile_id in tile_ids]
        paths = batch.get("path")
        if isinstance(paths, (list, tuple)) and len(paths) == batch_size:
            return [Path(str(path)).stem for path in paths]
        raise ValueError("deadhesion validation requires batch tile_id, img_id, or path metadata")

    def _write_deadhesion_val_summary(self, summary: dict[str, Any]) -> None:
        detail_dir = Path(self.config.weights_path) / "deadhesion_val"
        detail_dir.mkdir(parents=True, exist_ok=True)
        epoch = int(getattr(self, "current_epoch", 0)) + 1
        (detail_dir / f"epoch_{epoch:03d}.json").write_text(
            json.dumps(deadhesion_json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _loss_kwargs(self, batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
        kwargs = loss_batch_kwargs(batch, device)
        if not bool(getattr(self.loss, "requires_instance_supervision", False)):
            kwargs.pop("instance_maps", None)
            kwargs.pop("instance_pairs", None)
        if not bool(getattr(self.loss, "requires_ewc_supervision", False)):
            kwargs.pop("ewc_weights", None)
        if not bool(getattr(self.loss, "requires_wprf_supervision", False)):
            kwargs.pop("wprf_support", None)
            kwargs.pop("wprf_dense_components", None)
            kwargs.pop("wprf_pairs", None)
        return kwargs

    def _set_loss_epoch(self) -> None:
        callback = getattr(self.loss, "set_epoch", None)
        if callable(callback):
            callback(int(self.current_epoch))

    def on_train_epoch_start(self) -> None:
        self._set_loss_epoch()

    def on_validation_epoch_start(self) -> None:
        self._set_loss_epoch()

    def _predict_for_loss(
        self,
        image: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        loss_kwargs = self._loss_kwargs(batch, image.device)
        model_kwargs: dict[str, Any] = {}
        prepare = getattr(self.loss, "model_forward_kwargs", None)
        if callable(prepare):
            model_kwargs = dict(prepare(**loss_kwargs))
        return self.net(image, **model_kwargs), loss_kwargs

    def _instance_batch_predictions(
        self,
        prediction: Mapping[str, Any],
        *,
        size: tuple[int, int],
        score_threshold: float | None,
        valid_masks: torch.Tensor | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        from geoseg.models.mask2former_instance import postprocess_mask2former_instances

        return postprocess_mask2former_instances(
            prediction,
            size=size,
            score_threshold=score_threshold,
            mask_probability_threshold=self.instance_mask_threshold,
            max_instances=self.instance_num_queries,
            valid_masks=valid_masks,
        )

    def _instance_union_probabilities(
        self,
        prediction: Mapping[str, Any],
        *,
        size: tuple[int, int],
        valid_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        from geoseg.models.mask2former_instance import union_instance_masks

        instances = self._instance_batch_predictions(
            prediction,
            size=size,
            score_threshold=self.instance_score_threshold,
            valid_masks=valid_masks,
        )
        union = union_instance_masks(
            instances,
            size=size,
            device=get_main_logits(prediction).device,
        )
        return union.float(), instances

    def _binary_epoch_end(self, split: str) -> None:
        accumulator = self.metrics_train_binary if split == "train" else self.metrics_val_binary
        stats = accumulator.compute()
        eval_value = {
            "fg_iou": stats["fg_iou"],
            "bg_iou": stats["bg_iou"],
            "miou": stats["miou"],
            "fg_dice": stats["fg_dice"],
            "fg_precision": stats["fg_precision"],
            "fg_recall": stats["fg_recall"],
            "bg_fp_rate": stats["bg_fp_rate"],
        }
        if split == "val" and self.deadhesion_val_evaluator is not None:
            deadhesion_summary = self.deadhesion_val_evaluator.summary()
            eval_value.update(self.deadhesion_val_evaluator.flat_metrics(deadhesion_summary))
            self._write_deadhesion_val_summary(deadhesion_summary)
            self.deadhesion_val_evaluator.reset()
        if split == "val" and self.instance_segmentation:
            instance_stats = self.metrics_val_instance.compute()
            eval_value.update(
                {
                    "instance_ap": float(instance_stats["map"].detach().cpu()),
                    "instance_ap50": float(instance_stats["map_50"].detach().cpu()),
                    "instance_ap75": float(instance_stats["map_75"].detach().cpu()),
                    "instance_ar100": float(instance_stats["mar_100"].detach().cpu()),
                }
            )
            self.metrics_val_instance.reset()
        overall_score_weights = getattr(self.config, "overall_score_weights", None)
        if split == "val" and overall_score_weights is not None:
            if dict(overall_score_weights) != AQUADATASET_OVERALL_SCORE_WEIGHTS:
                raise ValueError("AquaDateset overall_score_weights must match the locked campaign formula")
            eval_value["overall_score"] = aquadataset_overall_score(eval_value)
        print(f"{split}:", eval_value)
        self.log_dict({f"{split}_{key}": value for key, value in eval_value.items()}, prog_bar=True)
        accumulator.reset()

    def training_step(self, batch: Mapping[str, Any], batch_idx: int):
        image, mask = batch["img"], batch["gt_semantic_seg"]
        prediction, loss_kwargs = self._predict_for_loss(image, batch)
        loss = self.loss(prediction, mask, **loss_kwargs)
        if self.binary_segmentation:
            logits = resize_logits_to_targets(get_main_logits(prediction).detach(), mask.detach())
            probabilities = (
                self._instance_union_probabilities(
                    prediction,
                    size=tuple(mask.shape[-2:]),
                    valid_masks=batch.get("valid_mask"),
                )[0]
                if self.instance_segmentation
                else torch.sigmoid(logits)
            )
            self.metrics_train_binary.update(
                logits,
                mask.detach(),
                probabilities=probabilities,
                valid_mask=batch.get("valid_mask"),
            )
            return {"loss": loss}

        output = prediction[0] if self.config.use_aux_loss else prediction
        self.metrics_train.update(output, mask)
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        audit = getattr(self.loss, "negative_gradient_audit", {})
        if audit.get("epoch") == int(self.current_epoch) + 1:
            directory = Path(self.config.weights_path) / "wprf_gradient_audit"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"epoch_{int(self.current_epoch) + 1:03d}.json").write_text(
                json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
        if self.binary_segmentation:
            self._binary_epoch_end("train")
            return
        self._multiclass_epoch_end("train")

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int):
        image, mask = batch["img"], batch["gt_semantic_seg"]
        prediction, loss_kwargs = self._predict_for_loss(image, batch)
        if self.binary_segmentation:
            logits = resize_logits_to_targets(get_main_logits(prediction).detach(), mask.detach())
            structure_instances: list[dict[str, torch.Tensor]] | None = None
            if self.instance_segmentation:
                from geoseg.losses.mask2former_instance import instance_metric_targets
                from geoseg.models.mask2former_instance import union_instance_masks

                size = tuple(mask.shape[-2:])
                valid_masks = loss_kwargs.get("valid_masks")
                ap_predictions = self._instance_batch_predictions(
                    prediction,
                    size=size,
                    score_threshold=None,
                    valid_masks=valid_masks,
                )
                structure_instances = []
                for instances in ap_predictions:
                    keep = instances["scores"] >= self.instance_score_threshold
                    structure_instances.append({"masks": instances["masks"][keep]})
                probabilities = union_instance_masks(
                    structure_instances, size=size, device=logits.device,
                ).float()
                ap_targets = instance_metric_targets(
                    loss_kwargs["instance_maps"],
                    max_instances=self.instance_num_queries,
                    valid_masks=valid_masks,
                )
                self.metrics_val_instance.update(ap_predictions, ap_targets)
            else:
                probabilities = torch.sigmoid(logits)
            if batch.get("valid_mask") is not None:
                valid = binary_valid_mask(batch["valid_mask"], probabilities)
                probabilities = probabilities.masked_fill(~valid, 0.0)
            self.metrics_val_binary.update(
                logits,
                mask.detach(),
                probabilities=probabilities,
                valid_mask=batch.get("valid_mask"),
            )
            cpu_buffer = None
            if self.deadhesion_val_evaluator is not None or self.validation_artifacts.enabled:
                cpu_buffer = build_binary_validation_cpu_buffer(
                    probabilities,
                    mask,
                    threshold=self.metric_threshold,
                )
            if self.deadhesion_val_evaluator is not None:
                tile_ids = self._batch_tile_ids(batch, batch_size=int(logits.shape[0]))
                if structure_instances is not None:
                    self.deadhesion_val_evaluator.update_instance_batch(
                        structure_instances,
                        cpu_buffer.targets.numpy(),
                        tile_ids,
                    )
                else:
                    self.deadhesion_val_evaluator.update_batch(
                        cpu_buffer.predictions.numpy(),
                        cpu_buffer.targets.numpy(),
                        tile_ids,
                    )
            if self.validation_artifacts.enabled and not getattr(self._trainer, "sanity_checking", False):
                self.validation_artifacts.save_binary_batch(
                    batch,
                    probabilities=probabilities,
                    mask=mask,
                    epoch=int(self.current_epoch) + 1,
                    global_step=int(self.global_step),
                    cpu_buffer=cpu_buffer,
                )
            loss_val = self.loss(prediction, mask, **loss_kwargs)
            return {"loss_val": loss_val}

        self.metrics_val.update(prediction, mask)
        loss_val = self.loss(prediction, mask, **loss_kwargs)
        return {"loss_val": loss_val}

    def on_validation_epoch_end(self) -> None:
        self.validation_artifacts.flush()
        if self.binary_segmentation:
            self._binary_epoch_end("val")
            return
        self._multiclass_epoch_end("val")

    def on_fit_end(self) -> None:
        self.validation_artifacts.close()

    def _multiclass_epoch_end(self, split: str) -> None:
        metrics = self.metrics_train if split == "train" else self.metrics_val
        drop_last_class = any(
            name in self.config.log_name
            for name in ("vaihingen", "potsdam", "whubuilding", "massbuilding", "cropland")
        )
        stats = metrics.compute()
        iou_per_class = stats["iou_per_class"]
        f1_per_class = stats["f1_per_class"]
        selected_iou = iou_per_class[:-1] if drop_last_class else iou_per_class
        selected_f1 = f1_per_class[:-1] if drop_last_class else f1_per_class
        values = {
            "mIoU": np.nanmean(selected_iou),
            "F1": np.nanmean(selected_f1),
            "OA": np.nanmean(stats["oa"]),
        }
        print(f"{split}:", values)
        print(dict(zip(self.config.classes, iou_per_class)))
        metrics.reset()
        self.log_dict({f"{split}_{key}": value for key, value in values.items()}, prog_bar=True)

    def _comparison_generators(self):
        generators = {}
        for split, loader in (("train", self.train_dataloader()), ("val", self.val_dataloader())):
            for name, owner in (("loader", loader), ("sampler", loader.sampler)):
                if generator := getattr(owner, "generator", None):
                    generators[f"{split}_{name}"] = generator
        return generators

    def on_save_checkpoint(self, checkpoint):
        if not getattr(self.config, "plain_checkpoint_names", False):
            return
        checkpoint["segformer_comparison_rng"] = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "generators": {key: generator.get_state() for key, generator in self._comparison_generators().items()},
        }

    def on_load_checkpoint(self, checkpoint):
        self._comparison_resume_rng = checkpoint.get("segformer_comparison_rng")

    def on_train_start(self):
        state = getattr(self, "_comparison_resume_rng", None)
        if state is None:
            return
        self._comparison_resume_rng = None
        torch.set_rng_state(state["torch"].cpu())
        if state["cuda"]:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
        random.setstate(state["python"])
        for key, generator in self._comparison_generators().items():
            generator.set_state(state["generators"][key].cpu())

    def configure_optimizers(self):
        return [self.config.optimizer], [self.config.lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader
