from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler, default_collate


CLASSES = ("background", "foreground")
PALETTE = [[0, 0, 0], [255, 255, 255]]

AQUAV3_MEAN = np.array([0.4251161849, 0.4954026022, 0.4909278856, 0.2641421088], dtype=np.float32)
AQUAV3_STD = np.array([0.2261392075, 0.2244978271, 0.229417928, 0.2071894144], dtype=np.float32)
AQUAV3_RAW_CLIP_MIN = 100.0
AQUAV3_RAW_CLIP_MAX = 60000.0
AQUAV3_RAW_OFFSET = 100.0
AQUAV3_RAW_SCALE = 59900.0


def _read_raster(path: Path, indexes: list[int]) -> np.ndarray:
    import rasterio

    with rasterio.open(str(path)) as ds:
        return ds.read(indexes=indexes).astype(np.float32)


def _manifest_valid_mask(raster, entry: dict) -> np.ndarray:
    valid = raster.dataset_mask() > 0
    height = int(entry.get("valid_height", valid.shape[0]))
    width = int(entry.get("valid_width", valid.shape[1]))
    if not 0 < height <= valid.shape[0] or not 0 < width <= valid.shape[1]:
        raise ValueError(f"Invalid valid rectangle for {entry['tile_id']}")
    valid[height:, :] = False
    valid[:, width:] = False
    return valid


def read_manifest_valid_mask(image_path: Path, entry: dict) -> np.ndarray:
    import rasterio

    with rasterio.open(image_path) as raster:
        return _manifest_valid_mask(raster, entry)


def _read_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        import rasterio

        with rasterio.open(str(path)) as ds:
            arr = ds.read(1)
    else:
        image = Image.open(path).convert("L")
        if image.size != (shape_hw[1], shape_hw[0]):
            image = image.resize((shape_hw[1], shape_hw[0]), Image.Resampling.NEAREST)
        arr = np.asarray(image)
    if arr.shape != shape_hw:
        raise ValueError(f"Mask shape {arr.shape} does not match image shape {shape_hw} for {path}")
    return (arr.astype(np.float32) > 0).astype(np.float32)


def normalized_aquav3_to_rgb(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().float().numpy()
    arr = arr * AQUAV3_STD[: arr.shape[0], None, None] + AQUAV3_MEAN[: arr.shape[0], None, None]
    if arr.shape[0] >= 3:
        rgb = arr[[2, 1, 0]]
    elif arr.shape[0] == 2:
        rgb = np.concatenate([arr, arr[:1]], axis=0)
    else:
        rgb = np.repeat(arr[:1], 3, axis=0)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def save_aquav3_prediction_preview(image: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = normalized_aquav3_to_rgb(image)
    gt = (mask.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8) * 255
    pred_arr = (pred.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8) * 255
    overlay = rgb.copy()
    overlay[pred_arr > 0] = (0.45 * overlay[pred_arr > 0] + 0.55 * np.array([255, 64, 32])).astype(np.uint8)
    overlay[gt > 0] = (0.55 * overlay[gt > 0] + 0.45 * np.array([32, 220, 120])).astype(np.uint8)
    canvas = np.concatenate(
        [
            rgb,
            np.repeat(gt[..., None], 3, axis=2),
            np.repeat(pred_arr[..., None], 3, axis=2),
            overlay,
        ],
        axis=1,
    )
    Image.fromarray(canvas).save(path)


class AquaV3ManifestDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 512,
        band_indices: list[int] | None = None,
        augment: bool = False,
        raw_clip_min: float = AQUAV3_RAW_CLIP_MIN,
        raw_clip_max: float = AQUAV3_RAW_CLIP_MAX,
        raw_offset: float = AQUAV3_RAW_OFFSET,
        raw_scale: float = AQUAV3_RAW_SCALE,
        normalize_mean: list[float] | np.ndarray | None = None,
        normalize_std: list[float] | np.ndarray | None = None,
        instance_target_root: str | Path | None = None,
        ewc_weight_root: str | Path | None = None,
        wprf_target_root: str | Path | None = None,
        wprf_dynamic_from_instance: bool = False,
        wprf_semantic_cache: str | Path | None = None,
        wprf_sampling_seed: int = 42,
        resize_mode: str = "resize",
        use_manifest_valid_mask: bool = False,
        instance_pair_policy: str = "all",
        load_instance_supervision: bool = True,
        skip_gdal_directory_scan: bool = False,
        image_cache_root: str | Path | None = None,
        model_image_size: int | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.root = Path(root)
        self.split = split
        self.image_size = int(image_size)
        self.model_image_size = model_image_size
        self.band_indices = list(band_indices if band_indices is not None else [0, 1, 2, 3])
        self.rasterio_indexes = [index + 1 for index in self.band_indices]
        self.augment = bool(augment)
        self.use_manifest_valid_mask = bool(use_manifest_valid_mask)
        self.load_instance_supervision = bool(load_instance_supervision)
        self.skip_gdal_directory_scan = bool(skip_gdal_directory_scan)
        self.image_cache_root = Path(image_cache_root) if image_cache_root is not None else None
        self.instance_pair_policy = str(instance_pair_policy)
        if self.instance_pair_policy not in {"all", "no_four_contact", "noncontact", "four_contact"}:
            raise ValueError(f"Unsupported instance pair policy: {instance_pair_policy}")
        self.resize_mode = str(resize_mode).strip().lower()
        if self.resize_mode not in {"resize", "pad"}:
            raise ValueError(f"Unsupported resize_mode: {resize_mode}")
        self.raw_clip_min = float(raw_clip_min)
        self.raw_clip_max = float(raw_clip_max)
        self.raw_offset = float(raw_offset)
        self.raw_scale = float(raw_scale)
        self.normalize_mean = np.asarray(normalize_mean if normalize_mean is not None else AQUAV3_MEAN, dtype=np.float32)
        self.normalize_std = np.asarray(normalize_std if normalize_std is not None else AQUAV3_STD, dtype=np.float32)
        if instance_target_root is None:
            self.instance_target_root = None
        else:
            instance_root = Path(instance_target_root)
            self.instance_target_root = instance_root if instance_root.is_absolute() else self.root / instance_root
        if ewc_weight_root is None:
            self.ewc_weight_root = None
        else:
            weight_root = Path(ewc_weight_root)
            self.ewc_weight_root = (
                weight_root if weight_root.is_absolute() else self.root / weight_root
            )
        if wprf_target_root is None:
            self.wprf_target_root = None
        else:
            structural_root = Path(wprf_target_root)
            self.wprf_target_root = (
                structural_root
                if structural_root.is_absolute()
                else self.root / structural_root
            )
        self.wprf_semantic_cache = wprf_semantic_cache
        self.wprf_sampling_seed = wprf_sampling_seed + {"train": 0, "val": 100_000, "test": 200_000}.get(split, 0)
        self.wprf_dynamic_from_instance = bool(wprf_dynamic_from_instance)
        if self.wprf_dynamic_from_instance and self.instance_target_root is None:
            raise ValueError("dynamic WPRF targets require instance_target_root")
        if self.raw_scale <= 0:
            raise ValueError("raw_scale must be positive")
        if len(self.normalize_mean) < len(self.band_indices) or len(self.normalize_std) < len(self.band_indices):
            raise ValueError("normalize_mean/std must cover selected bands")
        if np.any(self.normalize_std[: len(self.band_indices)] <= 0):
            raise ValueError("normalize_std values must be positive")
        self.entries = self._load_entries()
        if not self.entries:
            raise RuntimeError(f"No AquaV3 entries for split={split} under {self.root}")

    def _load_entries(self) -> list[dict[str, Any]]:
        split_path = self.root / "splits" / f"{self.split}.jsonl"
        manifest_path = split_path if split_path.exists() else self.root / "manifest.jsonl"
        entries: list[dict[str, Any]] = []
        with open(manifest_path, encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                if manifest_path.name == "manifest.jsonl" and record.get("split") != self.split:
                    continue
                entries.append(record)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def is_negative(self) -> list[bool]:
        return [bool(entry.get("is_background", False)) for entry in self.entries]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.skip_gdal_directory_scan:
            import rasterio

            # TRUE skips sibling enumeration while allowing named sidecar lookup.
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="TRUE"):
                return self._load_item(index)
        return self._load_item(index)

    def _load_item(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        image_path = self.root / entry["image"]
        read_path = image_path if self.image_cache_root is None else self.image_cache_root / image_path.name
        source_valid = None
        if self.use_manifest_valid_mask:
            import rasterio

            with rasterio.open(str(read_path)) as raster:
                raw = raster.read(indexes=self.rasterio_indexes).astype(np.float32)
                source_valid = torch.from_numpy(_manifest_valid_mask(raster, entry)).unsqueeze(0)
        else:
            raw = _read_raster(read_path, self.rasterio_indexes)
        arr = np.clip(raw, self.raw_clip_min, self.raw_clip_max)
        arr = (arr - self.raw_offset) / self.raw_scale
        mean = self.normalize_mean[: len(self.band_indices), None, None]
        std = self.normalize_std[: len(self.band_indices), None, None]
        image = torch.from_numpy(((arr - mean) / std).astype(np.float32, copy=False)).contiguous()

        shape_hw = tuple(image.shape[-2:])
        if source_valid is not None:
            image = image.masked_fill(~source_valid, 0.0)
        mask_path = entry.get("semantic")
        use_instance_mask = not mask_path and not bool(entry.get("is_background", False))
        instance_map, instance_pairs = (None, None)
        if self.load_instance_supervision or self.wprf_dynamic_from_instance or use_instance_mask:
            instance_map, instance_pairs = self._load_instance_target(entry, shape_hw)
        ewc_weight = self._load_ewc_weight(entry, shape_hw)
        wprf_target = (
            None
            if self.wprf_dynamic_from_instance
            else self._load_wprf_target(entry, shape_hw)
        )
        if mask_path:
            mask_arr = _read_mask(self.root / mask_path, shape_hw)
        elif instance_map is not None:
            mask_arr = (instance_map.squeeze(0).numpy() > 0).astype(np.float32)
        else:
            mask_arr = np.zeros(shape_hw, dtype=np.float32)
        mask = torch.from_numpy(mask_arr).unsqueeze(0).contiguous()
        if not self.load_instance_supervision and not self.wprf_dynamic_from_instance:
            instance_map, instance_pairs = None, None

        supervision_parts = [mask]
        if instance_map is not None:
            supervision_parts.append(instance_map.float())
        if ewc_weight is not None:
            supervision_parts.append(ewc_weight)
        if source_valid is not None:
            supervision_parts.append(source_valid.float())
        supervision = torch.cat(supervision_parts, dim=0)
        if self.augment:
            image, supervision, wprf_target = self._augment(
                image, supervision, wprf_target
            )
        if self.resize_mode == "pad":
            runtime_valid = torch.ones(
                (1, *image.shape[-2:]), dtype=torch.bool, device=image.device
            )
            image, supervision = self._pad(image, supervision)
            valid_mask, _ = self._pad(runtime_valid, runtime_valid)
        else:
            image, supervision = self._resize(image, supervision)
            valid_mask = None
        if source_valid is not None:
            stored_valid = supervision[-1:].bool()
            valid_mask = stored_valid if valid_mask is None else valid_mask & stored_valid
            image = image.masked_fill(~valid_mask, 0.0)
        mask = supervision[:1]
        if valid_mask is not None:
            mask = mask.masked_fill(~valid_mask, 0.0)
        supervision_index = 1
        if instance_map is not None:
            instance_map = (
                supervision[supervision_index : supervision_index + 1]
                .round()
                .to(dtype=torch.long)
                .contiguous()
            )
            supervision_index += 1
            if valid_mask is not None:
                instance_map = instance_map.masked_fill(~valid_mask, 0)
        if ewc_weight is not None:
            ewc_weight = supervision[
                supervision_index : supervision_index + 1
            ].float().contiguous()
            if valid_mask is not None:
                # EWC requires strictly positive pixel weights.  Runtime
                # padding is excluded from the loss by valid_mask, but it must
                # still carry the neutral multiplicative weight rather than
                # the zero used for masks and instance ids.
                ewc_weight = torch.where(
                    valid_mask,
                    ewc_weight,
                    torch.ones_like(ewc_weight),
                ).contiguous()
        if self.wprf_dynamic_from_instance:
            if instance_map is None:
                raise KeyError("dynamic WPRF target construction requires instance_map")
            from .aquadataset_wprf import build_wprf_target, wprf_pair_seed

            built = build_wprf_target(
                instance_map[0].detach().cpu().numpy() > 0,
                pair_seed=wprf_pair_seed(
                    str(entry["tile_id"]),
                    salt="vegasdenseclose-wprf-matched-v1",
                ),
            )
            wprf_target = {
                "support": torch.from_numpy(built["support"]).unsqueeze(0).bool(),
                "dense_components": torch.from_numpy(
                    built["dense_components"]
                ).unsqueeze(0).long(),
                "pairs": {
                    key: torch.from_numpy(value).long()
                    for key, value in built.items()
                    if key.startswith("pos_") or key.startswith("neg_")
                },
            }

        if self.wprf_semantic_cache is not None:
            from .segformer_wprf_targets import cached_semantic_target

            union = mask[0].numpy() > 0.5
            valid = np.ones_like(union) if valid_mask is None else valid_mask[0].numpy()
            built = cached_semantic_target(
                self.wprf_semantic_cache, union, valid, self.wprf_sampling_seed + index,
            )
            wprf_target = {
                "support": torch.from_numpy(built["support"]).unsqueeze(0),
                "dense_components": torch.from_numpy(built["dense_components"]).unsqueeze(0).long(),
                "pairs": {key: torch.from_numpy(value).long() for key, value in built.items()
                          if key.startswith(("pos_", "neg_"))},
            }

        if self.model_image_size is not None:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.model_image_size, self.model_image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).contiguous()

        item = {
            "img": image,
            "image": image,
            "gt_semantic_seg": mask,
            "mask": mask,
            "img_id": entry["tile_id"],
            "tile_id": entry["tile_id"],
            "path": str(image_path),
            "is_background": bool(entry.get("is_background", False)),
        }
        if instance_map is not None:
            item["instance_map"] = instance_map
            item["instance_pairs"] = instance_pairs
        if ewc_weight is not None:
            item["ewc_weight"] = ewc_weight
        if wprf_target is not None:
            item["wprf_support"] = wprf_target["support"]
            item["wprf_dense_components"] = wprf_target["dense_components"]
            item["wprf_pairs"] = wprf_target["pairs"]
        if valid_mask is not None:
            item["valid_mask"] = valid_mask
        return item

    def _load_wprf_target(
        self,
        entry: dict[str, Any],
        shape_hw: tuple[int, int],
    ) -> dict[str, Any] | None:
        if self.wprf_target_root is None:
            return None
        from .aquadataset_wprf import WPRF_GRID_STRIDE, WPRF_SCALES, load_wprf_target

        path = self.wprf_target_root / f"{entry['tile_id']}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing WPRF target: {path}")
        payload = load_wprf_target(path)
        expected = (
            self.image_size // WPRF_GRID_STRIDE,
            self.image_size // WPRF_GRID_STRIDE,
        )
        if payload["support"].shape != expected or payload["dense_components"].shape != expected:
            raise ValueError(f"WPRF grid shape mismatch in {path}")
        return {
            "support": torch.from_numpy(payload["support"]).unsqueeze(0).bool(),
            "dense_components": torch.from_numpy(
                payload["dense_components"]
            ).unsqueeze(0).long(),
            "pairs": {
                f"{kind}_{scale}": torch.from_numpy(payload[f"{kind}_{scale}"]).long()
                for scale in WPRF_SCALES
                for kind in ("pos", "neg")
            },
        }

    def _load_ewc_weight(
        self,
        entry: dict[str, Any],
        shape_hw: tuple[int, int],
    ) -> torch.Tensor | None:
        if self.ewc_weight_root is None:
            return None
        target_path = self.ewc_weight_root / f"{entry['tile_id']}.npz"
        if not target_path.is_file():
            raise FileNotFoundError(
                f"Missing EWC weight target for tile {entry['tile_id']}: {target_path}"
            )
        from .aquadataset_ewc import _load_ewc_target

        weight = _load_ewc_target(target_path)
        if weight.shape != shape_hw:
            raise ValueError(
                f"EWC weight shape {weight.shape} does not match image shape "
                f"{shape_hw}: {target_path}"
            )
        if not np.all(np.isfinite(weight)) or np.any(weight <= 0.0):
            raise ValueError(
                f"EWC weight must contain finite positive values: {target_path}"
            )
        return torch.from_numpy(weight).unsqueeze(0).contiguous()

    def _load_instance_target(
        self,
        entry: dict[str, Any],
        shape_hw: tuple[int, int],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.instance_target_root is None:
            return None, None
        tile_id = str(entry["tile_id"])
        if bool(entry.get("is_background", False)):
            label_map = np.zeros(shape_hw, dtype=np.int64)
            pairs = np.empty((0, 3), dtype=np.int64)
        else:
            target_path = self.instance_target_root / f"{tile_id}.npz"
            if not target_path.exists():
                raise FileNotFoundError(f"Missing PBCL instance target for positive tile {tile_id}: {target_path}")
            with np.load(target_path, allow_pickle=False) as target:
                if "instance_map" not in target or "instance_pairs" not in target:
                    raise ValueError(f"PBCL target must contain instance_map and instance_pairs: {target_path}")
                label_map = np.asarray(target["instance_map"], dtype=np.int64)
                pairs = np.asarray(target["instance_pairs"], dtype=np.int64)
            if label_map.shape != shape_hw:
                raise ValueError(
                    f"PBCL instance map shape {label_map.shape} does not match image shape {shape_hw}: {target_path}"
                )
            if np.any(label_map < 0):
                raise ValueError(f"PBCL instance map must contain non-negative ids: {target_path}")
            if pairs.ndim != 2 or pairs.shape[1] != 3:
                raise ValueError(f"PBCL instance pairs must have shape (N,3): {target_path}")
            if pairs.size:
                if np.any(pairs[:, :2] <= 0):
                    raise ValueError(f"PBCL pair instance ids must be positive: {target_path}")
                if np.any((pairs[:, 2] < 0) | (pairs[:, 2] > 2)):
                    raise ValueError(f"PBCL pair bucket codes must be 0, 1, or 2: {target_path}")
                present_ids = set(np.unique(label_map).tolist())
                referenced_ids = set(pairs[:, :2].reshape(-1).tolist())
                missing_ids = sorted(referenced_ids - present_ids)
                if missing_ids:
                    raise ValueError(f"PBCL pairs reference missing instance ids {missing_ids}: {target_path}")
            if self.split == "train" and self.instance_pair_policy != "all":
                from .geographic_holdout import filter_instance_pairs

                pairs = filter_instance_pairs(label_map, pairs, self.instance_pair_policy)
        return (
            torch.from_numpy(label_map).unsqueeze(0).contiguous(),
            torch.from_numpy(pairs).contiguous(),
        )

    @staticmethod
    def _transform_wprf_pairs(
        pairs: torch.Tensor,
        *,
        height: int,
        width: int,
        horizontal_flip: bool,
        vertical_flip: bool,
        turns: int,
    ) -> torch.Tensor:
        result = pairs.clone().long()
        if not result.numel():
            return result.reshape(-1, 4)
        if horizontal_flip:
            result[:, (1, 3)] = width - 1 - result[:, (1, 3)]
        if vertical_flip:
            result[:, (0, 2)] = height - 1 - result[:, (0, 2)]
        current_height, current_width = height, width
        for _ in range(int(turns) % 4):
            old_y = result[:, (0, 2)].clone()
            old_x = result[:, (1, 3)].clone()
            result[:, (0, 2)] = current_width - 1 - old_x
            result[:, (1, 3)] = old_y
            current_height, current_width = current_width, current_height
        return result

    def _augment(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        wprf_target: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
        horizontal_flip = random.random() < 0.5
        vertical_flip = random.random() < 0.25
        rotate = random.random() < 0.35
        turns = random.randint(0, 3) if rotate else 0
        if horizontal_flip:
            image = torch.flip(image, [2])
            mask = torch.flip(mask, [2])
        if vertical_flip:
            image = torch.flip(image, [1])
            mask = torch.flip(mask, [1])
        if turns:
            image = torch.rot90(image, turns, [1, 2])
            mask = torch.rot90(mask, turns, [1, 2])
        if wprf_target is not None:
            support = wprf_target["support"]
            dense = wprf_target["dense_components"]
            grid_height, grid_width = support.shape[-2:]
            if horizontal_flip:
                support = torch.flip(support, [2])
                dense = torch.flip(dense, [2])
            if vertical_flip:
                support = torch.flip(support, [1])
                dense = torch.flip(dense, [1])
            if turns:
                support = torch.rot90(support, turns, [1, 2])
                dense = torch.rot90(dense, turns, [1, 2])
            wprf_target = {
                "support": support.contiguous(),
                "dense_components": dense.contiguous(),
                "pairs": {
                    key: self._transform_wprf_pairs(
                        value,
                        height=grid_height,
                        width=grid_width,
                        horizontal_flip=horizontal_flip,
                        vertical_flip=vertical_flip,
                        turns=turns,
                    )
                    for key, value in wprf_target["pairs"].items()
                },
            }
        if random.random() < 0.35:
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.20)
            image = image * brightness
            mean_val = image.mean()
            image = (image - mean_val) * contrast + mean_val
        return image.contiguous(), mask.contiguous(), wprf_target

    def _resize(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if image.shape[-2:] == (self.image_size, self.image_size):
            return image, mask
        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(self.image_size, self.image_size), mode="nearest").squeeze(0)
        return image.contiguous(), mask.contiguous()

    def _pad(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = image.shape[-2:]
        if height > self.image_size or width > self.image_size:
            raise ValueError(
                f"Cannot pad shape {(height, width)} to {(self.image_size, self.image_size)}"
            )
        padding = (0, self.image_size - width, 0, self.image_size - height)
        return (
            F.pad(image, padding, mode="constant", value=0).contiguous(),
            F.pad(mask, padding, mode="constant", value=0).contiguous(),
        )


def make_balanced_sampler(
    dataset: AquaV3ManifestDataset,
    *,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    negatives = dataset.is_negative
    neg_count = sum(1 for value in negatives if value)
    pos_count = len(negatives) - neg_count
    weights = [0.5 / max(1, neg_count) if is_neg else 0.5 / max(1, pos_count) for is_neg in negatives]
    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def aquav3_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Default-collate fixed fields while preserving variable-length PBCL pairs."""
    if not batch:
        raise ValueError("aquav3_collate requires a non-empty batch")
    has_instance = ["instance_map" in item or "instance_pairs" in item for item in batch]
    if any(has_instance) and not all(
        "instance_map" in item and "instance_pairs" in item for item in batch
    ):
        raise ValueError("instance_map and instance_pairs must be present for every item in a batch")
    has_wprf = ["wprf_pairs" in item for item in batch]
    if any(has_wprf) and not all(has_wprf):
        raise ValueError("wprf_pairs must be present for every item in a WPRF batch")
    variable_keys = {"instance_pairs", "wprf_pairs"}
    aliases = {
        alias: key
        for alias, key in (("image", "img"), ("mask", "gt_semantic_seg"))
        if all(alias in item and key in item and item[alias] is item[key] for item in batch)
    }
    fixed_items = [
        {key: value for key, value in item.items() if key not in variable_keys and key not in aliases}
        for item in batch
    ]
    collated = default_collate(fixed_items)
    for alias, key in aliases.items():
        collated[alias] = collated[key]
    if any(has_instance):
        collated["instance_pairs"] = [item["instance_pairs"] for item in batch]
    if any(has_wprf):
        collated["wprf_pairs"] = [item["wprf_pairs"] for item in batch]
    return collated
