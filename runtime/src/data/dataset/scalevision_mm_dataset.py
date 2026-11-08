"""ScaleVision-MM multimodal detection dataset."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torchvision
from faster_coco_eval.core import mask as coco_mask
from faster_coco_eval.utils.pytorch import FasterCocoDetection
from PIL import Image

from ...core import register
from .._misc import convert_to_tv_tensor
from ._dataset import DetDataset
from .coco_dataset import load_coco_annotation_utf8

torchvision.disable_beta_transforms_warning()

__all__ = ["ScaleVisionMMDetection"]


DEFAULT_WEIGHT_FEATURES = (
    "delta_weight_g",
    "stable_before_g",
    "stable_after_g",
    "stable_std_g",
    "stable_range_g",
    "stable_drift_g",
    "stable_samples",
)


def _as_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class _WeightNormalizer:
    def __init__(self, annotations: Iterable[dict], feature_names: Sequence[str], eps: float = 1e-6) -> None:
        values: Dict[str, List[float]] = {name: [] for name in feature_names}
        for ann in annotations:
            features = ann.get("weight_features") or {}
            for name in feature_names:
                values[name].append(_as_float(features.get(name)))

        self.mean: Dict[str, float] = {}
        self.std: Dict[str, float] = {}
        for name, vals in values.items():
            if not vals:
                self.mean[name] = 0.0
                self.std[name] = 1.0
                continue
            tensor = torch.tensor(vals, dtype=torch.float32)
            self.mean[name] = float(tensor.mean().item())
            std = float(tensor.std(unbiased=False).item())
            self.std[name] = std if std > eps else 1.0

    def encode(self, features: dict, feature_names: Sequence[str]) -> List[float]:
        out = []
        for name in feature_names:
            value = _as_float((features or {}).get(name))
            out.append((value - self.mean[name]) / self.std[name])
        return out


class ConvertScaleVisionMMAnnotations:
    def __init__(
        self,
        normalizer: _WeightNormalizer,
        feature_names: Sequence[str],
        return_masks: bool = False,
        append_normalized_placement_order: bool = True,
    ) -> None:
        self.normalizer = normalizer
        self.feature_names = tuple(feature_names)
        self.return_masks = bool(return_masks)
        self.append_normalized_placement_order = bool(append_normalized_placement_order)

    def __call__(self, image: Image.Image, target, **kwargs):
        width, height = image.size
        image_id = torch.tensor([target["image_id"]])
        anno = [obj for obj in target["annotations"] if obj.get("iscrowd", 0) == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)

        category2label = kwargs.get("category2label", None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]
        labels = torch.tensor(labels, dtype=torch.int64)

        masks = None
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = _convert_coco_poly_to_mask(segmentations, height, width)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        keep_flags = keep.tolist()
        kept_anno = [obj for obj, flag in zip(anno, keep_flags) if flag]
        if self.return_masks and masks is not None:
            masks = masks[keep]

        ordered = sorted(
            kept_anno,
            key=lambda obj: (int(obj.get("placement_order", 0)), int(obj.get("id", 0))),
        )
        max_pos = max(1, len(ordered) - 1)
        event_index_by_ann_id = {int(obj["id"]): idx for idx, obj in enumerate(ordered)}

        weight_tokens = []
        weight_tokens_raw = []
        event_order = []
        for idx, obj in enumerate(ordered):
            features = obj.get("weight_features") or {}
            normalized_position = 0.0 if len(ordered) <= 1 else idx / max_pos
            token = self.normalizer.encode(features, self.feature_names)
            if self.append_normalized_placement_order:
                token = token + [normalized_position]
            weight_tokens.append(token)
            event_order.append(int(obj.get("placement_order", idx + 1)))
            weight_tokens_raw.append([
                _as_float(features.get(name))
                for name in self.feature_names
            ])

        event_indices = [
            event_index_by_ann_id.get(int(obj["id"]), -1)
            for obj in kept_anno
        ]

        out = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": torch.tensor([obj.get("area", 0.0) for obj in kept_anno], dtype=torch.float32),
            "iscrowd": torch.tensor([obj.get("iscrowd", 0) for obj in kept_anno], dtype=torch.int64),
            "orig_size": torch.as_tensor([int(width), int(height)]),
            "event_indices": torch.tensor(event_indices, dtype=torch.int64),
            "weight_tokens": torch.tensor(weight_tokens, dtype=torch.float32).reshape(
                -1,
                len(self.feature_names) + int(self.append_normalized_placement_order),
            ),
            "weight_tokens_raw": torch.tensor(weight_tokens_raw, dtype=torch.float32).reshape(
                -1,
                len(self.feature_names),
            ),
            "weight_mask": torch.ones((len(weight_tokens),), dtype=torch.bool),
            "event_order": torch.tensor(event_order, dtype=torch.int64),
        }
        if self.return_masks and masks is not None:
            out["masks"] = masks

        return image, out


@register()
class ScaleVisionMMDetection(FasterCocoDetection, DetDataset):
    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks=False,
        remap_mscoco_category=False,
        weight_stats_ann_file=None,
        scene_split_file=None,
        weight_stats_scene_split_file=None,
        weight_feature_names=None,
        append_normalized_placement_order=True,
        shuffle_weight_tokens=False,
        shuffle_weight_tokens_seed=None,
    ):
        super(FasterCocoDetection, self).__init__(img_folder, load_coco_annotation_utf8(ann_file))
        if scene_split_file:
            self._filter_coco_by_scene_file(Path(scene_split_file))
        self._transforms = transforms
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self.weight_feature_names = tuple(weight_feature_names or DEFAULT_WEIGHT_FEATURES)
        self.append_normalized_placement_order = bool(append_normalized_placement_order)
        self.shuffle_weight_tokens = bool(shuffle_weight_tokens)
        self.shuffle_weight_tokens_seed = 0 if shuffle_weight_tokens_seed is None else int(shuffle_weight_tokens_seed)

        stats_path = Path(weight_stats_ann_file or ann_file)
        if Path(ann_file) == stats_path and not weight_stats_scene_split_file:
            stats_dataset = self.coco.dataset
        else:
            stats_dataset = self._load_stats_dataset(stats_path)
        if weight_stats_scene_split_file:
            stats_dataset = self._filter_dataset_by_scene_file(stats_dataset, Path(weight_stats_scene_split_file))
        self.weight_normalizer = _WeightNormalizer(stats_dataset.get("annotations", []), self.weight_feature_names)
        self.prepare = ConvertScaleVisionMMAnnotations(
            self.weight_normalizer,
            self.weight_feature_names,
            return_masks=return_masks,
            append_normalized_placement_order=self.append_normalized_placement_order,
        )
        self._annotations_by_image_id = self._build_annotations_by_image_id()
        self._shuffle_image_id_by_image_id = (
            self._build_shuffle_image_id_map(self.shuffle_weight_tokens_seed)
            if self.shuffle_weight_tokens else {}
        )

    @staticmethod
    def _load_stats_dataset(path: Path) -> dict:
        import json

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _read_scene_ids(path: Path) -> set[str]:
        return {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    @classmethod
    def _filter_dataset_by_scene_file(cls, dataset: dict, path: Path) -> dict:
        scene_ids = cls._read_scene_ids(path)
        images = [
            img for img in dataset.get("images", [])
            if str(img.get("scene_id", Path(str(img.get("file_name", ""))).stem)) in scene_ids
        ]
        image_ids = {int(img["id"]) for img in images}
        annotations = [
            ann for ann in dataset.get("annotations", [])
            if int(ann.get("image_id", -1)) in image_ids
        ]
        filtered = dict(dataset)
        filtered["images"] = images
        filtered["annotations"] = annotations
        return filtered

    def _filter_coco_by_scene_file(self, path: Path) -> None:
        filtered = self._filter_dataset_by_scene_file(self.coco.dataset, path)
        self.coco.dataset["images"] = filtered["images"]
        self.coco.dataset["annotations"] = filtered["annotations"]
        if hasattr(self.coco, "createIndex"):
            self.coco.createIndex()
        self.ids = [int(img["id"]) for img in self.coco.dataset.get("images", [])]


    def _build_annotations_by_image_id(self) -> Dict[int, List[dict]]:
        grouped: Dict[int, List[dict]] = {}
        for ann in self.coco.dataset.get("annotations", []):
            grouped.setdefault(int(ann.get("image_id", -1)), []).append(ann)
        return grouped

    def _build_shuffle_image_id_map(self, seed: int) -> Dict[int, int]:
        by_count: Dict[int, List[int]] = {}
        for image_id in self.ids:
            count = len(self._annotations_by_image_id.get(int(image_id), []))
            by_count.setdefault(count, []).append(int(image_id))

        rng = random.Random(int(seed))
        mapping: Dict[int, int] = {}
        for image_ids in by_count.values():
            shuffled = list(image_ids)
            if len(shuffled) > 1:
                # Sattolo's algorithm creates one cycle, so every non-singleton
                # scene is guaranteed to receive a different scene's events.
                for index in range(len(shuffled) - 1, 0, -1):
                    swap_index = rng.randrange(index)
                    shuffled[index], shuffled[swap_index] = (
                        shuffled[swap_index],
                        shuffled[index],
                    )
            for src, dst in zip(image_ids, shuffled):
                mapping[int(src)] = int(dst)
        return mapping

    def _encode_weight_tokens_from_annotations(self, annotations: Sequence[dict]) -> torch.Tensor:
        ordered = sorted(
            [ann for ann in annotations if ann.get("iscrowd", 0) == 0],
            key=lambda obj: (int(obj.get("placement_order", 0)), int(obj.get("id", 0))),
        )
        max_pos = max(1, len(ordered) - 1)
        tokens = []
        for idx, obj in enumerate(ordered):
            features = obj.get("weight_features") or {}
            normalized_position = 0.0 if len(ordered) <= 1 else idx / max_pos
            token = self.weight_normalizer.encode(features, self.weight_feature_names)
            if self.append_normalized_placement_order:
                token = token + [normalized_position]
            tokens.append(token)
        return torch.tensor(tokens, dtype=torch.float32).reshape(
            -1,
            len(self.weight_feature_names) + int(self.append_normalized_placement_order),
        )

    def _encode_raw_weight_tokens_from_annotations(self, annotations: Sequence[dict]) -> torch.Tensor:
        ordered = sorted(
            [ann for ann in annotations if ann.get("iscrowd", 0) == 0],
            key=lambda obj: (int(obj.get("placement_order", 0)), int(obj.get("id", 0))),
        )
        tokens = [
            [
                _as_float((obj.get("weight_features") or {}).get(name))
                for name in self.weight_feature_names
            ]
            for obj in ordered
        ]
        return torch.tensor(tokens, dtype=torch.float32).reshape(
            -1,
            len(self.weight_feature_names),
        )

    @staticmethod
    def _encode_event_order_from_annotations(annotations: Sequence[dict]) -> torch.Tensor:
        ordered = sorted(
            [ann for ann in annotations if ann.get("iscrowd", 0) == 0],
            key=lambda obj: (int(obj.get("placement_order", 0)), int(obj.get("id", 0))),
        )
        return torch.tensor(
            [int(obj.get("placement_order", index + 1)) for index, obj in enumerate(ordered)],
            dtype=torch.int64,
        )

    def _apply_weight_token_shuffle(self, target: dict, image_id: int) -> dict:
        if not self.shuffle_weight_tokens:
            return target
        shuffled_image_id = self._shuffle_image_id_by_image_id.get(int(image_id), int(image_id))
        shuffled_annotations = self._annotations_by_image_id.get(int(shuffled_image_id), [])
        weight_tokens = self._encode_weight_tokens_from_annotations(shuffled_annotations)
        weight_tokens_raw = self._encode_raw_weight_tokens_from_annotations(shuffled_annotations)
        target["weight_tokens"] = weight_tokens
        target["weight_tokens_raw"] = weight_tokens_raw
        target["weight_mask"] = torch.ones((weight_tokens.shape[0],), dtype=torch.bool)
        target["event_order"] = self._encode_event_order_from_annotations(shuffled_annotations)
        target["shuffled_weight_image_id"] = torch.tensor([int(shuffled_image_id)])
        return target

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        image, target = super(FasterCocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}

        if self.remap_mscoco_category:
            from .coco_dataset import mscoco_category2label

            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        target["idx"] = torch.tensor([idx])
        image_info = self.coco.loadImgs([int(image_id)])[0]
        total_weight = image_info.get("scene_total_weight_g")
        empty_baseline = image_info.get("empty_baseline_g")
        total_value = _as_float(total_weight, float("nan"))
        baseline_value = _as_float(empty_baseline, float("nan"))
        total_available = math.isfinite(total_value) and math.isfinite(baseline_value)
        target["scene_total_weight_g"] = torch.tensor(total_value, dtype=torch.float32)
        target["empty_baseline_g"] = torch.tensor(baseline_value, dtype=torch.float32)
        target["scene_total_available"] = torch.tensor(total_available, dtype=torch.bool)
        target["scene_total_independent"] = torch.tensor(
            bool(image_info.get("scene_total_independent", False)), dtype=torch.bool
        )
        target = self._apply_weight_token_shuffle(target, image_id)

        if "boxes" in target:
            target["boxes"] = convert_to_tv_tensor(target["boxes"], key="boxes", spatial_size=image.size[::-1])

        if "masks" in target:
            target["masks"] = convert_to_tv_tensor(target["masks"], key="masks")

        return image, target

    def extra_repr(self) -> str:
        s = f" img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n"
        s += f" return_masks: {self.return_masks}\n"
        s += f" weight_feature_names: {list(self.weight_feature_names)}\n"
        s += f" append_normalized_placement_order: {self.append_normalized_placement_order}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        return s

    @property
    def categories(self):
        return self.coco.dataset["categories"]

    @property
    def category2name(self):
        return {cat["id"]: cat["name"] for cat in self.categories}

    @property
    def category2label(self):
        return {cat["id"]: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self):
        return {i: cat["id"] for i, cat in enumerate(self.categories)}
