#!/usr/bin/env python3
"""Build train-only physical-prior statistics for ScaleVision-MM-V3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "delta_weight_g",
    "stable_before_g",
    "stable_after_g",
    "stable_std_g",
    "stable_range_g",
    "stable_drift_g",
    "stable_samples",
)


def finite_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-ann",
        type=Path,
        default=Path(
            r"D:\MY_DATA\09-Dataset\ScaleVision-MM-V3\annotations\splits"
            r"\instances_multimodal_train.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"D:\MY_DATA\02-project\sci_project\RT_DETR\Official\RT-DETR"
            r"\rtdetrv2_pytorch\configs\dataset\scalevision_mm_v3_physical_prior_stats.json"
        ),
    )
    parser.add_argument("--sigma-floor-g", type=float, default=1.0)
    args = parser.parse_args()

    train_ann = args.train_ann.resolve()
    data = json.loads(train_ann.read_text(encoding="utf-8"))
    category_ids = sorted(int(category["id"]) for category in data["categories"])
    if category_ids != list(range(len(category_ids))):
        raise ValueError("M6 currently requires contiguous category ids starting at zero")

    feature_values = {name: [] for name in FEATURE_NAMES}
    delta_by_category = defaultdict(list)
    for annotation in data.get("annotations", []):
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        features = annotation.get("weight_features") or {}
        for name in FEATURE_NAMES:
            feature_values[name].append(finite_float(features.get(name)))
        delta_by_category[int(annotation["category_id"])].append(
            finite_float(features.get("delta_weight_g"))
        )

    feature_mean = []
    feature_std = []
    for name in FEATURE_NAMES:
        values = np.asarray(feature_values[name], dtype=np.float64)
        feature_mean.append(float(values.mean()))
        std = float(values.std(ddof=0))
        feature_std.append(std if std > 1e-6 else 1.0)

    delta_mean = feature_mean[0]
    delta_std = feature_std[0]
    class_mean_raw = []
    class_std_raw = []
    class_count = []
    for category_id in category_ids:
        values = np.asarray(delta_by_category[category_id], dtype=np.float64)
        if values.size == 0:
            raise ValueError(f"No delta_weight_g values for category {category_id}")
        class_mean_raw.append(float(values.mean()))
        class_std_raw.append(max(float(values.std(ddof=0)), args.sigma_floor_g))
        class_count.append(int(values.size))

    class_mean_normalized = [
        (value - delta_mean) / delta_std for value in class_mean_raw
    ]
    class_std_normalized = [value / delta_std for value in class_std_raw]

    result = {
        "schema_version": 1,
        "dataset": "ScaleVision-MM-V3",
        "source": {
            "train_annotation": str(train_ann),
            "train_annotation_sha256": sha256(train_ann),
            "split_policy": "train-only",
        },
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "delta_feature_index": 0,
        "quality_feature_indices": [3, 4, 5, 6],
        "category_ids": category_ids,
        "class_count": class_count,
        "class_mean_raw_g": class_mean_raw,
        "class_std_raw_g": class_std_raw,
        "class_mean_normalized": class_mean_normalized,
        "class_std_normalized": class_std_normalized,
        "sigma_floor_g": float(args.sigma_floor_g),
        "scene_aggregation": "event softmax then classwise max",
        "evidence": "log(scene_probability / uniform_class_probability)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    print(result["source"]["train_annotation_sha256"])


if __name__ == "__main__":
    main()
