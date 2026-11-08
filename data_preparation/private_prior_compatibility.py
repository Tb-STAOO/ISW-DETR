#!/usr/bin/env python3
"""Add explicit compatibility records to compact M6 train-only statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stats",
        nargs="?",
        type=Path,
        default=Path(
            r"D:\MY_DATA\02-project\sci_project\RT_DETR\Official\RT-DETR"
            r"\rtdetrv2_pytorch\configs\dataset"
            r"\scalevision_mm_v3_physical_prior_stats.json"
        ),
    )
    args = parser.parse_args()

    data = json.loads(args.stats.read_text(encoding="utf-8"))
    data["feature_stats"] = {
        name: {"mean": float(mean), "std": float(std)}
        for name, mean, std in zip(
            data["feature_names"], data["feature_mean"], data["feature_std"]
        )
    }
    data["class_delta_gaussians"] = [
        {
            "category_id": int(category_id),
            "count": int(count),
            "mean_g": float(mean_g),
            "std_g": float(std_g),
        }
        for category_id, count, mean_g, std_g in zip(
            data["category_ids"],
            data["class_count"],
            data["class_mean_raw_g"],
            data["class_std_raw_g"],
        )
    ]
    args.stats.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.stats)


if __name__ == "__main__":
    main()
