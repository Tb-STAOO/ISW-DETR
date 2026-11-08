"""Label-free frozen fusion preparation; attach training supervision afterwards."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


TRANSACTION_KEYS = (
    "weight_tokens_raw", "weight_mask", "event_order", "scene_total_weight_g",
    "empty_baseline_g", "scene_total_available", "scene_total_independent",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def modal_from_record(record, device="cpu"):
    result = {}
    for key in TRANSACTION_KEYS:
        if record.get(key) is None:
            continue
        dtype = torch.bool if key in ("weight_mask", "scene_total_available", "scene_total_independent") else torch.float32
        result[key] = torch.tensor(record[key], dtype=dtype, device=device).unsqueeze(0)
    return result


def annotation_maps(data):
    images = {int(image["id"]): image for image in data["images"]}
    annotations = defaultdict(list)
    for annotation in data["annotations"]:
        if not annotation.get("iscrowd", 0):
            annotations[int(annotation["image_id"])].append(annotation)
    for values in annotations.values():
        values.sort(key=lambda annotation: (int(annotation["placement_order"]), int(annotation["id"])))
    return images, annotations


def box_iou_matrix(boxes, annotations, image):
    ground = np.asarray([annotation["bbox"] for annotation in annotations], dtype=np.float64)
    ground[:, 2:] += ground[:, :2]
    boxes = np.asarray(boxes, dtype=np.float64)
    predicted = np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2,
                                boxes[:, :2] + boxes[:, 2:] / 2), axis=-1)
    predicted *= np.asarray([image["width"], image["height"]] * 2)
    low = np.maximum(predicted[:, None, :2], ground[None, :, :2])
    high = np.minimum(predicted[:, None, 2:], ground[None, :, 2:])
    overlap = np.maximum(high - low, 0).prod(-1)
    p_area = np.maximum(predicted[:, 2:] - predicted[:, :2], 0).prod(-1)
    g_area = np.maximum(ground[:, 2:] - ground[:, :2], 0).prod(-1)
    return overlap / np.maximum(p_area[:, None] + g_area[None] - overlap, 1e-12)


def maximum_matching(ious, query_indices):
    if not len(query_indices) or not ious.shape[1]:
        return {}
    current = ious[query_indices]
    valid = current >= .5
    rows, columns = linear_sum_assignment(-(valid * (min(current.shape) + 1) + current * valid))
    return {int(query_indices[row]): int(column) for row, column in zip(rows, columns) if valid[row, column]}


@torch.no_grad()
def prepare_forward(logits, boxes, record, module):
    """Only predicted query/box and observed physical inputs enter this function."""
    modal = modal_from_record(record)
    output = module(logits[None], boxes[None], modal)
    if not bool(output["integrity_gate"][0]):
        assert torch.equal(output["pred_logits"][0], logits)
        return None, output
    valid = modal["weight_mask"][0].nonzero().flatten()
    if len(valid) > 8:
        raise ValueError("More than eight valid records; no silent truncation.")
    candidates = output["candidate_mask"][0].nonzero().flatten()
    if not len(candidates):
        return None, output
    logp, probability, scene, _ = module.mass_evidence(
        modal["weight_tokens_raw"], modal["weight_mask"], modal)
    residual = (logp[0, valid] - probability[0].clamp_min(1e-8).log()).clamp(-6, 0)
    utility = torch.logsumexp((logits[candidates] + 2 * scene[0])[:, None] + residual[None], dim=-1)
    hard = logits.new_zeros((len(candidates), len(valid)))
    query_residual = logits.new_zeros((len(candidates), logits.shape[-1]))
    row_margin = logits.new_zeros(len(candidates))
    column_margin = logits.new_zeros(len(candidates))
    event_local = {int(original): local for local, original in enumerate(valid.tolist())}
    for local, query in enumerate(candidates.tolist()):
        original_event = int(output["assigned_event"][0, query])
        if original_event < 0:
            continue
        event = event_local[original_event]
        hard[local, event] = 1
        query_residual[local] = residual[event]
        value = utility[local, event]
        if len(valid) > 1:
            alternatives = utility[local].clone()
            alternatives[event] = -torch.inf
            row_margin[local] = value - alternatives.max()
        if len(candidates) > 1:
            alternatives = utility[:, event].clone()
            alternatives[local] = -torch.inf
            column_margin[local] = value - alternatives.max()
    prepared = {
        "z": logits[candidates], "boxes": boxes[candidates],
        "base": output["pred_logits"][0, candidates], "scene": scene[0],
        "logp": logp[0, valid], "raw": modal["weight_tokens_raw"][0, valid],
        "residual": residual, "hard_p": hard, "query_record_residual": query_residual,
        "row_margin": row_margin, "col_margin": column_margin,
        "candidates": candidates, "valid": valid,
    }
    assert all(torch.isfinite(value).all() for value in prepared.values())
    torch.testing.assert_close(prepared["base"], prepared["z"] + 2 * prepared["scene"] + .2 * query_residual,
                               rtol=0, atol=2e-6)
    return prepared, output


def attach_supervision(arrays, index, record, prepared, image, annotations, view_name):
    """Called after the image forward and label-free fusion preparation."""
    ious = box_iou_matrix(arrays["boxes"][index], annotations, image)
    candidates = prepared["candidates"].numpy()
    matching = maximum_matching(ious, candidates)
    labels = torch.full((len(candidates),), -1, dtype=torch.long)
    annotation_ids = torch.full_like(labels, -1)
    match_iou = torch.zeros(len(candidates))
    owners = torch.full((len(prepared["valid"]),), -1, dtype=torch.long)
    reverse = {int(query): local for local, query in enumerate(candidates)}
    for query, ground in matching.items():
        local = reverse[query]
        annotation = annotations[ground]
        labels[local] = int(annotation["category_id"])
        annotation_ids[local] = int(annotation["id"])
        match_iou[local] = float(ious[query, ground])
        for event, original in enumerate(prepared["valid"].tolist()):
            if int(record["event_order"][original]) == int(annotation["placement_order"]):
                owners[event] = local
    max_gt_iou = torch.tensor(ious[candidates].max(axis=1), dtype=torch.float32)
    unmatched = labels < 0
    return {
        "labels": labels, "owners": owners, "gt_annotation_ids": annotation_ids,
        "match_iou": match_iou, "foreground_mask": labels >= 0, "background_mask": labels < 0,
        "unmatched_mask": unmatched, "max_gt_iou": max_gt_iou,
        "true_background_mask": unmatched & (max_gt_iou < .1),
        "duplicate_or_overlapping_mask": unmatched & (max_gt_iou >= .5),
        "source": image["scene_id"], "source_weight": 1.,
        "image_id": int(arrays["image_ids"][index]), "source_image_id": int(image["id"]),
        "view_id": int(arrays["view_ids"][index]), "view_name": view_name,
        "ground_objects": len(annotations),
        "locatable_objects": len(maximum_matching(ious, np.arange(len(ious)))),
    }


def summarize(prepared, audits):
    result = {}
    for item, audit in zip(prepared, audits):
        if item is None:
            continue
        group = result.setdefault(audit["view_name"], {
            "scenes": 0, "ground_objects": 0, "matched_foreground": 0, "locatable_objects": 0,
            "visual_wrong": 0, "fusion_wrong": 0, "corrected": 0, "harmed": 0,
            "retained_correct": 0, "persistent_wrong": 0, "background_queries": 0,
            "unmatched_candidate_queries": 0, "unmatched_iou_lt_0.1": 0,
            "unmatched_iou_ge_0.5": 0,
            "background_score_increase_ge_0.1": 0,
            "background_crossed_score_0.05": 0,
        })
        label = audit["labels"]
        foreground = label >= 0
        background = ~foreground
        before = item["z"].argmax(-1) == label
        after = item["base"].argmax(-1) == label
        group["scenes"] += 1
        group["ground_objects"] += audit["ground_objects"]
        group["locatable_objects"] += audit["locatable_objects"]
        group["matched_foreground"] += int(foreground.sum())
        for key, mask in (
            ("visual_wrong", foreground & ~before), ("fusion_wrong", foreground & ~after),
            ("corrected", foreground & ~before & after), ("harmed", foreground & before & ~after),
            ("retained_correct", foreground & before & after),
            ("persistent_wrong", foreground & ~before & ~after),
        ):
            group[key] += int(mask.sum())
        old = item["z"].sigmoid().amax(-1)
        new = item["base"].sigmoid().amax(-1)
        group["background_queries"] += int(background.sum())
        group["unmatched_candidate_queries"] += int(background.sum())
        group["unmatched_iou_lt_0.1"] += int(audit["true_background_mask"].sum())
        group["unmatched_iou_ge_0.5"] += int(audit["duplicate_or_overlapping_mask"].sum())
        group["background_score_increase_ge_0.1"] += int((background & (new - old >= .1)).sum())
        group["background_crossed_score_0.05"] += int((background & (old < .05) & (new >= .05)).sum())
    return result
