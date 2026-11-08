"""Functions preserved from the original experiment implementation."""
import numpy as np
import torch
from geometry_matching import iou_matrix, maximum_matching, score_order_matching

def attach_supervision(arrays, index, record, prepared, images, annotations):
    """Attach training-only ownership labels after candidate selection is frozen."""
    image_id = int(arrays["image_ids"][index])
    image = images[image_id]
    ground = np.asarray([item["bbox"] for item in annotations[image_id]], dtype=float)
    ground[:, 2:] += ground[:, :2]
    center = arrays["boxes"][index]
    query_xyxy = np.concatenate((center[:, :2] - center[:, 2:] / 2, center[:, :2] + center[:, 2:] / 2), -1)
    query_xyxy *= np.array([image["width"], image["height"]] * 2)
    matching = maximum_matching(iou_matrix(query_xyxy, ground), prepared["candidates"].numpy())
    labels = torch.full((len(prepared["candidates"]),), -1, dtype=torch.long)
    owners = torch.full((len(prepared["valid"]),), -1, dtype=torch.long)
    for query, ground_index in matching.items():
        local = int(np.flatnonzero(prepared["candidates"].numpy() == query)[0])
        labels[local] = int(annotations[image_id][ground_index]["category_id"])
        for event, original in enumerate(prepared["valid"].tolist()):
            if int(record["event_order"][original]) == int(annotations[image_id][ground_index]["placement_order"]):
                owners[event] = local
    score_matching = score_order_matching(iou_matrix(query_xyxy, ground), arrays["logits"][index])
    return {
        "labels": labels,
        "owners": owners,
        "image_id": image_id,
        "source": image.get("source_scene_id", image["scene_id"]),
        "audit_pairs": [(query, int(annotations[image_id][ground_index]["category_id"])) for query, ground_index in score_matching.items()],
        "source_weight": 1.0,
    }

