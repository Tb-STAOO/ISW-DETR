"""Train a scalar benefit router from frozen, train-source query caches.

Only training annotations build supervision/sampling. Inference consumes frozen
query logits and physical fields. Model selection uses clean validation COCO AP;
this program deliberately has no internal/holdout evaluation entry point.
"""
from __future__ import annotations

import argparse
import os
import contextlib
import hashlib
import io
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("ISW_REPRO_WORKSPACE", str(Path.cwd() / "reproduction_workspace")))
NATIVE = HERE.parent / "inference/runtime"
DEFAULT_OUT = ROOT / "_active_runs/IMAGE_BENEFIT_ROUTER_20261013_R1"
OLD_PREPARED = ROOT / "_active_runs/CARA_CVSR_20260908_R2/prepared"
PACKAGE = Path(os.environ.get("ISW_PRIVATE_PROTOCOL", str(ROOT / "protocol")))
MODES = ("clean_only", "logit_noise", "image_uniform", "image_mass_hard",
         "image_mass_hard_no_mass_input")

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tensor(value, dtype=None):
    return torch.as_tensor(value, dtype=dtype).detach().cpu()


def load_cache(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("arrays", "prepared", "audits"):
        if key not in data:
            raise ValueError(f"Missing cache field {key}: {path}")
    n = len(data["prepared"])
    if len(data["audits"]) != n or len(data["arrays"]["logits"]) != n:
        raise ValueError("Prepared/audit/full-query rows are not aligned")
    return data


@torch.no_grad()
def predicted_assignment(z, scene, event_residual):
    k, e = len(z), len(event_residual)
    hard = z.new_zeros(k, e)
    utility = torch.logsumexp((z + 2.0 * scene)[:, None] + event_residual[None], -1)
    row_margin, col_margin = z.new_zeros(k), z.new_zeros(k)
    if k and e:
        rows, cols = linear_sum_assignment(-utility.numpy())
        hard[rows, cols] = 1.0
        for row, col in zip(rows.tolist(), cols.tolist()):
            others = torch.cat((utility[row, :col], utility[row, col + 1:]))
            row_margin[row] = utility[row, col] - others.max() if e > 1 else 0.0
            others = torch.cat((utility[:row, col], utility[row + 1:, col]))
            col_margin[row] = utility[row, col] - others.max() if k > 1 else 0.0
    return hard, row_margin, col_margin


def pack(data):
    """CPU padding; old event-residual caches and new query residuals are distinct."""
    items = data["prepared"]
    n = len(items)
    nonempty = [p for p in items if p is not None]
    if not nonempty:
        raise ValueError("No accepted transactions in cache")
    c = tensor(nonempty[0]["z"]).shape[-1]
    kmax = max(len(p["z"]) for p in nonempty)
    emax = max(len(p["logp"]) for p in nonempty)
    out = {"z": torch.full((n, kmax, c), -20.0),
           "base": torch.full((n, kmax, c), -20.0),
           "scene": torch.zeros(n, c), "logp": torch.full((n, emax, c), -100.0),
           "raw": torch.zeros(n, emax, 7),
           "event_residual": torch.zeros(n, emax, c),
           "query_record_residual": torch.zeros(n, kmax, c),
           "hard_p": torch.zeros(n, kmax, emax),
           "row_margin": torch.zeros(n, kmax), "col_margin": torch.zeros(n, kmax),
           "weight_mask": torch.zeros(n, emax, dtype=torch.bool),
           "candidate_mask": torch.zeros(n, kmax, dtype=torch.bool),
           "true_background_mask": torch.zeros(n, kmax, dtype=torch.bool),
           "background_supervision_available": torch.zeros(n, dtype=torch.bool),
           "transaction_gate": torch.zeros(n, dtype=torch.bool),
           "labels": torch.full((n, kmax), -1, dtype=torch.long)}
    for i, p in enumerate(items):
        if p is None:
            continue
        z, base = tensor(p["z"], torch.float32), tensor(p["base"], torch.float32)
        scene = tensor(p["scene"], torch.float32)
        lp = tensor(p["logp"], torch.float32)
        k, e = len(z), len(lp)
        candidates = tensor(p["candidates"], torch.long)
        if len(candidates.unique()) != k or (candidates < 0).any() or (candidates >= data["arrays"]["logits"].shape[1]).any():
            raise ValueError(f"Invalid/duplicate candidate query indexes at row {i}")
        residual = tensor(p["residual"], torch.float32)
        if residual.shape != (e, c):
            raise ValueError(f"Event residual must be E,C, got {residual.shape}")
        hard, row_margin, col_margin = predicted_assignment(z, scene, residual)
        if "hard_p" in p:
            supplied = tensor(p["hard_p"], torch.float32)
            if supplied.shape != (k, e) or not torch.equal(supplied, hard):
                raise ValueError(f"Prepared hard_p differs from predicted Hungarian at row {i}")
        qres = tensor(p.get("query_record_residual", (base - z - 2.0 * scene) / 0.2), torch.float32)
        expected = z + 2.0 * scene + 0.2 * (hard @ residual)
        if not torch.allclose(base, expected, atol=2e-5, rtol=0):
            raise ValueError(f"Prepared base is not frozen no-margin CVSR at row {i}")
        if qres.shape != (k, c) or not torch.allclose(qres, hard @ residual, atol=5e-5, rtol=0):
            raise ValueError(f"Query/event residual mapping inconsistent at row {i}")
        for key, value in (("z", z), ("base", base), ("query_record_residual", qres)):
            out[key][i, :k] = value
        out["scene"][i] = scene
        for key, value in (("logp", lp), ("raw", tensor(p["raw"], torch.float32)), ("event_residual", residual)):
            out[key][i, :e] = value
        out["hard_p"][i, :k, :e] = hard
        out["row_margin"][i, :k] = tensor(p.get("row_margin", row_margin), torch.float32)
        out["col_margin"][i, :k] = tensor(p.get("col_margin", col_margin), torch.float32)
        out["weight_mask"][i, :e] = True
        out["candidate_mask"][i, :k] = True
        out["transaction_gate"][i] = True
        audit = data["audits"][i]
        if audit is not None:
            labels = tensor(audit["labels"], torch.long)
            if labels.shape != (k,) or (labels >= c).any() or (labels < -1).any():
                raise ValueError(f"Invalid matched-query supervision at row {i}")
            out["labels"][i, :k] = labels
            if "foreground_mask" in audit and not torch.equal(tensor(audit["foreground_mask"], torch.bool), labels >= 0):
                raise ValueError("Foreground audit mask differs from matched labels")
            if "background_mask" in audit and not torch.equal(tensor(audit["background_mask"], torch.bool), labels < 0):
                raise ValueError("Background audit mask differs from unmatched query labels")
            if "true_background_mask" in audit:
                bg = tensor(audit["true_background_mask"], torch.bool)
                if bg.shape != (k,) or (bg & (labels >= 0)).any():
                    raise ValueError("True-background mask includes a matched foreground")
                if "max_gt_iou" not in audit:
                    raise ValueError("True-background audit requires maximum IoU against all GT")
                max_iou = tensor(audit["max_gt_iou"], torch.float32)
                if max_iou.shape != (k,) or not torch.equal(bg, (labels < 0) & (max_iou < .1)):
                    raise ValueError("True-background mask must be unmatched and maximum GT IoU <0.1")
                out["true_background_mask"][i, :k] = bg
                out["background_supervision_available"][i] = True
    return out


def model_class():
    sys.path.insert(0, str(HERE))
    from benefit_router import BenefitRouter
    return BenefitRouter


def take(packed, ids, device):
    ids = torch.as_tensor(ids, dtype=torch.long)
    return {key: value[ids].to(device) for key, value in packed.items()}


def forward(model, batch, constant_gate=None):
    return model(batch["z"], batch["base"], batch["candidate_mask"], batch["transaction_gate"],
                 scene_evidence=batch["scene"], event_log_probability=batch["logp"],
                 weight_tokens_raw=batch["raw"], weight_mask=batch["weight_mask"],
                 query_record_residual=batch["query_record_residual"], hard_p=batch["hard_p"],
                 row_margin=batch["row_margin"], col_margin=batch["col_margin"],
                 constant_gate=constant_gate, return_details=True)


@torch.no_grad()
def replay(model, data, packed, device, scale=1.0, constant_gate=None, batch_size=64):
    """Scatter only candidate updates into all frozen queries; never replay top-K."""
    logits = tensor(data["arrays"]["logits"], torch.float32).clone()
    gates = []
    for start in range(0, len(data["prepared"]), batch_size):
        ids = list(range(start, min(start + batch_size, len(data["prepared"]))))
        batch = take(packed, ids, device)
        out = forward(model, batch, constant_gate)
        if scale == 1.:
            corrected = out["pred_logits"]
        elif scale == 0.:
            corrected = batch["base"]
        else:
            corrected = batch["base"] + scale * (out["pred_logits"] - batch["base"])
        for j, i in enumerate(ids):
            p = data["prepared"][i]
            if p is not None:
                k = len(p["candidates"])
                logits[i, tensor(p["candidates"], torch.long)] = corrected[j, :k].cpu()
                gates.extend(out["gate"][j, :k].reshape(-1).cpu().tolist())
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite full-query prediction")
    return logits, gates


@torch.no_grad()
def coco_metrics(arrays, logits, annotation_path, device="cpu"):
    """Use the existing native postprocessor on full Q*C scores, then COCOeval."""
    sys.path.insert(0, str(NATIVE))
    from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    sizes = arrays.get("original_sizes", arrays.get("orig_sizes"))
    if sizes is None:
        raise ValueError("Full-query cache has no original sizes")
    post = RTDETRPostProcessor(num_classes=logits.shape[-1], num_top_queries=300,
                             remap_mscoco_category=False).to(device).eval()
    records = []
    for start in range(0, len(logits), 64):
        results = post({"pred_logits": logits[start:start + 64].to(device),
                        "pred_boxes": tensor(arrays["boxes"][start:start + 64], torch.float32).to(device)},
                       tensor(sizes[start:start + 64]).to(device))
        for image_id, result in zip(arrays["image_ids"][start:start + 64], results):
            boxes = result["boxes"].cpu().tolist()
            scores, labels = result["scores"].cpu().tolist(), result["labels"].cpu().tolist()
            for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
                records.append({"image_id": int(image_id), "category_id": int(label),
                                "bbox": [x1, y1, max(0., x2 - x1), max(0., y2 - y1)], "score": float(score)})
    annotation = read(annotation_path)
    wanted = set(int(i) for i in arrays["image_ids"])
    if len(wanted) != len(arrays["image_ids"]):
        raise ValueError("Validation image ids must be unique; do not evaluate perturbed views")
    dataset = {key: annotation.get(key, []) for key in ("licenses", "categories")}
    dataset["info"] = annotation.get("info", {})
    dataset["images"] = [image for image in annotation["images"] if int(image["id"]) in wanted]
    dataset["annotations"] = [ann for ann in annotation["annotations"] if int(ann["image_id"]) in wanted]
    if len(dataset["images"]) != len(wanted):
        raise ValueError("Validation cache image ids do not match annotation")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = dataset; gt.createIndex()
        dt = gt.loadRes(records)
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.imgIds = sorted(wanted)
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    stats = evaluator.stats
    return {"ap": float(stats[0]), "ap50": float(stats[1]), "ap75": float(stats[2]), "ar100": float(stats[8])}


def groups_for(data, packed):
    groups, clean = defaultdict(list), {}
    arrays = data["arrays"]
    for i, audit in enumerate(data["audits"]):
        if data["prepared"][i] is None or audit is None:
            continue
        source = str(audit.get("source", audit.get("source_image_id", arrays.get("source_image_ids", arrays["image_ids"])[i])))
        groups[source].append(i)
        view = audit.get("view_id", arrays.get("view_ids", [None] * len(data["prepared"]))[i])
        if str(view).lower() in ("0", "clean", "original", "orig"):
            clean[source] = i
    if not groups:
        raise ValueError("No train sources with matching supervision")
    missing = [s for s in groups if s not in clean]
    if missing:
        raise ValueError(f"Missing explicitly identified clean view for {len(missing)} sources")
    return dict(sorted(groups.items())), clean


@torch.no_grad()
def difficulty(packed):
    labels, z, base = packed["labels"], packed["z"], packed["base"]
    fg = packed["candidate_mask"] & (labels >= 0)
    rival = z.argmax(-1)
    physical = base - z
    truth = physical.gather(-1, labels.clamp_min(0)[..., None]).squeeze(-1)
    other = physical.gather(-1, rival[..., None]).squeeze(-1)
    # Predicted CVSR contrast is scene support plus predicted-record evidence.
    # It uses true/rival labels only to select TRAIN views, not router inputs.
    separability = ((truth - other).clamp(0., 6.) / 6.) * fg
    wrong = fg & (rival != labels)
    per_scene = (wrong * separability).sum(-1) / fg.sum(-1).clamp_min(1)
    weights = 1.0 + 8.0 * per_scene
    summary = {"foreground": int(fg.sum()), "visual_wrong": int(wrong.sum()),
               "mass_separable_wrong": int((wrong & (separability > 0)).sum()),
               "mean_sampling_weight": float(weights.mean()), "maximum_sampling_weight": float(weights.max())}
    return weights.numpy(), summary


def sample_epoch(groups, clean, weights, seed, epoch, mode):
    rng = np.random.default_rng(seed + 20261013 + 1009 * epoch)
    sources = list(groups)
    rng.shuffle(sources)
    selected, clean_ids = [], []
    for source in sources:
        ids = groups[source]
        if mode in ("clean_only", "logit_noise"):
            i = clean[source]
        else:
            probabilities = weights[ids] if mode.startswith("image_mass_hard") else np.ones(len(ids))
            probabilities = probabilities / probabilities.sum()
            i = int(rng.choice(ids, p=probabilities))
        selected.append(i); clean_ids.append(clean[source])
    return selected, clean_ids


@torch.no_grad()
def noisy_logits(batch, generator):
    batch = dict(batch)
    z = batch["z"].clone()
    fg = batch["candidate_mask"] & (batch["labels"] >= 0)
    chosen = fg & (torch.rand(fg.shape, device=z.device, generator=generator) < 0.65)
    ids = chosen.nonzero()
    if len(ids):
        labels = batch["labels"][chosen]
        rival = torch.randint(z.shape[-1] - 1, (len(ids),), device=z.device, generator=generator)
        rival = rival + (rival >= labels)
        a, b = ids[:, 0], ids[:, 1]
        correct = z[a, b, labels].clone()
        z[a, b, rival] = correct + torch.rand(len(ids), device=z.device, generator=generator) - 0.5
        z[a, b, labels] = correct - (0.5 + 3.5 * torch.rand(len(ids), device=z.device, generator=generator))
    batch["z"] = z
    batch["base"] = z + 2.0 * batch["scene"][:, None]
    batch["query_record_residual"] = torch.zeros_like(z)
    batch["hard_p"] = torch.zeros_like(batch["hard_p"])
    batch["row_margin"] = torch.zeros_like(batch["row_margin"])
    batch["col_margin"] = torch.zeros_like(batch["col_margin"])
    for i in range(len(z)):
        k, e = int(batch["candidate_mask"][i].sum()), int(batch["weight_mask"][i].sum())
        if k and e:
            hard, row, col = predicted_assignment(z[i, :k].cpu(), batch["scene"][i].cpu(), batch["event_residual"][i, :e].cpu())
            batch["hard_p"][i, :k, :e] = hard.to(z.device)
            batch["row_margin"][i, :k] = row.to(z.device)
            batch["col_margin"][i, :k] = col.to(z.device)
            qres = hard.to(z.device) @ batch["event_residual"][i, :e]
            batch["query_record_residual"][i, :k] = qres
            batch["base"][i, :k] += 0.2 * qres
    # Padded queries never enter loss/forward and retain their original baseline.
    return batch


def row_mean(values, mask):
    return (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def loss_for(model, batch, clean_batch):
    out = forward(model, batch)
    pred = out["pred_logits"]
    gate = out["gate"].reshape(batch["labels"].shape)
    labels = batch["labels"]
    fg = batch["candidate_mask"] & (labels >= 0)
    # One-to-one-unmatched candidates include duplicate objects. Only candidates
    # with max IoU against *every* GT below 0.1 receive an all-zero BCE target.
    bg = batch["true_background_mask"] & batch["candidate_mask"]
    supervised = fg | bg
    safe = labels.clamp_min(0)
    def ce(logits):
        return F.cross_entropy(logits.transpose(1, 2), safe, reduction="none")
    visual_ce, base_ce = ce(batch["z"]).detach(), ce(batch["base"]).detach()
    visual_correct = batch["z"].argmax(-1) == labels
    base_correct = batch["base"].argmax(-1) == labels
    corrected = fg & ~visual_correct & base_correct
    harmed = fg & visual_correct & ~base_correct
    both_correct = fg & visual_correct & base_correct
    target = torch.sigmoid((visual_ce - base_ce).clamp(-1., 1.) / 0.5)
    target = torch.where(both_correct, torch.full_like(target, 0.5), target)
    target = torch.where(corrected, torch.full_like(target, 0.9), target)
    target = torch.where(harmed, torch.full_like(target, 0.1), target)
    bg_visual = F.softplus(batch["z"]).mean(-1).detach()
    bg_base = F.softplus(batch["base"]).mean(-1).detach()
    target = torch.where(bg, torch.sigmoid((bg_visual - bg_base).clamp(-1., 1.) / 0.5), target)
    supervision_weight = torch.where(corrected | harmed, torch.ones_like(target), torch.full_like(target, 0.25))
    supervision_weight = torch.where(both_correct, torch.ones_like(target), supervision_weight)
    benefit = F.binary_cross_entropy((gate / model.gate_upper).clamp(1e-6, 1. - 1e-6), target, reduction="none")
    benefit_loss = row_mean(benefit * supervision_weight, supervised).mean()
    class_loss = row_mean(ce(pred), fg).mean()
    bg_loss = row_mean(F.softplus(pred).mean(-1), bg).mean()  # actual all-zero BCE
    clean_out = forward(model, clean_batch)
    clean_delta = (clean_out["pred_logits"] - clean_batch["base"]).square().mean(-1)
    clean_loss = row_mean(clean_delta, clean_batch["candidate_mask"]).mean()
    total = class_loss + 0.5 * benefit_loss + 0.1 * bg_loss + 0.02 * clean_loss
    result = {"loss": float(total.detach()), "class_ce": float(class_loss.detach()),
              "benefit_bce": float(benefit_loss.detach()), "background_bce": float(bg_loss.detach()),
              "clean_regularizer": float(clean_loss.detach()), "foreground": int(fg.sum()),
              "background": int(bg.sum()), "benefit_corrected": int(corrected.sum()),
              "benefit_harmed": int(harmed.sum()), "benefit_both_correct": int(both_correct.sum()),
              "benefit_both_wrong": int((fg & ~visual_correct & ~base_correct).sum()),
              "benefit_target_mean": float(target[supervised].mean().detach()) if bool(supervised.any()) else 0.,
              "gate_mean": float(gate[batch["candidate_mask"]].mean().detach())}
    return total, result


@torch.no_grad()
def transition_diagnostics(model, packed, device, scale=1.0, batch_size=64):
    totals = Counter()
    gates = defaultdict(list)
    for start in range(0, len(packed["z"]), batch_size):
        batch = take(packed, range(start, min(start + batch_size, len(packed["z"]))), device)
        out = forward(model, batch)
        pred = out["pred_logits"] if scale == 1. else batch["base"] if scale == 0. else batch["base"] + scale * (out["pred_logits"] - batch["base"])
        fg = batch["candidate_mask"] & (batch["labels"] >= 0)
        bg = batch["candidate_mask"] & ~fg
        true_bg = batch["true_background_mask"] & batch["candidate_mask"]
        labels = batch["labels"]
        visual, base, after = batch["z"].argmax(-1), batch["base"].argmax(-1), pred.argmax(-1)
        totals["matched_foreground"] += int(fg.sum())
        totals["visual_wrong"] += int((fg & (visual != labels)).sum())
        totals["base_wrong"] += int((fg & (base != labels)).sum())
        totals["router_wrong"] += int((fg & (after != labels)).sum())
        totals["corrected_vs_base"] += int((fg & (base != labels) & (after == labels)).sum())
        totals["harmed_vs_base"] += int((fg & (base == labels) & (after != labels)).sum())
        totals["background_queries"] += int(bg.sum())
        totals["true_background_queries"] += int(true_bg.sum())
        totals["background_visual_above_0p5"] += int((bg & (batch["z"].sigmoid().amax(-1) >= .5)).sum())
        totals["background_base_above_0p5"] += int((bg & (batch["base"].sigmoid().amax(-1) >= .5)).sum())
        totals["background_router_above_0p5"] += int((bg & (pred.sigmoid().amax(-1) >= .5)).sum())
        totals["true_background_router_above_0p5"] += int((true_bg & (pred.sigmoid().amax(-1) >= .5)).sum())
        gate = out["gate"].reshape(labels.shape)
        for name, mask in (("foreground", fg), ("background", bg),
                           ("base_corrected", fg & (visual != labels) & (base == labels)),
                           ("base_harmed", fg & (visual == labels) & (base != labels)),
                           ("both_correct", fg & (visual == labels) & (base == labels))):
            gates[name].extend(gate[mask].cpu().tolist())
    return {**dict(totals), "gate_by_group": {k: {"count": len(v), "mean": float(np.mean(v)) if v else None} for k, v in gates.items()}}


def protocol(args):
    return {"method": "train-source visual difficulty and benefit-supervised scalar CVSR router",
            "seed": args.seed, "mode": args.mode, "selection_split": "clean validation only",
            "epochs": args.epochs, "batch_size": args.batch_size, "optimizer": "AdamW",
            "learning_rate": args.lr, "weight_decay": 1e-4, "gradient_clip": 1.,
            "gate_upper": 2., "inference_scales": [0., .5, 1.],
            "scale_definition": "around no-margin base: effective_gate=1+scale*(learned_gate-1); scale0 is frozen base",
            "constant_gate_controls": [0., .5, 1., 1.5, 2.],
            "selection": "maximum unrounded validation AP; epoch -1 frozen base fallback; strict improvement",
            "loss": {"classification_ce": 1., "benefit_bce": .5, "true_background_bce": .1, "clean_delta_mse": .02},
            "benefit": "both correct -> g=1; corrected/harmed -> bounded 0.9/0.1 gate target; both wrong -> lower-weight bounded CE difference",
            "source_sampling": "each original source once per epoch; one view; all sources equal weight; matched budgets",
            "mass_hard_sampling": "1+8*mean(actual visual wrong * positive predicted-CVSR true/rival class contrast clipped 0..6 /6); train labels only",
            "background_target": "all-zero BCE only if unmatched and maximum IoU to every GT <0.1; other unmatched candidates excluded",
            "logit_noise": "same router; uniform rival score competition on clean matched foreground; predicted assignment recomputed",
            "inference_blocked_fields": ["labels", "owners", "source_image_id", "view_id", "annotation_ids", "GT boxes"],
            "frozen_detector": True, "test_or_holdout_access": False, "warmup": False,
            "limitations": "RGB perturbations are synthetic training views, not new physical acquisition; scalar routing and denoising have prior art"}


def lock_inputs(args, folder):
    files = [Path(__file__), HERE / "benefit_router.py", args.validation_cache, args.validation_annotations,
             NATIVE / "src/zoo/rtdetr/rtdetr_postprocessor.py"]
    if not args.baseline_only:
        files.append(args.cache)
    value = {"protocol": protocol(args), "files": {str(p.resolve()): sha(p) for p in files},
             "torch": torch.__version__, "device": args.device}
    path = folder / "INPUT_LOCK.json"
    if path.exists() and read(path) != value:
        raise RuntimeError("Input/protocol fingerprint changed: use another --out or mode/seed run folder")
    write(path, value)
    snapshot = folder / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for code in (Path(__file__), HERE / "benefit_router.py"):
        (snapshot / code.name).write_bytes(code.read_bytes())


def baseline_check(args, model, data, packed, folder):
    baseline, _ = replay(model, data, packed, args.device, constant_gate=1.)
    visual, _ = replay(model, data, packed, args.device, constant_gate=0.)
    expected = tensor(data["arrays"]["logits"], torch.float32)
    if not torch.equal(visual, expected):
        raise RuntimeError("g=0 replay does not exactly reproduce all frozen visual queries")
    for i, p in enumerate(data["prepared"]):
        if p is None:
            if not torch.equal(baseline[i], expected[i]):
                raise RuntimeError("Rejected transaction changed")
        else:
            ids = tensor(p["candidates"], torch.long)
            if not torch.equal(baseline[i, ids], tensor(p["base"], torch.float32)):
                raise RuntimeError("g=1 is not exactly no-margin candidate logits")
            mask = torch.ones(expected.shape[1], dtype=torch.bool); mask[ids] = False
            if not torch.equal(baseline[i, mask], expected[i, mask]):
                raise RuntimeError("Non-candidate query changed")
    init, gates = replay(model, data, packed, args.device)
    if not torch.equal(init, baseline):
        raise RuntimeError("Untrained router does not exactly reproduce frozen parent")
    stats = coco_metrics(data["arrays"], baseline, args.validation_annotations, args.device)
    if args.expected_baseline_ap is not None and abs(100. * stats["ap"] - args.expected_baseline_ap) > 0.001:
        raise RuntimeError(f"Baseline AP mismatch: {100*stats['ap']:.8f} vs {args.expected_baseline_ap}")
    value = {"seed": args.seed, "mode": args.mode, "epoch": -1, "scale": 0., **stats,
             "full_query_shape": list(baseline.shape), "visual_replay_exact": True,
             "base_replay_exact": True, "initial_router_replay_exact": True,
             "initial_gate_min": min(gates), "initial_gate_max": max(gates)}
    write(folder / "baseline.json", value)
    # Constants are pre-registered simple controls, not warmup checkpoints.
    constants = []
    for gate in (0., .5, 1., 1.5, 2.):
        logits, _ = replay(model, data, packed, args.device, constant_gate=gate)
        constants.append({"gate": gate, **coco_metrics(data["arrays"], logits, args.validation_annotations, args.device)})
    write(folder / "constant_gate_controls.json", constants)
    # A learned head must also be compared with this already-defined, cheap
    # confidence rule. Its threshold is fixed here and never tuned by this run.
    confidence_control = expected.clone()
    for i, p in enumerate(data["prepared"]):
        if p is not None:
            ids = tensor(p["candidates"], torch.long)
            keep = tensor(p["z"]).sigmoid().amax(-1) >= .05
            confidence_control[i, ids[keep]] = tensor(p["base"])[keep]
    confidence_stats = coco_metrics(data["arrays"], confidence_control, args.validation_annotations, args.device)
    write(folder / "confidence_0p05_control.json", {"threshold": .05, "training": False, **confidence_stats})
    print("BASELINE", json.dumps(value), flush=True)
    return stats, baseline, confidence_stats


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    folder = args.out / args.mode / f"seed{args.seed}"
    lock_inputs(args, folder)
    dval = load_cache(args.validation_cache)
    val = pack(dval)
    feature_mode = "no_mass_input" if args.mode.endswith("no_mass_input") else "full"
    model = model_class()(num_classes=val["z"].shape[-1], hidden_dim=32, feature_mode=feature_mode, gate_upper=2.).to(args.device)
    baseline, base_logits, confidence_baseline = baseline_check(args, model, dval, val, folder)
    if args.baseline_only:
        return
    dtrain = load_cache(args.cache)
    declared = dtrain.get("metadata", {}).get("split")
    if declared is not None and str(declared).lower() not in ("train", "training", "train_views", "train_original"):
        raise ValueError(f"Training cache declares forbidden split: {declared}")
    train_pack = pack(dtrain)
    if not bool(train_pack["background_supervision_available"][train_pack["transaction_gate"]].all()):
        raise ValueError("Training cache requires max_gt_iou and true_background_mask; unmatched alone is not true background")
    groups, clean = groups_for(dtrain, train_pack)
    weights, difficulty_summary = difficulty(train_pack)
    write(folder / "training_source_diagnostics.json", {"sources": len(groups), "cache_rows": len(dtrain["prepared"]),
          "views_per_source": dict(Counter(len(v) for v in groups.values())), **difficulty_summary})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 20261013)
    best = {"ap": baseline["ap"], "epoch": -1, "scale": 0.}
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history, first_step = [], None
    for epoch in range(args.epochs):
        started = time.time(); model.train()
        selected, clean_ids = sample_epoch(groups, clean, weights, args.seed, epoch, args.mode)
        sample_record = {"epoch": epoch, "row_indices": selected, "clean_row_indices": clean_ids,
                         "sha256": hashlib.sha256(json.dumps(selected).encode()).hexdigest()}
        write(folder / "sampling" / f"epoch{epoch:02d}.json", sample_record)
        rows = []
        for start in range(0, len(selected), args.batch_size):
            batch = take(train_pack, selected[start:start + args.batch_size], args.device)
            clean_batch = take(train_pack, clean_ids[start:start + args.batch_size], args.device)
            if args.mode == "logit_noise":
                batch = noisy_logits(batch, generator)
            before = {k: v.detach().clone() for k, v in model.named_parameters()} if first_step is None else None
            loss, diagnostics = loss_for(model, batch, clean_batch)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            gate_grad = {k: float(v.grad.norm()) for k, v in model.named_parameters() if v.grad is not None}
            optimizer.step()
            if first_step is None:
                changes = {k: float((v.detach() - before[k]).abs().max()) for k, v in model.named_parameters()}
                first_step = {"epoch": epoch, "batch": 0, "gradient_norm_before_clipping": grad_norm,
                              "parameter_gradient_norms": gate_grad, "parameter_max_updates": changes,
                              "any_gate_parameter_updated": any(v > 0 for v in changes.values()), **diagnostics}
                write(folder / "FIRST_STEP_UPDATE.json", first_step)
                if not first_step["any_gate_parameter_updated"] or not any(v > 0 for v in gate_grad.values()):
                    raise RuntimeError("Router gate failed to learn on first optimization step")
            rows.append({**diagnostics, "gradient_norm": grad_norm,
                         "gate_gradient_norm": float(sum(v * v for v in gate_grad.values()) ** .5)})
        model.eval()
        # Epoch checkpoints prove training took place, including nonselected ones.
        checkpoint = {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                      "config": {"num_classes": val["z"].shape[-1], "hidden_dim": 32, "feature_mode": feature_mode, "gate_upper": 2.},
                      "seed": args.seed, "mode": args.mode, "epoch": epoch}
        (folder / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, folder / "checkpoints" / f"epoch{epoch:02d}.pth")
        train_stats = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        train_stats["batches"] = len(rows)
        train_stats["source_scenes"] = len(selected)
        train_stats["optimization_steps_total"] = (epoch + 1) * len(rows)
        full_logits, _ = replay(model, dval, val, args.device)
        for scale in (.5, 1.):
            logits = full_logits if scale == 1. else base_logits + scale * (full_logits - base_logits)
            stats = coco_metrics(dval["arrays"], logits, args.validation_annotations, args.device)
            row = {"seed": args.seed, "mode": args.mode, "epoch": epoch, "scale": scale,
                   **stats, "training": train_stats, "epoch_seconds": time.time() - started}
            history.append(row)
            if stats["ap"] > best["ap"] + 1e-12:
                best = {"ap": stats["ap"], "epoch": epoch, "scale": scale}
                best_state = checkpoint["state_dict"]
        diagnostics = transition_diagnostics(model, train_pack, args.device)
        write(folder / "diagnostics" / f"epoch{epoch:02d}.json", diagnostics)
        write(folder / "history.json", history)
        print("EPOCH", json.dumps({"seed": args.seed, "mode": args.mode, "epoch": epoch,
              "best_ap_points": 100. * best["ap"], "gain_ap_points": 100. * (best["ap"] - baseline["ap"]),
              "training": train_stats, "transitions": diagnostics, "seconds": time.time() - started}), flush=True)
    model.load_state_dict(best_state); model.eval()
    final = {"state_dict": best_state, "config": checkpoint["config"], "class_name": "BenefitRouter",
             "inference_scale": best["scale"], "selection": {"split": "validation", "seed": args.seed,
             "mode": args.mode, **best, "baseline_ap": baseline["ap"]}, "first_step_updated": True}
    torch.save(final, folder / "best.pth")
    summary = {**final["selection"], "gain_ap_points": 100. * (best["ap"] - baseline["ap"]),
               "confidence_0p05_ap": confidence_baseline["ap"],
               "gain_vs_confidence_0p05_ap_points": 100. * (best["ap"] - confidence_baseline["ap"]),
               "parameters": sum(p.numel() for p in model.parameters()), "checkpoint_sha256": sha(folder / "best.pth"),
               "selected_fallback": best["epoch"] == -1,
               "train_diagnostics": transition_diagnostics(model, train_pack, args.device, best["scale"]),
               "validation_diagnostics": transition_diagnostics(model, val, args.device, best["scale"])}
    write(folder / "summary.json", summary)
    write(folder / "SELECTION_LOCK.json", {"selection_split": "clean validation", "finished": True, **summary})
    print("TRAIN_DONE", json.dumps(summary), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=MODES, default="image_uniform")
    parser.add_argument("--cache", type=Path, default=None, help="Train RGB-view prepared cache")
    parser.add_argument("--validation-cache", type=Path, default=None)
    parser.add_argument("--validation-annotations", type=Path, default=PACKAGE / "inputs/validation.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, choices=range(20, 31), default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu", help="Explicit CPU/GPU; defaults CPU to avoid export interference")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--expected-baseline-ap", type=float, default=None, help="Expected percentage AP for this seed, e.g. seed42 85.6659325")
    args = parser.parse_args()
    if args.validation_cache is None:
        args.validation_cache = OLD_PREPARED / f"validation_seed{args.seed}.pt"
    if args.cache is None:
        args.cache = args.out / "cache" / f"train_views_seed{args.seed}.pt"
    if args.batch_size < 1 or args.lr <= 0:
        parser.error("batch-size/lr must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
