from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageOps

FIELDNAMES = [
    "sample_id", "dataset", "version", "split", "task", "label",
    "image_path", "mask_path", "text_path", "source_dataset", "doc_type", "language",
    "field_name", "original_value", "forged_value", "generator", "spec_id", "new_id",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "notes",
]

def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return normalize_config(cfg, base_dir=config_path.parent.parent)

def normalize_config(cfg: dict[str, Any], base_dir: str | Path | None = None) -> dict[str, Any]:
    base = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()

    def resolve(value: str | Path, relative_to: Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (relative_to / path).resolve()

    root = resolve(cfg.get("project", {}).get("root", "."), base)
    cfg.setdefault("project", {})["root"] = str(root)
    data = cfg.setdefault("data", {})
    data["dataset_root"] = str(resolve(data.get("dataset_root", ".."), root))
    data["manifest_dir"] = str(resolve(data.get("manifest_dir", "data/manifests"), root))
    cfg.setdefault("outputs", {})
    outputs = cfg["outputs"]
    outputs["root"] = str(resolve(outputs.get("root", "outputs"), root))
    outputs["checkpoints"] = str(resolve(outputs.get("checkpoints", "outputs/checkpoints"), root))
    outputs["predictions"] = str(resolve(outputs.get("predictions", "outputs/predictions"), root))
    outputs["reports"] = str(resolve(outputs.get("reports", "outputs/reports"), root))
    return cfg

def ensure_dirs(cfg: dict[str, Any]) -> None:
    for p in [cfg["data"]["manifest_dir"], cfg["outputs"]["root"], cfg["outputs"]["checkpoints"], cfg["outputs"]["predictions"], cfg["outputs"]["reports"]]:
        Path(p).mkdir(parents=True, exist_ok=True)

def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
    return len(rows)

def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)

def assign_train_val(key: str, val_fraction: float) -> str:
    return "val" if stable_hash(key) % 100000 < int(val_fraction * 100000) else "train"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def _bbox(meta: dict[str, Any]) -> dict[str, Any]:
    box = meta.get("bbox_xyxy") or ["", "", "", ""]
    if len(box) != 4:
        box = ["", "", "", ""]
    return {"bbox_x1": box[0], "bbox_y1": box[1], "bbox_x2": box[2], "bbox_y2": box[3]}

def _aiforge_rows(dataset_root: Path, name: str, version: str, val_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ds = dataset_root / name
    rows, skipped = [], []
    meta_path = ds / "metadata.jsonl"
    if not meta_path.exists():
        return rows, [{"dataset": name, "reason": "metadata missing"}]
    for meta in read_jsonl(meta_path):
        raw_split = str(meta.get("split", "")).lower()
        logical = assign_train_val(f"{name}:{meta.get('spec_id')}:{meta.get('new_id')}", val_fraction) if raw_split == "training" else "test"
        image_rel = meta.get("image") or f"{'TrainingSet' if raw_split == 'training' else 'TestingSet'}/images/{meta.get('new_id')}.png"
        mask_rel = meta.get("mask") or f"{'TrainingSet' if raw_split == 'training' else 'TestingSet'}/masks/{meta.get('new_id')}.png"
        image_path, mask_path = ds / image_rel, ds / mask_rel
        if not image_path.exists() or not mask_path.exists():
            skipped.append({"dataset": name, "new_id": meta.get("new_id"), "reason": "missing image or mask", "image_path": str(image_path), "mask_path": str(mask_path)})
            continue
        row = {
            "sample_id": f"{name}:{raw_split}:forged:{meta.get('new_id')}:{meta.get('spec_id')}",
            "dataset": name, "version": version, "split": logical,
            "task": "tamper_localization", "label": 1,
            "image_path": str(image_path), "mask_path": str(mask_path), "text_path": "",
            "source_dataset": meta.get("source_dataset", ""), "doc_type": meta.get("doc_type", ""),
            "language": meta.get("language", ""), "field_name": meta.get("field_name", ""),
            "original_value": meta.get("original_value", ""), "forged_value": meta.get("forged_value", ""),
            "generator": meta.get("assigned_tool", ""), "spec_id": meta.get("spec_id", ""), "new_id": meta.get("new_id", ""),
            "notes": "forged_with_mask",
        }
        row.update(_bbox(meta))
        rows.append(row)
    for split_dir, raw_split in [("TrainingSet", "training"), ("TestingSet", "testing")]:
        auth_dir = ds / split_dir / "authentic"
        if not auth_dir.exists():
            continue
        for p in sorted(auth_dir.glob("*.png")):
            logical = assign_train_val(f"{name}:auth:{p.name}", val_fraction) if raw_split == "training" else "test"
            rows.append({
                "sample_id": f"{name}:{raw_split}:authentic:{p.stem}",
                "dataset": name, "version": version, "split": logical, "task": "image_classification", "label": 0,
                "image_path": str(p), "mask_path": "", "text_path": "",
                "source_dataset": "unknown", "doc_type": "document", "language": "unknown",
                "field_name": "", "original_value": "", "forged_value": "", "generator": "authentic",
                "spec_id": "", "new_id": p.stem, "bbox_x1": "", "bbox_y1": "", "bbox_x2": "", "bbox_y2": "",
                "notes": "authentic_no_mask",
            })
    return rows, skipped

def _gpt4o_rows(dataset_root: Path) -> list[dict[str, Any]]:
    ds = dataset_root / "gpt4o-receipt"
    if not ds.exists():
        return []
    pattern = re.compile(r"^receipt_.+_\d{4}\.png$", re.I)
    rows = []
    for p in sorted(ds.glob("*.png")):
        label = 1 if pattern.match(p.name) else 0
        bucket = stable_hash(f"gpt4o:{p.name}") % 100
        split = "val" if bucket < 12 else ("test" if bucket < 24 else "train")
        txt = p.with_suffix(".txt")
        rows.append({
            "sample_id": f"gpt4o-receipt:{p.stem}", "dataset": "gpt4o-receipt", "version": "1.0",
            "split": split, "task": "generated_receipt_classification", "label": label,
            "image_path": str(p), "mask_path": "", "text_path": str(txt) if txt.exists() else "",
            "source_dataset": "gpt4o-receipt", "doc_type": "receipt", "language": "en",
            "field_name": "", "original_value": "", "forged_value": "",
            "generator": "gpt4o_gpt-image-1" if label else "authentic_by_filename",
            "spec_id": "", "new_id": p.stem, "bbox_x1": "", "bbox_y1": "", "bbox_x2": "", "bbox_y2": "",
            "notes": "flat_receipt_dataset",
        })
    return rows

def build_manifests(cfg: dict[str, Any]) -> dict[str, Any]:
    ensure_dirs(cfg)
    dataset_root = Path(cfg["data"]["dataset_root"])
    val_fraction = float(cfg["data"].get("val_fraction_from_training", 0.12))
    rows, skipped = [], []
    if cfg["data"].get("include_aiforge_v1", True):
        r, s = _aiforge_rows(dataset_root, "AIForge-Doc-v1", "1.0", val_fraction)
        rows += r; skipped += s
    if cfg["data"].get("include_aiforge_v2", True):
        r, s = _aiforge_rows(dataset_root, "AIForge-Doc-v2", "2.0", val_fraction)
        rows += r; skipped += s
    if cfg["data"].get("include_gpt4o_receipt", True):
        rows += _gpt4o_rows(dataset_root)
    out = Path(cfg["data"]["manifest_dir"])
    write_csv_rows(out / "all_samples.csv", rows)
    for split in ["train", "val", "test"]:
        write_csv_rows(out / f"{split}.csv", [r for r in rows if r["split"] == split])
    summary = {"total_rows": len(rows), "skipped_rows": len(skipped), "splits": dict(Counter(r["split"] for r in rows)), "datasets": dict(Counter(r["dataset"] for r in rows))}
    write_json(out / "manifest_summary.json", {"summary": summary, "skipped": skipped})
    return {"summary": summary, "skipped": skipped}

def audit_dataset(cfg: dict[str, Any], sample_per_split: int = 250) -> dict[str, Any]:
    rows = read_csv_rows(Path(cfg["data"]["manifest_dir"]) / "all_samples.csv")
    report = {
        "total_rows": len(rows),
        "by_split": dict(Counter(r["split"] for r in rows)),
        "by_dataset": dict(Counter(r["dataset"] for r in rows)),
        "by_task": dict(Counter(r["task"] for r in rows)),
        "by_label": dict(Counter(r["label"] for r in rows)),
        "by_generator": dict(Counter(r["generator"] for r in rows)),
        "missing_files": [],
        "sample_stats": {},
    }
    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
        if not Path(row["image_path"]).exists():
            report["missing_files"].append({"sample_id": row["sample_id"], "kind": "image", "path": row["image_path"]})
        if row.get("mask_path") and not Path(row["mask_path"]).exists():
            report["missing_files"].append({"sample_id": row["sample_id"], "kind": "mask", "path": row["mask_path"]})
    for split, sr in by_split.items():
        widths, heights, ratios, bad_dims = [], [], [], []
        for row in sr[:sample_per_split]:
            try:
                with Image.open(row["image_path"]) as im:
                    widths.append(im.width); heights.append(im.height)
                    if row.get("mask_path"):
                        with Image.open(row["mask_path"]) as ma:
                            if ma.size != im.size:
                                bad_dims.append(row["sample_id"])
                            hist = ma.convert("L").histogram()
                            ratios.append(sum(hist[128:]) / max(1, ma.width * ma.height))
            except Exception as exc:
                report["missing_files"].append({"sample_id": row["sample_id"], "kind": f"open_error:{exc}", "path": row["image_path"]})
        report["sample_stats"][split] = {"sample_n": min(sample_per_split, len(sr)), "width_min_med_max": _mmm(widths), "height_min_med_max": _mmm(heights), "mask_ratio_min_med_max": _mmm(ratios), "bad_mask_dims": bad_dims[:10]}
    write_json(Path(cfg["outputs"]["reports"]) / "dataset_audit.json", report)
    md = ["# Dataset Audit", "", f"Rows: {report['total_rows']}", "", "## Counts"]
    for key in ["by_split", "by_dataset", "by_task", "by_label", "by_generator"]:
        md.append(f"### {key}")
        for k, v in report[key].items():
            md.append(f"- `{k}`: {v}")
    md.append(f"\nMissing files: {len(report['missing_files'])}\n")
    (Path(cfg["outputs"]["reports"]) / "dataset_audit.md").write_text("\n".join(md), encoding="utf-8")
    return report

def _mmm(values: list[float | int]) -> list[float | int] | None:
    if not values:
        return None
    vals = sorted(values)
    return [round(vals[0], 6), round(vals[len(vals)//2], 6), round(vals[-1], 6)]

def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F

class DocumentDataset:
    def __new__(cls, *args, **kwargs):
        torch, _, _ = _torch()
        from torch.utils.data import Dataset
        class _DS(Dataset):
            def __init__(self, rows, image_size=512, train=False):
                self.rows = rows; self.image_size = int(image_size); self.train = train
            def __len__(self): return len(self.rows)
            def __getitem__(self, idx):
                row = self.rows[idx]
                im = Image.open(row["image_path"]).convert("RGB")
                if row.get("mask_path") and Path(row["mask_path"]).exists():
                    ma = Image.open(row["mask_path"]).convert("L")
                else:
                    ma = Image.new("L", im.size, 0)
                im = im.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                ma = ma.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
                if self.train:
                    if random.random() < 0.25:
                        im = ImageOps.mirror(im); ma = ImageOps.mirror(ma)
                    im = ImageEnhance.Brightness(im).enhance(1 + random.uniform(-0.08, 0.08))
                    im = ImageEnhance.Contrast(im).enhance(1 + random.uniform(-0.08, 0.08))
                arr = np.asarray(im, dtype=np.float32) / 255.0
                mask = (np.asarray(ma, dtype=np.uint8) >= 128).astype(np.float32)
                return {
                    "image": torch.from_numpy(arr).permute(2, 0, 1),
                    "mask": torch.from_numpy(mask).unsqueeze(0),
                    "label": torch.tensor(float(row.get("label", 0)), dtype=torch.float32),
                    "sample_id": row.get("sample_id", str(idx)),
                    "image_path": row.get("image_path", ""),
                }
        return _DS(*args, **kwargs)

def dataloader(cfg, split, train=False, limit=None):
    torch, _, _ = _torch()
    from torch.utils.data import DataLoader
    rows = read_csv_rows(Path(cfg["data"]["manifest_dir"]) / f"{split}.csv")
    if limit:
        rows = rows[:int(limit)]
    ds = DocumentDataset(rows, image_size=int(cfg["training"].get("image_size", 512)), train=train)
    return DataLoader(ds, batch_size=int(cfg["training"].get("batch_size", 1)), shuffle=train, num_workers=int(cfg["training"].get("num_workers", 0)), pin_memory=torch.cuda.is_available())

def create_model(cfg):
    torch, nn, F = _torch()
    arch = str(cfg.get("model", {}).get("architecture", "tiny_unet")).lower()
    if arch not in {"tiny_unet", "tiny_unet_multitask"}:
        from ddfd.advanced_models import create_advanced_model
        return create_advanced_model(cfg)
    base = int(cfg["model"].get("base_channels", 32))
    drop = float(cfg["model"].get("dropout", 0.1))
    class Block(nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.SiLU(inplace=True), nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.SiLU(inplace=True), nn.Dropout2d(drop))
        def forward(self, x): return self.net(x)
    class Net(nn.Module):
        def __init__(self):
            super().__init__(); b = base
            self.e1 = Block(3, b); self.e2 = Block(b, b*2); self.e3 = Block(b*2, b*4); self.e4 = Block(b*4, b*8)
            self.pool = nn.MaxPool2d(2)
            self.u3 = nn.ConvTranspose2d(b*8, b*4, 2, 2); self.d3 = Block(b*8, b*4)
            self.u2 = nn.ConvTranspose2d(b*4, b*2, 2, 2); self.d2 = Block(b*4, b*2)
            self.u1 = nn.ConvTranspose2d(b*2, b, 2, 2); self.d1 = Block(b*2, b)
            self.seg = nn.Conv2d(b, 1, 1)
            self.cls = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(b*8, b*2), nn.SiLU(), nn.Dropout(drop), nn.Linear(b*2, 1))
        def forward(self, x):
            e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
            d3 = self.d3(torch.cat([self.u3(e4), e3], 1)); d2 = self.d2(torch.cat([self.u2(d3), e2], 1)); d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return {"seg_logits": self.seg(d1), "cls_logits": self.cls(e4).squeeze(1)}
    return Net()

def loss_fn(out, batch, cfg):
    torch, _, F = _torch()
    masks, labels = batch["mask"], batch["label"]
    posw = torch.tensor(float(cfg["training"].get("mask_pos_weight", 25.0)), device=masks.device)
    bce = F.binary_cross_entropy_with_logits(out["seg_logits"], masks, pos_weight=posw)
    probs = torch.sigmoid(out["seg_logits"])
    inter = (probs * masks).sum((1,2,3)); denom = probs.sum((1,2,3)) + masks.sum((1,2,3))
    dice = 1 - ((2 * inter + 1e-6) / (denom + 1e-6)).mean()
    cls = F.binary_cross_entropy_with_logits(out["cls_logits"], labels)
    total = float(cfg["training"].get("seg_loss_weight", 1.0)) * (bce + dice) + float(cfg["training"].get("cls_loss_weight", 0.35)) * cls
    return total, {"loss": float(total.detach().cpu()), "seg_loss": float((bce+dice).detach().cpu()), "cls_loss": float(cls.detach().cpu()), "dice_loss": float(dice.detach().cpu())}

def bin_metrics(y_true, y_score, threshold=0.5):
    yt = np.asarray(y_true).reshape(-1).astype(np.uint8); ys = np.asarray(y_score).reshape(-1)
    yp = (ys >= threshold).astype(np.uint8)
    tp = float(((yt==1)&(yp==1)).sum()); tn = float(((yt==0)&(yp==0)).sum()); fp = float(((yt==0)&(yp==1)).sum()); fn = float(((yt==1)&(yp==0)).sum())
    prec = tp / max(1, tp+fp); rec = tp / max(1, tp+fn); f1 = 2*prec*rec/max(1e-12, prec+rec); iou = tp / max(1, tp+fp+fn)
    out = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1, "iou": iou, "accuracy": (tp+tn)/max(1,tp+tn+fp+fn)}
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if len(np.unique(yt)) > 1:
            out["roc_auc"] = float(roc_auc_score(yt, ys)); out["pr_auc"] = float(average_precision_score(yt, ys))
    except Exception:
        pass
    return out

def device(cfg):
    torch, _, _ = _torch()
    return torch.device("cuda" if cfg["training"].get("prefer_cuda", True) and torch.cuda.is_available() else "cpu")

def save_ckpt(path, model, opt, epoch, cfg, metrics):
    torch, _, _ = _torch()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict() if opt else None, "epoch": epoch, "config": cfg, "metrics": metrics}, path)

def load_ckpt(path, cfg, dev):
    torch, _, _ = _torch()
    ck = torch.load(path, map_location=dev)
    m = create_model(cfg); m.load_state_dict(ck["model_state"]); return m

def evaluate(cfg, checkpoint, split="val", limit=None):
    torch, _, _ = _torch()
    dev = device(cfg); model = load_ckpt(checkpoint, cfg, dev).to(dev).eval(); loader = dataloader(cfg, split, False, limit)
    y_true=[]; y_score=[]
    threshold = float(cfg["inference"].get("threshold", 0.5))
    maxn = int(cfg["evaluation"].get("pixel_auc_max_samples", 2500000))
    sample_per_batch = max(512, maxn // max(1, len(loader)))
    rng = np.random.default_rng(42)
    auc_true=[]; auc_score=[]
    tp=tn=fp=fn=0.0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(dev); out = model(imgs)
            cls_scores = torch.sigmoid(out["cls_logits"]).cpu().numpy()
            y_true += batch["label"].cpu().numpy().tolist(); y_score += cls_scores.tolist()
            true = batch["mask"].cpu().numpy().reshape(-1).astype(np.uint8)
            score = torch.sigmoid(out["seg_logits"]).cpu().numpy().reshape(-1).astype(np.float32)
            pred = (score >= threshold).astype(np.uint8)
            tp += float(((true==1)&(pred==1)).sum()); tn += float(((true==0)&(pred==0)).sum())
            fp += float(((true==0)&(pred==1)).sum()); fn += float(((true==1)&(pred==0)).sum())
            if len(true) > sample_per_batch:
                idx = rng.choice(len(true), sample_per_batch, replace=False)
                auc_true.append(true[idx]); auc_score.append(score[idx])
            else:
                auc_true.append(true); auc_score.append(score)
    prec = tp / max(1, tp+fp); rec = tp / max(1, tp+fn); f1 = 2*prec*rec/max(1e-12, prec+rec); iou = tp / max(1, tp+fp+fn)
    pixel = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1, "iou": iou, "accuracy": (tp+tn)/max(1,tp+tn+fp+fn)}
    if auc_true:
        pt = np.concatenate(auc_true); ps = np.concatenate(auc_score)
        if len(pt) > maxn:
            idx = rng.choice(len(pt), maxn, replace=False); pt = pt[idx]; ps = ps[idx]
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            if len(np.unique(pt)) > 1:
                pixel["roc_auc"] = float(roc_auc_score(pt, ps)); pixel["pr_auc"] = float(average_precision_score(pt, ps))
        except Exception:
            pass
    metrics = {"split": split, "checkpoint": str(checkpoint), "n_samples": len(y_true), "image": bin_metrics(y_true, y_score, threshold), "pixel": pixel}
    write_json(Path(cfg["outputs"]["reports"]) / f"evaluation_{split}.json", metrics)
    return metrics

def train(cfg, epochs=None, limit_train=None, limit_val=None):
    torch, _, _ = _torch()
    ensure_dirs(cfg); set_seed(int(cfg["project"].get("seed", 42)))
    dev = device(cfg); model = create_model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"].get("learning_rate", 2e-4)), weight_decay=float(cfg["training"].get("weight_decay", 1e-5)))
    use_amp = bool(cfg["training"].get("mixed_precision", True)) and dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loader = dataloader(cfg, "train", True, limit_train); grad_steps = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
    epochs = int(epochs or cfg["training"].get("epochs", 20)); history=[]; best=-1; best_path=Path(cfg["outputs"]["checkpoints"]) / "best.pt"
    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad(set_to_none=True); run=Counter(); steps=0
        for i, batch in enumerate(loader, 1):
            batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(batch["image"]); loss, parts = loss_fn(out, batch, cfg); loss = loss / grad_steps
            scaler.scale(loss).backward()
            if i % grad_steps == 0 or i == len(loader):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("max_grad_norm", 1.0)))
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            run.update(parts); steps += 1
        save_ckpt(Path(cfg["outputs"]["checkpoints"]) / f"epoch_{ep:03d}.pt", model, opt, ep, cfg, dict(run))
        save_ckpt(best_path, model, opt, ep, cfg, dict(run))
        val = evaluate(cfg, best_path, "val", limit_val)
        score = float(val["pixel"].get("f1", 0)) + float(val["image"].get("roc_auc", 0) or 0)
        if score >= best:
            best = score; save_ckpt(best_path, model, opt, ep, cfg, val)
        rec = {"epoch": ep, "train": {k: v/max(1,steps) for k,v in run.items()}, "val": val}
        history.append(rec); write_json(Path(cfg["outputs"]["reports"]) / "training_history.json", history)
    return {"best_checkpoint": str(best_path), "best_score": best, "history": history}

def predict(cfg, checkpoint, image_path, output_dir=None):
    torch, _, _ = _torch()
    dev = device(cfg); model = load_ckpt(checkpoint, cfg, dev).to(dev).eval()
    image_path = Path(image_path); outdir = Path(output_dir or cfg["outputs"]["predictions"]) / image_path.stem; outdir.mkdir(parents=True, exist_ok=True)
    im = Image.open(image_path).convert("RGB"); size = int(cfg["training"].get("image_size", 512)); rz = im.resize((size,size), Image.Resampling.BILINEAR)
    arr = np.asarray(rz, dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = model(ten); score = float(torch.sigmoid(out["cls_logits"])[0].cpu()); mp = torch.sigmoid(out["seg_logits"])[0,0].cpu().numpy()
    thr = float(cfg["inference"].get("threshold", 0.5)); mask = (mp >= thr).astype(np.uint8) * 255
    mask_path = outdir / f"{image_path.stem}_mask.png"; overlay_path = outdir / f"{image_path.stem}_overlay.png"; report_path = outdir / f"{image_path.stem}_report.json"
    Image.fromarray(mask).save(mask_path)
    heat = np.zeros((size,size,3), dtype=np.uint8); heat[...,0] = (mp*255).astype(np.uint8)
    overlay = Image.blend(rz, Image.fromarray(heat), float(cfg["inference"].get("overlay_alpha", 0.45))); overlay.save(overlay_path)
    report = {"image_path": str(image_path), "fraud_score": score, "predicted_label": int(score >= thr), "mask_path": str(mask_path), "overlay_path": str(overlay_path), "mask_mean_probability": float(mp.mean()), "mask_max_probability": float(mp.max())}
    write_json(report_path, report); report["report_path"] = str(report_path); return report

MONEY_RE = re.compile(r"(?<!\w)(?:[$€£])?\s*(-?\d{1,4}(?:[, ]\d{3})*(?:\.\d{2})|-?\d+\.\d{2})(?!\w)")

def analyze_receipt_text(text: str) -> dict[str, Any]:
    text = re.sub(r"^```[^\n]*\n?", "", text, flags=re.M); text = re.sub(r"\n?```$", "", text, flags=re.M)
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    vals = []
    for line in lines:
        m = MONEY_RE.findall(line)
        if m:
            try: vals.append((line, float(m[-1].replace(",", "").replace(" ", ""))))
            except ValueError: pass
    def find(keys):
        got = None
        for line, val in vals:
            if any(k in line.lower() for k in keys): got = val
        return got
    subtotal, tax, total = find(["subtotal","sub total"]), find(["tax","gst","vat"]), find(["grand total","amount due","total"])
    warnings = []
    if subtotal is not None and tax is not None and total is not None and abs((subtotal+tax)-total) > max(.05, total*.02):
        warnings.append("subtotal_plus_tax_mismatch")
    if total is not None and total <= 0: warnings.append("non_positive_total")
    if len(vals) < 3: warnings.append("few_money_values_detected")
    return {"line_count": len(lines), "money_value_count": len(vals), "subtotal": subtotal, "tax": tax, "total": total, "warnings": warnings, "risk_score": min(1.0, .2*len(warnings))}

def analyze_receipts(cfg):
    rows = read_csv_rows(Path(cfg["data"]["manifest_dir"]) / "all_samples.csv"); reports=[]
    for row in rows:
        if row.get("dataset") == "gpt4o-receipt" and row.get("text_path"):
            txt = Path(row["text_path"]).read_text(encoding="utf-8", errors="ignore")
            rep = analyze_receipt_text(txt); rep.update({"sample_id": row["sample_id"], "label": row["label"], "text_path": row["text_path"]}); reports.append(rep)
    summary = {"n_texts": len(reports), "n_with_warnings": sum(1 for r in reports if r["warnings"]), "reports": reports}
    write_json(Path(cfg["outputs"]["reports"]) / "receipt_text_analysis.json", summary); return summary

def cli_common():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    return p
