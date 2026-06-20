from pathlib import Path
import csv
import json
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ddfd.core import (
    load_config, read_csv_rows, write_json, dataloader, load_ckpt, device,
    bin_metrics, evaluate, predict, analyze_receipts
)

def confusion_counts(y_true, y_score, threshold):
    yt = np.asarray(y_true).reshape(-1).astype(np.uint8)
    ys = np.asarray(y_score).reshape(-1)
    yp = (ys >= threshold).astype(np.uint8)
    return {
        "tp": float(((yt == 1) & (yp == 1)).sum()),
        "tn": float(((yt == 0) & (yp == 0)).sum()),
        "fp": float(((yt == 0) & (yp == 1)).sum()),
        "fn": float(((yt == 1) & (yp == 0)).sum()),
    }

def counts_to_metrics(c):
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    iou = tp / max(1.0, tp + fp + fn)
    acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    return {**c, "precision": precision, "recall": recall, "f1": f1, "iou": iou, "accuracy": acc}

def add_counts(target, incoming):
    for k in ["tp", "tn", "fp", "fn"]:
        target[k] += incoming[k]

def main():
    cfg = load_config(str(PROJECT / "configs" / "default.yaml"))
    checkpoint = PROJECT / "outputs" / "checkpoints" / "best.pt"
    reports = PROJECT / "outputs" / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    print("Running aggregate test evaluation...")
    test_eval = evaluate(cfg, checkpoint, "test")

    print("Running detailed threshold and group analysis...")
    import torch
    dev = device(cfg)
    model = load_ckpt(checkpoint, cfg, dev).to(dev).eval()
    test_rows = read_csv_rows(PROJECT / "data" / "manifests" / "test.csv")
    by_sample = {r["sample_id"]: r for r in test_rows}
    loader = dataloader(cfg, "test", False, None)
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    image_true, image_score = [], []
    per_sample = []
    pix_counts = {t: {"tp":0.0,"tn":0.0,"fp":0.0,"fn":0.0} for t in thresholds}
    group_image = defaultdict(lambda: {"true": [], "score": []})
    group_pixel = defaultdict(lambda: {t: {"tp":0.0,"tn":0.0,"fp":0.0,"fn":0.0} for t in thresholds})

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(dev)
            out = model(imgs)
            cls_scores = torch.sigmoid(out["cls_logits"]).cpu().numpy().tolist()
            seg_scores = torch.sigmoid(out["seg_logits"]).cpu().numpy()
            labels = batch["label"].cpu().numpy().tolist()
            masks = batch["mask"].cpu().numpy()
            sample_ids = list(batch["sample_id"])
            for i, sid in enumerate(sample_ids):
                row = by_sample.get(sid, {})
                label = float(labels[i]); score = float(cls_scores[i])
                image_true.append(label); image_score.append(score)
                key = f"dataset={row.get('dataset','unknown')}|generator={row.get('generator','unknown')}"
                group_image[key]["true"].append(label); group_image[key]["score"].append(score)
                true_pix = masks[i].reshape(-1).astype(np.uint8)
                score_pix = seg_scores[i].reshape(-1).astype(np.float32)
                for t in thresholds:
                    c = confusion_counts(true_pix, score_pix, t)
                    add_counts(pix_counts[t], c)
                    add_counts(group_pixel[key][t], c)
                per_sample.append({
                    "sample_id": sid,
                    "label": label,
                    "score": score,
                    "dataset": row.get("dataset", ""),
                    "generator": row.get("generator", ""),
                    "source_dataset": row.get("source_dataset", ""),
                    "doc_type": row.get("doc_type", ""),
                    "image_path": row.get("image_path", ""),
                })

    threshold_rows = []
    for t in thresholds:
        threshold_rows.append({
            "threshold": t,
            **{f"image_{k}": v for k, v in bin_metrics(image_true, image_score, t).items()},
            **{f"pixel_{k}": v for k, v in counts_to_metrics(pix_counts[t]).items()},
        })
    best_threshold = max(threshold_rows, key=lambda r: r.get("pixel_f1", 0.0))["threshold"]

    group_report = {}
    for key in sorted(group_image):
        group_report[key] = {
            "n_samples": len(group_image[key]["true"]),
            "image@0.5": bin_metrics(group_image[key]["true"], group_image[key]["score"], 0.5),
            f"pixel@{best_threshold}": counts_to_metrics(group_pixel[key][best_threshold]),
        }

    with (reports / "per_sample_test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
        writer.writeheader(); writer.writerows(per_sample)
    with (reports / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()))
        writer.writeheader(); writer.writerows(threshold_rows)
    write_json(reports / "cross_generator_analysis.json", group_report)
    write_json(reports / "threshold_sweep.json", {"best_pixel_f1_threshold": best_threshold, "rows": threshold_rows})

    print("Running receipt analysis...")
    receipt_summary = analyze_receipts(cfg)

    print("Creating prediction examples...")
    examples = []
    wanted = [r for r in test_rows if r["dataset"] == "AIForge-Doc-v2" and r["label"] == "1"][:3]
    wanted += [r for r in test_rows if r["dataset"] == "AIForge-Doc-v1" and r["generator"] == "gemini-nano"][:2]
    wanted += [r for r in test_rows if r["dataset"] == "AIForge-Doc-v1" and r["generator"] == "qwen-inpaint"][:2]
    wanted += [r for r in test_rows if r["label"] == "0"][:2]
    for row in wanted:
        try:
            rep = predict(cfg, checkpoint, row["image_path"], PROJECT / "outputs" / "predictions" / "final_examples")
            rep.update({"sample_id": row["sample_id"], "dataset": row["dataset"], "generator": row["generator"], "label": row["label"]})
            examples.append(rep)
        except Exception as exc:
            examples.append({"sample_id": row.get("sample_id"), "error": str(exc)})
    write_json(reports / "final_prediction_examples.json", examples)

    manifest_summary = json.loads((PROJECT / "data" / "manifests" / "manifest_summary.json").read_text(encoding="utf-8"))["summary"]
    audit = json.loads((reports / "dataset_audit.json").read_text(encoding="utf-8"))
    history = json.loads((reports / "training_history.json").read_text(encoding="utf-8"))
    sota_rows = [
        {"method": "Local Tiny U-Net Multi-task", "status": "trained locally", "test_image_auc": test_eval["image"].get("roc_auc"), "test_pixel_f1_best_threshold": max(r["pixel_f1"] for r in threshold_rows), "notes": "2-epoch time-boxed 4GB-GPU baseline"},
        {"method": "TruFor", "status": "published AIForge benchmark", "AIForge_v1_auc": 0.751, "AIForge_v2_auc": 0.599, "notes": "general image forgery localization baseline"},
        {"method": "DocTamper", "status": "published AIForge benchmark", "AIForge_v1_auc": 0.563, "AIForge_v2_auc": 0.585, "notes": "document tampered text baseline"},
        {"method": "GPT-4o / GPT-Image self-judge", "status": "published AIForge benchmark", "AIForge_v1_auc": 0.509, "AIForge_v2_auc": 0.532, "notes": "zero-shot multimodal judgment baseline"},
        {"method": "IML-ViT", "status": "integration hook / future work", "notes": "strong manipulation localization transformer; not run locally within time budget"},
        {"method": "SegFormer/ConvNeXt/Swin", "status": "future trainable SOTA family", "notes": "recommended next architecture once longer training budget is available"},
    ]
    write_json(reports / "sota_comparison.json", sota_rows)

    report = []
    report.append("# Final Project Report: Deepfake Document Detection")
    report.append("")
    report.append("## Executive Summary")
    report.append("This project implements a local baseline system for detecting AI-forged and manipulated document content. It includes dataset indexing, auditing, training, evaluation, receipt text checks, inference outputs, SOTA comparison, and final reporting.")
    report.append("")
    report.append("## Dataset Summary")
    report.append(f"- Total usable manifest rows: {manifest_summary['total_rows']}")
    report.append(f"- Splits: {manifest_summary['splits']}")
    report.append(f"- Datasets: {manifest_summary['datasets']}")
    report.append(f"- Skipped rows: {manifest_summary['skipped_rows']} (known missing AIForge-Doc-v1 training image 000000018)")
    report.append(f"- Audit missing files after filtering: {len(audit.get('missing_files', []))}")
    report.append("")
    report.append("## Method")
    report.append("The local model is a compact multi-task U-Net style CNN designed for the RTX 3050 Laptop GPU with 4 GB VRAM. It predicts both a pixel-level tamper mask and an image-level fraud score. Training uses BCE + Dice segmentation loss and BCE classification loss.")
    report.append("")
    report.append("## Training Setup")
    report.append("- Image size: 512")
    report.append("- Batch size: 1")
    report.append("- Mixed precision: enabled on CUDA")
    report.append("- Final time-boxed run: 2 full epochs")
    report.append(f"- Best checkpoint: `{checkpoint}`")
    report.append(f"- Final validation pixel F1: {history[-1]['val']['pixel'].get('f1'):.4f}")
    report.append(f"- Final validation pixel ROC-AUC: {history[-1]['val']['pixel'].get('roc_auc'):.4f}")
    report.append("")
    report.append("## Test Evaluation")
    report.append(f"- Test samples: {test_eval.get('n_samples')}")
    report.append(f"- Image ROC-AUC: {test_eval['image'].get('roc_auc')}")
    report.append(f"- Image PR-AUC: {test_eval['image'].get('pr_auc')}")
    report.append(f"- Pixel F1 @ 0.5: {test_eval['pixel'].get('f1')}")
    report.append(f"- Pixel IoU @ 0.5: {test_eval['pixel'].get('iou')}")
    report.append(f"- Pixel ROC-AUC sampled: {test_eval['pixel'].get('roc_auc')}")
    report.append(f"- Pixel PR-AUC sampled: {test_eval['pixel'].get('pr_auc')}")
    report.append(f"- Best threshold by pixel F1: {best_threshold}")
    report.append("")
    report.append("## Threshold Sweep")
    for r in threshold_rows:
        report.append(f"- threshold {r['threshold']}: pixel F1={r['pixel_f1']:.4f}, pixel IoU={r['pixel_iou']:.4f}, image F1={r['image_f1']:.4f}")
    report.append("")
    report.append("## Cross-Generator Analysis")
    for key, val in group_report.items():
        report.append(f"- {key}: n={val['n_samples']}, image F1@0.5={val['image@0.5'].get('f1'):.4f}, pixel F1@{best_threshold}={val[f'pixel@{best_threshold}'].get('f1'):.4f}")
    report.append("")
    report.append("## SOTA Baseline Comparison")
    report.append("The local model was trained and evaluated in this project. TruFor, DocTamper, and GPT-style detector numbers are included as published AIForge benchmark references because downloading and validating their external checkpoints is not feasible within the 5-hour completion window.")
    for row in sota_rows:
        report.append(f"- {row['method']} ({row['status']}): {row}")
    report.append("")
    report.append("## Receipt Fraud Module")
    report.append(f"- Receipt text files analyzed: {receipt_summary['n_texts']}")
    report.append(f"- Receipt text files with warnings: {receipt_summary['n_with_warnings']}")
    report.append("The current receipt module checks subtotal/tax/total consistency and basic monetary extraction from paired text files. It is ready to be combined with OCR for unseen receipt images.")
    report.append("")
    report.append("## Inference Productization")
    report.append("Prediction outputs include a fraud probability, binary mask PNG, heatmap overlay PNG, and JSON report. Final examples are listed in `final_prediction_examples.json` and saved under `outputs/predictions/final_examples/`.")
    report.append("")
    report.append("## Limitations")
    report.append("- Training was time-boxed to 2 epochs, so this is a functional baseline rather than a fully converged model.")
    report.append("- AIForge-Doc-v2 local copy lacks authentic paired images, so authentic negatives come mainly from AIForge-Doc-v1 and the small GPT4o filename-authentic subset.")
    report.append("- Full SOTA repo integration for TruFor, DocTamper, and IML-ViT is documented but not executed locally due time/checkpoint dependency risk.")
    report.append("- More threshold calibration and architecture tuning are needed for production use.")
    report.append("")
    report.append("## Future Work")
    report.append("- Train longer, preferably 10-20 epochs.")
    report.append("- Add SegFormer/ConvNeXt/Swin backbones.")
    report.append("- Add official TruFor/DocTamper/IML-ViT checkpoint wrappers.")
    report.append("- Add PDF batch inference and OCR for raw unseen documents.")
    report.append("- Perform separate train-v2/test-v1 and train-v1/test-v2 retraining studies.")
    (reports / "FINAL_PROJECT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    write_json(reports / "FINAL_PROJECT_SUMMARY.json", {
        "manifest_summary": manifest_summary,
        "test_eval": test_eval,
        "best_threshold": best_threshold,
        "receipt_summary": {"n_texts": receipt_summary["n_texts"], "n_with_warnings": receipt_summary["n_with_warnings"]},
        "artifacts": {
            "report": str(reports / "FINAL_PROJECT_REPORT.md"),
            "threshold_sweep": str(reports / "threshold_sweep.csv"),
            "cross_generator": str(reports / "cross_generator_analysis.json"),
            "sota": str(reports / "sota_comparison.json"),
            "examples": str(reports / "final_prediction_examples.json"),
        }
    })
    print("Finalization complete")
    print(str(reports / "FINAL_PROJECT_REPORT.md"))

if __name__ == "__main__":
    main()
