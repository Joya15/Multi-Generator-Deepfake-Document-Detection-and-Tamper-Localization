from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ddfd.core import load_config, read_csv_rows, write_csv_rows, write_json


def write_subset_manifests(base_manifest_dir: Path, out_dir: Path, train_dataset: str, test_dataset: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = read_csv_rows(base_manifest_dir / "all_samples.csv")
    train_rows = [r for r in all_rows if r["dataset"] == train_dataset and r["split"] in {"train", "val"}]
    # Keep validation from the selected training dataset's original validation rows.
    val_rows = [r for r in all_rows if r["dataset"] == train_dataset and r["split"] == "val"]
    train_rows = [r for r in train_rows if r["split"] == "train"]
    test_rows = [r for r in all_rows if r["dataset"] == test_dataset and r["split"] == "test"]
    for name, rows in [("all_samples.csv", train_rows + val_rows + test_rows), ("train.csv", train_rows), ("val.csv", val_rows), ("test.csv", test_rows)]:
        write_csv_rows(out_dir / name, rows)
    summary = {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "manifest_dir": str(out_dir),
    }
    write_json(out_dir / "study_manifest_summary.json", summary)
    return summary


def write_study_config(base_cfg_path: Path, manifest_dir: Path, out_name: str, architecture: str) -> Path:
    cfg = load_config(base_cfg_path)
    cfg["data"]["manifest_dir"] = str(manifest_dir)
    cfg["model"]["architecture"] = architecture
    cfg["training"]["epochs"] = 2
    cfg["outputs"]["checkpoints"] = str(PROJECT / "outputs" / "checkpoints" / out_name)
    cfg["outputs"]["reports"] = str(PROJECT / "outputs" / "reports" / out_name)
    cfg["outputs"]["predictions"] = str(PROJECT / "outputs" / "predictions" / out_name)
    cfg_path = PROJECT / "configs" / f"{out_name}.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=str(PROJECT / "configs" / "default.yaml"))
    parser.add_argument("--architecture", default="tiny_unet", choices=["tiny_unet", "segformer_lite", "convnext_tiny", "swin_t", "hf_segformer_b0"])
    parser.add_argument("--run", action="store_true", help="Actually run the two 2-epoch trainings and evaluations.")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    args = parser.parse_args()

    base_cfg = load_config(args.base_config)
    base_manifest_dir = Path(base_cfg["data"]["manifest_dir"])
    studies = [
        ("AIForge-Doc-v2", "AIForge-Doc-v1", f"study_train_v2_test_v1_{args.architecture}"),
        ("AIForge-Doc-v1", "AIForge-Doc-v2", f"study_train_v1_test_v2_{args.architecture}"),
    ]
    results = []
    py = PROJECT / ".venv" / "Scripts" / "python.exe"
    for train_ds, test_ds, name in studies:
        manifest_dir = PROJECT / "data" / "manifests" / name
        summary = write_subset_manifests(base_manifest_dir, manifest_dir, train_ds, test_ds)
        cfg_path = write_study_config(Path(args.base_config), manifest_dir, name, args.architecture)
        record = {"name": name, "summary": summary, "config": str(cfg_path)}
        if args.run:
            train_cmd = [str(py), str(PROJECT / "scripts" / "train.py"), "--config", str(cfg_path), "--epochs", "2"]
            if args.limit_train:
                train_cmd += ["--limit-train", str(args.limit_train)]
            if args.limit_val:
                train_cmd += ["--limit-val", str(args.limit_val)]
            subprocess.run(train_cmd, cwd=str(PROJECT), check=True)
            ckpt = PROJECT / "outputs" / "checkpoints" / name / "best.pt"
            eval_cmd = [str(py), str(PROJECT / "scripts" / "evaluate.py"), "--config", str(cfg_path), "--checkpoint", str(ckpt), "--split", "test"]
            if args.limit_test:
                eval_cmd += ["--limit", str(args.limit_test)]
            subprocess.run(eval_cmd, cwd=str(PROJECT), check=True)
            record["checkpoint"] = str(ckpt)
        results.append(record)
    out = PROJECT / "outputs" / "reports" / "cross_dataset_studies_plan.json"
    write_json(out, {"studies": results})
    print(json.dumps({"studies": results, "plan_path": str(out)}, indent=2))


if __name__ == "__main__":
    main()
