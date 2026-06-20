from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ddfd.core import load_config, build_manifests, audit_dataset, train, evaluate, predict, read_csv_rows
import argparse
p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/default.yaml"); a = p.parse_args()
cfg = load_config(a.config)
print("building manifests"); print(build_manifests(cfg))
print("auditing"); print(audit_dataset(cfg, 20)["total_rows"])
print("training smoke"); result = train(cfg, epochs=1, limit_train=8, limit_val=4); print(result)
print("evaluating smoke"); print(evaluate(cfg, result["best_checkpoint"], "val", 4))
rows = read_csv_rows(Path(cfg["data"]["manifest_dir"]) / "val.csv")
if rows: print("predicting smoke"); print(predict(cfg, result["best_checkpoint"], rows[0]["image_path"]))
