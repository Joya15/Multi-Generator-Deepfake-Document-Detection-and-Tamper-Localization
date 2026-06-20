from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ddfd.core import load_config, audit_dataset
import argparse
p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--sample-per-split", type=int, default=250); a = p.parse_args()
print(audit_dataset(load_config(a.config), a.sample_per_split))
