from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ddfd.core import load_config, analyze_receipts
import argparse
p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/default.yaml"); a = p.parse_args()
print(analyze_receipts(load_config(a.config)))
