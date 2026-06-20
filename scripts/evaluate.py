from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ddfd.core import load_config, evaluate
import argparse
p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--checkpoint", required=True); p.add_argument("--split", default="val"); p.add_argument("--limit", type=int); a = p.parse_args()
print(evaluate(load_config(a.config), a.checkpoint, a.split, a.limit))
