from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ddfd.core import load_config, train
import argparse
p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--epochs", type=int); p.add_argument("--limit-train", type=int); p.add_argument("--limit-val", type=int); a = p.parse_args()
print(train(load_config(a.config), a.epochs, a.limit_train, a.limit_val))
