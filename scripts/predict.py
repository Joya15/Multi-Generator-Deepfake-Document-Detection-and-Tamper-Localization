from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ddfd.core import load_config, predict
import argparse
p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--checkpoint", required=True); p.add_argument("--image", required=True); p.add_argument("--output-dir"); a = p.parse_args()
print(predict(load_config(a.config), a.checkpoint, a.image, a.output_dir))
