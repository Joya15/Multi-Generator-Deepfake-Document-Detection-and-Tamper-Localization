from pathlib import Path
import argparse
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ddfd.batch_inference import batch_infer_documents
from ddfd.core import load_config


parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(PROJECT / "configs" / "default.yaml"))
parser.add_argument("--checkpoint", default=str(PROJECT / "outputs" / "checkpoints" / "best.pt"))
parser.add_argument("--input", required=True, help="PDF, image, or folder of PDFs/images")
parser.add_argument("--output-dir", default=None)
parser.add_argument("--no-ocr", action="store_true")
parser.add_argument("--dpi", type=int, default=180)
args = parser.parse_args()

cfg = load_config(args.config)
summary = batch_infer_documents(cfg, args.checkpoint, args.input, args.output_dir, run_ocr=not args.no_ocr, dpi=args.dpi)
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
