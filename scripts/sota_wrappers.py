from pathlib import Path
import argparse
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ddfd.sota_wrappers import run_wrapper, wrapper_status


parser = argparse.ArgumentParser()
parser.add_argument("--status", action="store_true")
parser.add_argument("--method", choices=["trufor", "doctamper", "iml_vit"])
parser.add_argument("--image")
parser.add_argument("--output-dir", default=str(PROJECT / "outputs" / "sota"))
parser.add_argument("--python", default=str(PROJECT / ".venv" / "Scripts" / "python.exe"))
parser.add_argument("--command-template")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

if args.status or not args.method:
    print(json.dumps(wrapper_status(PROJECT), indent=2))
else:
    if not args.image:
        raise SystemExit("--image is required when running a wrapper")
    result = run_wrapper(args.method, args.image, args.output_dir, args.python, args.command_template, args.dry_run)
    out = PROJECT / "outputs" / "reports" / f"sota_{args.method}_last_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
