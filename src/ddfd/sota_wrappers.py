from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SotaMethod:
    name: str
    repo_url: str
    paper_url: str
    repo_env: str
    checkpoint_env: str
    default_repo: str
    default_checkpoint: str
    default_command: str
    notes: str


METHODS = {
    "trufor": SotaMethod(
        name="TruFor",
        repo_url="https://grip-unina.github.io/TruFor/",
        paper_url="https://arxiv.org/abs/2212.10957",
        repo_env="TRUFOR_REPO",
        checkpoint_env="TRUFOR_CHECKPOINT",
        default_repo="external/TruFor/TruFor_train_test",
        default_checkpoint="external/TruFor/TruFor_train_test/pretrained_models/weights/trufor.pth.tar",
        default_command=(
            "{python} {repo}/test.py -g 0 -in {image} -out {output_dir} "
            "-exp trufor_ph3 TEST.MODEL_FILE {checkpoint}"
        ),
        notes=(
            "Official TruFor inference runner. The repo targets an older Python/mmcv/mmseg stack, "
            "so use a dedicated TruFor environment if the project venv cannot satisfy those dependencies."
        ),
    ),
    "doctamper": SotaMethod(
        name="DocTamper",
        repo_url="https://github.com/qcf-568/DocTamper",
        paper_url="https://openaccess.thecvf.com/content/CVPR2023/html/Qu_Towards_Robust_Tampered_Text_Detection_in_Document_Image_New_Dataset_and_CVPR_2023_paper.html",
        repo_env="DOCTAMPER_REPO",
        checkpoint_env="DOCTAMPER_CHECKPOINT",
        default_repo="external/DocTamper/models",
        default_checkpoint="external/DocTamper/models/pths/dtd_doctamper.pth",
        default_command=(
            "{python} {repo}/eval_dtd.py --data_root {image} "
            "--lmdb_name DocTamperV1-TestingSet --pth {checkpoint} --minq 75"
        ),
        notes=(
            "Official DocTamper code evaluates LMDB datasets rather than arbitrary single images. "
            "Pass an LMDB data root as --image when using this generic wrapper, or adapt eval_dtd.py."
        ),
    ),
    "iml_vit": SotaMethod(
        name="IML-ViT",
        repo_url="https://github.com/SunnyHaze/IML-ViT",
        paper_url="https://arxiv.org/abs/2307.14863",
        repo_env="IMLVIT_REPO",
        checkpoint_env="IMLVIT_CHECKPOINT",
        default_repo="external/IML-ViT",
        default_checkpoint="external/IML-ViT/checkpoints/iml-vit_checkpoint.pth",
        default_command=(
            "{python} {repo}/main_train.py --eval --resume {checkpoint} "
            "--test_data_path {image} --output_dir {output_dir}"
        ),
        notes=(
            "Official IML-ViT repo is notebook/training oriented and expects datasets in its loader format. "
            "The downloaded checkpoint is available for adapting Demo.ipynb or main_train.py evaluation."
        ),
    ),
}


def _resolve_default(project_root: str | Path, rel_path: str) -> Path:
    return (Path(project_root) / rel_path).expanduser()


def _resolve_repo_and_checkpoint(method: SotaMethod, project_root: str | Path) -> tuple[Path, Path]:
    repo_env = os.environ.get(method.repo_env)
    ckpt_env = os.environ.get(method.checkpoint_env)
    repo = Path(repo_env).expanduser() if repo_env else _resolve_default(project_root, method.default_repo)
    ckpt = Path(ckpt_env).expanduser() if ckpt_env else _resolve_default(project_root, method.default_checkpoint)
    return repo, ckpt


def wrapper_status(project_root: str | Path) -> dict[str, Any]:
    rows = {}
    for key, method in METHODS.items():
        repo, ckpt = _resolve_repo_and_checkpoint(method, project_root)
        rows[key] = {
            "name": method.name,
            "repo_url": method.repo_url,
            "paper_url": method.paper_url,
            "repo_env": method.repo_env,
            "checkpoint_env": method.checkpoint_env,
            "repo_path": str(repo),
            "checkpoint_path": str(ckpt),
            "repo_exists": repo.exists(),
            "checkpoint_exists": ckpt.exists(),
            "using_repo_env": bool(os.environ.get(method.repo_env)),
            "using_checkpoint_env": bool(os.environ.get(method.checkpoint_env)),
            "default_command_template": method.default_command,
            "notes": method.notes,
        }
    out_path = Path(project_root) / "outputs" / "reports" / "sota_wrapper_status.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def run_wrapper(
    method_key: str,
    image: str | Path,
    output_dir: str | Path,
    python: str | Path,
    command_template: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if method_key not in METHODS:
        raise ValueError(f"Unknown SOTA method: {method_key}. Valid: {sorted(METHODS)}")
    method = METHODS[method_key]
    project_root = Path(__file__).resolve().parents[2]
    repo, checkpoint = _resolve_repo_and_checkpoint(method, project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = (command_template or method.default_command).format(
        python=str(python),
        repo=str(repo),
        checkpoint=str(checkpoint),
        image=str(image),
        output_dir=str(output_dir),
    )
    result = {
        "method": method.name,
        "method_key": method_key,
        "repo_url": method.repo_url,
        "repo_path": str(repo),
        "checkpoint_path": str(checkpoint),
        "image": str(image),
        "output_dir": str(output_dir),
        "command": command,
        "dry_run": dry_run,
    }
    if dry_run:
        result["status"] = "dry_run"
        return result
    if not repo.exists():
        result["status"] = "missing_repo"
        result["message"] = f"Set {method.repo_env} to a local clone of {method.repo_url}."
        return result
    if not checkpoint.exists():
        result["status"] = "missing_checkpoint"
        result["message"] = f"Set {method.checkpoint_env} to the official checkpoint path."
        return result
    proc = subprocess.run(shlex.split(command), cwd=str(repo), capture_output=True, text=True)
    result.update(
        {
            "status": "ok" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    )
    return result
