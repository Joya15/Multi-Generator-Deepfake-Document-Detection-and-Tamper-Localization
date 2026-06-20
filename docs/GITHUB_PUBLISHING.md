# Publishing This Repository

## Preflight

The root `.gitignore` excludes raw datasets, generated manifests, downloaded
third-party repositories, model checkpoints, predictions, reports, virtual
environments, and the internal project handover. Keep those files local.

Before making the repository public, choose a source-code license. No project
license is selected by default because the correct choice belongs to the
project owner. Dataset, checkpoint, and third-party licenses remain separate
even after a source-code license is added.

## Command-Line Publishing

Run these commands from the repository root after installing Git and creating
an empty GitHub repository:

```powershell
git init
git add .
git status
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/<username>/<repository>.git
git push -u origin main
```

Inspect `git status` before committing. It should list source code,
configuration, documentation, and placeholder README files only. It must not
list `.venv`, `data/manifests`, `external` model contents, `outputs` artifacts,
`PROJECT_HANDOVER_FOR_NEW_CHAT.md`, or checkpoint files.

Large trained weights should be distributed separately through a versioned
release or model registry only after verifying redistribution rights. Do not
force-add ignored assets with `git add -f`.
