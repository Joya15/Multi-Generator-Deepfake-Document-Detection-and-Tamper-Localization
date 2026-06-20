# External SOTA Models

Third-party repositories and checkpoints are intentionally excluded from Git.
Clone or download them locally only when you need the official comparison
wrappers:

| Method | Official source | Environment variables |
|---|---|---|
| TruFor | https://grip-unina.github.io/TruFor/ | `TRUFOR_REPO`, `TRUFOR_CHECKPOINT` |
| DocTamper | https://github.com/qcf-568/DocTamper | `DOCTAMPER_REPO`, `DOCTAMPER_CHECKPOINT` |
| IML-ViT | https://github.com/SunnyHaze/IML-ViT | `IMLVIT_REPO`, `IMLVIT_CHECKPOINT` |

The wrapper defaults also recognize the local layouts documented in
`src/ddfd/sota_wrappers.py`. Check readiness with:

```powershell
.\.venv\Scripts\python.exe scripts\sota_wrappers.py --status
```

These methods may require separate environments because their official
dependency stacks differ from this project. Their source code and weights
remain governed by their respective licenses.
