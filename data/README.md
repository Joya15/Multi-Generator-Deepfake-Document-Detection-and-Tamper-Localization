# Local Dataset Layout

Raw datasets and generated manifests are intentionally excluded from Git. The
default configuration expects this repository and the three datasets to share
one parent directory:

```text
dataset-root/
|-- AIForge-Doc-v1/
|-- AIForge-Doc-v2/
|-- gpt4o-receipt/
`-- deepfake-document-detection/
```

If your layout differs, update `data.dataset_root` in the selected YAML config.
Run the following command after arranging the datasets:

```powershell
.\.venv\Scripts\python.exe scripts\build_manifests.py --config configs\default.yaml
```

This creates machine-specific CSV manifests under `data/manifests/`. Do not
commit those files: they contain absolute paths to local dataset samples.

Dataset sources, citations, and license cautions are documented in the root
`README.md`. Obtain each dataset from its original publisher and comply with
its access and redistribution terms.
