from __future__ import annotations

import csv
import html
import json
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from ddfd.core import analyze_receipt_text, predict, write_json

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_EASYOCR_READER = None


def render_pdf_pages(pdf_path: str | Path, output_dir: str | Path, dpi: int = 180) -> list[Path]:
    import fitz

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir) / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages: list[Path] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        out = output_dir / f"{pdf_path.stem}_page_{i + 1:03d}.png"
        pix.save(out)
        pages.append(out)
    doc.close()
    return pages


def ocr_image(image_path: str | Path) -> dict[str, Any]:
    try:
        import pytesseract

        text = pytesseract.image_to_string(Image.open(image_path).convert("RGB"))
        analysis = analyze_receipt_text(text)
        return {"available": True, "engine": "tesseract", "text": text, "analysis": analysis}
    except Exception as tesseract_exc:
        try:
            global _EASYOCR_READER
            import easyocr

            if _EASYOCR_READER is None:
                project_root = Path(__file__).resolve().parents[2]
                model_dir = project_root / "external" / "easyocr_models"
                model_dir.mkdir(parents=True, exist_ok=True)
                _EASYOCR_READER = easyocr.Reader(
                    ["en"],
                    gpu=False,
                    verbose=False,
                    model_storage_directory=str(model_dir),
                    user_network_directory=str(model_dir / "user_network"),
                )
            ocr_input = str(image_path)
            temp_path = None
            with Image.open(image_path).convert("RGB") as ocr_image_obj:
                if max(ocr_image_obj.size) > 1200:
                    ocr_image_obj.thumbnail((1200, 1200), Image.Resampling.BILINEAR)
                    temp = tempfile.NamedTemporaryFile(prefix="ddfd_easyocr_", suffix=".png", delete=False)
                    temp_path = Path(temp.name)
                    temp.close()
                    ocr_image_obj.save(temp_path)
                    ocr_input = str(temp_path)
            try:
                lines = _EASYOCR_READER.readtext(ocr_input, detail=0, paragraph=True)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
            text = "\n".join(str(line) for line in lines)
            analysis = analyze_receipt_text(text)
            return {
                "available": True,
                "engine": "easyocr",
                "text": text,
                "analysis": analysis,
                "fallback_from": "tesseract",
                "fallback_reason": str(tesseract_exc),
            }
        except Exception as easyocr_exc:
            return {
                "available": False,
                "engine": "",
                "text": "",
                "analysis": {},
                "error": f"tesseract: {tesseract_exc}; easyocr: {easyocr_exc}",
            }


def collect_inputs(input_path: str | Path, render_dir: str | Path) -> list[dict[str, Any]]:
    input_path = Path(input_path)
    items: list[dict[str, Any]] = []
    paths = sorted(input_path.rglob("*")) if input_path.is_dir() else [input_path]
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            items.append({"source_path": str(path), "page_image": str(path), "source_type": "image", "page": 1})
        elif suffix == ".pdf":
            for i, page in enumerate(render_pdf_pages(path, render_dir), start=1):
                items.append({"source_path": str(path), "page_image": str(page), "source_type": "pdf", "page": i})
    return items


def batch_infer_documents(
    cfg: dict[str, Any],
    checkpoint: str | Path,
    input_path: str | Path,
    output_dir: str | Path | None = None,
    run_ocr: bool = True,
    dpi: int = 180,
) -> dict[str, Any]:
    output_dir = Path(output_dir or Path(cfg["outputs"]["predictions"]) / "batch_documents")
    render_dir = output_dir / "_rendered_pages"
    output_dir.mkdir(parents=True, exist_ok=True)
    items = collect_inputs(input_path, render_dir)
    rows: list[dict[str, Any]] = []
    for item in items:
        pred = predict(cfg, checkpoint, item["page_image"], output_dir / "pages")
        ocr = ocr_image(item["page_image"]) if run_ocr else {"available": False, "text": "", "analysis": {}}
        row = {
            **item,
            "fraud_score": pred.get("fraud_score"),
            "predicted_label": pred.get("predicted_label"),
            "mask_path": pred.get("mask_path"),
            "overlay_path": pred.get("overlay_path"),
            "report_path": pred.get("report_path"),
            "ocr_available": ocr.get("available", False),
            "ocr_engine": ocr.get("engine", ""),
            "ocr_error": ocr.get("error", ""),
            "ocr_risk_score": ocr.get("analysis", {}).get("risk_score", ""),
            "ocr_warnings": ";".join(ocr.get("analysis", {}).get("warnings", [])),
        }
        if run_ocr:
            text_path = output_dir / "ocr_text" / (Path(item["page_image"]).stem + ".txt")
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(ocr.get("text", ""), encoding="utf-8", errors="ignore")
            row["ocr_text_path"] = str(text_path)
        rows.append(row)
    summary = {
        "input_path": str(input_path),
        "checkpoint": str(checkpoint),
        "n_pages": len(rows),
        "n_predicted_forged": sum(1 for r in rows if int(r.get("predicted_label") or 0) == 1),
        "rows": rows,
    }
    write_json(output_dir / "batch_report.json", summary)
    _write_csv(output_dir / "batch_report.csv", rows)
    _write_html(output_dir / "batch_report.html", rows)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Document Fraud Batch Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:6px;font-size:13px}img{max-width:220px}</style>",
        "</head><body><h1>Document Fraud Batch Report</h1><table>",
        "<tr><th>Source</th><th>Page</th><th>Fraud score</th><th>OCR warnings</th><th>Overlay</th></tr>",
    ]
    for row in rows:
        overlay = html.escape(str(row.get("overlay_path", "")))
        parts.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('source_path', '')))}</td>"
            f"<td>{html.escape(str(row.get('page', '')))}</td>"
            f"<td>{html.escape(str(row.get('fraud_score', '')))}</td>"
            f"<td>{html.escape(str(row.get('ocr_warnings', '')))}</td>"
            f"<td><a href='{overlay}'><img src='{overlay}'></a></td>"
            "</tr>"
        )
    parts.append("</table></body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
