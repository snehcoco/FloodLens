"""Grounded situation-report generation."""

from __future__ import annotations

from pathlib import Path


REQUIRED_ATTRIBUTIONS = (
    "Contains modified Copernicus Sentinel data 2026.",
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH "
    "2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.",
    "© OpenStreetMap contributors.",
    "Kuro Siwo training data: cite Bountos et al., 2024 if this dataset was used.",
    "European Union, Copernicus Emergency Management Service data (EMSR927) — checking only; never used as input.",
)


def write_situation_report(output_dir: str | Path | None = None, **stats: object) -> Path:
    """Write report values exactly as supplied by completed pipeline stages."""
    root = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[2] / "outputs" / "reports"
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "situation_report.txt"
    lines = [
        "FloodLens Situation Report",
        "==========================",
        "Potential impacts are satellite/OSM overlap estimates, not confirmed damage.",
        "",
    ]
    for key, value in stats.items():
        lines.append(f"{key}: {value}")
    lines.extend(["", "Data attributions (apply when the corresponding data are used):", *REQUIRED_ATTRIBUTIONS])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def create_pdf_report(text_report: str | Path, output_path: str | Path | None = None) -> Path:
    """Render an existing text report as a PDF using ReportLab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError("PDF export requires ReportLab; install it with `python -m pip install reportlab`.") from exc
    source = Path(text_report)
    destination = Path(output_path) if output_path else source.with_suffix(".pdf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(destination), pagesize=letter)
    width, height = letter
    margin = 38
    max_width = width - 2 * margin
    y = height - margin
    for index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
        if not line:
            y -= 4
            continue
        font = "Helvetica-Bold" if index == 0 or line.startswith("Data attributions") else "Helvetica"
        font_size = 10 if index == 0 else 6.7
        pdf.setFont(font, font_size)
        words = line.split()
        wrapped: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and stringWidth(candidate, font, font_size) > max_width:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        if current:
            wrapped.append(current)
        for wrapped_line in wrapped:
            if y < margin:
                raise RuntimeError("Report is too long for a one-page PDF; shorten the supplied report values.")
            pdf.drawString(margin, y, wrapped_line)
            y -= 9
    pdf.save()
    return destination


def translate_report_to_nepali(
    text_report: str | Path,
    output_path: str | Path | None = None,
    *,
    local_files_only: bool = False,
) -> Path:
    """Translate the report with Helsinki-NLP MarianMT; model weights are cached locally."""
    try:
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError as exc:
        raise RuntimeError("Nepali translation requires transformers and sentencepiece from requirements.txt.") from exc
    model_name = "Helsinki-NLP/opus-mt-en-ne"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = MarianMTModel.from_pretrained(model_name, local_files_only=local_files_only)
    except OSError as exc:
        hint = "Download the model once while online, then use local_files_only=True." if local_files_only else "Check network access and model availability."
        raise RuntimeError(f"Could not load the offline Nepali translation model. {hint}") from exc

    source = Path(text_report)
    lines = source.read_text(encoding="utf-8").splitlines()
    pieces: list[tuple[str, str | None]] = []
    for line in lines:
        if not line:
            pieces.append(("", None))
        elif ": " in line:
            label, value = line.split(": ", 1)
            pieces.append((label + ":", value))
        else:
            pieces.append((line, None))
    translated_labels: list[str] = []
    text_to_translate = [label for label, _ in pieces if label]
    for start in range(0, len(text_to_translate), 8):
        batch = text_to_translate[start:start + 8]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        generated = model.generate(**encoded, max_new_tokens=512)
        translated_labels.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    translated: list[str] = []
    translated_iterator = iter(translated_labels)
    for _, value in pieces:
        label = pieces[len(translated)][0]
        if not label:
            translated.append("")
        else:
            translated_label = next(translated_iterator)
            translated.append(f"{translated_label} {value}" if value is not None else translated_label)
    destination = Path(output_path) if output_path else source.with_name("situation_report_ne.txt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(translated) + "\n", encoding="utf-8")
    return destination
