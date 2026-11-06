"""
ocr_processor.py - OCR (Optical Character Recognition) module.

This module ONLY turns images into text using Tesseract. It does not know
about PDFs, DOCX files or AI. file_parser.py decides WHEN to call it.
"""

import os
import shutil
from dataclasses import dataclass, field

import pytesseract
from dotenv import load_dotenv
from PIL import Image, ImageOps
from pytesseract import Output

# Typical install locations on Windows (used if Tesseract is not on PATH).
_WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

LOW_CONFIDENCE_THRESHOLD = 60.0  # Average word confidence (0-100) below this = warning.
MIN_UPSCALE_WIDTH = 1200         # Small images are enlarged for better accuracy.


class OCRError(Exception):
    """Base class for OCR problems."""


class TesseractNotInstalledError(OCRError):
    """Tesseract could not be found on this computer."""


@dataclass
class OCRResult:
    text: str
    confidence: float | None  # Average word confidence, or None if no words found.
    word_count: int
    warnings: list[str] = field(default_factory=list)


def configure_tesseract() -> str | None:
    """
    Tell pytesseract where tesseract.exe is. Order of search:
    1. TESSERACT_CMD from .env   2. system PATH   3. default Windows folders.
    Returns the path, or None if not found.
    """
    load_dotenv()
    candidates = [os.getenv("TESSERACT_CMD", "").strip().strip('"'),
                  shutil.which("tesseract"), *_WINDOWS_PATHS]
    for path in candidates:
        if path and os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return path
    return None


def check_tesseract() -> tuple[bool, str]:
    """Return (is_available, message) for display in the UI."""
    path = configure_tesseract()
    if not path:
        return False, ("Tesseract OCR was not found. Install it and add it to PATH, "
                       "or set TESSERACT_CMD in your .env file.")
    try:
        return True, f"Tesseract {pytesseract.get_tesseract_version()} ({path})"
    except Exception as error:
        return False, f"Tesseract was found but could not run: {error}"


def preprocess_image(image: Image.Image) -> Image.Image:
    """Make an image easier to read: fix rotation, grayscale, contrast, size."""
    image = ImageOps.exif_transpose(image)  # Respect the phone/camera rotation.
    image = image.convert("L")              # Grayscale.
    image = ImageOps.autocontrast(image)    # Stretch contrast.
    if image.width < MIN_UPSCALE_WIDTH:
        scale = MIN_UPSCALE_WIDTH / image.width
        image = image.resize((MIN_UPSCALE_WIDTH, int(image.height * scale)),
                             Image.LANCZOS)
    return image


def ocr_image(image: Image.Image, language: str = "eng") -> OCRResult:
    """Run Tesseract on one PIL image and return text plus a quality estimate."""
    if not configure_tesseract():
        raise TesseractNotInstalledError(
            "Tesseract OCR is not installed or not configured. See the README.")

    try:
        prepared = preprocess_image(image)
        data = pytesseract.image_to_data(prepared, lang=language,
                                         output_type=Output.DICT)
    except pytesseract.TesseractNotFoundError as error:
        raise TesseractNotInstalledError(str(error)) from error
    except Exception as error:
        raise OCRError(f"OCR failed: {error}") from error

    # Rebuild text line by line, keeping paragraph breaks.
    lines: list[str] = []
    current: list[str] = []
    current_key = None
    last_block_par = None
    confidences: list[float] = []

    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        block_par = (data["block_num"][i], data["par_num"][i])
        if key != current_key:
            if current:
                lines.append(" ".join(current))
            if last_block_par is not None and block_par != last_block_par:
                lines.append("")  # blank line between paragraphs
            current, current_key, last_block_par = [], key, block_par
        current.append(word)
        try:
            conf = float(data["conf"][i])
            if conf >= 0:
                confidences.append(conf)
        except (ValueError, TypeError):
            pass
    if current:
        lines.append(" ".join(current))

    text = "\n".join(lines).strip()
    word_count = len(text.split())
    avg = sum(confidences) / len(confidences) if confidences else None

    warnings = ocr_quality_warnings(text, avg)
    return OCRResult(text=text, confidence=avg, word_count=word_count, warnings=warnings)


def ocr_quality_warnings(text: str, confidence: float | None) -> list[str]:
    """Basic checks to spot unreliable OCR output."""
    warnings = []
    if not text.strip():
        warnings.append("OCR found no readable text.")
        return warnings
    if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
        warnings.append(
            f"Low OCR confidence ({confidence:.0f}/100). Please check the text carefully.")
    letters = sum(ch.isalnum() for ch in text)
    if len(text) > 20 and letters / len(text) < 0.5:
        warnings.append("The OCR text contains many symbols and may be unreliable.")
    return warnings
