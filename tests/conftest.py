"""Shared test helpers: builds sample documents in memory (no real files needed)."""

import io
import sys
from pathlib import Path

import pymupdf
import pytest
from docx import Document
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import ocr_processor  # noqa: E402

TESSERACT_OK = ocr_processor.check_tesseract()[0]
needs_tesseract = pytest.mark.skipif(not TESSERACT_OK, reason="Tesseract is not installed")

SAMPLE_TEXT = ("Senior Python Developer. We need an engineer with Python and Django "
               "experience to build REST APIs on AWS. You will work with Git and Docker.")


def make_text_image(text: str, width: int = 1400) -> Image.Image:
    """Draw text on a white image (large font so OCR can read it)."""
    try:
        font = ImageFont.load_default(size=42)
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()
    image = Image.new("RGB", (width, 420), "white")
    draw = ImageDraw.Draw(image)
    y = 30
    for line in text.split("\n"):
        draw.text((40, y), line, fill="black", font=font)
        y += 70
    return image


def image_bytes(text: str, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    make_text_image(text).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def text_pdf() -> bytes:
    doc = pymupdf.open()
    for i in range(2):
        page = doc.new_page()
        page.insert_text((72, 100), f"Page {i + 1}. " + SAMPLE_TEXT * 2, fontsize=11)
    return doc.tobytes()


@pytest.fixture
def scanned_pdf() -> bytes:
    """A PDF whose only content is a picture of text (no selectable text)."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=image_bytes("Python Developer\nDjango and AWS\nGit and Docker"))
    return doc.tobytes()


@pytest.fixture
def mixed_pdf() -> bytes:
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((72, 100), SAMPLE_TEXT * 2, fontsize=11)
    page2 = doc.new_page()
    page2.insert_image(page2.rect, stream=image_bytes("Scanned page\nKubernetes and Linux\nTeam leadership"))
    return doc.tobytes()


@pytest.fixture
def docx_bytes() -> bytes:
    doc = Document()
    doc.add_paragraph(SAMPLE_TEXT)
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Skill"
    table.rows[0].cells[1].text = "Python"
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
