"""Tests 1-9: document parsing and OCR."""

import pytest

from tests.conftest import SAMPLE_TEXT, image_bytes, needs_tesseract
from utils.file_parser import (CorruptedDocumentError, DocumentError,
                               EmptyDocumentError, UnsupportedFileError,
                               clean_text, extract_text, validate_upload)


# 1. Normal text-based PDF
def test_text_pdf_uses_no_ocr(text_pdf):
    result = extract_text("job.pdf", text_pdf)
    assert result.method_summary == "Normal text parsing"
    assert len(result.pages) == 2
    assert all(p.method == "text" for p in result.pages)
    assert "Django" in result.text and "--- Page 2 ---" in result.text


# 2. Scanned PDF
@needs_tesseract
def test_scanned_pdf_uses_ocr(scanned_pdf):
    result = extract_text("scan.pdf", scanned_pdf)
    assert result.method_summary == "OCR (Tesseract)"
    assert "django" in result.text.lower()


# 3. Mixed PDF: OCR only on the scanned page
@needs_tesseract
def test_mixed_pdf_ocr_only_where_needed(mixed_pdf):
    result = extract_text("mixed.pdf", mixed_pdf)
    assert [p.method for p in result.pages] == ["text", "ocr"]
    assert result.method_summary == "Mixed: text parsing + OCR"
    assert "kubernetes" in result.text.lower()


def test_scanned_pdf_without_ocr_is_reported(scanned_pdf):
    with pytest.raises(EmptyDocumentError):
        extract_text("scan.pdf", scanned_pdf, use_ocr=False)


# 4. DOCX (paragraphs + tables)
def test_docx(docx_bytes):
    result = extract_text("cv.docx", docx_bytes)
    assert "Django" in result.text
    assert "Skill | Python" in result.text


# 5. TXT
def test_txt():
    result = extract_text("cv.txt", (SAMPLE_TEXT + "  \n\n\n\nMore").encode("utf-8"))
    assert "More" in result.text and "\n\n\n" not in result.text


def test_txt_other_encoding():
    result = extract_text("cv.txt", "Café résumé Python developer".encode("cp1252"))
    assert "Café" in result.text


# 6. JPG and PNG images
@needs_tesseract
@pytest.mark.parametrize("name,fmt", [("a.png", "PNG"), ("a.jpg", "JPEG"), ("a.jpeg", "JPEG")])
def test_images(name, fmt):
    result = extract_text(name, image_bytes("Python Developer\nDjango and AWS", fmt))
    assert result.method_summary == "OCR (Tesseract)"
    assert "python" in result.text.lower()


# 7. Empty documents
def test_zero_byte_file():
    with pytest.raises(EmptyDocumentError):
        extract_text("empty.txt", b"")


def test_whitespace_only_txt():
    with pytest.raises(EmptyDocumentError):
        extract_text("blank.txt", b"   \n \n  ")


# 8. Corrupted documents
def test_corrupted_pdf():
    with pytest.raises(CorruptedDocumentError):
        extract_text("bad.pdf", b"this is not a pdf at all" * 20)


def test_corrupted_docx():
    with pytest.raises(CorruptedDocumentError):
        extract_text("bad.docx", b"not a zip file" * 20)


def test_corrupted_image():
    with pytest.raises(CorruptedDocumentError):
        extract_text("bad.png", b"not an image" * 20)


# 9. Unsupported file and size limit
def test_unsupported_extension():
    with pytest.raises(UnsupportedFileError):
        validate_upload("malware.exe", 100)


def test_file_too_large():
    with pytest.raises(DocumentError):
        validate_upload("big.pdf", 11 * 1024 * 1024)


def test_clean_text():
    assert clean_text("a   b\t\tc\r\n\r\n\r\n\r\nd  ") == "a b c\n\nd"
