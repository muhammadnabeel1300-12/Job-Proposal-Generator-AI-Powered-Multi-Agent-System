"""
file_parser.py - Document parsing (PDF, DOCX, TXT, images).

Responsibilities:
- Validate uploads (extension, size).
- Detect the file type and extract text with the right parser.
- For PDFs: extract selectable text page by page and call OCR ONLY for pages
  that do not contain enough text (scanned or mixed PDFs).
- Clean the text and report how every page was processed.

OCR itself lives in ocr_processor.py. AI processing lives in agents/.
"""

import io
import os
import re
import zipfile
from dataclasses import dataclass, field

try:
    import pymupdf  # PyMuPDF
except ImportError:
    pymupdf = None
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader

from utils import ocr_processor

# ---------------------------------------------------------------- settings
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "jpg", "jpeg", "png"}
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE_MB = 10
MIN_CHARS_PER_PAGE = 40   # Fewer characters than this -> page treated as scanned.
MAX_PDF_PAGES = 50        # Protects against very slow OCR on huge files.
OCR_DPI = 300             # Resolution used to render scanned PDF pages.


# --------------------------------------------------------------- exceptions
class DocumentError(Exception):
    """Base class for all document problems (message is user-friendly)."""


class UnsupportedFileError(DocumentError):
    pass


class EmptyDocumentError(DocumentError):
    pass


class CorruptedDocumentError(DocumentError):
    pass


# --------------------------------------------------------------- data classes
@dataclass
class PageResult:
    page_number: int
    method: str                 # "text", "ocr", "failed" or "empty"
    text: str = ""
    confidence: float | None = None
    warning: str = ""

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class ExtractionResult:
    filename: str
    file_type: str
    pages: list[PageResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Combined text. Page markers are added only for multi-page documents."""
        usable = [p for p in self.pages if p.text.strip()]
        if len(self.pages) <= 1:
            return "\n\n".join(p.text for p in usable)
        return "\n\n".join(f"--- Page {p.page_number} ---\n{p.text}" for p in usable)

    @property
    def method_summary(self) -> str:
        methods = {p.method for p in self.pages if p.method in ("text", "ocr")}
        if methods == {"text"}:
            return "Normal text parsing"
        if methods == {"ocr"}:
            return "OCR (Tesseract)"
        if methods == {"text", "ocr"}:
            return "Mixed: text parsing + OCR"
        return "None (no text extracted)"

    @property
    def failed_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.method == "failed"]


# --------------------------------------------------------------- validation
def get_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower().lstrip(".")


def validate_upload(filename: str, size_bytes: int) -> None:
    """Raise a DocumentError if the file should not be accepted."""
    ext = get_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(e.upper() for e in ALLOWED_EXTENSIONS))
        raise UnsupportedFileError(
            f"'{filename}' is not supported. Allowed formats: {allowed}.")
    if size_bytes == 0:
        raise EmptyDocumentError(f"'{filename}' is empty (0 bytes).")
    if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise DocumentError(
            f"'{filename}' is larger than the {MAX_FILE_SIZE_MB} MB limit.")


# --------------------------------------------------------------- text cleaning
def clean_text(text: str) -> str:
    """Remove extra spaces and blank lines while keeping paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = text.replace("\x00", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------- main entry
def extract_text(filename: str, data: bytes, use_ocr: bool = True) -> ExtractionResult:
    """
    Extract text from an uploaded file.
    Raises DocumentError subclasses with friendly messages on failure.
    """
    validate_upload(filename, len(data))
    ext = get_extension(filename)

    if ext == "pdf":
        result = _parse_pdf(filename, data, use_ocr)
    elif ext == "docx":
        result = _parse_docx(filename, data)
    elif ext == "txt":
        result = _parse_txt(filename, data)
    else:
        result = _parse_image(filename, data, use_ocr)

    if not result.text.strip():
        details = " ".join(result.warnings)
        raise EmptyDocumentError(
            f"No readable text was found in '{filename}'. {details}".strip())
    if len(result.text.strip()) < 100:
        result.warnings.append(
            "Very little text was extracted. Please check it before continuing.")
    return result


# --------------------------------------------------------------- PDF
def _parse_pdf(filename: str, data: bytes, use_ocr: bool) -> ExtractionResult:
    result = ExtractionResult(filename=filename, file_type="pdf")
    if pymupdf is None:
        return _parse_pdf_with_pypdf(filename, data)
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        return _parse_pdf_with_pypdf(filename, data)  # Fallback parser.

    with doc:
        if doc.needs_pass:
            raise CorruptedDocumentError(
                f"'{filename}' is password protected. Remove the password and try again.")
        total = doc.page_count
        if total == 0:
            raise CorruptedDocumentError(f"'{filename}' has no pages.")
        if total > MAX_PDF_PAGES:
            result.warnings.append(
                f"The PDF has {total} pages. Only the first {MAX_PDF_PAGES} were processed.")

        ocr_ok = use_ocr and ocr_processor.check_tesseract()[0]
        ocr_missing_warned = False

        for index in range(min(total, MAX_PDF_PAGES)):
            number = index + 1
            try:
                page = doc[index]
                text = clean_text(page.get_text("text"))
            except Exception as error:
                result.pages.append(PageResult(number, "failed",
                                               warning=f"Could not read page: {error}"))
                continue

            # Enough selectable text -> no OCR needed.
            if len(text) >= MIN_CHARS_PER_PAGE:
                result.pages.append(PageResult(number, "text", text))
                continue

            # Not enough text: scanned / image-only page.
            if not ocr_ok:
                if not ocr_missing_warned:
                    result.warnings.append(
                        "Some pages look scanned but OCR is unavailable "
                        "(Tesseract missing or disabled).")
                    ocr_missing_warned = True
                result.pages.append(PageResult(
                    number, "failed", text, warning="Needs OCR but OCR is unavailable."))
                continue

            result.pages.append(_ocr_pdf_page(page, number, text))

    _add_page_warnings(result)
    return result


def _ocr_pdf_page(page, number: int, fallback_text: str) -> PageResult:
    """Render one PDF page to an image and run OCR on it."""
    try:
        pix = page.get_pixmap(dpi=OCR_DPI)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr = ocr_processor.ocr_image(image)
    except Exception as error:
        return PageResult(number, "failed", fallback_text,
                          warning=f"OCR failed: {error}")

    if not ocr.text.strip():
        # A truly blank page is not an error.
        return PageResult(number, "empty", "", ocr.confidence,
                          "No text found (blank page?).")
    return PageResult(number, "ocr", clean_text(ocr.text), ocr.confidence,
                      " ".join(ocr.warnings))


def _parse_pdf_with_pypdf(filename: str, data: bytes) -> ExtractionResult:
    """Fallback if PyMuPDF cannot open the file. Text only, no OCR."""
    result = ExtractionResult(filename=filename, file_type="pdf")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise CorruptedDocumentError(
                f"'{filename}' is password protected. Remove the password and try again.")
        for i, page in enumerate(reader.pages[:MAX_PDF_PAGES], start=1):
            text = clean_text(page.extract_text() or "")
            method = "text" if text else "empty"
            result.pages.append(PageResult(i, method, text))
    except DocumentError:
        raise
    except Exception as error:
        raise CorruptedDocumentError(
            f"'{filename}' appears to be corrupted or is not a valid PDF.") from error
    result.warnings.append("The PDF was read with the fallback parser (no OCR).")
    return result


def _add_page_warnings(result: ExtractionResult) -> None:
    if result.failed_pages:
        result.warnings.append(
            f"Pages that could not be processed: {', '.join(map(str, result.failed_pages))}.")
    for page in result.pages:
        if page.method == "ocr" and page.warning:
            result.warnings.append(f"Page {page.page_number}: {page.warning}")


# --------------------------------------------------------------- DOCX
def _parse_docx(filename: str, data: bytes) -> ExtractionResult:
    try:
        doc = Document(io.BytesIO(data))
    except (zipfile.BadZipFile, KeyError, ValueError, Exception) as error:
        raise CorruptedDocumentError(
            f"'{filename}' is not a valid DOCX file or is corrupted.") from error

    blocks: list[str] = []
    # Walk the body in order so tables stay in the right place.
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            blocks.append(Paragraph(child, doc).text)
        elif child.tag.endswith("}tbl"):
            for row in Table(child, doc).rows:
                cells: list[str] = []
                for cell in row.cells:
                    value = cell.text.strip()
                    if value and value not in cells:  # merged cells repeat
                        cells.append(value)
                if cells:
                    blocks.append(" | ".join(cells))

    text = clean_text("\n".join(blocks))
    result = ExtractionResult(filename=filename, file_type="docx")
    result.pages.append(PageResult(1, "text" if text else "empty", text))
    return result


# --------------------------------------------------------------- TXT
def _parse_txt(filename: str, data: bytes) -> ExtractionResult:
    text = None
    encodings = ["utf-8-sig", "cp1252", "latin-1"]
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):  # UTF-16 only if it has a BOM
        encodings.insert(0, "utf-16")
    for encoding in encodings:
        try:
            text = data.decode(encoding)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if text is None:
        raise CorruptedDocumentError(f"'{filename}' could not be decoded as text.")
    text = clean_text(text)
    result = ExtractionResult(filename=filename, file_type="txt")
    result.pages.append(PageResult(1, "text" if text else "empty", text))
    return result


# --------------------------------------------------------------- IMAGES
def _parse_image(filename: str, data: bytes, use_ocr: bool) -> ExtractionResult:
    if not use_ocr:
        raise DocumentError("Images need OCR, but OCR is turned off.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # Forces full decoding so corrupted files fail here.
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise CorruptedDocumentError(
            f"'{filename}' is not a readable image or is corrupted.") from error

    try:
        ocr = ocr_processor.ocr_image(image)
    except ocr_processor.OCRError as error:
        raise DocumentError(f"OCR failed for '{filename}': {error}") from error

    result = ExtractionResult(filename=filename, file_type=get_extension(filename))
    result.pages.append(PageResult(
        1, "ocr" if ocr.text.strip() else "empty",
        clean_text(ocr.text), ocr.confidence, " ".join(ocr.warnings)))
    result.warnings.extend(ocr.warnings)
    return result
