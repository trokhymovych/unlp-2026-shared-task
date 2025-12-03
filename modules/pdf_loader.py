"""PDF loading utilities.

Provides `parse_pdf` which extracts text, tables and images per page.

Implementation notes:
- Uses `pdfplumber` for text and table extraction.
- Uses `PyMuPDF` (imported as `fitz`) for robust image extraction when available.
- Falls back to returning image metadata from `pdfplumber` if `fitz` is not installed.
"""
from typing import Any, Dict, List, Optional
import os

import pdfplumber
import pandas as pd

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except Exception:
    _HAS_FITZ = False


def _extract_images_with_fitz(pdf_path: str, page_number: int, images_dir: Optional[str]) -> List[Dict[str, Any]]:
    """Extract images from a single page using PyMuPDF.

    Returns a list of dicts with keys: `bytes`, `ext`, `xref`, and optionally `path`.
    """
    results: List[Dict[str, Any]] = []
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        imgs = page.get_images(full=True)
        for idx, img in enumerate(imgs):
            xref = img[0]
            base = doc.extract_image(xref)
            image_bytes = base.get("image")
            ext = base.get("ext", "png")
            info = {"bytes": image_bytes, "ext": ext, "xref": xref, "index": idx}
            if images_dir:
                os.makedirs(images_dir, exist_ok=True)
                filename = f"page_{page_number}_img_{idx}.{ext}"
                out_path = os.path.join(images_dir, filename)
                with open(out_path, "wb") as f:
                    f.write(image_bytes)
                info["path"] = out_path
            results.append(info)
    finally:
        doc.close()
    return results


def parse_pdf(
    pdf_path: str,
    extract_images: bool = True,
    images_dir: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse a PDF and return a structure with per-page text, tables and images.

    Args:
        pdf_path: Path to the PDF file.
        extract_images: Whether to try to extract images (requires PyMuPDF for full extraction).
        images_dir: If provided and images are extracted, save images to this directory and add `path`.
        doc_id: Optional document ID to include in the output.

    Returns:
        A dict with `file_path` and `doc_id` (if provided) and `pages` list. Each page is a dict:
        `{page_number, text, tables, images}`.
        - `text` is a string ('' if no text found).
        - `tables` is a list of pandas.DataFrame (may contain empty frames).
        - `images` is a list of dicts. If `fitz` is available, entries contain `bytes`, `ext`, and maybe `path`.
          Otherwise entries have `metadata` from pdfplumber's image info.
    """
    pages: List[Dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            # Extract tables
            raw_tables = page.extract_tables() or []
            tables: List[pd.DataFrame] = []
            for raw in raw_tables:
                try:
                    df = pd.DataFrame(raw)
                except Exception:
                    df = pd.DataFrame()
                tables.append(df)

            # Extract images
            images: List[Dict[str, Any]] = []
            if extract_images and _HAS_FITZ:
                try:
                    images = _extract_images_with_fitz(pdf_path, page_idx, images_dir)
                except Exception:
                    # on any fitz error, fall back to pdfplumber metadata
                    images = [{"metadata": img} for img in page.images]
            else:
                # pdfplumber provides image metadata (no image bytes)
                images = [{"metadata": img} for img in page.images]

            pages.append({"page_number": page_idx, "doc_id": doc_id, "text": text, "tables": tables, "images": images})

    return {"file_path": pdf_path, "doc_id": doc_id, "pages": pages}


if __name__ == "__main__":
    # Quick local test (won't run in library import)
    import sys

    if len(sys.argv) > 1:
        result = parse_pdf(sys.argv[1], extract_images=_HAS_FITZ)
        print(f"Parsed {len(result['pages'])} pages from {result['file_path']}")
