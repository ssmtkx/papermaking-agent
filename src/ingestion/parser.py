"""PDF document parser using PyMuPDF + pdfplumber fallback."""

from pathlib import Path


class PDFParser:
    """Parse paper-making PDF documents into plain text."""

    @staticmethod
    def parse(file_path: str) -> str:
        """Extract text from a single PDF file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        # PyMuPDF first (fast), pdfplumber as fallback
        text = PDFParser._pymupdf_extract(path)
        if not text.strip():
            text = PDFParser._pdfplumber_extract(path)
        return text

    @staticmethod
    def parse_directory(dir_path: str) -> list[dict]:
        """Parse all PDFs in a directory, return [{filename, text}, ...]."""
        results = []
        for pdf_file in Path(dir_path).glob("*.pdf"):
            text = PDFParser.parse(str(pdf_file))
            results.append({"filename": pdf_file.name, "text": text})
        return results

    @staticmethod
    def _pymupdf_extract(path: Path) -> str:
        import fitz

        doc = fitz.open(str(path))
        texts = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(texts)

    @staticmethod
    def _pdfplumber_extract(path: Path) -> str:
        import pdfplumber

        texts = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
        return "\n\n".join(texts)
