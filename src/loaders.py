"""Input adapters: turn a file on disk into plain text for the extractor.

Text and markdown pass through untouched. PDFs go through pypdf. HTML goes
through readability-lxml so navigation chrome and boilerplate do not pollute
the prompt. Images are transcribed by a NIM vision model
(meta/llama-3.2-90b-vision-instruct) -- opt-in via the ``--ocr`` flag because
it costs an extra API call. Swap ``VISION_MODEL`` for any other NIM model that
accepts ``image_url`` content parts if you want to wire a different one.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Optional, Union

TEXT_EXTENSIONS = {".txt", ".md"}
HTML_EXTENSIONS = {".html", ".htm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PDF_EXTENSIONS = {".pdf"}
LOADABLE_EXTENSIONS = (
    TEXT_EXTENSIONS | HTML_EXTENSIONS | IMAGE_EXTENSIONS | PDF_EXTENSIONS
)

VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"

OCR_PROMPT = (
    "Transcribe every piece of text in this document image, top to bottom, "
    "left to right. Preserve line breaks and column alignment as well as you "
    "can. Output the raw text only -- no commentary, no markdown."
)


class LoaderError(Exception):
    """A document could not be turned into text."""


def load_document(
    path: Union[str, Path],
    *,
    ocr: bool = False,
    client: Optional[Any] = None,
) -> str:
    """Read ``path`` and return its text content.

    ``ocr=True`` enables the NIM vision path for images. ``client`` lets
    callers (and tests) inject a chat client for that path; when omitted a
    real ``NIMChat`` is built, which requires ``NVIDIA_API_KEY``.
    """
    p = Path(path)
    if not p.exists():
        raise LoaderError(f"file not found: {p}")
    suffix = p.suffix.lower()

    if suffix in TEXT_EXTENSIONS:
        text = p.read_text(encoding="utf-8", errors="replace")
    elif suffix in PDF_EXTENSIONS:
        text = _load_pdf(p)
    elif suffix in HTML_EXTENSIONS:
        text = _load_html(p)
    elif suffix in IMAGE_EXTENSIONS:
        if not ocr:
            raise LoaderError(
                f"{p.name} is an image; pass --ocr to transcribe it with the "
                f"NIM vision model ({VISION_MODEL}). This costs one extra API call."
            )
        text = _ocr_image(p, client=client)
    else:
        raise LoaderError(
            f"unsupported file type '{suffix}'; supported: "
            f"{', '.join(sorted(LOADABLE_EXTENSIONS))}"
        )

    if not text.strip():
        hint = (
            " -- if this is a scanned PDF, export its pages as images and use --ocr"
            if suffix in PDF_EXTENSIONS
            else ""
        )
        raise LoaderError(f"{p.name} produced no text{hint}")
    return text


def _load_pdf(p: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise LoaderError("pypdf is not installed; run: pip install pypdf") from exc
    reader = PdfReader(str(p))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)


def _load_html(p: Path) -> str:
    try:
        from lxml import html as lxml_html
        from readability import Document
    except ImportError as exc:
        raise LoaderError(
            "readability-lxml is not installed; run: "
            "pip install readability-lxml \"lxml[html_clean]\""
        ) from exc
    raw = p.read_text(encoding="utf-8", errors="replace")
    doc = Document(raw)
    title = (doc.short_title() or "").strip()
    tree = lxml_html.fromstring(doc.summary())
    body = "\n".join(
        line.strip() for line in tree.text_content().splitlines() if line.strip()
    )
    return f"{title}\n\n{body}" if title else body


def _ocr_image(p: Path, client: Optional[Any] = None) -> str:
    """Transcribe an image with a NIM vision model.

    This is the "wire a NIM vision model here" seam: any model on
    build.nvidia.com that accepts ``image_url`` content parts works -- change
    ``VISION_MODEL`` or pass a custom ``client`` to swap it out.
    """
    if client is None:
        from src.extractor import NIMChat  # local import avoids a hard cycle

        client = NIMChat()
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    encoded = base64.b64encode(p.read_bytes()).decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                },
            ],
        }
    ]
    return client.chat(messages, model=VISION_MODEL, temperature=0.0, max_tokens=4096)
