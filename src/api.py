"""FastAPI service exposing the extractor over HTTP.

Run from the repo root:

    python -m src.api
    # or, with reload during development:
    uvicorn src.api:app --reload

Endpoints:

- ``GET  /health``             liveness + configured model
- ``GET  /schemas``            JSON Schema for every doctype
- ``POST /extract/{doctype}``  multipart file upload -> validated JSON

``POST /extract/{doctype}`` returns 422 (with full validation detail, the
per-field confidence map and the extracted data) when the extraction validated
against the schema but its overall confidence sits below the threshold -- the
caller decides whether to route it to a human.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is in requirements.txt but is not load-bearing
    pass

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src.extractor import (
    DEFAULT_MODEL,
    ExtractionError,
    Extractor,
    MissingAPIKeyError,
)
from src.loaders import LOADABLE_EXTENSIONS, LoaderError, load_document
from src.schemas import DOCTYPES, schema_catalog

DEFAULT_MIN_CONFIDENCE = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))

app = FastAPI(
    title="structured-extraction-agents",
    description=(
        "Turn messy documents into validated JSON. Upload a file, get back "
        "schema-validated data with per-field confidence."
    ),
    version="1.0.0",
)

_extractor: Extractor | None = None


def _engine() -> Extractor:
    """Lazy singleton so the app can boot (and serve /schemas) without a key."""
    global _extractor
    if _extractor is None:
        try:
            _extractor = Extractor()
        except MissingAPIKeyError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _extractor


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": os.environ.get("NIM_MODEL", DEFAULT_MODEL),
        "doctypes": sorted(DOCTYPES),
    }


@app.get("/schemas")
def schemas() -> dict:
    return schema_catalog()


@app.post("/extract/{doctype}")
async def extract_document(
    doctype: str,
    file: UploadFile = File(...),
    min_confidence: float = Query(
        DEFAULT_MIN_CONFIDENCE,
        ge=0.0,
        le=1.0,
        description="Reject extractions whose overall confidence is below this.",
    ),
    ocr: bool = Query(
        False, description="Transcribe image uploads with the NIM vision model."
    ),
) -> dict:
    if doctype not in DOCTYPES:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown doctype '{doctype}'; expected one of: "
                f"{', '.join(sorted(DOCTYPES))}"
            ),
        )

    suffix = Path(file.filename or "upload.txt").suffix.lower() or ".txt"
    if suffix not in LOADABLE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported file type '{suffix}'; supported: "
                f"{', '.join(sorted(LOADABLE_EXTENSIONS))}"
            ),
        )

    payload = await file.read()
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(payload)
        tmp.close()
        engine = _engine()
        try:
            text = await asyncio.to_thread(load_document, tmp.name, ocr=ocr)
            result = await asyncio.to_thread(engine.extract, doctype, text)
        except LoaderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ExtractionError as exc:
            raise HTTPException(
                status_code=502, detail=f"extraction failed: {exc}"
            ) from exc
    finally:
        os.unlink(tmp.name)

    body = result.to_dict()
    if result.overall_confidence < min_confidence:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    f"overall confidence {result.overall_confidence:.3f} is below "
                    f"the threshold {min_confidence:.2f}; the extraction validated "
                    "against the schema but should be reviewed by a human"
                ),
                "confidence": body["confidence"],
                "warnings": body["warnings"],
                "data": body["data"],
            },
        )
    return body


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="127.0.0.1", port=8000)
