#!/usr/bin/env python
"""Command-line interface for structured-extraction-agents.

Examples:

    python cli.py extract invoice samples/invoice_01.txt --pretty
    python cli.py extract receipt photo_of_receipt.jpg --ocr --pretty
    python cli.py batch invoice ./inbox --out results/invoices.jsonl --concurrency 6
    python cli.py schemas --doctype contract
    python cli.py serve --port 8000

Exit codes: 0 success, 1 error, 2 finished but below the confidence bar
(extract with --min-confidence) or with failed files (batch).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is in requirements.txt but is not load-bearing

    def load_dotenv() -> None:
        return None


from src.batch import run_batch
from src.extractor import (
    ExtractionError,
    Extractor,
    MissingAPIKeyError,
    UnknownDoctypeError,
)
from src.loaders import LoaderError, load_document
from src.schemas import DOCTYPES, get_schema, schema_catalog


def cmd_extract(args: argparse.Namespace) -> int:
    try:
        engine = Extractor(score_confidence=not args.no_judge)
        text = load_document(args.path, ocr=args.ocr, client=engine.client)
        result = engine.extract(args.doctype, text)
    except MissingAPIKeyError as exc:
        print(exc, file=sys.stderr)
        return 1
    except (LoaderError, UnknownDoctypeError, ExtractionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    indent = 2 if args.pretty else None
    print(json.dumps(result.to_dict(), indent=indent, ensure_ascii=False))

    if (
        args.min_confidence is not None
        and result.overall_confidence < args.min_confidence
    ):
        print(
            f"warning: overall confidence {result.overall_confidence:.3f} is "
            f"below {args.min_confidence:.2f}",
            file=sys.stderr,
        )
        return 2
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    try:
        summary = asyncio.run(
            run_batch(
                args.folder,
                args.doctype,
                args.out,
                concurrency=args.concurrency,
                ocr=args.ocr,
                resume=not args.no_resume,
            )
        )
    except MissingAPIKeyError as exc:
        print(exc, file=sys.stderr)
        return 1
    except NotADirectoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print()
    print(summary.format())
    print(f"\nresults: {args.out}")
    return 0 if summary.failed == 0 else 2


def cmd_schemas(args: argparse.Namespace) -> int:
    if args.doctype:
        try:
            print(json.dumps(get_schema(args.doctype), indent=2))
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        print(json.dumps(schema_catalog(), indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Turn messy documents into validated JSON.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract one document to JSON.")
    p_extract.add_argument("doctype", choices=sorted(DOCTYPES))
    p_extract.add_argument(
        "path", help="Path to a .txt/.md/.pdf/.html file, or an image with --ocr."
    )
    p_extract.add_argument(
        "--pretty", action="store_true", help="Indent the JSON output."
    )
    p_extract.add_argument(
        "--ocr",
        action="store_true",
        help="Transcribe image input with the NIM vision model first.",
    )
    p_extract.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the confidence-judge pass (faster, one less API call).",
    )
    p_extract.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Exit with status 2 when overall confidence lands below this value.",
    )
    p_extract.set_defaults(func=cmd_extract)

    p_batch = sub.add_parser(
        "batch", help="Extract every document in a folder to a JSONL file."
    )
    p_batch.add_argument("doctype", choices=sorted(DOCTYPES))
    p_batch.add_argument("folder", help="Folder containing the documents.")
    p_batch.add_argument("--out", default="results/batch.jsonl")
    p_batch.add_argument("--concurrency", type=int, default=4)
    p_batch.add_argument("--ocr", action="store_true")
    p_batch.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess everything even when the output file already has results.",
    )
    p_batch.set_defaults(func=cmd_batch)

    p_schemas = sub.add_parser(
        "schemas", help="Print the JSON Schema(s) the extractor enforces."
    )
    p_schemas.add_argument("--doctype", choices=sorted(DOCTYPES), default=None)
    p_schemas.set_defaults(func=cmd_schemas)

    p_serve = sub.add_parser("serve", help="Run the FastAPI service.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
