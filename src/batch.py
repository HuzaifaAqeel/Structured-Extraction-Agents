"""Batch mode: extract every document in a folder into a JSONL results file.

Design notes:

- asyncio with a semaphore caps concurrent NIM calls. The chat client is
  synchronous, so each call runs in a worker thread via ``asyncio.to_thread``.
- Every file is isolated: one bad document writes an error record and the run
  continues.
- Records append as each file finishes, so a crashed run leaves a valid
  partial JSONL behind. Re-running with resume enabled (the default) skips
  files that already succeeded and retries the ones that failed. A file that
  failed and later succeeded appears twice; consumers should keep the last
  record per file name.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from src.extractor import Extractor
from src.loaders import LOADABLE_EXTENSIONS, load_document


@dataclass
class BatchSummary:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    mean_confidence: float = 0.0
    rounds_histogram: dict[int, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def format(self) -> str:
        lines = [
            "Batch summary",
            f"  files:      {self.total} ({self.skipped} skipped, already done)",
            f"  succeeded:  {self.succeeded}",
            f"  failed:     {self.failed}",
        ]
        if self.succeeded:
            lines.append(f"  confidence: {self.mean_confidence:.3f} mean overall")
            rounds = ", ".join(
                f"round {r}: {n}" for r, n in sorted(self.rounds_histogram.items())
            )
            lines.append(f"  repairs:    {rounds}")
        lines.append(f"  elapsed:    {self.elapsed_seconds:.1f}s")
        return "\n".join(lines)


def _already_succeeded(output_path: Path) -> set[str]:
    """File names with an ``ok: true`` record in an existing JSONL output."""
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn line from a crashed run; that file gets retried
            if record.get("ok") and record.get("file"):
                done.add(record["file"])
    return done


async def run_batch(
    input_dir: Union[str, Path],
    doctype: str,
    output_path: Union[str, Path],
    *,
    concurrency: int = 4,
    ocr: bool = False,
    resume: bool = True,
    extractor: Optional[Extractor] = None,
) -> BatchSummary:
    """Extract every loadable file in ``input_dir`` and append JSONL records.

    Each record is either ``{"file", "ok": true, "data", "confidence",
    "rounds", "warnings"}`` or ``{"file", "ok": false, "error"}``.
    """
    folder = Path(input_dir)
    if not folder.is_dir():
        raise NotADirectoryError(f"not a directory: {folder}")
    out = Path(output_path)
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in LOADABLE_EXTENSIONS
    )
    done = _already_succeeded(out) if resume else set()
    pending = [p for p in files if p.name not in done]

    summary = BatchSummary(total=len(files), skipped=len(files) - len(pending))
    if not pending:
        return summary

    engine = extractor if extractor is not None else Extractor()
    semaphore = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    confidences: list[float] = []
    rounds: Counter = Counter()
    started = time.perf_counter()

    mode = "a" if (resume and out.exists()) else "w"
    with out.open(mode, encoding="utf-8") as handle:

        async def write_record(record: dict) -> None:
            async with write_lock:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

        async def process(path: Path) -> None:
            async with semaphore:
                try:
                    text = await asyncio.to_thread(load_document, path, ocr=ocr)
                    result = await asyncio.to_thread(engine.extract, doctype, text)
                except Exception as exc:  # noqa: BLE001 -- per-file isolation is the point
                    summary.failed += 1
                    print(f"[err] {path.name}: {exc}")
                    await write_record(
                        {"file": path.name, "ok": False, "error": str(exc)}
                    )
                    return
                summary.succeeded += 1
                confidences.append(result.overall_confidence)
                rounds[result.rounds] += 1
                print(
                    f"[ok]  {path.name} "
                    f"(confidence {result.overall_confidence:.2f}, round {result.rounds})"
                )
                await write_record({"file": path.name, "ok": True, **result.to_dict()})

        await asyncio.gather(*(process(path) for path in pending))

    summary.elapsed_seconds = time.perf_counter() - started
    summary.mean_confidence = (
        sum(confidences) / len(confidences) if confidences else 0.0
    )
    summary.rounds_histogram = dict(rounds)
    return summary
