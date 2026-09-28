#!/usr/bin/env python
"""Offline demo of the extraction pipeline -- no API key required.

Runs the full pipeline (prompt -> extract -> Pydantic validate -> self-repair
loop -> confidence judge) with a scripted stand-in for the LLM, so you can
watch the self-repair behavior on a messy document without any network access.

Usage:
    python demo_offline.py
"""

from __future__ import annotations

import copy
import json

from src.extractor import Extractor
from src.schemas import Invoice


class ScriptedClient:
    """Replays scripted LLM responses; records each conversation."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list = []

    def chat(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        if not self.responses:
            raise AssertionError("ScriptedClient ran out of scripted responses")
        return self.responses.pop(0)


def invoice_payload(bad_subtotal: bool = False) -> dict:
    payload = {
        "vendor_name": "Cascada Cloudworks S.A.",
        "vendor_tax_id": "155712345-2-2026",
        "invoice_number": "INV-2026-0187",
        "issue_date": "2026-05-02",
        "due_date": "2026-06-01",
        "currency": "usd",
        "line_items": [
            {
                "description": "Cloud infrastructure management",
                "quantity": 1,
                "unit_price": 1200.0,
                "amount": 1200.0,
            },
            {
                "description": "Senior DevOps engineering hours",
                "quantity": 16,
                "unit_price": 85.0,
                "amount": 1360.0,
            },
        ],
        "subtotal": 2560.0,
        "tax": 179.2,
        "total": 2739.2,
    }
    if bad_subtotal:  # the "OCR smudge": subtotal disagrees with line items
        payload["subtotal"] = 2500.0
    return payload


def judge_scores() -> str:
    return json.dumps({name: 0.92 for name in Invoice.model_fields})


def show(label: str, doctype: str, document: str, scripted: list[str]) -> None:
    client = ScriptedClient(scripted)
    result = Extractor(client=client).extract(doctype, document)
    print(f"\n{'=' * 64}")
    print(label)
    print(f"{'=' * 64}")
    print(f"rounds taken     : {result.rounds}")
    print(f"confidence       : {result.overall_confidence:.2f} (overall)")
    print(f"warnings         : {result.warnings or ['none']}")
    print("extracted total  :", result.data.total, "| subtotal:", result.data.subtotal)
    if result.rounds > 1:
        repair_msg = client.calls[1][-1]["content"]
        first_line = next(
            (ln for ln in repair_msg.splitlines() if ln.strip()), repair_msg
        )
        print("repair feedback  :", first_line[:100])


def main() -> None:
    print("Structured Extraction Agents -- offline demo (scripted LLM, real pipeline)")

    with open("samples/invoice_01.txt", encoding="utf-8") as fh:
        clean = fh.read()

    # 1) Clean invoice: model answers correctly on the first try.
    show(
        "Scenario 1: clean invoice",
        "invoice",
        clean,
        [json.dumps(invoice_payload()), judge_scores()],
    )

    # 2) Messy invoice: first reply has a bad subtotal, Pydantic rejects it,
    #    and the repair loop feeds the exact validation error back.
    show(
        "Scenario 2: OCR-smudged invoice (self-repair kicks in)",
        "invoice",
        clean,
        [json.dumps(invoice_payload(bad_subtotal=True)), json.dumps(invoice_payload()), judge_scores()],
    )

    print("\nDone. Same flow runs live against NVIDIA NIM with NVIDIA_API_KEY set.")


if __name__ == "__main__":
    main()
