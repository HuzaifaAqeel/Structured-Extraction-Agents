"""Offline test suite: schema rules, the repair loop, the judge, and loaders.

No network, no API key. The LLM is replaced by ``FakeClient``, which replays a
scripted list of responses and records every conversation it was given.
"""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from src.extractor import (
    ExtractionError,
    Extractor,
    UnknownDoctypeError,
    _parse_json_block,
)
from src.loaders import LoaderError, load_document
from src.schemas import DOCTYPES, Contract, Invoice, Receipt, Resume


class FakeClient:
    """Replays scripted responses; records a deep copy of each conversation."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        if not self.responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        return self.responses.pop(0)


# --------------------------------------------------------------------------- #
# Payload factories
# --------------------------------------------------------------------------- #


def make_invoice_payload(**overrides):
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
    payload.update(overrides)
    return payload


def make_resume_payload(**overrides):
    payload = {
        "full_name": "Marta Esquivel Rios",
        "email": "marta.esquivel@example.com",
        "skills": ["JS", "Py", "React", "react", "  PostgreSQL "],
        "experience": [
            {
                "company": "Istmo Digital Labs",
                "title": "Senior Software Engineer",
                "start_date": "2023-03-01",
                "end_date": None,
                "highlights": [],
            },
            {
                "company": "Canalside Analytics",
                "title": "Software Engineer",
                "start_date": "2020-06-01",
                "end_date": "2023-02-28",
                "highlights": [],
            },
        ],
    }
    payload.update(overrides)
    return payload


def make_contract_payload(**overrides):
    payload = {
        "title": "Master Services Agreement",
        "parties": [
            {"name": "Faro Norte Logistics, S.A.", "role": "Client"},
            {"name": "Quetzal Software Group LLC", "role": "Contractor"},
        ],
        "effective_date": "2026-03-01",
        "termination_date": "2027-02-28",
        "obligations": [
            {
                "party": "Quetzal Software Group LLC",
                "description": "Deliver monthly feature releases.",
                "deadline": None,
            },
            {
                "party": "Client",
                "description": "Pay invoices within 30 days.",
                "deadline": None,
            },
        ],
        "governing_law": "Republic of Panama",
        "payment_terms": "USD 4,500 per month, net 30",
        "auto_renews": False,
    }
    payload.update(overrides)
    return payload


def make_receipt_payload(**overrides):
    payload = {
        "merchant_name": "La Terraza del Puerto",
        "purchase_date": "2026-07-11",
        "purchase_time": "20:42",
        "items": [
            {"name": "Corvina a la plancha", "quantity": 1, "price": 18.50},
            {"name": "Limonada de coco", "quantity": 2, "price": 9.00},
            {"name": "Patacones", "quantity": 1, "price": 6.00},
        ],
        "subtotal": 33.50,
        "tax": 2.35,
        "tip": 3.35,
        "total": 39.20,
        "payment_method": "VISA",
    }
    payload.update(overrides)
    return payload


GOOD_INVOICE_JSON = json.dumps(make_invoice_payload())
JUDGE_ALL_HIGH = json.dumps({name: 0.9 for name in Invoice.model_fields})


# --------------------------------------------------------------------------- #
# Schema validation logic
# --------------------------------------------------------------------------- #


def test_invoice_valid_payload_passes_and_normalizes_currency():
    invoice = Invoice.model_validate(make_invoice_payload())
    assert invoice.currency == "USD"
    assert invoice.total == pytest.approx(2739.2)


def test_invoice_total_mismatch_raises():
    with pytest.raises(ValidationError, match="total"):
        Invoice.model_validate(make_invoice_payload(total=9999.0))


def test_invoice_subtotal_must_match_line_items():
    with pytest.raises(ValidationError, match="subtotal"):
        Invoice.model_validate(
            make_invoice_payload(subtotal=3000.0, total=3179.2)
        )


def test_invoice_line_amount_must_match_quantity_times_price():
    payload = make_invoice_payload()
    payload["line_items"][0]["amount"] = 111.0
    with pytest.raises(ValidationError, match="quantity"):
        Invoice.model_validate(payload)


def test_invoice_due_date_before_issue_date_raises():
    with pytest.raises(ValidationError, match="due_date"):
        Invoice.model_validate(make_invoice_payload(due_date="2026-04-01"))


def test_invoice_rejects_bogus_currency():
    with pytest.raises(ValidationError, match="ISO 4217"):
        Invoice.model_validate(make_invoice_payload(currency="dollars"))


def test_resume_skills_normalized_and_deduped():
    resume = Resume.model_validate(make_resume_payload())
    assert resume.skills == ["javascript", "python", "react", "postgresql"]


def test_resume_experience_sorted_newest_first():
    payload = make_resume_payload()
    payload["experience"] = list(reversed(payload["experience"]))
    resume = Resume.model_validate(payload)
    assert resume.experience[0].company == "Istmo Digital Labs"
    assert resume.experience[0].end_date is None  # current role floats to the top


def test_resume_rejects_backwards_experience_dates():
    payload = make_resume_payload()
    payload["experience"][1]["end_date"] = "2019-01-01"  # before its start
    with pytest.raises(ValidationError, match="end_date"):
        Resume.model_validate(payload)


def test_resume_rejects_malformed_email():
    with pytest.raises(ValidationError, match="email"):
        Resume.model_validate(make_resume_payload(email="not-an-email"))


def test_contract_valid_payload_passes():
    contract = Contract.model_validate(make_contract_payload())
    assert contract.auto_renews is False
    assert len(contract.parties) == 2


def test_contract_termination_must_follow_effective_date():
    with pytest.raises(ValidationError, match="termination_date"):
        Contract.model_validate(
            make_contract_payload(termination_date="2026-01-01")
        )


def test_contract_obligation_party_must_exist():
    payload = make_contract_payload()
    payload["obligations"].append(
        {
            "party": "Banco del Istmo",
            "description": "Guarantee the payments.",
            "deadline": None,
        }
    )
    with pytest.raises(ValidationError, match="Banco del Istmo"):
        Contract.model_validate(payload)


def test_contract_obligation_party_matches_by_role():
    # 'Client' is a role, not a legal name -- the validator accepts both.
    contract = Contract.model_validate(make_contract_payload())
    assert contract.obligations[1].party == "Client"


def test_receipt_payment_method_normalized():
    receipt = Receipt.model_validate(make_receipt_payload())
    assert receipt.payment_method == "credit_card"


def test_receipt_unknown_payment_method_becomes_other():
    receipt = Receipt.model_validate(make_receipt_payload(payment_method="cowrie shells"))
    assert receipt.payment_method == "other"


def test_receipt_total_cross_check_raises_on_mismatch():
    with pytest.raises(ValidationError, match="total"):
        Receipt.model_validate(make_receipt_payload(total=99.0))


def test_receipt_item_sum_must_match_subtotal():
    with pytest.raises(ValidationError, match="subtotal"):
        Receipt.model_validate(make_receipt_payload(subtotal=50.0, total=55.70))


def test_receipt_time_zero_padded_and_validated():
    receipt = Receipt.model_validate(make_receipt_payload(purchase_time="8:05"))
    assert receipt.purchase_time == "08:05"
    with pytest.raises(ValidationError, match="HH:MM"):
        Receipt.model_validate(make_receipt_payload(purchase_time="25:99"))


def test_every_doctype_produces_a_json_schema():
    for name, model in DOCTYPES.items():
        schema = model.model_json_schema()
        assert schema.get("properties"), f"{name} schema has no properties"


# --------------------------------------------------------------------------- #
# JSON parsing helpers
# --------------------------------------------------------------------------- #


def test_parse_json_block_strips_markdown_fences():
    raw = 'Sure! Here is the JSON:\n```json\n{"a": 1}\n```\nHope that helps.'
    assert _parse_json_block(raw) == {"a": 1}


def test_parse_json_block_extracts_embedded_object():
    raw = 'The result is {"a": {"b": 2}} as requested.'
    assert _parse_json_block(raw) == {"a": {"b": 2}}


def test_parse_json_block_rejects_non_json():
    with pytest.raises(json.JSONDecodeError):
        _parse_json_block("no braces here at all")


# --------------------------------------------------------------------------- #
# Repair loop (mocked LLM)
# --------------------------------------------------------------------------- #


def test_first_round_success_has_no_repair_warnings():
    fake = FakeClient([GOOD_INVOICE_JSON, JUDGE_ALL_HIGH])
    result = Extractor(client=fake).extract("invoice", "pretend invoice text")
    assert result.rounds == 1
    assert not any("repair" in w for w in result.warnings)
    assert result.data.vendor_name == "Cascada Cloudworks S.A."


def test_repair_loop_recovers_after_unparseable_round():
    fake = FakeClient(["this is not json at all", GOOD_INVOICE_JSON, JUDGE_ALL_HIGH])
    result = Extractor(client=fake).extract("invoice", "pretend invoice text")
    assert result.rounds == 2
    assert any("repair" in w for w in result.warnings)
    # The second extraction call must carry the parse feedback.
    repair_turn = fake.calls[1][-1]
    assert repair_turn["role"] == "user"
    assert "not parseable" in repair_turn["content"]


def test_repair_loop_feeds_validation_errors_back():
    bad = json.dumps(make_invoice_payload(total=9999.0))
    fake = FakeClient([bad, GOOD_INVOICE_JSON, JUDGE_ALL_HIGH])
    result = Extractor(client=fake).extract("invoice", "pretend invoice text")
    assert result.rounds == 2
    repair_prompt = fake.calls[1][-1]["content"]
    assert "failed schema validation" in repair_prompt
    assert "total" in repair_prompt


def test_repair_loop_gives_up_after_max_rounds():
    fake = FakeClient(["nope", "still nope", "absolutely not json"])
    with pytest.raises(ExtractionError, match="3 rounds"):
        Extractor(client=fake).extract("invoice", "pretend invoice text")
    assert len(fake.calls) == 3


def test_unknown_doctype_raises_before_any_llm_call():
    fake = FakeClient([])
    with pytest.raises(UnknownDoctypeError, match="passport"):
        Extractor(client=fake).extract("passport", "text")
    assert fake.calls == []


def test_empty_document_raises():
    fake = FakeClient([])
    with pytest.raises(ExtractionError, match="empty"):
        Extractor(client=fake).extract("invoice", "   \n  ")


# --------------------------------------------------------------------------- #
# Confidence judge (mocked LLM)
# --------------------------------------------------------------------------- #


def test_judge_defaults_missing_fields_and_clamps_out_of_range():
    scores = {name: 0.9 for name in Invoice.model_fields}
    scores.pop("tax")          # judge forgot one field
    scores["total"] = 1.7      # judge over-enthused; clamp to 1.0
    fake = FakeClient([GOOD_INVOICE_JSON, json.dumps(scores)])
    result = Extractor(client=fake).extract("invoice", "text")
    assert result.field_confidence["tax"] == 0.5
    assert result.field_confidence["total"] == 1.0
    assert any("tax" in w for w in result.warnings)


def test_unparseable_judge_reply_defaults_everything_to_neutral():
    fake = FakeClient([GOOD_INVOICE_JSON, "the judge rambles instead of scoring"])
    result = Extractor(client=fake).extract("invoice", "text")
    assert set(result.field_confidence) == set(Invoice.model_fields)
    assert all(v == 0.5 for v in result.field_confidence.values())
    assert result.overall_confidence == 0.5
    assert any("unparseable" in w for w in result.warnings)


def test_low_confidence_fields_generate_warnings():
    scores = {name: 0.95 for name in Invoice.model_fields}
    scores["vendor_tax_id"] = 0.2
    fake = FakeClient([GOOD_INVOICE_JSON, json.dumps(scores)])
    result = Extractor(client=fake).extract("invoice", "text")
    assert any(
        "vendor_tax_id" in w and "low confidence" in w.lower()
        for w in result.warnings
    )


def test_disabling_judge_makes_a_single_llm_call():
    fake = FakeClient([GOOD_INVOICE_JSON])
    result = Extractor(client=fake, score_confidence=False).extract("invoice", "text")
    assert len(fake.calls) == 1
    assert result.overall_confidence == 1.0


# --------------------------------------------------------------------------- #
# Loaders (no network)
# --------------------------------------------------------------------------- #


def test_txt_loader_passes_text_through(tmp_path):
    doc = tmp_path / "note.txt"
    doc.write_text("INVOICE 42\nTotal: 10.00\n", encoding="utf-8")
    assert load_document(doc) == "INVOICE 42\nTotal: 10.00\n"


def test_markdown_loader_passes_text_through(tmp_path):
    doc = tmp_path / "note.md"
    doc.write_text("# Receipt\n\n- coffee 2.50\n", encoding="utf-8")
    assert "coffee 2.50" in load_document(doc)


def test_empty_file_raises(tmp_path):
    doc = tmp_path / "blank.txt"
    doc.write_text("   \n\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="no text"):
        load_document(doc)


def test_missing_file_raises(tmp_path):
    with pytest.raises(LoaderError, match="not found"):
        load_document(tmp_path / "ghost.txt")


def test_unsupported_extension_raises(tmp_path):
    doc = tmp_path / "resume.docx"
    doc.write_bytes(b"PK\x03\x04 pretend docx")
    with pytest.raises(LoaderError, match="unsupported"):
        load_document(doc)


def test_image_without_ocr_flag_raises(tmp_path):
    img = tmp_path / "receipt.png"
    img.write_bytes(b"\x89PNG pretend image bytes")
    with pytest.raises(LoaderError, match="--ocr"):
        load_document(img)


def test_image_with_ocr_calls_vision_client(tmp_path):
    img = tmp_path / "receipt.png"
    img.write_bytes(b"\x89PNG pretend image bytes")
    fake = FakeClient(["MINI SUPER EL VECINO\nTOTAL 22.45"])
    text = load_document(img, ocr=True, client=fake)
    assert text.startswith("MINI SUPER EL VECINO")
    content = fake.calls[0][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
