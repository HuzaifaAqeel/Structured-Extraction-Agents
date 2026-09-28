"""Pydantic v2 schemas for the four supported document types.

Field descriptions double as micro-prompts: the extraction engine embeds the
JSON Schema (descriptions included) directly in the LLM prompt, so every
description below is written as an instruction to the model, not just as
documentation. Validators encode business rules -- totals that must add up,
dates that must be ordered, parties that must exist -- so bad extractions
fail fast and trigger the repair loop instead of silently polluting your data.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Absolute drift allowed when cross-checking money math. Covers rounding on
# per-line tax and unit prices printed with two decimals.
MONEY_TOLERANCE = 0.05


# --------------------------------------------------------------------------- #
# Invoice
# --------------------------------------------------------------------------- #


class InvoiceLineItem(BaseModel):
    description: str = Field(
        ..., description="The line item text exactly as written on the invoice."
    )
    quantity: float = Field(
        1.0,
        ge=0,
        description="Units billed. Use 1 when the invoice does not state a quantity.",
    )
    unit_price: float = Field(..., description="Price per single unit, before tax.")
    amount: float = Field(
        ..., description="Line total. Must equal quantity * unit_price."
    )

    @model_validator(mode="after")
    def amount_matches_quantity_times_price(self) -> "InvoiceLineItem":
        expected = round(self.quantity * self.unit_price, 2)
        if abs(expected - round(self.amount, 2)) > MONEY_TOLERANCE:
            raise ValueError(
                f"amount {self.amount} does not equal quantity ({self.quantity}) "
                f"x unit_price ({self.unit_price}) = {expected}"
            )
        return self


class Invoice(BaseModel):
    """A commercial invoice. Totals are cross-checked against the line items."""

    vendor_name: str = Field(
        ..., description="Legal name of the company issuing the invoice."
    )
    vendor_tax_id: Optional[str] = Field(
        None, description="Vendor tax / VAT / RUC identifier if printed, else null."
    )
    invoice_number: str = Field(
        ...,
        description="The invoice identifier exactly as printed, e.g. 'INV-2026-0042'.",
    )
    issue_date: date = Field(
        ..., description="Date the invoice was issued, ISO 8601 (YYYY-MM-DD)."
    )
    due_date: Optional[date] = Field(
        None, description="Payment due date if stated, ISO 8601, else null."
    )
    currency: str = Field(
        ...,
        description="Three-letter ISO 4217 code, e.g. 'USD'. Infer from currency symbols or country context when not spelled out.",
    )
    line_items: list[InvoiceLineItem] = Field(
        ..., min_length=1, description="Every billable line on the invoice."
    )
    subtotal: float = Field(
        ..., description="Sum of all line amounts before tax."
    )
    tax: float = Field(
        0.0, ge=0, description="Total tax charged. Use 0 when no tax appears."
    )
    total: float = Field(..., description="Grand total due: subtotal + tax.")

    @field_validator("currency")
    @classmethod
    def currency_is_iso4217(cls, value: str) -> str:
        code = value.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", code):
            raise ValueError(
                f"'{value}' is not a three-letter ISO 4217 currency code"
            )
        return code

    @model_validator(mode="after")
    def totals_cross_check(self) -> "Invoice":
        line_sum = round(sum(item.amount for item in self.line_items), 2)
        if abs(line_sum - round(self.subtotal, 2)) > MONEY_TOLERANCE:
            raise ValueError(
                f"subtotal {self.subtotal} does not match the sum of line "
                f"amounts ({line_sum}); re-read the line items"
            )
        expected_total = round(self.subtotal + self.tax, 2)
        if abs(expected_total - round(self.total, 2)) > MONEY_TOLERANCE:
            raise ValueError(
                f"total {self.total} does not equal subtotal + tax ({expected_total})"
            )
        if self.due_date is not None and self.due_date < self.issue_date:
            raise ValueError(
                f"due_date {self.due_date} is before issue_date {self.issue_date}"
            )
        return self


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #

# Canonical spellings for skills the model (or the candidate) tends to
# abbreviate. Keys are compared lowercase after stripping.
SKILL_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "python3": "python",
    "golang": "go",
    "node": "node.js",
    "nodejs": "node.js",
    "reactjs": "react",
    "react.js": "react",
    "vuejs": "vue",
    "vue.js": "vue",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "tf": "terraform",
    "gcp": "google cloud",
    "ml": "machine learning",
}

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ExperienceEntry(BaseModel):
    company: str = Field(..., description="Employer name.")
    title: str = Field(..., description="Job title held at this employer.")
    start_date: date = Field(
        ...,
        description="Start of the role, ISO 8601. Use the first day of the month when only month/year is given.",
    )
    end_date: Optional[date] = Field(
        None,
        description="End of the role, ISO 8601. Null when the role is current ('present').",
    )
    highlights: list[str] = Field(
        default_factory=list,
        description="Notable achievements or responsibilities, one string each.",
    )

    @model_validator(mode="after")
    def dates_are_ordered(self) -> "ExperienceEntry":
        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError(
                f"end_date {self.end_date} is before start_date {self.start_date}"
            )
        return self


class Resume(BaseModel):
    """A candidate resume. Skills are normalized; experience is sorted newest-first."""

    full_name: str = Field(
        ..., description="Candidate's full name as written at the top of the resume, without nicknames."
    )
    email: Optional[str] = Field(
        None, description="Contact email address if present, else null."
    )
    phone: Optional[str] = Field(
        None, description="Contact phone number if present, else null."
    )
    location: Optional[str] = Field(
        None, description="City and/or country if stated, else null."
    )
    summary: Optional[str] = Field(
        None, description="The professional summary paragraph, if the resume has one."
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Technical and professional skills, one skill per entry.",
    )
    experience: list[ExperienceEntry] = Field(
        ..., min_length=1, description="Every job listed on the resume."
    )
    education: list[str] = Field(
        default_factory=list,
        description="Degrees and certifications, one string each, e.g. 'BSc Computer Science, UTP, 2019'.",
    )

    @field_validator("email")
    @classmethod
    def email_looks_valid(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        candidate = value.strip()
        if not _EMAIL_RE.match(candidate):
            raise ValueError(f"'{value}' does not look like an email address")
        return candidate.lower()

    @field_validator("skills")
    @classmethod
    def normalize_skills(cls, values: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for raw in values:
            skill = raw.strip().lower()
            if not skill:
                continue
            skill = SKILL_ALIASES.get(skill, skill)
            seen.setdefault(skill, None)
        return list(seen)

    @model_validator(mode="after")
    def sort_experience_newest_first(self) -> "Resume":
        self.experience.sort(
            key=lambda entry: (entry.end_date or date.max, entry.start_date),
            reverse=True,
        )
        return self


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


class ContractParty(BaseModel):
    name: str = Field(
        ..., description="Legal name of the party exactly as written in the contract."
    )
    role: str = Field(
        ...,
        description="The party's role label in the contract, e.g. 'Client', 'Contractor', 'Licensor', 'the Company'.",
    )


class Obligation(BaseModel):
    party: str = Field(
        ...,
        description="Name or role of the party that owes this obligation. Must match one of the contract's parties.",
    )
    description: str = Field(
        ..., description="What the party must do, stated in one sentence."
    )
    deadline: Optional[date] = Field(
        None,
        description="Deadline for the obligation if the contract states one, ISO 8601, else null.",
    )


class Contract(BaseModel):
    """A commercial agreement: who signed, when it runs, who owes what, which law governs."""

    title: str = Field(
        ..., description="The contract's title or heading, e.g. 'Master Services Agreement'."
    )
    parties: list[ContractParty] = Field(
        ..., min_length=2, description="Every party to the agreement."
    )
    effective_date: date = Field(
        ..., description="Date the agreement takes effect, ISO 8601."
    )
    termination_date: Optional[date] = Field(
        None,
        description="Fixed end date if the contract has one, else null (e.g. indefinite or auto-renewing terms).",
    )
    obligations: list[Obligation] = Field(
        default_factory=list,
        description="Concrete duties the contract imposes, one per entry.",
    )
    governing_law: str = Field(
        ...,
        description="Jurisdiction whose law governs, e.g. 'Republic of Panama' or 'State of Delaware, USA'.",
    )
    payment_terms: Optional[str] = Field(
        None, description="Payment amount and schedule in one sentence, if stated."
    )
    auto_renews: Optional[bool] = Field(
        None,
        description="True if the contract renews automatically, false if it explicitly does not, null if unstated.",
    )

    @model_validator(mode="after")
    def dates_and_parties_are_consistent(self) -> "Contract":
        if (
            self.termination_date is not None
            and self.termination_date <= self.effective_date
        ):
            raise ValueError(
                f"termination_date {self.termination_date} is not after "
                f"effective_date {self.effective_date}"
            )
        known = {p.name.strip().lower() for p in self.parties} | {
            p.role.strip().lower() for p in self.parties
        }
        for obligation in self.obligations:
            owner = obligation.party.strip().lower()
            if owner not in known and not any(
                owner in name or name in owner for name in known
            ):
                raise ValueError(
                    f"obligation party '{obligation.party}' does not match any "
                    f"contract party (parties: {sorted(known)})"
                )
        return self


# --------------------------------------------------------------------------- #
# Receipt
# --------------------------------------------------------------------------- #

PAYMENT_METHODS = ("cash", "credit_card", "debit_card", "mobile", "other")

_PAYMENT_ALIASES = {
    "visa": "credit_card",
    "mastercard": "credit_card",
    "amex": "credit_card",
    "american express": "credit_card",
    "credit": "credit_card",
    "credit card": "credit_card",
    "cc": "credit_card",
    "tarjeta": "credit_card",
    "debit": "debit_card",
    "debit card": "debit_card",
    "apple pay": "mobile",
    "google pay": "mobile",
    "yappy": "mobile",
    "efectivo": "cash",
}

_TIME_RE = re.compile(r"^(?:[01]?\d|2[0-3]):[0-5]\d$")


class ReceiptItem(BaseModel):
    name: str = Field(
        ...,
        description="Item name as printed, expanded from abbreviations when the expansion is obvious.",
    )
    quantity: float = Field(
        1.0, gt=0, description="Units bought. Use 1 when not printed."
    )
    price: float = Field(
        ..., ge=0, description="Line total for this item (not the unit price)."
    )


class Receipt(BaseModel):
    """A point-of-sale receipt. The total is cross-checked when a subtotal is present."""

    merchant_name: str = Field(
        ..., description="Store or restaurant name at the top of the receipt."
    )
    merchant_address: Optional[str] = Field(
        None, description="Merchant address if printed, else null."
    )
    purchase_date: date = Field(..., description="Date of purchase, ISO 8601.")
    purchase_time: Optional[str] = Field(
        None, description="Time of purchase as 'HH:MM' (24-hour) if printed, else null."
    )
    items: list[ReceiptItem] = Field(
        ..., min_length=1, description="Every purchased item on the receipt."
    )
    subtotal: Optional[float] = Field(
        None, description="Pre-tax subtotal if printed, else null."
    )
    tax: Optional[float] = Field(
        None, description="Tax amount if printed, else null."
    )
    tip: Optional[float] = Field(
        None, description="Tip / service charge if printed, else null."
    )
    total: float = Field(..., description="Final amount paid.")
    payment_method: str = Field(
        "other",
        description="One of: cash, credit_card, debit_card, mobile, other. Card brand names count as credit_card.",
    )

    @field_validator("purchase_time")
    @classmethod
    def time_is_hh_mm(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        candidate = value.strip()
        if not _TIME_RE.match(candidate):
            raise ValueError(f"'{value}' is not a valid 24-hour HH:MM time")
        hours, minutes = candidate.split(":")
        return f"{int(hours):02d}:{minutes}"

    @field_validator("payment_method", mode="before")
    @classmethod
    def normalize_payment_method(cls, value: object) -> str:
        if value is None:
            return "other"
        method = str(value).strip().lower().replace("-", " ")
        method = _PAYMENT_ALIASES.get(method, method)
        if method not in PAYMENT_METHODS:
            return "other"
        return method

    @model_validator(mode="after")
    def total_cross_check(self) -> "Receipt":
        item_sum = round(sum(item.price for item in self.items), 2)
        if self.subtotal is not None:
            if abs(item_sum - round(self.subtotal, 2)) > MONEY_TOLERANCE:
                raise ValueError(
                    f"subtotal {self.subtotal} does not match the sum of item "
                    f"prices ({item_sum})"
                )
            expected = round(
                self.subtotal + (self.tax or 0.0) + (self.tip or 0.0), 2
            )
            if abs(expected - round(self.total, 2)) > MONEY_TOLERANCE:
                raise ValueError(
                    f"total {self.total} does not equal subtotal + tax + tip "
                    f"({expected})"
                )
        return self


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

DOCTYPES: dict[str, type[BaseModel]] = {
    "invoice": Invoice,
    "resume": Resume,
    "contract": Contract,
    "receipt": Receipt,
}


def get_schema(doctype: str) -> dict:
    """JSON Schema for one doctype, via Pydantic's model_json_schema()."""
    try:
        return DOCTYPES[doctype].model_json_schema()
    except KeyError:
        raise KeyError(
            f"unknown doctype '{doctype}'; expected one of: {', '.join(sorted(DOCTYPES))}"
        ) from None


def schema_catalog() -> dict[str, dict]:
    """JSON Schemas for every doctype, keyed by name. Served by GET /schemas."""
    return {name: model.model_json_schema() for name, model in DOCTYPES.items()}
