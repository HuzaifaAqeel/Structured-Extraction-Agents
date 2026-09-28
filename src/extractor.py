"""Schema-guided extraction with self-repair and confidence scoring.

The engine turns a Pydantic model into an extraction prompt, asks a NIM-hosted
LLM for JSON, validates the reply against the model, and feeds validation
errors back to the LLM for up to three repair rounds. A second, cheaper judge
pass scores per-field confidence so callers can decide what to trust and what
to route to a human.

The chat client is injected (see ``ChatClient``), so the whole pipeline is
testable without a network connection -- ``tests/test_extractor.py`` drives it
with a scripted fake.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Optional, Protocol

from pydantic import BaseModel, ValidationError

from src.schemas import DOCTYPES

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
DEFAULT_JUDGE_MODEL = "meta/llama-3.1-8b-instruct"
MAX_REPAIR_ROUNDS = 3
LOW_CONFIDENCE = 0.6

FRIENDLY_KEY_MESSAGE = """\
NVIDIA_API_KEY is not set.

Getting a free key takes about two minutes:
  1. Go to https://build.nvidia.com and sign up (free account, no card needed).
  2. Open any model page and click "Get API Key".
  3. Copy the key -- it starts with `nvapi-`.
  4. Put it in a .env file in the repo root:
       NVIDIA_API_KEY=nvapi-XXXXXXXXXXXXXXXXXXXXXXXX
"""


class MissingAPIKeyError(RuntimeError):
    """NVIDIA_API_KEY is absent from the environment."""


class UnknownDoctypeError(ValueError):
    """The requested doctype has no registered schema."""


class ExtractionError(RuntimeError):
    """The model could not produce valid JSON within the repair budget."""


class ChatClient(Protocol):
    """Anything that can answer a chat conversation with a string.

    ``messages`` follows the OpenAI chat format. ``content`` may be a plain
    string or a list of content parts (used by the vision/OCR path).
    """

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        ...


class NIMChat:
    """Thin wrapper over the OpenAI SDK pointed at NVIDIA NIM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = NIM_BASE_URL,
        default_model: Optional[str] = None,
    ) -> None:
        key = (api_key or os.environ.get("NVIDIA_API_KEY", "")).strip()
        if not key:
            raise MissingAPIKeyError(FRIENDLY_KEY_MESSAGE)
        from openai import OpenAI  # local import keeps offline tests dependency-free

        self._client = OpenAI(api_key=key, base_url=base_url)
        self.default_model = default_model or os.environ.get("NIM_MODEL", DEFAULT_MODEL)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


@dataclass
class ExtractionResult:
    """What the engine hands back: validated data plus trust metadata."""

    data: BaseModel
    field_confidence: dict[str, float]
    overall_confidence: float
    rounds: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": self.data.model_dump(mode="json"),
            "confidence": {
                "overall": self.overall_confidence,
                "fields": self.field_confidence,
            },
            "rounds": self.rounds,
            "warnings": self.warnings,
        }


SYSTEM_PROMPT = (
    "You are a document extraction engine. You read a document and return a "
    "single JSON object that conforms exactly to the provided JSON Schema. "
    "Rules: output raw JSON only -- no markdown fences, no commentary; use "
    "ISO 8601 dates (YYYY-MM-DD); use null for optional fields the document "
    "does not state; never invent values that are not supported by the "
    "document text; copy names and identifiers verbatim."
)

JUDGE_SYSTEM = (
    "You audit data extracted from a document. For each field you are asked "
    "about, return a confidence between 0.0 and 1.0 that the extracted value "
    "is correct and actually supported by the document. 1.0 = stated "
    "verbatim; 0.5 = plausible but indirect; 0.0 = contradicted or absent. "
    "Return raw JSON only: an object mapping field names to numbers."
)


def build_extraction_prompt(model_cls: type[BaseModel], document_text: str) -> str:
    """Embed the JSON Schema (with field descriptions) next to the document.

    The descriptions in ``src/schemas.py`` are written as instructions, so the
    schema itself carries most of the prompt engineering.
    """
    schema = json.dumps(model_cls.model_json_schema(), indent=2)
    return (
        "JSON Schema (the field descriptions tell you exactly what belongs in "
        "each field):\n"
        f"{schema}\n\n"
        "Document:\n"
        f"<<<\n{document_text}\n>>>\n\n"
        "Return the JSON object now."
    )


def _parse_json_block(raw: str) -> Any:
    """Pull a JSON object out of an LLM reply.

    Tolerates markdown fences and stray prose around the object, because even
    at temperature 0 models occasionally wrap their answer.
    """
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("no JSON object found in reply", raw, 0)
    return json.loads(text[start : end + 1])


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"- {loc}: {err['msg']}")
    return "\n".join(lines)


def _repair_message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        detail = _format_validation_error(exc)
        return (
            "Your JSON failed schema validation with these errors:\n"
            f"{detail}\n\n"
            "Fix every error and return the corrected JSON object. Re-read the "
            "document before changing numeric fields -- the validators "
            "cross-check totals, so recompute them from the line items rather "
            "than guessing. Output raw JSON only."
        )
    return (
        f"Your reply was not parseable as JSON ({exc}). Return only the JSON "
        "object -- no markdown fences, no explanation, no trailing text."
    )


class Extractor:
    """Doctype-aware extraction engine with a bounded self-repair loop."""

    def __init__(
        self,
        client: Optional[ChatClient] = None,
        *,
        model: Optional[str] = None,
        judge_model: Optional[str] = None,
        score_confidence: bool = True,
    ) -> None:
        self.client: ChatClient = client if client is not None else NIMChat()
        self.model = model or os.environ.get("NIM_MODEL", DEFAULT_MODEL)
        self.judge_model = judge_model or os.environ.get(
            "NIM_JUDGE_MODEL", DEFAULT_JUDGE_MODEL
        )
        self.score_confidence = score_confidence

    def extract(self, doctype: str, document_text: str) -> ExtractionResult:
        """Extract ``doctype`` data from ``document_text``.

        Raises ``UnknownDoctypeError`` for unregistered doctypes and
        ``ExtractionError`` when the model cannot satisfy the schema within
        ``MAX_REPAIR_ROUNDS`` rounds.
        """
        try:
            model_cls = DOCTYPES[doctype]
        except KeyError:
            raise UnknownDoctypeError(
                f"unknown doctype '{doctype}'; expected one of: "
                f"{', '.join(sorted(DOCTYPES))}"
            ) from None
        if not document_text or not document_text.strip():
            raise ExtractionError("document text is empty; nothing to extract")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_extraction_prompt(model_cls, document_text)},
        ]
        last_error: Optional[Exception] = None

        for round_num in range(1, MAX_REPAIR_ROUNDS + 1):
            raw = self.client.chat(messages, model=self.model, temperature=0.0)
            try:
                payload = _parse_json_block(raw)
                instance = model_cls.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": _repair_message(exc)})
                continue

            warnings: list[str] = []
            if round_num > 1:
                warnings.append(
                    f"output required {round_num - 1} repair round(s) before validating"
                )

            if self.score_confidence:
                field_conf, judge_warnings = self._judge_confidence(
                    model_cls, document_text, instance
                )
                warnings.extend(judge_warnings)
            else:
                field_conf = {name: 1.0 for name in model_cls.model_fields}
                warnings.append(
                    "confidence scoring disabled; per-field confidences are placeholders"
                )

            overall = round(fmean(field_conf.values()), 3) if field_conf else 0.0
            for name, score in sorted(field_conf.items()):
                if score < LOW_CONFIDENCE:
                    warnings.append(
                        f"low confidence on field '{name}' ({score:.2f}); verify manually"
                    )

            return ExtractionResult(
                data=instance,
                field_confidence=field_conf,
                overall_confidence=overall,
                rounds=round_num,
                warnings=warnings,
            )

        summary = str(last_error)
        if isinstance(last_error, ValidationError):
            summary = _format_validation_error(last_error)
        raise ExtractionError(
            f"model output failed validation after {MAX_REPAIR_ROUNDS} rounds; "
            f"last errors:\n{summary}"
        )

    def _judge_confidence(
        self,
        model_cls: type[BaseModel],
        document_text: str,
        instance: BaseModel,
    ) -> tuple[dict[str, float], list[str]]:
        """Second pass with a cheaper model: score each top-level field.

        The judge never blocks a result -- unparseable or partial replies
        degrade to a neutral 0.5 with an explicit warning.
        """
        field_names = list(model_cls.model_fields)
        prompt = (
            "Document:\n"
            f"<<<\n{document_text}\n>>>\n\n"
            "Extracted data:\n"
            f"{instance.model_dump_json(indent=2)}\n\n"
            f"Score these fields: {', '.join(field_names)}\n"
            'Return JSON like {"field_name": 0.95, ...} with one entry per field.'
        )
        warnings: list[str] = []
        raw = self.client.chat(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            model=self.judge_model,
            temperature=0.0,
            max_tokens=1024,
        )
        try:
            scores = _parse_json_block(raw)
            if not isinstance(scores, dict):
                raise json.JSONDecodeError("judge reply is not a JSON object", raw, 0)
        except json.JSONDecodeError:
            warnings.append(
                "confidence judge reply was unparseable; defaulted every field to 0.5"
            )
            return {name: 0.5 for name in field_names}, warnings

        confidence: dict[str, float] = {}
        for name in field_names:
            value = scores.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                confidence[name] = round(min(max(float(value), 0.0), 1.0), 3)
            else:
                confidence[name] = 0.5
                warnings.append(f"judge did not score field '{name}'; defaulted to 0.5")
        return confidence, warnings
