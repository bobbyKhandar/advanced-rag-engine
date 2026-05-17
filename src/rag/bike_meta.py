"""Bike identity extraction from first pages of a PDF manual."""

import json
import logging
import re
from typing import Optional

from openai import OpenAI

from src.config.bike_prompts import (
    BIKE_EXTRACTION_SYSTEM_PROMPT,
    BIKE_EXTRACTION_USER_PROMPT,
)

logger = logging.getLogger(__name__)


def _regex_extract(text: str) -> Optional[dict]:
    """Regex fallback for bike info extraction from first-page text."""
    # Pattern: KTM <model> <year> or KTM <year> <model>
    m = re.search(
        r"(KTM)\s+(.+?)\s+(\d{4})\b",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    make = m.group(1).upper()
    model = m.group(2).strip()
    year = m.group(3)
    full_name = f"{make} {model} {year}"
    return {"make": make, "model": model, "year": year, "full_name": full_name}


def extract_bike_info(
    pages_text: str,
    api_key: str,
    base_url: str = "https://api.groq.com/openai/v1",
    model: str = "llama-3.3-70b-versatile",
) -> dict:
    """Extract bike make/model/year from first pages of a manual.

    Uses LLM first, falls back to regex if the LLM call fails.
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": BIKE_EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": BIKE_EXTRACTION_USER_PROMPT.format(
                        pages_text=pages_text[:4000]
                    ),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        result = json.loads(raw)
        if not result.get("make") and not result.get("model"):
            raise ValueError("LLM returned empty bike info")
        logger.info(
            "Extracted bike info via LLM: %s %s %s",
            result.get("make"),
            result.get("model"),
            result.get("year", ""),
        )
        return result
    except Exception as exc:
        logger.warning("LLM bike extraction failed (%s), trying regex fallback", exc)
        result = _regex_extract(pages_text)
        if result:
            logger.info(
                "Extracted bike info via regex: %s %s %s",
                result["make"],
                result["model"],
                result["year"],
            )
            return result
        logger.warning("Regex fallback also failed, returning empty bike info")
        return {"make": "", "model": "", "year": None, "full_name": ""}


def decompose_query(
    question: str,
    api_key: str,
    base_url: str = "https://api.groq.com/openai/v1",
    model: str = "llama-3.3-70b-versatile",
) -> dict:
    """Extract bike info and clean question from a user query.

    Returns {"make": ..., "model": ..., "question": ...}
    """
    from src.config.bike_prompts import (
        BIKE_QUERY_DECOMPOSE_SYSTEM_PROMPT,
        BIKE_QUERY_DECOMPOSE_USER_PROMPT,
    )

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": BIKE_QUERY_DECOMPOSE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": BIKE_QUERY_DECOMPOSE_USER_PROMPT.format(
                        question=question
                    ),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        result = json.loads(raw)
        logger.info(
            "Decomposed query: make=%s model=%s question=%.50s",
            result.get("make"),
            result.get("model"),
            result.get("question", ""),
        )
        return result
    except Exception as exc:
        logger.warning("Query decomposition failed: %s", exc)
        return {"make": None, "model": None, "question": question}
