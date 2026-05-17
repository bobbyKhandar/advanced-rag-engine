"""Prompt templates for bike metadata extraction and query decomposition."""

BIKE_EXTRACTION_SYSTEM_PROMPT = (
    "You extract motorcycle identity information from manual page content. "
    "Return valid JSON only with these fields:\n"
    '  "make" — manufacturer (e.g. KTM, Honda, Yamaha)\n'
    '  "model" — full model name (e.g. "350 SX-F", "RC 125", "250 Duke")\n'
    '  "year" — model year (4-digit number or null)\n'
    '  "full_name" — combined make + model + year (e.g. "KTM RC 125 2024")\n\n'
    "If you cannot determine a field, set it to null."
)

BIKE_EXTRACTION_USER_PROMPT = (
    "Extract the motorcycle identity from these manual pages:\n\n"
    "{pages_text}"
)

BIKE_QUERY_DECOMPOSE_SYSTEM_PROMPT = (
    "You extract motorcycle search intent from a user question. "
    "Return valid JSON only with these fields:\n"
    '  "make" — manufacturer (e.g. KTM, Honda) or null if not mentioned\n'
    '  "model" — model name (e.g. "RC 125", "350 SX-F", "Duke 250") or null\n'
    '  "question" — the actual question without the bike preamble\n\n'
    "Handle partial/fuzzy matches — 'ktm 125' refers to model 'RC 125', "
    "'ktm 350' refers to '350 SX-F', etc. "
    "If no bike is mentioned, set make and model to null "
    "and return the full question as-is."
)

BIKE_QUERY_DECOMPOSE_USER_PROMPT = (
    "Decompose this question into bike info and search question:\n\n"
    "{question}"
)
