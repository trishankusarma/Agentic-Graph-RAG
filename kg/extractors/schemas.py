"""
kg/extractors/schemas.py

JSON Schemas describing extractor output.
"""

from .config import MAX_FACT_ARITY, MIN_FACT_ARITY

FACT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": MIN_FACT_ARITY,
                "maxItems": MAX_FACT_ARITY,
            },
            "relation":       {"type": "string"},
            "sentence_index": {"type": "integer", "minimum": 0},
            "confidence":     {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["entities", "relation", "sentence_index", "confidence"],
        "additionalProperties": False,
    },
}

ENTITY_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
}