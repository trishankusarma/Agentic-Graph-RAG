"""
kg/text.py

String normalization used everywhere an entity label, article title or answer
is compared.
"""


def normalize(label: str) -> str:
    """Lowercase and strip. The canonical form for every graph node id."""
    return label.lower().strip()