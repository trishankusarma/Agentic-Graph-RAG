"""
kg/detection/pairing.py

Maps each question entity to the gold article title it refers to, so
comparison questions ("Are X and Y both Z?") split into one chain per side.
Bridge questions don't need this — they only ever have one gold-title chain.
"""

from typing import Optional

from .utils import normalize


class Pairing:
    """3-tier entity → gold-title matcher. Titles consumed greedily."""

    def __init__(self, path_finder):
        self.path_finder = path_finder

    def pair_entities_to_titles(
        self,
        question_entities: list[str],
        gold_titles:       list[str],
    ) -> list[tuple[str, Optional[str]]]:
        """
        Tier 1 — exact string match.
        Tier 2 — substring, either direction.
        Tier 3 — shares a graph neighbor with the title's own entities.

        Each title is matched at most once — a later entity can't steal a
        title an earlier entity already claimed.
        """
        remaining = list(gold_titles)
        pairs: list[tuple[str, Optional[str]]] = []

        for entity in question_entities:
            entity_norm = normalize(entity)
            matched = None

            # tier 1 — exact
            for title in remaining:
                if normalize(title) == entity_norm:
                    matched = title
                    break

            # tier 2 — substring
            if matched is None:
                for title in remaining:
                    title_norm = normalize(title)
                    if entity_norm in title_norm or title_norm in entity_norm:
                        matched = title
                        break

            # tier 3 — graph neighbor
            if matched is None:
                neighbors = self.path_finder.neighbors(entity_norm)
                for title in remaining:
                    if neighbors.intersection(
                        self.path_finder.entities_for_title(title)
                    ):
                        matched = title
                        break

            if matched is not None:
                remaining.remove(matched)
            pairs.append((entity, matched))

        return pairs