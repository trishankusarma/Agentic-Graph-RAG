"""
kg/detection/chain_builder.py

Stitches src -> gold[0] -> ... -> gold[n] and hands off the terminal hop.
Owns graph-connectivity policy: what counts as "reachable" between two
waypoints, and when a broken segment gets healed by skipping ahead.
"""

from kg.data_loader import HotpotSample
from kg.reasoning_models.chains import Chain, Segment
from kg.reasoning_models.graph_models import PathResult

from .path_policy import best_path


class ChainBuilder:
    def __init__(self, path_finder, terminal):
        self.path_finder = path_finder
        self.terminal     = terminal   # a TerminalResolver

    def build(
        self,
        sample:      HotpotSample,
        src:         str,
        gold_titles: list[str],
    ) -> Chain:
        """
        Stitch src -> gold[0] -> ... -> gold[n], then resolve the terminal.
        On segment failure, tries one skip-ahead fallback.
        """
        resolved_src = self.path_finder.resolve_entity(src)
        src_ents = resolved_src or {src}
        waypoints: list[tuple[str, set[str]]] = [("src", src_ents)]

        for title in gold_titles:
            waypoints.append((title, self.path_finder.entities_for_title(title)))

        segments: list[Segment] = []
        i = 0
        while i < len(waypoints) - 1:
            from_node, from_ents = waypoints[i]
            to_node,   to_ents   = waypoints[i + 1]

            seg = self._segment_between(from_node, from_ents, to_node, to_ents)

            if seg.is_broken and i + 2 < len(waypoints):
                skip_node, skip_ents = waypoints[i + 2]
                fallback = self._segment_between(
                    from_node, from_ents, skip_node, skip_ents
                )
                fallback.is_fallback = True

                if not fallback.is_broken:
                    segments.append(seg)       # record the break
                    segments.append(fallback)  # record the successful skip
                    i += 2
                    continue

            segments.append(seg)
            i += 1

        if len(waypoints) > 1:
            reached_final = bool(segments) and not segments[-1].is_broken
            last_ents = waypoints[-1][1] if reached_final else set()
        else:
            last_ents = src_ents

        terminal = self.terminal.resolve(sample, last_ents, gold_titles)

        return Chain(segments=segments, terminal=terminal)

    def _segment_between(
        self,
        from_node: str,
        from_ents: set[str],
        to_node:   str,
        to_ents:   set[str],
    ) -> Segment:
        """Shortest clean path between two waypoint entity sets."""
        if not from_ents or not to_ents:
            return self.broken_segment(from_node, to_node)

        best = best_path(self.path_finder, from_ents, to_ents)
        if best is None:
            f0 = next(iter(from_ents))
            t0 = next(iter(to_ents))
            return Segment(
                from_node=from_node, to_node=to_node,
                path_result=PathResult(src=f0, dst=t0, hops=[], found=False),
                is_broken=True,
            )

        return Segment(
            from_node=from_node, to_node=to_node,
            path_result=best, is_broken=False,
        )

    @staticmethod
    def broken_segment(from_node: str, to_node: str) -> Segment:
        """
        Public (was `_broken_segment`): detector.py also needs this, for the
        no-question-entities / no-gold-titles cases where there's no waypoint
        to build a real chain from at all.
        """
        return Segment(
            from_node=from_node, to_node=to_node,
            path_result=PathResult(src="", dst="", hops=[], found=False),
            is_broken=True,
        )