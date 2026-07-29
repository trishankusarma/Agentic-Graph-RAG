"""
Graph-R1/agent/tool/tools/repair_tool.py

The repair action: when the agent hits a dead end in the graph, re-read the source
passage conditioned on what it is trying to reach, and insert any genuinely new fact.
"""

import json
from typing import Dict, List

from agent.tool.tool_base import Tool

from kg.hypergraph_builder import add_fact_to_graph


class GraphRepairTool(Tool):
    """
    Args:
        state:                  Episode-scoped mutable graph state. MUST be the same
                                 object the episode's search tool holds — that shared
                                 reference is what makes a repair visible to the next
                                 search call within the same episode. Repairs do not
                                 persist across episodes: each episode deep-copies its
                                 own graph, so a repair made for question A is gone
                                 before question B starts. What transfers between
                                 questions is the POLICY (when to reach for repair),
                                 not the repaired edge itself.
        max_new_facts_per_call: Cap on edges one repair call can add. Bounds the blast
                                 radius of a single action, same motivation as
                                 MAX_FACT_ARITY bounding clique expansion.
        max_chunks_per_call:    Cap on source passages re-read per call. An entity
                                 appearing in many chunks would otherwise make one
                                 repair action cost many LLM calls.
    """

    def __init__(
        self,
        state,
        max_new_facts_per_call: int = 4,
        max_chunks_per_call: int = 3,
    ):
        super().__init__(
            name="repair",
            description=(
                "When you cannot find a connection you need in the knowledge graph, "
                "re-read the source text to look for it. Give the entity you are "
                "reasoning from, what you are trying to reach, and why."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "src": {
                        "type": "string",
                        "description": "The entity you are reasoning from",
                    },
                    "dst": {
                        "type": "string",
                        "description": (
                            "What you are trying to reach. May be a description "
                            "rather than a name if you do not know the name yet "
                            "(e.g. 'the film's director')."
                        ),
                    },
                    "goal": {
                        "type": "string",
                        "description": "What you are trying to establish, in one line",
                    },
                },
                "required": ["src", "dst"],
            },
        )
        self.state = state
        self.max_new_facts_per_call = max_new_facts_per_call
        self.max_chunks_per_call = max_chunks_per_call

    def validate_args(self, args: Dict):
        if not isinstance(args, dict):
            return False, "args must be an object"
        for key in ("src", "dst"):
            if key not in args or not str(args[key]).strip():
                return False, f"Missing required parameter: {key}"
        return True, ""

    def execute(self, args: Dict) -> str:
        src  = str(args["src"]).strip()
        dst  = str(args["dst"]).strip()
        goal = str(args.get("goal", "")).strip()

        # Resolve src to real nodes so we know WHICH passages to re-read. dst is
        # deliberately NOT resolved — it is frequently a description of something the
        # agent cannot name yet, which is often the whole reason it is stuck.
        resolved = self.state.path_finder.resolve_entity(src)
        if not resolved:
            return json.dumps({
                "repaired": False,
                "reason": f"'{src}' does not appear in this question's source text.",
            })

        chunk_ids: set[str] = set()
        for entity_id in resolved:
            node = self.state.graph.nodes.get(entity_id)
            if node:
                chunk_ids.update(node.chunks)

        chunks = [
            self.state.chunks_by_id[cid]
            for cid in sorted(chunk_ids)          # sorted: deterministic across rollouts
            if cid in self.state.chunks_by_id
        ][: self.max_chunks_per_call]

        if not chunks:
            return json.dumps({
                "repaired": False,
                "reason": "found the entity but no source passage to re-read.",
            })

        new_edges = []
        for chunk in chunks:
            try:
                facts = self.state.extractor.extract_targeted(
                    chunk, src=src, dst=dst, goal=goal,
                )
            except Exception:
                continue   # a failed repair is a no-op, never a crashed rollout

            for fact in facts:
                edge = add_fact_to_graph(self.state.graph, fact, chunk)
                if edge is not None:      # None => degenerate, or already known
                    new_edges.append(edge)
                    if len(new_edges) >= self.max_new_facts_per_call:
                        break
            if len(new_edges) >= self.max_new_facts_per_call:
                break

        if not new_edges:
            # Expected and legitimate: the prompt explicitly instructs the model to
            # return [] rather than invent an edge when the passage does not support
            # one. Reported plainly so the agent can move on instead of retrying.
            return json.dumps({
                "repaired": False,
                "reason": "re-read the source text; no new connection found there.",
            })

        self.state.invalidate_edge_index()
        self.state.repairs_made.append({
            "src": src, "dst": dst, "goal": goal,
            "new_facts": [
                {"relation": e.relation, "entities": e.entities} for e in new_edges
            ],
        })
        return json.dumps({
            "repaired": True,
            "new_facts": [f"{e.entities} — {e.relation}" for e in new_edges],
        })

    def batch_execute(self, args_list: List[Dict]) -> List[str]:
        # Sequential on purpose. Each episode has its own EpisodeGraphState, so there
        # is no cross-trajectory race — but within one trajectory these calls mutate
        # the same graph object and must not interleave.
        return [self.execute(a) for a in args_list]

    def calculate_reward(self, args: Dict, result: str) -> float:
        # 0.0 matches SearchTool's own convention in this codebase. Graph-R1's reward
        # (verl/utils/reward_score/qa_em_and_format.py) only ever inspects tag
        # structure and the final answer — intermediate tool identity cannot affect
        # it — so GRPO's group-relative advantage is what reinforces useful repairs.
        #
        # Whether that sparse signal is sufficient for an action this rare is an open
        # question, not a settled design choice: ProGraph-R1's ablation shows dense
        # process-level reward beating sparse outcome reward for RETRIEVAL steps
        # (60.47 vs 58.45 F1 on 2Wiki). If repair proves hard to learn, this is the
        # first place to intervene — and the detector's gold-based verdict is
        # available at TRAINING time to shape it, since the reward function is
        # allowed to see gold even though the policy is not.
        return 0.0