"""
kg/extractors/prompts.py

All prompt text, isolated so prompt iteration is a single-file edit and the
history of what has already been tried does not get lost.
"""

from .config import MAX_FACT_ARITY, MAX_FACTS_PER_SENTENCE

EXTRACTION_SYSTEM_PROMPT = f"""You are a Knowledge Graph extraction engine.

You will receive a passage where every sentence is prefixed with a sentence index.

Extract every relational fact from the passage.
Treat every numbered sentence independently.
Never combine entities or information from different sentence indices.

For each fact return:
- "entities":       A list of two or more participating entities.
- "relation":       A concise snake_case relation label.
- "sentence_index": The integer index of the sentence this fact came from.
- "confidence":     A float 0-1 representing extraction confidence.

Rules for "relation":
- Derive the label from the main verb or predicate of the sentence you are
  extracting from. Build it out of words that actually appear in that sentence.
- Two sentences that express different things must not share a label.

Rules for "entities":
- Use proper nouns, named concepts, dates, or quantities.
- Keep surface forms exactly as they appear in the sentence.
- A fact may hold a subject, an object, and one or two qualifiers such as a
  date or a place. NEVER put more than {MAX_FACT_ARITY} entities in a single
  fact. If a sentence links more, split it into several separate facts.
- Do not repeat the same entity twice within one fact.

Rules for how much to extract:
- Emit at most {MAX_FACTS_PER_SENTENCE} facts per sentence. Prefer the most
  specific and informative facts. Do NOT enumerate every possible pairing of
  entities in a sentence.
- Every fact must originate from exactly one sentence.
- Do not invent facts or merge information across sentences.
- Do not output generic facts such as "is a person" or "exists".
- Return ONLY a JSON array, no explanation, no markdown fences.

Output format (the angle-bracket values below are PLACEHOLDERS showing shape
only — never emit them literally, and never treat them as example labels):

[
  {{
    "entities": ["<entity_from_sentence>", "<another_entity>", "<a_date_or_place>"],
    "relation": "<snake_case_verb_from_that_sentence>",
    "sentence_index": 0,
    "confidence": 0.95
  }}
]

Every value you emit must come from the passage you are given."""


ENTITY_EXTRACTION_PROMPT = """You are a named entity extractor.

Given a question, extract the key named entities that the question is ABOUT.
These are the entities that would be the starting points for graph traversal.

Rules:
- For bridge questions (who/what/where did X do?): return the main subject entity
- For comparison questions (did X and Y share property Z?): return BOTH X and Y
- Return proper nouns only (people, places, orgs, works, dates)
- NEVER return a descriptive phrase. "Hawaiian surfer", "Senator", "race track",
  "British singer-songwriter", "Indian cricketer" are all WRONG answers — they
  describe an entity rather than naming one. If the question only describes the
  subject, return the named work, place or person it refers to, or return an
  empty array. An empty array is better than a description.
- Return ONLY a JSON array of entity strings, nothing else, no markdown fences.

Examples:
Q: "Who directed the film starring Shirley Temple as Corliss Archer?"
→ ["Shirley Temple"]

Q: "Were Scott Derrickson and Ed Wood of the same nationality?"
→ ["Scott Derrickson", "Ed Wood"]

Q: "What year was the director of Inception born?"
→ ["Inception"]

Q: "What science fantasy young adult series has a companion book narrated by Tobias?"
→ ["Tobias"]

Q: "What American professional Hawaiian surfer won the Rip Curl Pro Portugal?"
→ ["Rip Curl Pro Portugal"]

Q: "In which year was the King who made the 1925 Birthday Honours born?"
→ ["1925 Birthday Honours"]"""


def build_extraction_prompt(sentences: list[str]) -> str:
    """Number the sentences so the model can cite sentence_index."""
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return f"Extract all relational facts from the following passage:\n\n{numbered}"


# --------------------------------------------------------------------------- #
# Targeted repair extraction
# --------------------------------------------------------------------------- #

REPAIR_SYSTEM_PROMPT = f"""You are a Knowledge Graph repair engine.

A reasoning system is traversing a knowledge graph and has hit a dead end: it needs
to get from one entity to another, but no edge connecting them exists in the graph.
Your job is to re-read the source passage and find the connection that was missed.

You will receive a passage where every sentence is prefixed with a sentence index,
plus a SOURCE entity and a TARGET the system is trying to reach.

Return facts that help connect SOURCE to TARGET. A fact helps if it:
- directly relates SOURCE and TARGET, or
- relates SOURCE (or TARGET) to an intermediate entity that plausibly bridges them.

Rules for "relation":
- Derive the label from the main verb or predicate of the sentence you are
  extracting from. Build it out of words that actually appear in that sentence.

Rules for "entities":
- Use proper nouns, named concepts, dates, or quantities.
- Keep surface forms exactly as they appear in the sentence.
- NEVER put more than {MAX_FACT_ARITY} entities in a single fact.
- Do not repeat the same entity twice within one fact.

Critical constraints:
- Every fact MUST come from a single sentence in the passage below. Do NOT use
  outside knowledge, and do NOT combine information across sentence indices.
- If the passage genuinely does not support any connection, return an empty array [].
  An empty array is the correct answer when the link is not in the text. Do NOT
  invent a plausible-sounding edge to satisfy the request.
- Return at most {MAX_FACTS_PER_SENTENCE} facts total. Prefer the single most direct
  connection over many weak ones.
- Return ONLY a JSON array, no explanation, no markdown fences.

Output format (the angle-bracket values are PLACEHOLDERS showing shape only — never
emit them literally, and never treat them as example labels):

[
  {{
    "entities": ["<entity_from_sentence>", "<another_entity>"],
    "relation": "<snake_case_verb_from_that_sentence>",
    "sentence_index": 0,
    "confidence": 0.95
  }}
]"""

def build_repair_prompt(
    sentences: list[str],
    src: str,
    dst: str,
    goal: str = "",
) -> str:
    """
    Args:
        sentences: The source chunk's sentences (numbered for sentence_index).
        src:       Entity the agent is reasoning FROM.
        dst:       Entity or description the agent is trying to reach. May be a
                    description rather than a named entity ("the film's director") —
                    that is fine and often the point, since the agent frequently does
                    not yet know the name of what it is looking for.
        goal:      Optional free-text statement of what the agent is trying to
                    establish. Passed straight through from the agent's own tool call,
                    so it reflects the actual sub-goal rather than a reconstruction.
    """
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    goal_line = f"\nWHAT THE SYSTEM IS TRYING TO ESTABLISH: {goal}" if goal.strip() else ""
    return (
        f"SOURCE entity: {src}\n"
        f"TARGET to reach: {dst}"
        f"{goal_line}\n\n"
        f"Passage:\n\n{numbered}\n\n"
        f"Find facts in this passage that connect SOURCE toward TARGET. "
        f"Return [] if the passage does not support any connection."
    )