"""
kg/data_loader.py

Loads HotpotQA and prepares chunked context paragraphs for hypergraph extraction.
"""

import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_NAME   = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"


@dataclass
class Chunk:
    """A single text chunk ready for hypergraph extraction."""
    chunk_id:              str
    sample_id:             str
    title:                 str        # wikipedia article title
    start_sentence:        int        # index of this chunk's first sentence
    sentences:             list[str]  # original sentences in this chunk
    gold_sentence_offsets: list[int]  # supporting-fact indices, chunk-relative

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


@dataclass
class HotpotSample:
    """One HotpotQA sample with all context chunks attached."""
    sample_id:      str
    question:       str
    answer:         str
    hop_type:       str                   # "bridge" | "comparison"
    level:          str                   # "easy" | "medium" | "hard"
    gold_sentences: dict[str, list[int]]  # article title -> sentence ids
    chunks:         list[Chunk] = field(default_factory=list)


class HotpotQALoader:
    """
    Loads HotpotQA and prepares chunked context paragraphs.

    Args:
        split:       "train" | "validation"
        chunk_size:  max sentences per chunk
        overlap:     sentence overlap between consecutive chunks
        max_samples: cap on number of samples to load (None = all)
        cache_path:  if set, saves/loads processed samples as JSONL

    Cache validity:
        The cache stores the chunking parameters in a header line and is
        rejected on mismatch. Without that check, changing chunk_size and
        rerunning would silently reload chunks built under the old setting —
        which would quietly invalidate any chunk-size ablation.
    """

    def __init__(
        self,
        split:       str           = "validation",
        chunk_size:  int           = 5,
        overlap:     int           = 1,
        max_samples: Optional[int] = None,
        cache_path:  Optional[str] = None,
    ):
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be < chunk_size ({chunk_size}) — "
                "otherwise the window never advances"
            )
        self.split       = split
        self.chunk_size  = chunk_size
        self.overlap     = overlap
        self.max_samples = max_samples
        self.cache_path  = cache_path

    # ------------------------------------------------------------------ #

    def _cache_key(self) -> dict:
        """Parameters that change the produced chunks."""
        return {
            "dataset":     DATASET_NAME,
            "config":      DATASET_CONFIG,
            "split":       self.split,
            "chunk_size":  self.chunk_size,
            "overlap":     self.overlap,
            "max_samples": self.max_samples,
        }

    def load(self) -> list[HotpotSample]:
        """Main entry point. Returns a list of HotpotSample objects."""
        if self.cache_path and os.path.exists(self.cache_path):
            cached = self._load_from_cache(self.cache_path)
            if cached is not None:
                logger.info(
                    f"Loaded {len(cached)} cached samples from {self.cache_path}"
                )
                return cached
            logger.warning("Cache parameters differ — re-downloading")

        logger.info(f"Downloading HotpotQA ({DATASET_CONFIG} / {self.split})...")
        raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split=self.split)

        if self.max_samples:
            raw = raw.select(range(min(self.max_samples, len(raw))))
            logger.info(f"Capped at {self.max_samples} samples")

        samples = [self._process_sample(row) for row in raw]
        logger.info(
            f"Loaded {len(samples)} samples, "
            f"{sum(len(s.chunks) for s in samples)} total chunks"
        )

        if self.cache_path:
            self._save_to_cache(samples, self.cache_path)
            logger.info(f"Cached to {self.cache_path}")

        return samples

    # ------------------------------------------------------------------ #

    def _process_sample(self, row: dict) -> HotpotSample:
        """Convert one raw HotpotQA row into a HotpotSample."""
        gold_sentences_dict: dict[str, list[int]] = defaultdict(list)
        for title, sentence_id in zip(
            row["supporting_facts"]["title"],
            row["supporting_facts"]["sent_id"],
        ):
            gold_sentences_dict[title].append(sentence_id)

        chunks: list[Chunk] = []
        chunk_idx = 0
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            article_chunks, chunk_idx = self._chunk_sentences(
                sample_id      = row["id"],
                title          = title,
                sentences      = sents,
                gold_sentences = gold_sentences_dict.get(title, []),
                chunk_idx      = chunk_idx,
            )
            chunks.extend(article_chunks)

        return HotpotSample(
            sample_id      = row["id"],
            question       = row["question"],
            answer         = row["answer"],
            hop_type       = row["type"],
            level          = row["level"],
            chunks         = chunks,
            gold_sentences = dict(gold_sentences_dict),
        )

    def _chunk_sentences(
        self,
        sample_id:      str,
        title:          str,
        sentences:      list[str],
        gold_sentences: list[int],
        chunk_idx:      int,
    ) -> tuple[list[Chunk], int]:
        """
        Slide a window of `chunk_size` sentences over the article
        with `overlap` sentences shared between consecutive windows.
        """
        chunks: list[Chunk] = []
        step = max(1, self.chunk_size - self.overlap)

        for start in range(0, len(sentences), step):
            window = sentences[start : start + self.chunk_size]
            if not window:
                continue
            gold_sentence_offsets = [
                sent_id - start
                for sent_id in gold_sentences
                if start <= sent_id < start + len(window)
            ]
            chunks.append(
                Chunk(
                    chunk_id              = f"{sample_id}_{chunk_idx}",
                    sample_id             = sample_id,
                    title                 = title,
                    start_sentence        = start,
                    sentences             = window,
                    gold_sentence_offsets = gold_sentence_offsets,
                )
            )
            chunk_idx += 1
        return chunks, chunk_idx

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #

    def _load_from_cache(self, path: str) -> Optional[list[HotpotSample]]:
        """Returns None if the cache was written with different parameters."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                header_line = f.readline()
                if not header_line:
                    return None
                header = json.loads(header_line)
                if header.get("_cache_key") != self._cache_key():
                    logger.info(
                        f"Cache key mismatch:\n"
                        f"  on disk: {header.get('_cache_key')}\n"
                        f"  wanted : {self._cache_key()}"
                    )
                    return None
                return [
                    self._dict_to_sample(json.loads(line))
                    for line in f if line.strip()
                ]
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Unreadable cache at {path}: {e}")
            return None

    def _save_to_cache(self, samples: list[HotpotSample], path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_cache_key": self._cache_key()}) + "\n")
            for sample in samples:
                f.write(json.dumps(asdict(sample)) + "\n")

    @staticmethod
    def _dict_to_sample(d: dict) -> HotpotSample:
        return HotpotSample(
            sample_id      = d["sample_id"],
            question       = d["question"],
            answer         = d["answer"],
            hop_type       = d["hop_type"],
            level          = d["level"],
            gold_sentences = d["gold_sentences"],
            chunks         = [Chunk(**c) for c in d["chunks"]],
        )

    # ------------------------------------------------------------------ #

    def get_all_chunks(self, samples: list[HotpotSample]) -> list[Chunk]:
        """Flatten all chunks across all samples into a single list."""
        return [chunk for sample in samples for chunk in sample.chunks]

    def summary(self, samples: list[HotpotSample]) -> dict:
        """Quick stats about the loaded dataset."""
        all_chunks = self.get_all_chunks(samples)
        return {
            "total_samples":         len(samples),
            "bridge":                sum(1 for s in samples if s.hop_type == "bridge"),
            "comparison":            sum(1 for s in samples if s.hop_type == "comparison"),
            "total_chunks":          len(all_chunks),
            "avg_chunks_per_sample": round(len(all_chunks) / max(len(samples), 1), 1),
        }


if __name__ == "__main__":
    loader = HotpotQALoader(
        split="validation",
        chunk_size=5,
        overlap=1,
        max_samples=5,
        cache_path="data/data_loader_cache.jsonl",
    )
    samples = loader.load()

    print("\n=== Summary ===")
    for k, v in loader.summary(samples).items():
        print(f"    {k}: {v}")

    for index, sample in enumerate(samples):
        print(f"\n=== Sample {index} ===")
        print(f" Q : {sample.question}")
        print(f" A : {sample.answer}")
        print(f" type : {sample.hop_type}")
        print(f" gold titles : {list(sample.gold_sentences.keys())}")
        print(f" chunks : {len(sample.chunks)}")
        if sample.chunks:
            print(f"\n First chunk text")
            print(f" {sample.chunks[0].sentences[0][:300]}")