import os
import json
import logging
from dataclasses import dataclass, field
from datasets import load_dataset
from typing import Optional

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Hyper parameters
DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "fullwiki"

@dataclass
class Chunk:
    """A single text chunk ready for hypergraph extraction"""
    chunk_id:    str
    sample_id:   str
    title:       str          # wikipedia article title
    text:        str          # chunk text
    sentences:   list[str]    # original sentences in this chunk
    is_gold:     bool = False # whether this chunk contains a supporting fact

@dataclass
class HotpotSample:
    """One HotpotQA sample with all context chunks attached."""
    sample_id:      str
    question:       str
    answer:         str
    hop_type:       str         # "bridge" or "comparison"
    level:          str         # "easy" | "medium" | "hard"
    gold_titles:    list[str]   # the two gold wikipedia articles
    chunks:         list[Chunk] = field(default_factory=list)

class HotpotQALoader:
    """Loads HotpotQA and prepares chunked context paragraphs.

    Args:
        split:      "train" | "validation"
        chunk_size:  max sentences per chunk
        overlap:     sentence overlap between consecutive chunks
        max_samples: cap on number of samples to load (None = all)
        cache_path:  if set, saves/loads processed samples as JSONL"""

    def __init__(
        self,
        split: str = "validation",
        chunk_size: int = 5,
        overlap: int = 1,
        max_samples: Optional[int] = None,
        cache_path: Optional[str] = None,
    ):
        self.split = split
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_samples = max_samples
        self.cache_path = cache_path

    def load(self) -> list[HotpotSample]:
        """Main entry point. Return a list of HotpotSample objects."""
        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"Loading cached samples from {self.cache_path}")
            return self._load_from_cache(self.cache_path)
        
        logger.info(f"Downloading HotpotQA ({DATASET_CONFIG} / {self.split})...")

        raw = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split=self.split,
        )

        if self.max_samples:
            raw = raw.select(range(min(self.max_samples, len(raw))))
            logger.info(f"Capped at {self.max_samples} samples")
        
        samples = [self._process_sample(row) for row in raw]
        logger.info(f"Loaded {len(samples)} samples, "
                    f"{sum(len(s.chunks) for s in samples)} total chunks")
        
        if self.cache_path:
            self._save_to_chache(samples. self.cache_path)
            logger.info(f"Cached to {self.cache_path}")
        
        return samples
        
    def _process_sample(self, row: dict) -> HotpotSample:
        """Convert one raw HotpotQA row into a HotpotSample."""
        gold_titles = set(row["supporting_facts"]["title"])

        chunks: list[Chunk] = []
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]

        for title, sents in zip(titles, sentences):
            is_gold = title in gold_titles
            article_chunks = self._chunk_sentences(
                sample_id = row["id"],
                title = title,
                sentences = sents,
                is_gold = is_gold,
            )
            chunks.extend(article_chunks)
        
        return HotpotSample(
            sample_id=row["id"],
            question=row["question"],
            answer=row["answer"],
            hop_type=row["type"],
            level=row["level"],
            gold_titles=list(gold_titles),
            chunks = chunks
        )
    
    def _chunk_sentences(
        self,
        sample_id: str,
        title: str,
        sentences: list[str],
        is_gold: bool
    ) -> list[Chunk]:
        """
        Slide a window of 'chunk size' sentences over the article
        with 'overlap' sentence overlap between consecutive windows
        """
        chunks = []
        step = max(1, self.chunk_size - self.overlap)

        for i, start in enumerate(range(0, len(sentences), step)):
            window = sentences[start : start + self.chunk_size]
            if not window:
                continue
            chunks.append(
                Chunk(
                    chunk_id    = f"{sample_id}_{title}_{i}",
                    sample_id   = sample_id,
                    title       = title,
                    text        = " ".join(window).strip(),
                    sentences   = window,
                    is_gold     = is_gold
                )
            )
        return chunks
    
    def _load_from_cache(self, path: str) -> list[HotpotSample]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(self._dict_to_sample(json.loads(line)))
        return samples

    @staticmethod
    def _dict_to_sample(d: dict) -> HotpotSample:
        chunks = [Chunk(**c) for c in d["chunks"]]
        return HotpotSample(
            sample_id=d["sample_id"],
            question=d["question"],
            answer=d["answer"],
            hop_type=d["hop_type"],
            level=d["level"],
            gold_titles=d["gold_titles"],
            chunks=chunks
        )
    
    def get_all_chunks(self, samples: list[HotpotSample]) -> list[Chunk]:
        """Flatten all chunks across all samples into a single list"""
        return [chunk for sample in samples for chunk in sample.chunks]
    
    def get_gold_chunks(self, samples: list[HotpotSample]) -> list[Chunk]:
        """Return only chunks that contain supporting facts"""
        return [c for c in self.get_all_chunks(samples) if c.is_gold]
    
    def summary(self, samples: list[HotpotSample]) -> dict:
        """Quick stats about the loaded dataset."""
        all_chunks = self.get_all_chunks(samples)
        gold_chunks = self.get_gold_chunks(samples)
        bridge = sum(1 for s in samples if s.hop_type == "bridge")
        comparison = sum(1 for s in samples if s.hop_type == "comparison")

        return {
            "total_samples" : len(samples),
            "bridge"        : bridge,
            "comparison"    : comparison,
            "total_chunks"  : len(all_chunks),
            "gold_chunks"   : len(gold_chunks),
            "avg_chunks_per_sample" : round(len(all_chunks) / len(samples), 1)
        }

if __name__ == "__main__":
    loader = HotpotQALoader(
        split="validation",
        chunk_size=5,
        overlap=1,
        max_samples=10,
    )
    samples = loader.load()

    print("\n=== Summary ===")
    for k, v in loader.summary(samples).items():
        print(f"    {k}: {v}")
    
    print("\n=== First Sample ===")
    s = samples[0]
    print(f" Q : {s.question}")
    print(f" A : {s.answer}")
    print(f" type : {s.hop_type}")
    print(f" gold : {s.gold_titles}")
    print(f" chunks : {len(s.chunks)}")
    print(f"\n First chunk text")
    print(f" {s.chunks[0].text[:300]}")