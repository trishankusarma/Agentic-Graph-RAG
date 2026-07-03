import os
import json
import logging
from dataclasses import dataclass, field
from datasets import load_dataset
from typing import Optional
from dataclasses import asdict
from collections import defaultdict

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
    start_sentence: int       # the initial index of the chunk
    sentences:   list[str]    # original sentences in this chunk
    relative_gold_sentences: list[int] # index of sentences that are in supporting facts

@dataclass
class HotpotSample:
    """One HotpotQA sample with all context chunks attached."""
    sample_id:      str
    question:       str
    answer:         str
    hop_type:       str         # "bridge" or "comparison"
    level:          str         # "easy" | "medium" | "hard"
    gold_sentences: dict[str, list[int]]        # dictionary :: str -> list[int]
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
            self._save_to_cache(samples, self.cache_path)
            logger.info(f"Cached to {self.cache_path}")
        
        return samples
        
    def _process_sample(self, row: dict) -> HotpotSample:
        """Convert one raw HotpotQA row into a HotpotSample."""
        gold_sentences_dict = defaultdict(list)

        for title, sentence_id in zip(
            row["supporting_facts"]["title"],
            row["supporting_facts"]["sent_id"]
        ):
            gold_sentences_dict[title].append(sentence_id)

        chunks: list[Chunk] = []
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        chunk_idx = 0

        for title, sents in zip(titles, sentences):
            gold_sentences = gold_sentences_dict.get(title, [])
            article_chunks, chunk_idx = self._chunk_sentences(
                sample_id = row["id"],
                title = title,
                sentences = sents,
                gold_sentences = gold_sentences,
                chunk_idx = chunk_idx,
            )
            chunks.extend(article_chunks)
        
        return HotpotSample(
            sample_id=row["id"],
            question=row["question"],
            answer=row["answer"],
            hop_type=row["type"],
            level=row["level"],
            chunks = chunks,
            gold_sentences = dict(gold_sentences_dict)
        )
    
    def _chunk_sentences(
        self,
        sample_id: str,
        title: str,
        sentences: list[str],
        gold_sentences: list[int],
        chunk_idx: int,
    ) -> tuple[list[Chunk], int]:
        """
        Slide a window of 'chunk size' sentences over the article
        with 'overlap' sentence overlap between consecutive windows
        """
        chunks = []
        step = max(1, self.chunk_size - self.overlap)

        for start in range(0, len(sentences), step):
            window = sentences[start : start + self.chunk_size]
            relative_gold_sentences = [ # lets store the index of the sentences in the chunk order
                sent_id - start
                for sent_id in gold_sentences
                if start <= sent_id < start + len(window)
            ]
            if not window:
                continue
            chunks.append(
                Chunk(
                    chunk_id    = f"{sample_id}_{chunk_idx}",
                    sample_id   = sample_id,
                    title       = title,
                    start_sentence= start,
                    sentences   = window,
                    relative_gold_sentences = relative_gold_sentences
                )
            )
            chunk_idx += 1
        return chunks, chunk_idx
    
    def _load_from_cache(self, path: str) -> list[HotpotSample]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(self._dict_to_sample(json.loads(line)))
        return samples
    
    def _save_to_cache(self, samples, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(asdict(sample)) + "\n")

    @staticmethod
    def _dict_to_sample(d: dict) -> HotpotSample:
        chunks = [Chunk(**c) for c in d["chunks"]]
        return HotpotSample(
            sample_id=d["sample_id"],
            question=d["question"],
            answer=d["answer"],
            hop_type=d["hop_type"],
            level=d["level"],
            gold_sentences=d["gold_sentences"],
            chunks=chunks
        )
    
    def get_all_chunks(self, samples: list[HotpotSample]) -> list[Chunk]:
        """Flatten all chunks across all samples into a single list"""
        return [chunk for sample in samples for chunk in sample.chunks]
    
    def summary(self, samples: list[HotpotSample]) -> dict:
        """Quick stats about the loaded dataset."""
        all_chunks = self.get_all_chunks(samples)
        bridge = sum(1 for s in samples if s.hop_type == "bridge")
        comparison = sum(1 for s in samples if s.hop_type == "comparison")

        return {
            "total_samples" : len(samples),
            "bridge"        : bridge,
            "comparison"    : comparison,
            "total_chunks"  : len(all_chunks),
            "avg_chunks_per_sample" : round(len(all_chunks) / len(samples), 1)
        }

if __name__ == "__main__":
    loader = HotpotQALoader(
        split="validation",
        chunk_size=5,
        overlap=1,
        max_samples=5,
        cache_path="data/hypergraph_cache.json",
    )
    samples = loader.load()

    print("\n=== Summary ===")
    for k, v in loader.summary(samples).items():
        print(f"    {k}: {v}")
    
    for index, sample in enumerate(samples):
        print(f"\n=== Sample {index}===")
        print(f" Q : {sample.question}")
        print(f" A : {sample.answer}")
        print(f" type : {sample.hop_type}")
        print(f" gold titles : {sample.gold_sentences.keys()}")
        print(f" chunks : {len(sample.chunks)}")
        print(f"\n First chunk text")
        print(f" {sample.chunks[0].sentences[0][:300]}")