# data_preprocess_cuad.py

import json
import re
from typing import List, Dict


class CUADPreprocessor:
    """
    Preprocess CUAD dataset into retriever corpus chunks.
    """

    def __init__(self, chunk_size: int = 300):
        self.chunk_size = chunk_size

    def load_json(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # recursive chunking
    def recursive_chunk(self, text: str) -> List[str]:
        """
        Recursive chunking:
        优先按结构切（段落/换行），再降级到句子，再到词
        """

        separators = ["\n\n", "\n", ". ", " "]

        def split_text(t, sep_idx):
            # 如果已经足够小 or 没有更细粒度
            if len(t.split()) <= self.chunk_size or sep_idx >= len(separators):
                return [t]

            sep = separators[sep_idx]
            parts = t.split(sep)

            chunks = []
            current = ""

            for p in parts:
                p = p.strip()
                if not p:
                    continue

                candidate = (current + sep + p) if current else p

                if len(candidate.split()) <= self.chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.extend(split_text(current, sep_idx + 1))
                    current = p

            if current:
                chunks.extend(split_text(current, sep_idx + 1))

            return chunks

        chunks = split_text(text, 0)

        # 清理空chunk
        return [c.strip() for c in chunks if c.strip()]

    def chunk_text(self, text: str) -> List[str]:
        """
        使用 recursive chunking 替代 sliding window
        """
        return self.recursive_chunk(text)

    def build_corpus(self, cuad_data: Dict) -> List[Dict]:
        """
        Convert CUAD into retriever corpus.
        Output format:
        {
            "id": int,
            "text": str,
            "source": str
        }
        """

        corpus = []
        idx = 0

        for item in cuad_data["data"]:
            doc_name = item.get("title", "contract")

            for para in item["paragraphs"]:
                context = para["context"]

                chunks = self.chunk_text(context)

                for chunk in chunks:
                    corpus.append({
                        "id": idx,
                        "text": chunk,
                        "source": doc_name
                    })
                    idx += 1

        return corpus

    def save_corpus(self, corpus: List[Dict], path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)

    def run(self, input_path: str, output_path: str):
        data = self.load_json(input_path)
        corpus = self.build_corpus(data)
        self.save_corpus(corpus, output_path)
        return corpus