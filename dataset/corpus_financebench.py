import json
from typing import List, Dict


class FinanceBenchPreprocessor:
    """
    Build corpus from FinanceBench evidence_text_full_page
    """

    def __init__(self, chunk_size: int = 300):
        self.chunk_size = chunk_size

    def load_jsonl(self, path: str):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    # recursive chunking
    def recursive_chunk(self, text: str) -> List[str]:
        separators = ["\n\n", "\n", " "]  # ❗去掉 ". "

        def split_text(t, sep_idx):
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

        return [c.strip() for c in split_text(text, 0) if c.strip()]

    def chunk_text(self, text: str) -> List[str]:
        return self.recursive_chunk(text)

    def build_corpus(self, data: List[Dict]) -> List[Dict]:
        """
        Output format:
        {
            "id": int,
            "text": str,
            "source": str
        }
        """

        corpus = []
        idx = 0

        seen_pages = set()  # 避免重复page

        for item in data:
            doc_name = item.get("doc_name", "")
            evidences = item.get("evidence", [])

            for e in evidences:
                page_text = e.get("evidence_text_full_page", "").strip()

                if not page_text:
                    continue

                # 去重（同一页可能被多个问题引用）
                key = (doc_name, page_text)
                if key in seen_pages:
                    continue
                seen_pages.add(key)

                chunks = self.chunk_text(page_text)

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
        data = self.load_jsonl(input_path)
        corpus = self.build_corpus(data)
        self.save_corpus(corpus, output_path)
        return corpus