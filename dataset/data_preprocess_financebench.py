import json
from datasets import load_dataset

def preprocess_financebench(save_path="financebench_rag.json"):
    dataset = load_dataset("PatronusAI/financebench", split="train")

    processed = []

    for i, item in enumerate(dataset):
        question = item["question"]
        answer = item["answer"]

        # 支持文档（可能字段名不同，需检查）
        docs = []
        if "context" in item:
            docs = [item["context"]]
        elif "documents" in item:
            docs = item["documents"]
        elif "evidence" in item:
            docs = [item["evidence"]]

        # fallback
        docs = [d for d in docs if d and len(d.strip()) > 20]

        processed.append({
            "id": f"finance_{i}",
            "question": question.strip(),
            "answer": answer.strip(),
            "documents": docs
        })

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(processed)} samples to {save_path}")