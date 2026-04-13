import json


def preprocess_financebench(
    json_path="./financebench_merged.jsonl",
    save_path="financebench_rag.json",
    max_samples=1000
):
    """
    Convert FinanceBench → RAG-ready dataset

    Output format (保持不变):
    {
        id,
        question,
        answers: [...],
        documents: [...],
        context,
        title
    }
    """

    processed = []
    count = 0
    id_seen = set()

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if count >= max_samples:
                break

            sample = json.loads(line)

            qa_id = sample.get("financebench_id", f"financebench_{count}")
            if qa_id in id_seen:
                raise ValueError(f"Duplicate ID found: {qa_id}")
            id_seen.add(qa_id)

            question = sample.get("question", "").strip()
            if not question:
                continue

            # answer
            answer = sample.get("answer", "").strip()
            answer_texts = [answer] if answer else ["NONE"]

            # evidence
            evidences = sample.get("evidence", [])

            documents = []
            context_list = []

            for e in evidences:
                # 优先使用精确证据
                text = e.get("evidence_text", "").strip()
                full_page = e.get("evidence_text_full_page", "").strip()

                if text:
                    documents.append(text)
                elif full_page:
                    documents.append(full_page)

                if full_page:
                    context_list.append(full_page)

            # fallback
            if not documents:
                # 用 justification 或空
                fallback = sample.get("justification", "").strip()
                if fallback:
                    documents = [fallback]
                else:
                    continue

            # context：拼接所有 full_page
            context = "\n".join(context_list) if context_list else documents[0]

            # title → 用 doc_name
            title = sample.get("doc_name", "")

            processed.append({
                "id": qa_id,
                "question": question,
                "answers": answer_texts,
                "documents": documents,
                "context": context,
                "title": title
            })

            count += 1

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(processed)} samples to {save_path}")

    return processed


def save_dataset(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_dataset_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    FINANCEBENCH_PATH = "./financebench/financebench_merged.jsonl"

    preprocess_financebench(
        json_path=FINANCEBENCH_PATH,
        save_path="financebench_rag.json",
        max_samples=30000
    )