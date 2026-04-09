import json
from datasets import load_dataset


def preprocess_cuad_hf(
    json_path="./cuad/train_separate_questions.json",
    save_path="cuad_rag.json",
    max_samples=1000
):
    """
    Convert local CUAD/AOK-style dataset → RAG-ready dataset

    Output format:
    {
        id,
        question,
        answers: [...],        # single answer OR ["NONE"]
        documents: [...],
        context,
        title
    }
    """

    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    processed = []
    count = 0

    id_seen = set()

    # dataset level
    for doc in dataset["data"]:

        title = doc.get("title", "")

        paragraphs = doc.get("paragraphs", [])
        if not paragraphs:
            continue

        # paragraph level
        for para in paragraphs:

            context = para.get("context", "")

            qas = para.get("qas", [])
            if not qas:
                continue

            # QA level
            for qa in qas:
                qa_id = qa.get("id")
                if qa_id in id_seen:
                    assert 0, f"Duplicate ID found: {qa_id}"

                if count >= max_samples:
                    break

                question = qa.get("question", "").strip()
                if not question:
                    continue

                is_impossible = qa.get("is_impossible", False)
                answers = qa.get("answers", [])

                # 无答案或不可回答的样本，统一标记为 ["NONE"]
                if is_impossible or not answers:
                    answer_texts = ["NONE"]
                else:
                    if(len(answers) > 1):
                        print(f"Warning: Multiple answers found for question '{question}'. Only the first one will be used.")
                        assert 0
                    text = answers[0].get("text", "").strip()
                    answer_texts = [text] if text else ["NONE"]

                # RAG document source
                source_text = context if context else title
                if not source_text:
                    continue

                processed.append({
                    "id": qa.get("id", f"cuad_{count}"),

                    "question": question,

                    "answers": answer_texts,

                    "documents": [source_text],

                    "context": context,
                    "title": title
                })

                count += 1

        if count >= max_samples:
            break

    # assert 0, f"count: {count}"
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
    # save path
    CUAD_path = "./cuad/train_separate_questions.json"
    preprocess_cuad_hf(json_path=CUAD_path, save_path="cuad_raw.json", max_samples=30000) # 共22450个samples