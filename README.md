# RAG / Self-RAG QA Pipeline

三种方法:

* **LLM-only**
* **Vanilla RAG**
* **Self-RAG (with self-judgment)**

目前仅支持 CUAD 数据集。

---

# Project Structure

```
pipeline/
├── main.py                 # Entry point
├── methods.py              # RAG / LLM / Self-RAG implementations
├── retriever.py            # Hybrid retriever
├── llm.py                  # Model loading & generation
├── prompt.py               # Prompt construction
├── eval.py                 # Evaluation metrics
├── dataset/
│   ├── cuad_rag.json
│   ├── cuad_corpus.json
│   ├── chunking.py
│   ├── corpus_cuad.py
│   ├── data_preprocess_cuad.py
│   └── main.py             # Generate jsons in need
├── models/                 # Include Llama & Qwen
└── results/
```

---

# ⚙️ Arguments
- 带 `--batch` 就是批量化推理，比单个跑快，如果只想测试一个样本则去掉（但 self rag 因为设计限制不论如何都很慢）；
- 批量化推理需要注意显存别爆炸；
- 建议开 `--debug`，全样本推理得很久很久；
- 选 `--eval_only` 时，需要附带已存在的推理结果 `--input_results`;

| Argument          | Description               |
| ----------------- | ------------------------- |
| `--method`        | 选`rag` / `llm` / `selfrag`之一 |
| `--batch`         | Enable batch inference    |
| `--batch_size`    | Batch size (default=8)    |
||
| `--model_path`    | Path to local LLM         |
| `--dataset_path`  | 数据集本体，含问题回答引用gt |
| `--corpus_path`   | 语料库                     |
| `--topk`          | Retrieval top-k           |
||
| `--debug`         | Use small subset          |
| `--debug_size`    | 选择使用的数据集长度n，即前n个样本 |
||
| `--eval_only`     | Only run evaluation       |
| `--input_results` | Path of existing results file |



---

# Quick Start

## 1️⃣ Run Self-RAG (default)
```bash
python main.py --method selfrag --batch
```

---

## 2️⃣ Run Vanilla RAG

```bash
python main.py --method rag --batch --topk 5
```

---

## 3️⃣ Run LLM-only baseline

```bash
python main.py --method llm --batch
```



# Evaluate Existing Results

```bash
python main.py \
  --eval_only \
  --input_results ./results/xxx/selfrag_results.json
```


# Self-RAG Overview

Self-RAG introduces **adaptive retrieval**:

1. Generate initial answer (LLM-only)
2. Self-judge (YES / NO)
3. If NO → retrieve documents
4. Generate final answer

---

# Output Format

Each prediction:

```json
{
  "id": "...",
  "question": "...",
  "gold_answer": [...],
  "pred_answer": [{"text": "..."}],
  "docs": [...],
  "self_rag_judge": "YES/NO"
}
```


# Model Setup

Example:

```bash
--model_path ./models/LLM-Research/Meta-Llama-3___1-8B-Instruct
```

```bash
--model_path ./models/Qwen/Qwen2.5-7B-Instruct
```