"""
This module defines the main RAG pipeline methods, including:
- run_rag: standard RAG inference
- run_rag_batch: batch version of RAG inference for speedup
- run_llm_only: inference without retrieval (LLM-only)
- run_llm_only_batch: batch version of LLM-only inference
- run_self_rag: Self-RAG inference with selective retrieval
- run_self_rag_batch: batch version of Self-RAG inference
"""

from tqdm import tqdm
from prompt import build_prompt
from retriever import HybridRetriever
from llm import generate_answer, generate_batch

def run_rag(dataset, retriever, tokenizer, model, k=5):
    results = []

    for sample in tqdm(dataset):
        question = sample["question"]

        # retrieve
        docs = retriever.retrieve(question, k=k)

        # prompt
        prompt = build_prompt(question, docs)

        # generate
        answer = generate_answer(tokenizer, model, prompt)

        results.append({
            "id": sample["id"],
            "question": question,
            "gold_answer": sample["answers"],
            "pred_answer": [{"text": answer}],
            "docs": docs
        })
        
        # print("-" * 50)
        # print(f'\n【id】: {sample["id"]}')
        # print(f"\n【Question】: {question}")
        # print(f"【Gold Answer】: {sample['answers']}")
        # print(f"【Pred Answer】: {answer}")
        # print(f"【Docs】: {docs}")
        # print("-" * 50)

    return results


def run_rag_batch(dataset, retriever, tokenizer, model, k=5, batch_size=8):
    results = []

    # 按长度排序
    dataset = sorted(dataset, key=lambda x: len(x["question"]))

    for i in tqdm(range(0, len(dataset), batch_size)):
        # dataset questions batch
        batch = dataset[i:i + batch_size]
        questions = [s["question"] for s in batch]

        # retrieve
        batch_docs = retriever.retrieve_batch(questions, k=k)

        # prompt
        prompts = [build_prompt(q, d) for q, d in zip(questions, batch_docs)]

        # generate (batch)
        answers = generate_batch(tokenizer, model, prompts)

        for sample, docs, ans in zip(batch, batch_docs, answers):
            results.append({
                "id": sample["id"],
                "question": sample["question"],
                "gold_answer": sample["answers"],
                "pred_answer": [{"text": ans}],
                "docs": docs
            })

    return results

def run_llm_only(dataset, tokenizer, model):
    results = []

    for sample in tqdm(dataset):
        question = sample["question"]

        prompt = build_prompt(question, docs=None)  # 无docs

        answer = generate_answer(tokenizer, model, prompt)

        results.append({
            "id": sample["id"],
            "question": question,
            "gold_answer": sample["answers"],
            "pred_answer": [{"text": answer}]
        })

    return results

def run_llm_only_batch(dataset, tokenizer, model, batch_size=8):
    results = []

    dataset = sorted(dataset, key=lambda x: len(x["question"]))

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]

        questions = [s["question"] for s in batch]

        # build prompts
        prompts = [
            build_prompt(q, docs=None)
            for q in questions
        ]

        # generate batch
        answers = generate_batch(tokenizer, model, prompts)

        for sample, ans in zip(batch, answers):
            results.append({
                "id": sample["id"],
                "question": sample["question"],
                "gold_answer": sample["answers"],
                "pred_answer": [{"text": ans}]
            })

    return results


def run_self_rag(dataset, retriever, tokenizer, model, k=5):
    results = []

    for sample in tqdm(dataset):
        question = sample["question"]

        # Step 1: LLM-only initial answer
        prompt_no_docs = build_prompt(question, docs=None)
        draft_answer = generate_answer(tokenizer, model, prompt_no_docs)

        # Step 2: Self-judge (是否需要检索)
        judge_prompt = f"""
You are a helpful assistant.

Question:
{question}

Current Answer:
{draft_answer}

Is this answer sufficient and correct?
Answer only "YES" or "NO".
"""

        judge = generate_answer(tokenizer, model, judge_prompt)

        # Step 3: decide whether to retrieve
        if "NO" in judge.upper():
            docs = retriever.retrieve(question, k=k)

            prompt = build_prompt(question, docs)
            final_answer = generate_answer(tokenizer, model, prompt)
        else:
            docs = []
            final_answer = draft_answer

        results.append({
            "id": sample["id"],
            "question": question,
            "gold_answer": sample["answers"],
            "pred_answer": [{"text": final_answer}],
            "docs": docs,
            "self_rag_judge": judge,
            "draft_answer": draft_answer
        })

    return results

def run_self_rag_batch(dataset, retriever, tokenizer, model, k=5, batch_size=8):
    results = []

    dataset = sorted(dataset, key=lambda x: len(x["question"]))

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        questions = [s["question"] for s in batch]

        # Step 1: LLM-only
        prompts_no_docs = [build_prompt(q, None) for q in questions]
        draft_answers = generate_batch(tokenizer, model, prompts_no_docs)

        # Step 2: judge
        judge_prompts = [
            f"""
You are a helpful assistant.

Question:
{q}

Current Answer:
{a}

Is this answer sufficient and correct?
Answer only "YES" or "NO".
"""
            for q, a in zip(questions, draft_answers)
        ]

        judges = generate_batch(tokenizer, model, judge_prompts)

        # Step 3: selective retrieval
        need_retrieval = ["NO" in j.upper() for j in judges]

        retrieve_questions = [q for q, flag in zip(questions, need_retrieval) if flag]

        if len(retrieve_questions) > 0:
            retrieved_docs = retriever.retrieve_batch(retrieve_questions, k=k)
        else:
            retrieved_docs = []

        # Step 4: final generation
        final_answers = []
        docs_idx = 0

        for idx, (q, draft, flag) in enumerate(zip(questions, draft_answers, need_retrieval)):
            if flag:
                docs = retrieved_docs[docs_idx]
                docs_idx += 1

                prompt = build_prompt(q, docs)
                ans = generate_answer(tokenizer, model, prompt)
            else:
                docs = []
                ans = draft

            final_answers.append((ans, docs))

        for sample, (ans, docs), judge, draft in zip(batch, final_answers, judges, draft_answers):
            results.append({
                "id": sample["id"],
                "question": sample["question"],
                "gold_answer": sample["answers"],
                "pred_answer": [{"text": ans}],
                "docs": docs,
                "self_rag_judge": judge,
                "draft_answer": draft
            })

    return results