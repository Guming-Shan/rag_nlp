"""
RAG Experiment Pipeline

Supports:
- rag / llm / selfrag
- batch inference
- evaluation only mode
"""

import json
import os
import argparse
from datetime import datetime

from retriever import HybridRetriever
from llm import load_llm_local
from eval import evaluate_all
from methods import (
    run_rag, run_rag_batch,
    run_llm_only, run_llm_only_batch,
    run_self_rag_batch
)


def parse_args():
    parser = argparse.ArgumentParser()

    # method
    parser.add_argument(
        "--method",
        type=str,
        default="selfrag",
        choices=["rag", "llm", "selfrag"],
        help="Choose inference method"
    )

    # batch
    parser.add_argument("--batch", action="store_true", help="Use batch inference")
    parser.add_argument("--batch_size", type=int, default=8)

    # model
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/LLM-Research/Meta-Llama-3___1-8B-Instruct"
    )

    # dataset
    parser.add_argument("--dataset_path", type=str, default="./dataset/cuad_rag.json")
    parser.add_argument("--corpus_path", type=str, default="./dataset/cuad_corpus.json")

    # debug
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_size", type=int, default=80)

    # evaluate only
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--input_results", type=str, default=None)

    # retrieval
    parser.add_argument("--topk", type=int, default=5)

    return parser.parse_args()


def main():
    args = parse_args()

    # results directory
    timeline = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f"./results/{timeline}"
    os.makedirs(results_path, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.debug:
        dataset = dataset[:args.debug_size]

    # Evaluate-only mode
    if args.eval_only:
        assert args.input_results is not None, "Need --input_results"

        print("Loading existing results...")
        with open(args.input_results, "r", encoding="utf-8") as f:
            results = json.load(f)

        print("Evaluating...")
        eval_results = evaluate_all(dataset, results)
        print(eval_results)
        return

    # Build retriever (only if needed)
    if args.method in ["rag", "selfrag"]:
        print("Loading retriever corpus...")
        with open(args.corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        print("Building retriever...")
        retriever = HybridRetriever(corpus)
    else:
        retriever = None

    # Load model
    print("Loading LLM...")
    tokenizer, model = load_llm_local(args.model_path)

    # Run method
    print(f"Running method: {args.method}")

    if args.method == "rag":
        if args.batch:
            results = run_rag_batch(
                dataset, retriever, tokenizer, model,
                k=args.topk,
                batch_size=args.batch_size
            )
        else:
            results = run_rag(
                dataset, retriever, tokenizer, model,
                k=args.topk
            )

    elif args.method == "llm":
        if args.batch:
            results = run_llm_only_batch(
                dataset, tokenizer, model,
                batch_size=args.batch_size
            )
        else:
            results = run_llm_only(
                dataset, tokenizer, model
            )

    elif args.method == "selfrag":
        results = run_self_rag_batch(
            dataset, retriever, tokenizer, model,
            k=args.topk,
            batch_size=args.batch_size
        )

    else:
        raise ValueError("Unknown method")

    # Save results
    result_file = os.path.join(results_path, f"{args.method}_results.json")

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {result_file}")

    # Evaluate
    print("Evaluating...")
    eval_results = evaluate_all(dataset, results)
    print("Evaluation Results:", eval_results)

    eval_file = os.path.join(results_path, "evaluation.json")
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation saved to {eval_file}")


if __name__ == "__main__":
    main()