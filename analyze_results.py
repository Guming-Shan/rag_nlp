import os
import json
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "./results"


# Load
def load_experiment(exp_path):
    eval_path = os.path.join(exp_path, "evaluation.json")

    if not os.path.exists(eval_path):
        return None

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    args = data.get("args", {})
    metrics = data.get("metrics", {})

    record = {
        "exp_name": os.path.basename(exp_path),

        # args
        "method": args.get("method"),
        "dataset": "financebench" if "financebench" in args.get("dataset_path", "") else "cuad",
        "model": os.path.basename(args.get("model_path", "")),
        "topk": args.get("topk"),
        "batch": args.get("batch", False),

        # metrics
        **metrics
    }

    return record


def load_all_experiments():
    records = []

    for folder in os.listdir(RESULTS_DIR):
        exp_path = os.path.join(RESULTS_DIR, folder)

        if not os.path.isdir(exp_path):
            continue

        record = load_experiment(exp_path)
        if record:
            records.append(record)

    df = pd.DataFrame(records)

    # 补齐缺失列，防止 KeyError
    needed_cols = [
        "hit_rate", "factscore", "hallucination_rate",
        "semantic_similarity", "bertscore",
        "citation_precision", "citation_recall",
        "soft_hit_rate"
    ]

    for col in needed_cols:
        if col not in df.columns:
            df[col] = None

    # ✅ 增加 citation F1
    df["citation_f1"] = 2 * df["citation_precision"] * df["citation_recall"] / (
        df["citation_precision"] + df["citation_recall"] + 1e-8
    )

    return df


# Analysis
def analyze_hallucination(df):
    print("\n===== Hallucination Analysis =====")
    table = df.groupby("method")[["hallucination_rate", "factscore"]].mean()
    print(table.round(4))


def analyze_hallucination_improvement(df):
    print("\n===== Hallucination Improvement (LLM vs RAG) =====")

    pivot = df.pivot_table(
        index=["model", "dataset", "topk"],
        columns="method",
        values="hallucination_rate"
    )

    if "llm" in pivot.columns and "rag" in pivot.columns:
        pivot["improvement"] = pivot["llm"] - pivot["rag"]
        print(pivot.round(4))
    else:
        print("Need both llm and rag for comparison.")


def analyze_by_domain(df):
    print("\n===== Domain (Dataset) Analysis =====")

    table = df.groupby(["dataset", "method"])[
        ["citation_precision", "citation_recall", "citation_f1", "factscore"]
    ].mean()

    print(table.round(4))


def analyze_rag_vs_selfrag(df):
    print("\n===== RAG vs Self-RAG =====")

    subset = df[df["method"].isin(["rag", "selfrag"])]

    table = subset.groupby(["dataset", "model"])[
        ["factscore", "hallucination_rate", "citation_f1"]
    ].mean()

    print(table.round(4))


def analyze_topk(df):
    print("\n===== Top-k Analysis =====")

    table = df.groupby(["topk", "method"])[
        ["factscore", "hallucination_rate"]
    ].mean()

    print(table.round(4))


def analyze_model(df):
    print("\n===== Model Sensitivity =====")

    table = df.groupby(["model", "method"])[
        ["factscore", "hallucination_rate"]
    ].mean()

    print(table.round(4))


def analyze_correlation(df):
    print("\n===== Correlation =====")

    corr = df[[
        "hit_rate",
        "factscore",
        "hallucination_rate",
        "semantic_similarity",
        "bertscore"
    ]].corr()

    print(corr.round(3))


# Plot
def plot_metrics(df):
    metrics = [
        "hit_rate",
        "factscore",
        "hallucination_rate",
        "soft_hit_rate"
    ]

    df = df.dropna(subset=["method"])

    grouped = df.groupby("method")[metrics].mean()

    grouped.plot(kind="bar", figsize=(10, 6))
    plt.title("Overall Metrics Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)

    os.makedirs("./analysis", exist_ok=True)
    plt.savefig("./analysis/metrics_overall.png", dpi=300, bbox_inches="tight")
    plt.clf()

    print("[Saved] ./analysis/metrics_overall.png")


def plot_by_dataset(df):
    metrics = ["factscore", "hallucination_rate"]

    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]

        grouped = subset.groupby("method")[metrics].mean()

        grouped.plot(kind="bar", figsize=(8, 5))
        plt.title(f"{dataset} Comparison")
        plt.ylabel("Score")
        plt.xticks(rotation=0)

        plt.savefig(f"./analysis/{dataset}_comparison.png", dpi=300)
        plt.clf()

        print(f"[Saved] ./analysis/{dataset}_comparison.png")


def plot_topk(df):
    for metric in ["factscore", "hallucination_rate"]:
        pivot = df.pivot_table(
            index="topk",
            columns="method",
            values=metric
        )

        pivot.plot(marker="o")
        plt.title(f"{metric} vs Top-k")
        plt.ylabel(metric)

        plt.savefig(f"./analysis/{metric}_topk.png", dpi=300)
        plt.clf()

        print(f"[Saved] ./analysis/{metric}_topk.png")


# Save
def save_tables(df):
    os.makedirs("./analysis", exist_ok=True)

    df.to_csv("./analysis/experiment_summary.csv", index=False)

    print("[Saved] ./analysis/experiment_summary.csv")


def print_latex_table(df):
    metrics = [
        "hit_rate",
        "factscore",
        "hallucination_rate",
        "citation_f1"
    ]

    table = df.groupby("method")[metrics].mean().round(3)

    print("\n===== LaTeX Table =====\n")
    print(table.to_latex())


def plot_detailed_by_dataset_model(df):
    os.makedirs("./analysis", exist_ok=True)

    metrics = ["factscore", "hallucination_rate"]

    for dataset in df["dataset"].unique():
        for topk in sorted(df["topk"].dropna().unique()):

            subset = df[(df["dataset"] == dataset) & (df["topk"] == topk)]

            if subset.empty:
                continue

            pivot = subset.pivot_table(
                index="method",
                columns="model",
                values=metrics
            )

            # 多指标拆开画
            for metric in metrics:
                pivot_metric = pivot[metric]

                pivot_metric.plot(kind="bar", figsize=(8, 5))
                plt.title(f"{metric} | dataset={dataset}, topk={topk}")
                plt.ylabel(metric)
                plt.xticks(rotation=0)

                filename = f"./analysis/{metric}_{dataset}_topk{topk}.png"
                plt.savefig(filename, dpi=300)
                plt.clf()

                print(f"[Saved] {filename}")

def plot_topk_by_dataset_model(df):
    os.makedirs("./analysis", exist_ok=True)

    for dataset in df["dataset"].unique():
        for model in df["model"].unique():

            subset = df[(df["dataset"] == dataset) & (df["model"] == model)]

            if subset.empty:
                continue

            pivot = subset.pivot_table(
                index="topk",
                columns="method",
                values="factscore"
            )

            pivot.plot(marker="o")
            plt.title(f"FactScore vs Top-k\n{dataset} | {model}")
            plt.ylabel("factscore")

            filename = f"./analysis/topk_curve_{dataset}_{model}.png"
            plt.savefig(filename, dpi=300)
            plt.clf()

            print(f"[Saved] {filename}")

def plot_hallucination_grid(df):
    os.makedirs("./analysis", exist_ok=True)

    for dataset in df["dataset"].unique():
        for model in df["model"].unique():

            subset = df[(df["dataset"] == dataset) & (df["model"] == model)]

            if subset.empty:
                continue

            grouped = subset.groupby("method")["hallucination_rate"].mean()

            grouped.plot(kind="bar", figsize=(6, 4))
            plt.title(f"Hallucination Rate\n{dataset} | {model}")
            plt.ylabel("hallucination_rate")
            plt.xticks(rotation=0)

            filename = f"./analysis/hallu_{dataset}_{model}.png"
            plt.savefig(filename, dpi=300)
            plt.clf()

            print(f"[Saved] {filename}")

# Main
def main():
    print("Loading experiments...")
    df = load_all_experiments()

    if df.empty:
        print("No experiments found!")
        return

    print("\n===== Raw Summary =====")
    print(df)

    save_tables(df)

    # ===== Analysis =====
    analyze_hallucination(df)
    analyze_hallucination_improvement(df)
    analyze_by_domain(df)
    analyze_rag_vs_selfrag(df)
    analyze_topk(df)
    analyze_model(df)
    analyze_correlation(df)

    # ===== Plot =====
    plot_metrics(df)
    plot_by_dataset(df)
    plot_topk(df)
    plot_detailed_by_dataset_model(df)
    plot_topk_by_dataset_model(df)
    plot_hallucination_grid(df)

    # ===== LaTeX =====
    print_latex_table(df)


if __name__ == "__main__":
    main()