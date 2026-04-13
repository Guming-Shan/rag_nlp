import os
import json
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "./results"


def load_experiment(exp_path):
    """
    Load one experiment folder
    """
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

    return pd.DataFrame(records)


def save_tables(df):
    os.makedirs("./analysis", exist_ok=True)

    csv_path = "./analysis/experiment_summary.csv"
    xlsx_path = "./analysis/experiment_summary.xlsx"

    df.to_csv(csv_path, index=False)
    # df.to_excel(xlsx_path, index=False)

    print(f"[Saved] CSV → {csv_path}")
    print(f"[Saved] Excel → {xlsx_path}")


def plot_metrics(df):
    """
    Plot key metrics comparison
    """

    metrics = [
        "hit_rate",
        "factscore",
        "hallucination_rate",
        "soft_hit_rate"
    ]

    df = df.dropna(subset=["method"])

    grouped = df.groupby("method")[metrics].mean()

    grouped.plot(kind="bar", figsize=(10, 6))
    plt.title("RAG Evaluation Metrics Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")

    os.makedirs("./analysis", exist_ok=True)
    plt.savefig("./analysis/metrics_comparison.png", dpi=300, bbox_inches="tight")

    print("[Saved] Plot → ./analysis/metrics_comparison.png")


def print_latex_table(df):
    """
    Optional: LaTeX table for paper
    """
    metrics = ["hit_rate", "factscore", "hallucination_rate"]

    table = df.groupby("method")[metrics].mean().round(3)

    print("\n===== LaTeX Table =====\n")
    print(table.to_latex())


def main():
    print("Loading experiments...")
    df = load_all_experiments()

    if df.empty:
        print("No experiments found!")
        return

    print("\n===== Raw Summary =====")
    print(df)

    save_tables(df)
    plot_metrics(df)
    print_latex_table(df)


if __name__ == "__main__":
    main()