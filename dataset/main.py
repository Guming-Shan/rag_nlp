from data_preprocess_cuad import preprocess_cuad_hf, save_dataset, load_dataset_json
from chunking import apply_chunking
from corpus_cuad import CUADPreprocessor
from data_preprocess_financebench import preprocess_financebench
from corpus_financebench import FinanceBenchPreprocessor

import json


def build_dataset(output_path="cuad_rag.json", max_samples=1000, datatype="cuad"):

    if datatype == "cuad":
        print("Loading CUAD...")
        CUAD_path = "./cuad/train_separate_questions.json"
        data = preprocess_cuad_hf(
            json_path=CUAD_path,
            save_path="cuad_raw.json",
            max_samples=max_samples
        )

    elif datatype == "financebench":
        print("Loading FinanceBench...")
        FB_path = "./financebench/financebench_merged.jsonl"

        data = preprocess_financebench(
            json_path=FB_path,
            save_path="financebench_raw.json",
            max_samples=max_samples
        )

    else:
        raise ValueError(f"Unsupported datatype: {datatype}")

    print(f"Raw samples: {len(data)}")

    print("Applying chunking...")
    data = apply_chunking(data, datatype=datatype)

    print("Saving dataset...")
    save_dataset(data, output_path)

    print(f"Done: saved to {output_path}")


def build_corpus(output_path="cuad_corpus.json", chunk_size=300, datatype="cuad"):

    if datatype == "cuad":
        print("Building CUAD corpus...")
        preprocessor = CUADPreprocessor(chunk_size=chunk_size)

        cuad_raw_path = "./cuad/train_separate_questions.json"
        corpus = preprocessor.run(cuad_raw_path, output_path)

    elif datatype == "financebench":
        print("Building FinanceBench corpus...")

        preprocessor = FinanceBenchPreprocessor(chunk_size=chunk_size)

        fb_path = "./financebench/financebench_merged.jsonl"

        corpus = preprocessor.run(fb_path, output_path)

    else:
        raise ValueError(f"Unsupported datatype: {datatype}")

    print(f"Corpus built with {len(corpus)} documents, saved to {output_path}")


if __name__ == "__main__":

    # DATASET
    # build_dataset(
    #     output_path="cuad_rag.json",
    #     max_samples=30000,
    #     datatype="cuad"
    # )

    # build_dataset(
    #     output_path="financebench_rag.json",
    #     max_samples=30000,
    #     datatype="financebench"
    # )

    # CORPUS
    # build_corpus(
    #     output_path="cuad_corpus.json",
    #     chunk_size=300,
    #     datatype="cuad"
    # )

    build_corpus(
        output_path="financebench_corpus.json",
        chunk_size=300,
        datatype="financebench"
    )