# import json
# from chunking import apply_chunking
# from data_preprocess_cuad import preprocess_cuad, preprocess_cuad_local
# from data_preprocess_financebench import preprocess_financebench


# def build_all_datasets(CUAD_path="", FinanceBench_path=""):
#     if not CUAD_path and not FinanceBench_path:
#         print("No dataset path provided. Exiting.")
#         return

#     list_of_datasets = []
#     if FinanceBench_path:
#         preprocess_financebench(FinanceBench_path)
#         list_of_datasets.append(FinanceBench_path)

#     if CUAD_path:
#         preprocess_cuad_local(save_path=CUAD_path)
#         list_of_datasets.append(CUAD_path)

#     # chunk
#     for file in list_of_datasets:
#         with open(file, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         data = apply_chunking(data)

#         with open(f"{file.split('_')[0]}_rag.json", "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)

#         print(f"{file.split('_')[0]} processed with chunking.")

# if __name__ == "__main__":
#     # save path
#     CUAD_path = "cuad_raw.json"
#     FinanceBench_path = ""

#     build_all_datasets(CUAD_path, FinanceBench_path)

# main.py

from data_preprocess_cuad import preprocess_cuad_hf, save_dataset, load_dataset_json
from chunking import apply_chunking
from corpus_cuad import CUADPreprocessor
import json


def build_dataset(output_path="cuad_rag.json", max_samples=1000):

    print("Loading CUAD...")
    CUAD_path = "./cuad/train_separate_questions.json"
    data = preprocess_cuad_hf(json_path=CUAD_path, save_path="cuad_raw.json", max_samples=max_samples)

    print(f"Raw samples: {len(data)}")

    print("Applying chunking...")
    data = apply_chunking(data)

    print("Saving dataset...")
    save_dataset(data, output_path)

    print(f"Done: saved to {output_path}")


def build_corpus(output_path="cuad_corpus.json", chunk_size=300):
    corpus = []
    preprocessor = CUADPreprocessor(chunk_size=chunk_size)

    cuad_raw_path = "./cuad/train_separate_questions.json"
    corpus = preprocessor.run(cuad_raw_path, output_path)
    
    print(f"Corpus built with {len(corpus)} documents, saved to {output_path}")

if __name__ == "__main__":

    # build_dataset(
    #     output_path="cuad_rag.json",
    #     max_samples=30000
    # )

    build_corpus(
        output_path="cuad_corpus.json",
        chunk_size=300
    )