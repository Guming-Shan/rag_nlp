# hybrid_retriever.py

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    """
    Hybrid Retriever:
    - Dense retrieval (FAISS + embeddings)
    - Sparse retrieval (BM25)
    - Score fusion
    """

    def __init__(self,
                 corpus,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):

        self.corpus = corpus
        self.texts = [c["text"] for c in corpus]

        # Dense model
        self.encoder = SentenceTransformer(embedding_model)

        # Sparse model
        tokenized_corpus = [doc.split() for doc in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # FAISS index
        self.embeddings = self.encoder.encode(self.texts, show_progress_bar=True)
        self.embeddings = np.array(self.embeddings).astype("float32")

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    # Dense retrieval
    def dense_search(self, query, k=20):
        q_emb = self.encoder.encode([query]).astype("float32")

        D, I = self.index.search(q_emb, k)

        return [(i, 1 / (1 + D[0][idx])) for idx, i in enumerate(I[0])]

    def dense_search_batch(self, queries, k=20):
        q_embs = self.encoder.encode(
            queries,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype("float32")

        D, I = self.index.search(q_embs, k)

        results = []
        for qi in range(len(queries)):
            res = [(i, 1 / (1 + D[qi][idx])) for idx, i in enumerate(I[qi])]
            results.append(res)

        return results

    # Sparse retrieval
    def sparse_search(self, query, k=20):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        top_k_idx = np.argsort(scores)[::-1][:k]

        return [(i, scores[i]) for i in top_k_idx]
    
    def sparse_search_batch(self, queries, k=20):
        results = []
        for q in queries:
            tokenized_query = q.split()
            scores = self.bm25.get_scores(tokenized_query)

            top_k_idx = np.argsort(scores)[::-1][:k]
            results.append([(i, scores[i]) for i in top_k_idx])

        return results

    # Hybrid fusion
    def hybrid_search(self, query, k=5, alpha=0.5):
        """
        alpha: dense weight
        (1 - alpha): bm25 weight
        """

        dense_results = self.dense_search(query, k=50)
        sparse_results = self.sparse_search(query, k=50)

        score_dict = {}

        # normalize dense
        for idx, score in dense_results:
            score_dict[idx] = score_dict.get(idx, 0) + alpha * score

        # normalize sparse
        max_sparse = max([s for _, s in sparse_results]) if sparse_results else 1

        for idx, score in sparse_results:
            norm_score = score / max_sparse
            score_dict[idx] = score_dict.get(idx, 0) + (1 - alpha) * norm_score

        ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

        top_docs = [self.corpus[i] for i, _ in ranked[:k]]

        return top_docs

    def hybrid_search_batch(self, queries, k=5, alpha=0.5):
        dense_batch = self.dense_search_batch(queries, k=50)
        sparse_batch = self.sparse_search_batch(queries, k=50)

        all_results = []

        for dense_results, sparse_results in zip(dense_batch, sparse_batch):

            score_dict = {}

            # dense
            for idx, score in dense_results:
                score_dict[idx] = score_dict.get(idx, 0) + alpha * score

            # sparse
            max_sparse = max([s for _, s in sparse_results]) if sparse_results else 1

            for idx, score in sparse_results:
                norm_score = score / max_sparse
                score_dict[idx] = score_dict.get(idx, 0) + (1 - alpha) * norm_score

            ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

            top_docs = [self.corpus[i] for i, _ in ranked[:k]]
            all_results.append(top_docs)

        return all_results

    # main API
    def retrieve(self, query, k=5):
        return self.hybrid_search(query, k=k)

    def retrieve_batch(self, queries, k=5):
        return self.hybrid_search_batch(queries, k=k)