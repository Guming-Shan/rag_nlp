import re
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score


# utils
def detect_answer_type(text):
    text = text.strip().lower()

    # yes/no
    if text in ["yes", "no"]:
        return "binary"

    # numeric (%, $, number)
    if re.search(r"\d", text):
        return "numeric"

    return "text"

def extract_number(text):
    """
    extract first number in string
    """
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(nums[0]) if nums else None

def numeric_match(pred, gold, tol=0.05):
    """
    relative tolerance match
    """
    p = extract_number(pred)
    g = extract_number(gold)

    if p is None or g is None:
        return 0

    if g == 0:
        return abs(p - g) == 0

    return abs(p - g) / abs(g) <= tol

def binary_match(pred, gold):
    return pred.strip().lower() == gold.strip().lower()


def normalize(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_pred_list(pred_answer):
    """
    unify prediction format -> List[{"text": str}]
    """
    if pred_answer is None:
        return []

    if isinstance(pred_answer, str):
        return [{"text": pred_answer}]

    if isinstance(pred_answer, list):
        if len(pred_answer) == 0:
            return []
        if isinstance(pred_answer[0], str):
            return [{"text": x} for x in pred_answer]
        return pred_answer

    return []


def split_claims(text):
    sents = re.split(r"[。.!?;]", text)
    return [s.strip() for s in sents if s.strip()]


# semantic support
def semantic_support(claim, documents, model, threshold=0.5):
    if not documents:
        return False

    claim_emb = model.encode(claim, normalize_embeddings=True)

    doc_embs = model.encode(documents, normalize_embeddings=True)

    sims = np.dot(doc_embs, claim_emb)

    return np.max(sims) > threshold


# smaller utility
def semantic_match(a, b):
    vec = TfidfVectorizer().fit([a, b])
    v = vec.transform([a, b])
    return cosine_similarity(v[0], v[1])[0][0]


def doc_contains_answer(doc_text, answers):
    doc_text = normalize(doc_text)
    for a in answers:
        if normalize(a) in doc_text:
            return True
    return False


# metrics
def exact_match(pred, gold):
    """
    strict exact match after normalization
    """
    return normalize(pred) == normalize(gold)

def soft_hit_rate(preds, golds, threshold=0.3):
    """
    semantic version of hit_rate
    """
    if not preds or not golds:
        return 0

    pred_texts = [p["text"] for p in preds]

    for g in golds:
        for p in pred_texts:
            vec = TfidfVectorizer().fit([g, p])
            v = vec.transform([g, p])
            sim = cosine_similarity(v[0], v[1])[0][0]

            if sim > threshold:
                return 1

    return 0

def hit_rate(preds, golds):
    if not preds or not golds:
        return 0

    pred_texts = [p["text"] for p in preds]

    for g in golds:
        g_type = detect_answer_type(g)

        for p in pred_texts:

            p_type = detect_answer_type(p)

            # numeric
            if g_type == "numeric":
                if numeric_match(p, g):
                    return 1

            # binary
            elif g_type == "binary":
                if binary_match(p, g):
                    return 1

            # text fallback (semantic)
            else:
                vec = TfidfVectorizer().fit([g, p])
                v = vec.transform([g, p])
                sim = cosine_similarity(v[0], v[1])[0][0]

                if sim > 0.15:
                    return 1

    return 0


def hallucination_rate(preds, documents, model):
    """
    embedding-based hallucination detection
    """

    if not preds:
        return 0

    total_claims = 0
    unsupported_claims = 0

    for p in preds:
        claims = split_claims(p["text"])

        for c in claims:
            total_claims += 1

            if not semantic_support(c, documents, model):
                unsupported_claims += 1

    return unsupported_claims / max(total_claims, 1)


def factscore(preds, documents, model):
    """
    embedding-based factual consistency
    """

    if not preds:
        return 0

    total = 0
    supported = 0

    for p in preds:
        claims = split_claims(p["text"])

        for c in claims:
            total += 1

            if semantic_support(c, documents, model):
                supported += 1

    return supported / max(total, 1)


def citation_metrics(pred_docs, gold_docs, gold_answers=None, threshold=0.3):
    """
    Better version: evidence-level retrieval evaluation
    """

    if not pred_docs:
        return 0, 0

    # Precision: retrieved docs 是否接近 gold evidence
    precision_hit = 0

    for d in pred_docs:
        d_text = d.get("text", "")

        best_sim = 0
        for g in gold_docs:
            sim = semantic_match(d_text, g)
            best_sim = max(best_sim, sim)

        if best_sim > threshold:   # threshold
            precision_hit += 1

    precision = precision_hit / len(pred_docs)

    # Recall: gold evidence 是否被覆盖
    recall_hit = 0

    for g in gold_docs:
        for d in pred_docs:
            if semantic_match(d.get("text", ""), g) > threshold:
                recall_hit += 1
                break

    recall = recall_hit / len(gold_docs) if gold_docs else 0

    return precision, recall


# embedding metrics (ROUGE + BERTScore)
def bertscore_metric(preds, golds):
    if not preds or not golds:
        return 0

    pred_text = " ".join([p["text"] for p in preds])
    gold_text = " ".join(golds)

    P, R, F1 = bert_score([pred_text], [gold_text], lang="en")
    return F1.mean().item()


def semantic_similarity(preds, golds):
    """
    TF-IDF cosine similarity
    """
    if not preds or not golds:
        return 0

    pred_text = " ".join([p["text"] for p in preds])
    gold_text = " ".join(golds)

    vec = TfidfVectorizer().fit([pred_text, gold_text])
    v = vec.transform([pred_text, gold_text])

    return cosine_similarity(v[0], v[1])[0][0]


# grouping

def build_gold_index(dataset):
    gold_index = defaultdict(list)

    for item in dataset:
        gold_index[item["question"]].append(item)

    return gold_index


# main eval

def evaluate_all(dataset, model_outputs):

    gold_index = build_gold_index(dataset)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    results = {
        "EM": 0,
        "hit_rate": 0,
        "citation_precision": 0,
        "citation_recall": 0,
        "hallucination_rate": 0,
        "factscore": 0,
        "soft_hit_rate": 0,
        "semantic_similarity": 0,
        "bertscore": 0
    }

    n = 0

    for pred_obj in tqdm(model_outputs):
        question = pred_obj["question"]

        if question not in gold_index:
            continue

        items = gold_index[question]
        n += 1

        pred_answer = pred_obj.get("pred_answer")
        pred_docs = pred_obj.get("docs", [])

        preds = to_pred_list(pred_answer)

        gold_answers = []
        gold_docs = []

        for item in items:
            gold_answers.extend(item.get("answers", []))
            gold_docs.extend(item.get("documents", []))

        gold_answers = list(set(gold_answers))

    
        # metrics aggregation
        results["EM"] += max([exact_match(p["text"], g) for p in preds for g in gold_answers])
        
        results["hit_rate"] += hit_rate(preds, gold_answers)
        results["soft_hit_rate"] += soft_hit_rate(preds, gold_answers, threshold=0.3)

        cp, cr = citation_metrics(pred_docs, gold_docs, gold_answers)
        results["citation_precision"] += cp
        results["citation_recall"] += cr

        results["hallucination_rate"] += hallucination_rate(preds, gold_docs, model)

        results["factscore"] += factscore(preds, gold_docs, model)

        results["semantic_similarity"] += semantic_similarity(preds, gold_answers)

        results["bertscore"] += bertscore_metric(preds, gold_answers)

    for k in results:
        results[k] /= max(n, 1)

    return results