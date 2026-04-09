# recursive chunking
def chunk_text(text, max_words=200):
    separators = ["\n\n", "\n", ". ", " "]

    def split_text(t, sep_idx):
        if len(t.split()) <= max_words or sep_idx >= len(separators):
            return [t]

        sep = separators[sep_idx]
        parts = t.split(sep)

        chunks = []
        current = ""

        for p in parts:
            if len((current + sep + p).split()) <= max_words:
                current += (sep + p)
            else:
                if current:
                    chunks.extend(split_text(current, sep_idx + 1))
                current = p

        if current:
            chunks.extend(split_text(current, sep_idx + 1))

        return chunks

    return [c.strip() for c in split_text(text, 0) if c.strip()]


def apply_chunking(dataset):
    """
    Apply chunking to all documents in dataset.

    Input:
        dataset: list of {
            question,
            answers,
            documents: [context]
        }

    Output:
        same structure but documents replaced by chunks
    """

    new_dataset = []

    for item in dataset:
        all_chunks = []

        for doc in item["documents"]:
            chunks = chunk_text(doc)
            all_chunks.extend(chunks)

        # optional: deduplicate chunks
        seen = set()
        dedup_chunks = []
        for c in all_chunks:
            if c not in seen:
                dedup_chunks.append(c)
                seen.add(c)

        item["documents"] = dedup_chunks
        new_dataset.append(item)

    return new_dataset