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


def chunk_text_finance(text, max_words=300):
    separators = ["\n\n", "\n", " "]

    def split_text(t, sep_idx):
        if len(t.split()) <= max_words or sep_idx >= len(separators):
            return [t]

        sep = separators[sep_idx]
        parts = t.split(sep)

        chunks = []
        current = ""

        for p in parts:
            candidate = (current + sep + p) if current else p

            if len(candidate.split()) <= max_words:
                current = candidate
            else:
                if current:
                    chunks.extend(split_text(current, sep_idx + 1))
                current = p

        if current:
            chunks.extend(split_text(current, sep_idx + 1))

        return chunks

    return [c.strip() for c in split_text(text, 0) if c.strip()]


def add_overlap(chunks, overlap_words=50):
    new_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            new_chunks.append(chunk)
            continue

        prev_words = chunks[i-1].split()[-overlap_words:]
        new_chunk = " ".join(prev_words) + " " + chunk
        new_chunks.append(new_chunk)

    return new_chunks

def apply_chunking(dataset, datatype="financebench"):
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
            if datatype == "financebench":
                # chunks = chunk_text_finance(doc)
                # chunks = add_overlap(chunks)
                chunk = doc
            else:
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