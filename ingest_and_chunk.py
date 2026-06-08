"""
ingest_and_chunk.py
-------------------
Milestone 3 — Ingestion and Chunking Pipeline
McNeese State University Professor Reviews RAG System

Pipeline:
  1. Load each .txt file from the documents/ directory
  2. Parse the header block (professor-level metadata)
  3. Split on '---' delimiter to isolate individual reviews
  4. Parse per-review metadata fields (course, date, rating)
  5. Apply content-based noise filter
  6. Return a list of chunk dicts ready for ChromaDB ingestion
"""

import os
import re

# ── Configuration ────────────────────────────────────────────────────────────

DOCUMENTS_DIR = "documents_clean"

# Noise filter: a chunk must contain at least one of these signals to be kept
DOMAIN_KEYWORDS = {
    "exam", "grade", "lecture", "homework", "attendance",
    "textbook", "quiz", "test", "midterm", "final", "syllabus",
    "assignment", "project", "lab", "notes", "slides", "office hours"
}

SENTIMENT_KEYWORDS = {
    "loved", "hated", "dreaded", "amazing", "awful", "worst", "best",
    "terrible", "great", "horrible", "fantastic", "avoid", "recommend",
    "excellent", "outstanding", "disappointing", "frustrating", "helpful",
    "useless", "brilliant", "incompetent", "caring", "rude", "kind",
    "disorganized", "organized", "clear", "confusing"
}

# Regex to detect a numeric rating anywhere in the chunk text
RATING_PATTERN = re.compile(r'\b\d+(\.\d+)?\s*/\s*5\b|\bquality\s*:\s*\d', re.IGNORECASE)

# Regex to detect a course code (e.g. CSCI309, BIOL225, MATH190)
COURSE_CODE_PATTERN = re.compile(r'\b[A-Z]{2,5}\s*\d{3,4}[A-Z]?\b')


# ── Helper: parse the header block ───────────────────────────────────────────

def parse_header(header_text: str) -> dict:
    """
    Extract professor-level metadata from the header block at the top
    of each .txt file.

    Returns a dict with keys: professor, department
    (overall_rating and summary are professor-level, not stored per chunk)
    """
    metadata = {
        "professor": "",
        "department": "",
    }

    for line in header_text.splitlines():
        line = line.strip()
        if line.lower().startswith("professor:"):
            metadata["professor"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("department:"):
            metadata["department"] = line.split(":", 1)[1].strip()

    return metadata


# ── Helper: parse a single review block ──────────────────────────────────────

def parse_review_block(block: str) -> dict:
    """
    Extract per-review metadata and review text from a single review block.

    Expected fields in block:
        Course, Date, Quality, Difficulty, Grade, Tags, Review

    Returns a dict with keys:
        course, date, rating, text
    """
    data = {
        "course": "Unknown",
        "date": "Unknown",
        "rating": None,
        "text": ""
    }

    lines = block.strip().splitlines()
    review_lines = []
    in_review = False

    for line in lines:
        stripped = line.strip()

        if in_review:
            review_lines.append(stripped)
            continue

        lower = stripped.lower()

        if lower.startswith("course:"):
            data["course"] = stripped.split(":", 1)[1].strip()
        elif lower.startswith("date:"):
            data["date"] = stripped.split(":", 1)[1].strip()
        elif lower.startswith("quality:"):
            try:
                data["rating"] = float(stripped.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif lower.startswith("review:"):
            # Everything after "Review:" on this line + subsequent lines
            first_part = stripped.split(":", 1)[1].strip()
            if first_part:
                review_lines.append(first_part)
            in_review = True

    data["text"] = " ".join(review_lines).strip()
    return data


# ── Noise filter ─────────────────────────────────────────────────────────────

def passes_noise_filter(review_text: str) -> bool:
    """
    A chunk passes if its review text contains at least ONE of:
      - a numeric quality rating (e.g. "Quality: 4.0" already parsed, but
        check raw text too for inline mentions like "3/5")
      - a course code (e.g. CSCI309, BIOL225)
      - a domain keyword
      - a sentiment keyword

    Empty or near-empty reviews are always rejected.
    """
    if len(review_text.strip()) < 10:
        return False

    text_lower = review_text.lower()

    # Check for inline rating pattern (e.g. "4/5", "3.5/5")
    if RATING_PATTERN.search(review_text):
        return True

    # Check for course code in the review text itself
    if COURSE_CODE_PATTERN.search(review_text):
        return True

    # Check for domain keywords
    words = set(re.findall(r'\b\w+\b', text_lower))
    if words & DOMAIN_KEYWORDS:
        return True

    # Check for sentiment keywords
    if words & SENTIMENT_KEYWORDS:
        return True

    return False


# ── Main: load and chunk a single file ───────────────────────────────────────

def ingest_file(filepath: str) -> list[dict]:
    """
    Load one professor .txt file and return a list of chunk dicts.

    Each chunk dict has:
        text       : the review text (no professor name injected)
        professor  : str  ← metadata only
        department : str  ← metadata only
        course     : str  ← metadata only
        date       : str  ← metadata only
        rating     : float or None  ← metadata only
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into header and reviews section on "--- REVIEWS ---"
    parts = re.split(r"---\s*REVIEWS\s*---", content, maxsplit=1, flags=re.IGNORECASE)

    if len(parts) != 2:
        print(f"  [WARNING] Could not find '--- REVIEWS ---' in {filepath}. Skipping.")
        return []

    header_text, reviews_text = parts
    professor_meta = parse_header(header_text)

    # Split individual reviews on lines that are just dashes (--- or ---)
    # The delimiter in the files is a line containing only "---"
    raw_blocks = re.split(r'\n---+\n', reviews_text)

    chunks = []
    kept = 0
    dropped = 0

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue

        review_data = parse_review_block(block)

        # Skip blocks with no review text
        if not review_data["text"]:
            dropped += 1
            continue

        # Apply noise filter
        if not passes_noise_filter(review_data["text"]):
            dropped += 1
            continue

        chunk = {
            "text": review_data["text"],
            "professor": professor_meta["professor"],
            "department": professor_meta["department"],
            "course": review_data["course"],
            "date": review_data["date"],
            "rating": review_data["rating"],
        }
        chunks.append(chunk)
        kept += 1

    print(f"  {professor_meta['professor']}: {kept} chunks kept, {dropped} dropped")
    return chunks


# ── Main: load all files ──────────────────────────────────────────────────────

def ingest_all(documents_dir: str = DOCUMENTS_DIR) -> list[dict]:
    """
    Load all .txt files from documents_dir and return all chunks.
    """
    all_chunks = []

    txt_files = sorted([
        f for f in os.listdir(documents_dir) if f.endswith(".txt")
    ])

    if not txt_files:
        print(f"[ERROR] No .txt files found in '{documents_dir}'")
        return []

    print(f"[INFO] Found {len(txt_files)} files in '{documents_dir}':\n")

    for filename in txt_files:
        filepath = os.path.join(documents_dir, filename)
        chunks = ingest_file(filepath)
        all_chunks.extend(chunks)

    print(f"\n[INFO] Total chunks after noise filter: {len(all_chunks)}")
    return all_chunks


# ── Inspection: print 5 sample chunks ────────────────────────────────────────

def print_sample_chunks(chunks: list[dict], n: int = 5):
    """
    Print n evenly-spaced sample chunks for manual inspection.
    For each chunk, ask: does this make sense on its own?
    Could someone answer a question from this chunk alone?
    """
    print("\n" + "="*70)
    print(f"SAMPLE CHUNKS ({n} of {len(chunks)} total)")
    print("="*70)

    step = max(1, len(chunks) // n)
    samples = [chunks[i * step] for i in range(n) if i * step < len(chunks)]

    for i, chunk in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"  Professor  : {chunk['professor']}")
        print(f"  Department : {chunk['department']}")
        print(f"  Course     : {chunk['course']}")
        print(f"  Date       : {chunk['date']}")
        print(f"  Rating     : {chunk['rating']}")
        print(f"  Text       : {chunk['text']}")
    print("\n" + "="*70)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chunks = ingest_all()

    if chunks:
        print_sample_chunks(chunks, n=5)

        # Quick stats
        professors = {}
        for c in chunks:
            professors.setdefault(c["professor"], 0)
            professors[c["professor"]] += 1

        print("\nChunks per professor:")
        for prof, count in sorted(professors.items()):
            print(f"  {prof}: {count}")
