"""
load_and_clean.py
-----------------
Milestone 3 — Step 2 & 3: Load and Clean Documents
McNeese State University Professor Reviews RAG System

This script:
  1. Loads each raw .txt file from documents/
  2. Cleans each document:
       - Decodes to UTF-8, normalizes line endings
       - Strips leftover HTML entities (&amp; &nbsp; etc.)
       - Removes excessive blank lines
       - Strips trailing whitespace from every line
       - Keeps all substantive content: review text, ratings,
         course codes, professor name, department
  3. Saves each cleaned file to documents_clean/
  4. Prints a before/after diff summary so you can verify nothing
     important was removed
"""

import os
import re
import html

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_DIR   = "documents"
CLEAN_DIR = "documents_clean"

# ── Cleaning logic ────────────────────────────────────────────────────────────

def clean_document(raw_text: str) -> str:
    """
    Apply all cleaning steps to a single document string.

    Steps:
      1. Normalize line endings (Windows CRLF → LF)
      2. Decode HTML entities  (&amp; → & , &nbsp; → space, etc.)
      3. Strip trailing whitespace from every line
      4. Collapse runs of 3+ blank lines down to 2
      5. Strip leading/trailing whitespace from the whole document
    """

    # 1. Normalize line endings
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Decode HTML entities (handles &amp; &nbsp; &lt; &gt; &#39; etc.)
    text = html.unescape(text)

    # 3. Strip trailing whitespace from every line
    lines = [line.rstrip() for line in text.split("\n")]

    # 4. Collapse runs of more than 2 consecutive blank lines into 2
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    # 5. Strip leading/trailing whitespace from the whole document
    text = "\n".join(cleaned_lines).strip()

    return text


def summarize_diff(original: str, cleaned: str, filename: str):
    """
    Print a short before/after summary for manual verification.
    """
    orig_lines   = original.count("\n")
    clean_lines  = cleaned.count("\n")
    orig_chars   = len(original)
    clean_chars  = len(cleaned)
    removed_chars = orig_chars - clean_chars

    print(f"  {filename}")
    print(f"    Lines  : {orig_lines} → {clean_lines}  (diff: {clean_lines - orig_lines})")
    print(f"    Chars  : {orig_chars} → {clean_chars}  (removed: {removed_chars})")

    # Warn if we lost a suspiciously large amount of content
    if removed_chars > 500:
        print(f"    ⚠ Large reduction — review this file manually")
    elif removed_chars == 0:
        print(f"    ✓ No changes needed")
    else:
        print(f"    ✓ Minor whitespace/encoding cleanup only")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_and_clean_all():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    txt_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".txt")])

    if not txt_files:
        print(f"[ERROR] No .txt files found in '{RAW_DIR}'")
        return

    print(f"[INFO] Loading and cleaning {len(txt_files)} files...\n")

    for filename in txt_files:
        raw_path   = os.path.join(RAW_DIR,   filename)
        clean_path = os.path.join(CLEAN_DIR, filename)

        with open(raw_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        cleaned_text = clean_document(raw_text)

        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        summarize_diff(raw_text, cleaned_text, filename)
        print()

    print(f"[INFO] Cleaned files saved to '{CLEAN_DIR}/'")
    print(f"[INFO] Next step: run ingest_and_chunk.py pointing at '{CLEAN_DIR}/'")


if __name__ == "__main__":
    load_and_clean_all()
