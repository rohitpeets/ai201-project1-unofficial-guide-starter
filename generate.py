"""
generate.py
-----------
Milestone 5 — Generation and Interface
McNeese State University Professor Reviews RAG System
"""

import os
from dotenv import load_dotenv
from groq import Groq
import gradio as gr

from embed_and_retrieve import retrieve, embed_and_store
from ingest_and_chunk import ingest_all

# ── Load environment variables ────────────────────────────────────────────────

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file.")

groq_client = Groq(api_key=GROQ_API_KEY)

# ── Load and embed chunks on startup ─────────────────────────────────────────

print("[INFO] Loading and embedding chunks...")
chunks = ingest_all()
embed_and_store(chunks)
print("[INFO] Ready.\n")

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant for McNeese State University students.
You help students make informed decisions about which professors to take based on real student reviews.

STRICT GROUNDING RULES:
- Answer ONLY using the information provided in the retrieved reviews below.
- Do NOT use your general training knowledge about professors, universities, or teaching.
- If the retrieved reviews do not contain enough information to answer the question, say exactly:
  "I don't have enough information in the current reviews to answer that question."
- Do NOT make up or infer anything not explicitly stated in the reviews.
- Do NOT give advice beyond what the reviews actually say.

SOURCE ATTRIBUTION RULES:
- After your answer, always include a "Sources:" section.
- List each review you drew from, formatted as:
  - [Professor Name] | [Course] | [Date] | Rating: [X/5]
- Only list sources you actually used in your answer.

Keep your answer clear, helpful, and concise."""


# ── Core ask function ─────────────────────────────────────────────────────────

def ask(query: str) -> dict:
    results = retrieve(query)

    if not results:
        return {
            "answer": "I don't have reviews for that professor in the current dataset.",
            "sources": []
        }

    context_blocks = []
    for i, r in enumerate(results, 1):
        block = (
            f"Review {i}:\n"
            f"Professor: {r['professor']}\n"
            f"Course: {r['course']}\n"
            f"Date: {r['date']}\n"
            f"Rating: {r['rating']}/5\n"
            f"Text: {r['text']}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    user_message = f"""Retrieved reviews:

{context}

---

Student question: {query}

Answer the question using ONLY the reviews above. Cite your sources at the end."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    sources = [
        f"{r['professor']} | {r['course']} | {r['date']} | Rating: {r['rating']}/5"
        for r in results
    ]

    return {"answer": answer, "sources": sources}


# ── Gradio handler ────────────────────────────────────────────────────────────

def handle_query(question: str):
    if not question.strip():
        return "Please enter a question.", ""

    result  = ask(question)
    answer  = result["answer"]
    sources = "\n".join(f"• {s}" for s in result["sources"])
    return answer, sources


# ── Custom CSS ────────────────────────────────────────────────────────────────

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Merriweather:wght@700&display=swap');

:root {
    --bg:        #f5f5f0;
    --surface:   #ffffff;
    --border:    #e0ddd6;
    --accent:    #2d6a4f;
    --accent-lt: #40916c;
    --text:      #1a1a1a;
    --muted:     #6b7280;
    --label:     #2d6a4f;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
}

.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* Header */
.app-header {
    background: var(--surface);
    border-bottom: 2px solid var(--accent);
    padding: 2.5rem 3rem 2rem;
    text-align: center;
}

.app-header .eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.6rem;
}

.app-header h1 {
    font-family: 'Merriweather', serif !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    margin-bottom: 0.5rem !important;
}

.app-header .subtitle {
    font-size: 0.88rem;
    color: var(--muted);
    font-weight: 300;
}

/* Body */
.app-body {
    padding: 2rem 3rem;
    background: var(--bg);
}

/* Input card */
.input-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

textarea, input[type="text"] {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.85rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(45, 106, 79, 0.12) !important;
    outline: none !important;
}

textarea::placeholder {
    color: #aaa !important;
}

/* Button */
button.primary {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 6px !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2rem !important;
    margin-top: 1rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.1s !important;
}

button.primary:hover {
    background: var(--accent-lt) !important;
    transform: translateY(-1px) !important;
}

/* Output cards */
.output-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

.output-card label,
.output-card label span {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--label) !important;
    font-family: 'Inter', sans-serif !important;
}

.output-card textarea {
    border: none !important;
    background: transparent !important;
    color: var(--text) !important;
    font-size: 0.92rem !important;
    font-weight: 400 !important;
    line-height: 1.75 !important;
    resize: none !important;
    padding: 0.5rem 0 0 !important;
}

.sources-card label,
.sources-card label span {
    color: var(--muted) !important;
}

/* Footer */
.app-footer {
    background: var(--surface);
    border-top: 1px solid var(--border);
    padding: 1.25rem 3rem;
    text-align: center;
}

.app-footer p {
    font-size: 0.75rem;
    color: var(--muted);
    line-height: 1.8;
}

.app-footer .accent-text {
    color: var(--accent);
    font-weight: 600;
}

footer { display: none !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(css=custom_css, title="McNeese Professor Guide") as demo:

    gr.HTML("""
    <div class="app-header">
        <div class="eyebrow">McNeese State University</div>
        <h1>Unofficial Professor Guide</h1>
        <div class="subtitle">Answers grounded in real student reviews — no hallucination, no guessing.</div>
    </div>
    """)

    with gr.Column(elem_classes="app-body"):
        with gr.Group(elem_classes="input-card"):
            inp = gr.Textbox(
                label="Your Question",
                placeholder='e.g. "Does Hardee give partial credit on exams?"',
                lines=2,
            )
            btn = gr.Button("Ask", variant="primary")

        with gr.Row(equal_height=True):
            with gr.Group(elem_classes="output-card"):
                answer = gr.Textbox(
                    label="Answer",
                    lines=12,
                    interactive=False,
                )
            with gr.Group(elem_classes="output-card sources-card"):
                sources = gr.Textbox(
                    label="Retrieved From",
                    lines=12,
                    interactive=False,
                )

    gr.HTML("""
    <div class="app-footer">
        <p><span class="accent-text">Available Professors</span><br>
        Andrew Mudd &middot; Bei Xie &middot; Constance Kersten &middot; Jennifer Lavergne &middot; Lara Guidroz
        &middot; Lyle Hardee &middot; Shaikh Samad &middot; Susie Beasley &middot; Tristan Salinas &middot; Vipin Menon</p>
    </div>
    """)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch()
