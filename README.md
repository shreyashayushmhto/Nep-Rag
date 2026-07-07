# Nep-RAG — Streamlit Frontend

नेपाली दस्तावेज पुनःप्राप्ति प्रणाली · Nepali Document Retrieval Augmented Generation UI.

## Run it

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What's real vs. mocked

- **Real:** the entire UI — layout, bilingual (Nepali/English) copy, sidebar
  upload + knowledge-base panel, Chat vs. Search mode toggle, chat transcript,
  citation "ticket" cards, dark/light theme, PDF text extraction on upload.
- **Mocked (search `REPLACE WITH REAL BACKEND CALL` in `app.py`):**
  - `retrieve()` — currently a keyword-overlap search over a tiny in-memory
    corpus + your uploaded files. Swap in your embedding model + vector
    store (FAISS, Milvus, pgvector, Qdrant, etc.).
  - `generate_answer()` — currently stitches retrieved snippets into a
    template sentence. Swap in a real LLM call (an example using the
    Anthropic API is commented directly in that function).
  - Image OCR on upload is a no-op placeholder — wire in `pytesseract` (with
    the `nep` Devanagari trained data) or a hosted OCR API there.

## Design notes

The visual identity is built around Nepal's own document culture rather than
generic AI-app styling:

- **Seal mark** — the "न" logo is styled like an official government stamp
  (छाप), echoing that Gorkhapatra is Nepal's national gazette.
- **Citation tickets** — sources are shown as archive-ticket cards with a
  dotted "perforation" top edge, echoing physical newspaper clippings and
  library index cards — the same motif as the seal, used sparingly.
- **Palette** — lokta-paper cream, sindoor red, Himalayan indigo, and a
  muted gold, pulled from the Nepali flag and tika/temple palette rather than
  a generic light-cream + terracotta AI-app template.
- **Type** — Yatra One for the Devanagari display mark, Noto Sans Devanagari
  for readable body text, Fraunces + Work Sans for English headings/labels.

All theming lives in `inject_css()` near the top of `app.py` — change the
`tokens` dicts there to adjust the palette for light/dark mode.
