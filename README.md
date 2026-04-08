# KDD
# KDD RAG Prototype

Minimal prototype for a structure-aware adaptive retrieval paper idea:

- retrieve an initial set of documents
- analyze evidence structure
- estimate evidence sufficiency
- retrieve more only when needed

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/main.py
```

## Current scope

- toy retriever
- evidence features: relevance, redundancy, diversity, supportiveness
- rule-based sufficiency estimator
- two actions: `answer_now` or `retrieve_more`

## Next steps

- replace toy retriever with FAISS or BM25
- replace placeholder generator with a real LLM call
- connect a real QA dataset such as HotpotQA or Natural Questions
