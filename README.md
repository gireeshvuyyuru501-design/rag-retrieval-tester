# RAG Retrieval Tester

A reproducible, dependency-free **offline retrieval benchmark** built using Python TF-IDF and the public SQuAD v1.1 development set.

## What it does

- Downloads real SQuAD question/passage pairs on first run.
- Indexes passage text using normalized TF-IDF vectors.
- Retrieves the top-k passages for each question.
- Measures **Hit@K** and **MRR@K** using the passage containing the original question as the relevant passage.
- Writes `retrieval_report.json` locally.

## Run

```bash
python -m unittest discover -s tests -v
python retrieval.py --limit 100 --k 5
```

Internet is required only for the first dataset download. The unit tests run offline.

## Limitations

This is **not a production RAG system** and does not generate answers or call Gemini. The relevance labels are derived from SQuAD's question/passage association. This is a simple lexical baseline, not a learned embedding model. Benchmark results depend on the selected question subset and should not be represented as production performance.

## Roadmap

FastAPI endpoint, BM25/embedding baselines, retrieval error analysis, and integration with a shared React dashboard.
