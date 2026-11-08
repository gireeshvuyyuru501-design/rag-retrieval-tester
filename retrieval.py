"""Dependency-free TF-IDF retrieval evaluation on public SQuAD v1.1."""
import argparse
import json
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path

URL = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def load_squad(path):
    path = Path(path)
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    raw = json.loads(path.read_text(encoding='utf-8'))
    contexts, questions = [], []
    for article in raw['data']:
        for paragraph in article['paragraphs']:
            idx = len(contexts)
            contexts.append(paragraph['context'])
            for qa in paragraph['qas']:
                questions.append((qa['question'], idx))
    return contexts, questions

def build_index(contexts):
    docs = [Counter(tokenize(c)) for c in contexts]
    df = Counter()
    for doc in docs:
        df.update(doc.keys())
    n = len(docs)
    idf = {t: math.log((n + 1) / (f + 1)) + 1 for t, f in df.items()}
    vectors = []
    for doc in docs:
        vec = {t: (1 + math.log(freq)) * idf[t] for t, freq in doc.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1
        vectors.append({t: v / norm for t, v in vec.items()})
    return idf, vectors

def retrieve(question, index, k=5):
    idf, vectors = index
    q = Counter(tokenize(question))
    qvec = {t: (1 + math.log(freq)) * idf[t] for t, freq in q.items() if t in idf}
    norm = math.sqrt(sum(v * v for v in qvec.values())) or 1
    qvec = {t: v / norm for t, v in qvec.items()}
    scores = [(sum(weight * doc.get(t, 0) for t, weight in qvec.items()), i) for i, doc in enumerate(vectors)]
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [i for _, i in scores[:k]]

def metrics(ranks, k):
    if not ranks:
        return {'questions': 0, 'hit_at_k': 0.0, 'mrr_at_k': 0.0}
    return {'questions': len(ranks), 'hit_at_k': sum(r is not None and r <= k for r in ranks)/len(ranks),
            'mrr_at_k': sum(1/r for r in ranks if r is not None and r <= k)/len(ranks)}

def evaluate(contexts, questions, k=5):
    index = build_index(contexts)
    ranks = []
    for question, relevant in questions:
        retrieved = retrieve(question, index, k)
        ranks.append(retrieved.index(relevant)+1 if relevant in retrieved else None)
    return metrics(ranks, k)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--k', type=int, default=5)
    args = p.parse_args()
    if args.limit < 1 or args.k < 1:
        p.error('--limit and --k must be positive')
    contexts, questions = load_squad('squad_dev.json')
    report = {'dataset': 'SQuAD v1.1 dev', 'source': URL, 'retriever': 'offline TF-IDF cosine similarity',
              'corpus_passages': len(contexts), 'k': args.k, **evaluate(contexts, questions[:args.limit], args.k)}
    Path('retrieval_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
