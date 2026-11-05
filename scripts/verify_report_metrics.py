import csv
from pathlib import Path

def load_csv(name):
    p = Path('powerbi/exports') / name
    with open(p, 'r', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

docs = load_csv('documents.csv')
ai = load_csv('ai_analytics.csv')
ingest = load_csv('ingestion_analytics.csv')
vectors = load_csv('semantic_vectors.csv')
sim = load_csv('semantic_similarity.csv')

print(f"Total Manuscripts: {len(set(d['document_id'] for d in docs))}")
print(f"Total Collections: {len(set(d['collection_id'] for d in docs if d['collection_id']))}")
print(f"Total Archives: {len(set(d['archive_id'] for d in docs if d['archive_id']))}")

from collections import Counter
print("Eras:", dict(Counter(d['era'] for d in docs)))
print("Physical Conditions:", dict(Counter(d['physical_condition'] for d in docs)))
print("Archives:", dict(Counter(d['archive_name'] for d in docs)))

ocr_conf = [float(r['ocr_confidence']) for r in ai if r.get('ocr_confidence')]
print("OCR confidence values count:", len(ocr_conf))
print("Ingestion jobs count:", len(ingest))
print("Semantic vectors count:", len(vectors))
sim_vals = [float(r['cosine_similarity']) for r in sim]
print(f"Semantic similarity pairs: {len(sim_vals)}, Min: {min(sim_vals):.4f}, Max: {max(sim_vals):.4f}")

