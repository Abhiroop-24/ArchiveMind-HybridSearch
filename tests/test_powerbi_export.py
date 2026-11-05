"""Offline reporting contracts, using the repository's actual schema and seeds."""
from contextlib import closing
import csv
import hashlib
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services.powerbi_export import HEADERS, duration, export_powerbi
from database.db_manager import ArchiveMindDB
from unittest.mock import patch


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / 'input.db'
        self.out = Path(self.temp.name) / 'exports'
        # Use existing seed metadata without any model initialization or fallback vectors.
        with patch.object(ArchiveMindDB, '_seed_ai_models', lambda *args: None):
            ArchiveMindDB(str(self.db))

    def read(self, name):
        with (self.out / (name + '.csv')).open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, HEADERS[name])
            return list(reader)

    def test_counts_privacy_readonly_and_rerun(self):
        before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        result = export_powerbi(self.db, self.out)
        docs = self.read('documents')
        with closing(sqlite3.connect(self.db)) as c, c:
            self.assertEqual(len(docs), c.execute('SELECT COUNT(*) FROM document').fetchone()[0])
        self.assertEqual(len({r['document_id'] for r in docs}), len(docs))
        self.assertTrue(any(';' in r['author_names'] for r in docs))
        for name in HEADERS:
            self.read(name)
            content = (self.out / (name + '.csv')).read_text(encoding='utf-8-sig')
            for private in ['password_hash', 'ip_address', '@pes.edu', '10.42.0.116', 'hash_teacher_admin']:
                self.assertNotIn(private, content)
        self.assertEqual(result['row_counts']['semantic_vectors'], 0)
        self.assertTrue(all(r['ocr_confidence'] == '' and r['prediction_confidence'] == '' for r in self.read('ai_analytics')))
        initial = {n: (self.out / (n+'.csv')).read_bytes() for n in HEADERS}
        export_powerbi(self.db, self.out)
        self.assertEqual(initial, {n: (self.out / (n+'.csv')).read_bytes() for n in HEADERS})
        self.assertEqual(before, hashlib.sha256(self.db.read_bytes()).hexdigest())

    def test_latest_records_and_duration(self):
        with closing(sqlite3.connect(self.db)) as c, c:
            c.execute("INSERT INTO document_image(image_id,document_id,original_path,sha256) VALUES(1,1,'private','checksum')")
            for i in [1,2]:
                c.execute("INSERT INTO ocr_result(ocr_id,image_id,engine,model_name,raw_text,corrected_text,confidence,created_at) VALUES(?,1,?,'model','text',?,?,'2025-01-01')", (i, f'engine{i}', '' if i==2 else None, i/10))
                c.execute("INSERT INTO ml_prediction(prediction_id,document_id,model_name,model_version,prediction_type,predicted_label,confidence,created_at) VALUES(?,1,'model','v1','Physical Preservation Assessment',?,?,'2025-01-01')", (i, f'label{i}', i/10))
            c.execute("INSERT INTO ml_prediction(document_id,model_name,model_version,prediction_type,predicted_label,confidence,created_at) VALUES(1,'other','v2','unrelated','ignore',1,'2026-01-01')")
            c.execute("INSERT INTO ingestion_job(job_id,scan_id,original_image_path,status,created_at,completed_at) VALUES('job','scan','private','COMPLETED','2025-01-01T00:00:00Z','2025-01-01T00:00:12Z')")
        export_powerbi(self.db, self.out)
        row = self.read('ai_analytics')[0]
        self.assertEqual((row['ocr_engine'],row['predicted_condition'],row['correction_status']), ('engine2','label2','Corrected'))
        self.assertEqual(float(self.read('ingestion_analytics')[0]['processing_duration_seconds']), 12)
        self.assertIsNone(duration('broken', '2025-01-01'))
        self.assertIsNone(duration('2025-01-02', '2025-01-01'))
        self.assertEqual(duration('2025-01-01T00:00:00+01:00', '2024-12-31T23:00:05Z'), 5)

    def test_vectors_cosine_not_projected_distance(self):
        with closing(sqlite3.connect(self.db)) as c, c:
            for i, sign in [(1,1),(2,-1)]:
                vec = np.zeros(384, dtype=np.float32); vec[0] = sign
                c.execute('INSERT INTO document_chunk(chunk_id,document_id,chunk_index,chunk_text) VALUES(?,?,0,?)', (i,i,'fixture'))
                c.execute("INSERT INTO chunk_embedding(chunk_id,model_name,embedding_blob) VALUES(?,'test-fixture',?)", (i,vec.tobytes()))
        export_powerbi(self.db, self.out)
        vectors = self.read('semantic_vectors')
        self.assertEqual(len(vectors),2)
        for row in vectors:
            self.assertTrue(np.isfinite([float(row['x']),float(row['y'])]).all())
        pairs = self.read('semantic_similarity')
        self.assertEqual(len(pairs),1)
        self.assertAlmostEqual(float(pairs[0]['cosine_similarity']), -1)
        with closing(sqlite3.connect(self.db)) as c, c:
            c.execute('DELETE FROM chunk_embedding WHERE chunk_id=2')
        export_powerbi(self.db, self.out)  # Single-vector projection must work too.
        self.assertEqual(len(self.read('semantic_vectors')),1)
        self.assertEqual(len(self.read('semantic_similarity')),0)

    def test_empty_optional_tables_and_invalid_embeddings(self):
        with closing(sqlite3.connect(self.db)) as c, c:
            c.execute("INSERT INTO document_chunk(chunk_id,document_id,chunk_index,chunk_text) VALUES(1,1,0,'fixture')")
            c.execute("INSERT INTO chunk_embedding(chunk_id,model_name,embedding_blob) VALUES(1,'test',?)", (np.full(384,np.nan,dtype=np.float32).tobytes(),))
        export_powerbi(self.db,self.out)
        self.assertEqual(self.read('semantic_vectors'),[])
        with closing(sqlite3.connect(self.db)) as c, c:
            for table in ['ocr_result','ml_prediction','ingestion_job','device','chunk_embedding','document_chunk']:
                c.execute('DROP TABLE '+table)
            c.execute('DELETE FROM document')
        export_powerbi(self.db,self.out)
        for name in HEADERS:
            self.assertEqual(self.read(name),[])

    def test_cli_actionable_missing_database(self):
        result = subprocess.run([sys.executable,str(ROOT/'scripts/export_powerbi.py'),'--database',str(self.db.with_name('missing.db'))], capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('--database',result.stderr)
        self.assertFalse(self.db.with_name('missing.db').exists())


if __name__ == '__main__':
    unittest.main()
