"""Read-only CSV reporting. Never initializes ArchiveMindDB or loads ML models."""
from contextlib import closing
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADERS = {
    'documents': 'document_id,title,doc_type,date_written,digitized_at,physical_condition,collection_name,archive_name,era,author_names,ocr_status,collection_id,archive_id'.split(','),
    'ai_analytics': 'document_id,document_title,ocr_engine,ocr_confidence,predicted_condition,prediction_confidence,model_name,correction_status'.split(','),
    'ingestion_analytics': 'job_id,scan_id,device_name,status,created_at,completed_at,processing_duration_seconds'.split(','),
    'semantic_vectors': 'document_id,title,domain,collection,era,x,y'.split(','),
    'semantic_similarity': 'document_id_a,document_id_b,title_a,title_b,cosine_similarity'.split(','),
}


def duration(start, end):
    try:
        def parse(value):
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        seconds = (parse(end) - parse(start)).total_seconds()
        return seconds if seconds >= 0 else None
    except (ValueError, TypeError, AttributeError, OverflowError):
        return None


def export_powerbi(db_path=ROOT / 'database/archivemind.db', output=ROOT / 'powerbi/exports'):
    path = Path(db_path).resolve()
    if not path.is_file():
        raise ValueError(f'SQLite database not found: {path}. Pass --database PATH to an existing ArchiveMind database.')
    output = Path(output)
    warnings = []
    data = {name: [] for name in HEADERS}
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA query_only=ON')
        conn.execute('BEGIN')
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {'document', 'collection', 'archive', 'person', 'document_author'}
        if required - tables:
            raise ValueError('Not an ArchiveMind database; missing tables: ' + ', '.join(sorted(required - tables)))
        data['documents'] = [dict(r) for r in conn.execute('''
            SELECT d.document_id,d.title,d.doc_type,d.date_written,d.digitized_at,
                   d.physical_condition,c.name collection_name,a.name archive_name,c.era,
                   (SELECT group_concat(full_name, '; ') FROM
                       (SELECT p.full_name FROM document_author da JOIN person p USING(person_id)
                        WHERE da.document_id=d.document_id ORDER BY p.full_name,p.person_id)) author_names,
                   d.ocr_status,c.collection_id,a.archive_id
            FROM document d LEFT JOIN collection c USING(collection_id)
            LEFT JOIN archive a USING(archive_id) ORDER BY d.document_id''')]
        for doc in data['documents']:
            row = dict.fromkeys(HEADERS['ai_analytics'])
            row.update(document_id=doc['document_id'], document_title=doc['title'])
            if {'ocr_result', 'document_image'} <= tables:
                ocr = conn.execute('''SELECT o.* FROM ocr_result o JOIN document_image i USING(image_id)
                    WHERE i.document_id=? ORDER BY julianday(o.created_at) DESC,o.ocr_id DESC LIMIT 1''', (doc['document_id'],)).fetchone()
                if ocr:
                    row.update(ocr_engine=ocr['engine'], ocr_confidence=ocr['confidence'],
                               correction_status='Corrected' if ocr['corrected_text'] is not None else 'Uncorrected')
            if 'ml_prediction' in tables:
                pred = conn.execute('''SELECT * FROM ml_prediction WHERE document_id=?
                    AND prediction_type='Physical Preservation Assessment'
                    ORDER BY julianday(created_at) DESC,prediction_id DESC LIMIT 1''', (doc['document_id'],)).fetchone()
                if pred:
                    row.update(predicted_condition=pred['predicted_label'], prediction_confidence=pred['confidence'], model_name=pred['model_name'])
            data['ai_analytics'].append(row)
        if 'ingestion_job' in tables:
            device = 'device' in tables
            query = '''SELECT j.job_id,j.scan_id,{} device_name,j.status,j.created_at,j.completed_at
                       FROM ingestion_job j {} ORDER BY j.created_at,j.job_id'''.format(
                           'd.device_name' if device else 'NULL',
                           'LEFT JOIN device d USING(device_id)' if device else '')
            for r in conn.execute(query):
                row = dict(r)
                row['processing_duration_seconds'] = duration(row['created_at'], row['completed_at'])
                data['ingestion_analytics'].append(row)
        # Shared service reads the SAME transaction, with no global DB_PATH mutation.
        if {'chunk_embedding', 'document_chunk'} <= tables:
            from backend.services import vector_viz
            space = vector_viz.compute_vector_space(connection=conn)
            data['semantic_vectors'] = space['nodes']
            matrix = vector_viz.compute_similarity_matrix(connection=conn)
            docs = matrix['documents']
            for i, a in enumerate(docs):
                for j in range(i + 1, len(docs)):
                    b = docs[j]
                    data['semantic_similarity'].append(dict(document_id_a=a['document_id'], document_id_b=b['document_id'],
                        title_a=a['title'], title_b=b['title'], cosine_similarity=matrix['matrix'][i][j]))
        if not data['semantic_vectors']:
            warnings.append('No valid 384-dimensional nonzero finite embeddings; semantic CSVs contain headers only.')
        warnings.append('Embedding provenance is unverified: existing ArchiveMind seed/offline paths store fallback vectors under the MiniLM model name. These exports do not certify genuine MiniLM inference. Domains are title/collection keyword labels, not learned clusters.')
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in data.items():
        target = output / (name + '.csv')
        temp = target.with_suffix('.csv.tmp')
        with temp.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS[name], extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(target)
    manifest = dict(exported_at=datetime.now(timezone.utc).isoformat(), row_counts={n: len(r) for n,r in data.items()},
                    warnings=warnings, evidence='SQLite snapshot; provenance unverified', similarity='Upper triangle, excludes self; actual 384-D cosine [-1,1]')
    (output / 'export_summary.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
