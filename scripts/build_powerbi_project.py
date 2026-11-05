"""Build a native PBIP/PBIR report from the CSV import assets (no extra packages).

Open the generated project in Desktop and Refresh to populate the imported model.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services.powerbi_export import HEADERS


def build(exports, destination):
    exports, destination = Path(exports).resolve(), Path(destination)
    if (destination / "ArchiveMind.pbip").exists():
        print("Existing project preserved. ExportFolder can be changed in Desktop; use a new --output folder to generate a separate draft.")
        return destination / "ArchiveMind.pbip"
    report = destination / 'ArchiveMind.Report'
    model = destination / 'ArchiveMind.SemanticModel'

    def write(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

    def schema(kind, version):
        return f'https://developer.microsoft.com/json-schemas/fabric/item/report/definition/{kind}/{version}/schema.json'

    write(destination / 'ArchiveMind.pbip', {'version': '1.0', 'artifacts': [{'report': {'path': 'ArchiveMind.Report'}}], 'settings': {'enableAutoRecovery': True}})
    write(report / 'definition.pbir', {'version': '4.0', 'datasetReference': {'byPath': {'path': '../ArchiveMind.SemanticModel'}}})
    write(model / 'definition.pbism', {'version': '1.0', 'settings': {}})
    measures = []
    for line in (ROOT / 'powerbi/dax/measures.dax').read_text(encoding='utf-8-sig').splitlines():
        if not line or line.startswith('//'):
            continue
        name, expression = line.split(' = ', 1)
        fmt = '0.0%' if 'Confidence' in name or 'Percentage' in name else '0.0' if 'Duration' in name else '0'
        if name == 'Pair Similarity':
            fmt = '0.000'
        measures.append({'name': name, 'expression': expression, 'formatString': fmt})
    ints = {'document_id', 'collection_id', 'archive_id', 'document_id_a', 'document_id_b'}
    decimals = {'ocr_confidence', 'prediction_confidence', 'processing_duration_seconds', 'x', 'y', 'cosine_similarity'}
    dates = {'digitized_at', 'created_at', 'completed_at'}
    tables = []
    for name in [*HEADERS, 'similarity_heatmap']:
        headers = HEADERS.get(name, HEADERS['semantic_similarity'])
        columns = [{'name': col, 'dataType': 'int64' if col in ints else 'double' if col in decimals else 'dateTime' if col in dates else 'string',
                    'sourceColumn': col, 'summarizeBy': 'none'} for col in headers]
        query = (ROOT / 'powerbi/powerquery' / (name + '.pq')).read_text(encoding='utf-8-sig')
        table = {'name': name, 'columns': columns, 'partitions': [{'name': name, 'mode': 'import', 'source': {'type': 'm', 'expression': query.splitlines()}}]}
        if name == 'documents':
            table['measures'] = measures
        tables.append(table)
    relationships = [{'name': name + '_document', 'fromTable': name, 'fromColumn': 'document_id',
                      'toTable': 'documents', 'toColumn': 'document_id', 'crossFilteringBehavior': 'oneDirection'}
                     for name in ['ai_analytics', 'semantic_vectors']]
    write(model / 'model.bim', {'name': 'ArchiveMind', 'compatibilityLevel': 1567, 'model': {
        'culture': 'en-US', 'defaultPowerBIDataSourceVersion': 'powerBI_V3',
        'expressions': [{'name': 'ExportFolder', 'kind': 'm', 'expression': json.dumps(exports.as_posix()) + ' meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]'}],
        'tables': tables, 'relationships': relationships}})
    write(report / 'definition/version.json', {'$schema': schema('versionMetadata','1.0.0'), 'version': '2.0.0'})
    write(report / 'definition/report.json', {'$schema': schema('report', '3.0.0'), 'themeCollection': {}, 'settings': {'useStylableVisualContainerHeader': True}})
    pages = ['overview', 'processing', 'semantic']
    write(report / 'definition/pages/pages.json', {'$schema': schema('pagesMetadata', '1.0.0'), 'pageOrder': pages, 'activePageName': 'overview'})

    def column(table, col):
        return {'Column': {'Expression': {'SourceRef': {'Entity': table}}, 'Property': col}}

    def measure(name):
        return {'Measure': {'Expression': {'SourceRef': {'Entity': 'documents'}}, 'Property': name}}

    def literal(value):
        return {'expr': {'Literal': {'Value': json.dumps(value) if not isinstance(value, str) else "'" + value.replace("'", "''") + "'"}}}

    def visual(page, name, kind, title, roles, x, y, w, h):
        state = {}
        for role, fields in roles.items():
            state[role] = {'projections': [{'field': field, 'queryRef': f'{role}_{i}', 'nativeQueryRef': f'{role}_{i}'} for i, field in enumerate(fields)]}
        v = {'$schema': schema('visualContainer', '2.4.0'), 'name': name,
             'position': {'x': x, 'y': y, 'z': 0, 'height': h, 'width': w, 'tabOrder': 0},
             'visual': {'visualType': kind, 'query': {'queryState': state},
                        'visualContainerObjects': {'title': [{'properties': {'show': literal(True), 'text': literal(title), 'fontSize': literal(12)}}]}}}
        write(report / 'definition/pages' / page / 'visuals' / name / 'visual.json', v)
        return v

    names = ['Archival Overview', 'AI and Document Processing', 'Semantic Search and Similarity']
    for page, name in zip(pages, names):
        write(report / 'definition/pages' / page / 'page.json', {'$schema': schema('page', '2.0.0'), 'name': page, 'displayName': name, 'displayOption': 'FitToPage', 'width': 1280, 'height': 800})
    for i, name in enumerate(['Total Manuscripts', 'Total Collections', 'Total Archives']):
        visual('overview', f'kpi{i}', 'card', name, {'Values': [measure(name)]}, 20 + i*280, 15, 260, 110)
    for i, col in enumerate(['collection_name', 'era', 'physical_condition']):
        visual('overview', f'slicer{i}', 'slicer', col.replace('_', ' ').title(), {'Values': [column('documents', col)]}, 900, 15 + i*245, 360, 230)
    for i, (col, kind, title) in enumerate([('physical_condition','donutChart','Manuscripts by Physical Condition'), ('era','barChart','Manuscripts by Era'), ('archive_name','columnChart','Manuscripts by Archive'), ('digitized_at','lineChart','Digitization Timeline')]):
        visual('overview', f'chart{i}', kind, title, {'Category': [column('documents', col)], 'Y': [measure('Total Manuscripts')]}, 20+(i%2)*435, 145+(i//2)*315, 415, 290)
    for i, name in enumerate(['Average OCR Confidence', 'Average Processing Duration']):
        visual('processing', f'kpi{i}', 'card', name + ' (blank = no recorded data)', {'Values': [measure(name)]}, 20+i*620, 15, 600, 100)
    specs = [('ai_analytics','ocr_engine','barChart','OCR Engine Usage','OCR Results'), ('ai_analytics','predicted_condition','donutChart','Predicted Document Conditions','Predicted Documents'), ('ai_analytics','predicted_condition','columnChart','ML Prediction Confidence','Average ML Confidence'), ('ingestion_analytics','status','columnChart','Ingestion Jobs by Status','Total Ingestion Jobs'), ('ingestion_analytics','created_at','lineChart','Ingestion Jobs Over Time','Total Ingestion Jobs')]
    for i,(table,col,kind,title,m) in enumerate(specs):
        roles = {'Category': [column(table,col)], 'Y': [measure(m)]}
        if i == 3:
            roles['Series'] = [column(table,'status')]
        visual('processing', f'chart{i}', kind, title, roles, 20+(i%3)*420, 135+(i//3)*325, 400, 300)
    visual('semantic','scatter','scatterChart','Stored embeddings: provenance unverified; domains use title keywords', {'X':[{'Aggregation':{'Expression':column('semantic_vectors','x'),'Function':1}}], 'Y':[{'Aggregation':{'Expression':column('semantic_vectors','y'),'Function':1}}], 'Category':[column('semantic_vectors','document_id')], 'Series':[column('semantic_vectors','domain')], 'Tooltips':[column('semantic_vectors','title')]},20,15,760,400)
    visual('semantic','domains','barChart','Semantic Domain Distribution',{'Category':[column('semantic_vectors','domain')], 'Y':[measure('Semantic Documents')]},800,15,460,400)
    heat = visual('semantic','heatmap','pivotTable','Pairwise cosine similarity (384-D); diagonal intentionally blank',{'Rows':[column('similarity_heatmap','title_a')], 'Columns':[column('similarity_heatmap','title_b')], 'Values':[{'Aggregation':{'Expression':column('similarity_heatmap','cosine_similarity'),'Function':3}}]},20,435,760,345)
    # Conditional formatting is a field-based rule bound to the actual matrix value.
    gradient = {'linearGradient2': {
        'min': {'color': {'Literal': {'Value': "'#F5F7FA'"}}, 'value': {'Literal': {'Value': '-1D'}}},
        'max': {'color': {'Literal': {'Value': "'#00A86B'"}}, 'value': {'Literal': {'Value': '1D'}}}}}
    rule = {'FillRule': {'Input': {'Aggregation': {'Expression': column('similarity_heatmap','cosine_similarity'), 'Function':3}}, 'FillRule': gradient}}
    heat['visual']['objects'] = {'values': [{'selector': {'metadata': 'Values_0'},
        'properties': {'backColor': {'solid': {'color': {'expr': rule}}}}}]}
    write(report / 'definition/pages/semantic/visuals/heatmap/visual.json', heat)
    ranking = visual('semantic','ranking','tableEx','Similarity Ranking (unique pairs)',{'Values':[column('semantic_similarity',c) for c in ['title_a','title_b','cosine_similarity']]},800,435,460,345)
    ranking['visual']['query']['sortDefinition'] = {'sort':[{'field':column('semantic_similarity','cosine_similarity'),'direction':'Descending'}]}
    write(report / 'definition/pages/semantic/visuals/ranking/visual.json', ranking)
    return destination / 'ArchiveMind.pbip'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', type=Path, default=ROOT / 'powerbi/exports')
    parser.add_argument('--output', type=Path, default=ROOT / 'powerbi/ArchiveMind')
    args = parser.parse_args()
    print(build(args.exports, args.output))
