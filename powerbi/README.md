# ArchiveMind Power BI — Aayaan Kamil

The committed report contains Archival Overview, Document Analytics, and Semantic Search and Similarity.
The report was opened and CSV-refresh tested in Desktop; later small slicer/label formatting edits
were validated from file definitions. The development vectors are offline fallback embeddings.

CSV integration: **SQLite → exporter → CSV → Power BI Desktop Refresh**.
No live refresh, cloud publishing, credentials, or embedded report is configured.

## Quick demonstration

1. From the repository root, use your existing environment:
   ```bash
   .venv/bin/python scripts/export_powerbi.py --database database/archivemind.db --output powerbi/exports
   .venv/bin/python scripts/build_powerbi_project.py --exports powerbi/exports
   ```
   Windows equivalents: `.venv\Scripts\python.exe scripts/export_powerbi.py --output powerbi/exports`
   and `.venv\Scripts\python.exe scripts/build_powerbi_project.py --exports powerbi/exports`.
   Requires Python 3.11+ and the project's existing NumPy dependency. No model downloads.
   The exporter never initializes, seeds, or writes to the database. A missing database is an error;
   supply an existing ArchiveMind SQLite file with `--database`.
2. Open `powerbi/ArchiveMind/ArchiveMind.pbip` in Power BI Desktop. The generated model
   uses a configurable ExportFolder parameter. Before the first refresh, change **Transform data → Manage parameters → ExportFolder**.
3. Click **Home → Refresh** to load CSVs. If Desktop requests a privacy classification, choose it
   yourself according to the data you are importing. No online sign-in is needed for local reporting.
4. Review the three pages. Document Analytics shows available metadata and preservation risk.
5. After a new manuscript is ingested, rerun the exporter, then click Desktop **Refresh**.
   CSV changes alone do not update the imported model. Save the project after refreshing.

The PBIP contains native visuals and an imported semantic model. Do not run the builder while
editing the same project: it regenerates report definitions. Save UI edits first and keep a backup.
PBIP format reference: https://learn.microsoft.com/en-us/power-bi/developer/projects/projects-report

## Import manually (or rebuild the report)

1. In Desktop choose **Get data → Text/CSV** for each of the five files in `powerbi/exports`.
   Use UTF-8 and comma delimiter. Name tables exactly after the filename without `.csv`.
   Alternatively create a Text parameter `ExportFolder`, then five Blank Queries using
   **Advanced Editor** and the matching `powerquery/*.pq` files; these set column types and nulls.
   Keep `date_written` as Text because historical dates can be year-only.
2. In Model view, create single-direction relationships:
   `documents[document_id] (1) → ai_analytics[document_id] (*)` and
   `documents[document_id] (1) → semantic_vectors[document_id] (*)`.
   Leave ingestion and pairwise similarity disconnected: jobs do not reliably identify documents,
   and pair rows have two document roles. Do not create ambiguous relationships to both pair IDs.
3. **View → Themes → Browse for themes**, select `archivemind_theme.json`.
4. **Modeling → New measure**: paste each measure separately from `dax/measures.dax`.
   Format confidence and At-Risk Percentage as Percentage; source confidence values are proportions,
   not 0–100 scores. Duration is seconds. Missing confidence stays BLANK, never zero-filled.
   At-risk means Fragile or Critical-Decay; this is an explicit reporting definition.
5. Create the pages below. Each KPI is a Card. Use measures rather than summed IDs.

### 1. Archival Overview

| Visual | Category / axis | Value |
|---|---|---|
| Card | — | Total Manuscripts |
| Card | — | Total Collections |
| Card | — | Total Archives |
| Donut | documents[physical_condition] | Total Manuscripts |
| Bar | documents[era] | Total Manuscripts |
| Column | documents[archive_name] | Total Manuscripts |
| Line | documents[digitized_at] | Total Manuscripts |

Add slicers for collection_name, era, physical_condition. Timeline uses digitization dates,
not historical writing dates. Collections and archives count distinct stable IDs represented
in the filtered manuscripts, not distinct display names or empty entities without documents.

### 2. AI and Document Processing

| Visual | Category / axis | Value |
|---|---|---|
| Card | — | Average OCR Confidence |
| Bar | ai_analytics[ocr_engine] | OCR Results |
| Donut | ai_analytics[predicted_condition] | Predicted Documents |
| Column | ai_analytics[predicted_condition] | Average ML Confidence |
| Stacked column | ingestion_analytics[status] | Total Ingestion Jobs; legend=status |
| Line | ingestion_analytics[created_at] | Total Ingestion Jobs |
| Card | — | Average Processing Duration |

AI rows are one per manuscript, including manuscripts with no results. OCR Results counts
nonblank recorded confidence values, so manuscripts without OCR do not inflate engine usage.
Latest means created_at parsed by SQLite, descending, then highest primary key for ties.
OCR is selected across images; condition predictions are filtered to the application's
`Physical Preservation Assessment` type and selected independently. model_name is the condition
prediction model, not the OCR model. A non-null corrected_text is Corrected (including empty text).
Duration uses valid ISO timestamps, treats naive database timestamps as UTC, and excludes missing,
invalid or negative elapsed times. It measures job elapsed time, including queue time.
The current repository classifier is heuristic; a stored model name alone does not prove neural inference.

### 3. Semantic Search and Similarity

| Visual | Fields |
|---|---|
| Scatter | X=x, Y=y (Average), Legend=domain, Details=document_id, Tooltip=title |
| Matrix heatmap | Rows=title_a, Columns=title_b, Values=MAX(cosine_similarity) |
| Ranking table | title_a, title_b, cosine_similarity; descending similarity |
| Bar | domain; Semantic Documents |

Use semantic_vectors for scatter/distribution and semantic_similarity for the ranking.
For a symmetric heatmap, create a Blank Query named `similarity_heatmap` from
`powerquery/similarity_heatmap.pq`. It mirrors exported unordered pairs. Use that query in the
matrix and apply Values → cosine_similarity → Conditional formatting → Background color,
minimum -1, maximum 1. The self diagonal is intentionally blank; no artificial diagonal is added.
The PBIP already includes this derived query. Scatter interaction filters the domain chart;
the pair table is deliberately disconnected. Pair filtering can be added using separate role dimensions.

Coordinates reuse vector_viz PCA/SVD, normalized by its existing scale. First chunk is selected
by chunk_index then chunk_id consistently for projection and cosine. Vectors must contain 384
finite float32 values with nonzero norm. Zero, malformed and nonfinite vectors are omitted.
Cosine uses original 384-D vectors, **not 2D plot distance**; negative values are retained.
Domains are the existing title/collection keyword labels, not model-learned clusters.

## Evidence and privacy

This checkout initially had no SQLite database (Git ignores it). The delivered development CSV
snapshot was exported from the repository's 10 seed manuscripts, using the existing **offline
hash-projection fallback**, not genuine MiniLM inference. There are no recorded OCR results,
ML predictions or ingestion jobs in that snapshot. No confidences or processing jobs were invented.
Replace the input with your existing production/development database and rerun for your actual records.
Existing seed code labels fallback vectors with the MiniLM model name, so generic exports warn
that provenance cannot be certified from that field. `export_summary.json` records row counts and caveats.

Only allowlisted reporting columns are written. No users, email addresses, passwords, IP columns,
raw OCR text, image paths or error traces are exported. Historical author names and manuscript titles
are intentional reporting fields. Export links follow the application's existing local access model;
review data access before exposing the server outside its intended network.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p test_powerbi_export.py -v
```

Tests use isolated SQLite fixtures, the real schema and seed metadata; no GPU/network/hardware.
They cover headers/counts, multi-author grain, latest results, missing optional tables, valid/invalid
elapsed times, single/invalid vectors, negative cosine, sensitive fields, repeatability and unchanged DB bytes.

## Portable setup and final presentation

The committed ExportFolder default is `C:/ArchiveMind-HybridSearch/powerbi/exports`.
Set **Transform data → Manage parameters → ExportFolder** to your checkout's absolute
`powerbi/exports` folder, then Apply and Refresh. This is a local parameter, not a live connection.
The builder preserves an existing customized project; choose a different --output to create a new draft.
The publication copy uses the generic default; the contributor's working local parameter is preserved.
The final unchanged presentation is [ArchiveMind_VDUI.pptx](../docs/presentations/ArchiveMind_VDUI.pptx).
