# ArchiveMind — AI-Powered Hybrid SQL–Vector Semantic Search Engine
**VDUI Project**

---

## 1. System Architecture Overview

ArchiveMind is a complete **Distributed Edge-AI, Computer Vision, Real Semantic Search, and Archival Forensics Platform with a Nothing OS Brutalist Archival User Interface**.

```
┌──────────────────────────────────────────────┐          Direct Ethernet Link (eno1)          ┌────────────────────────────────────────────────────────┐
│        Raspberry Pi 3B+ Edge Device          │ ───────────────────────────────────────────── │       Central AI Processing Server (Laptop)            │
│  IP: 10.42.0.116 (archivemind-pi)            │       HTTP API & Ingestion Payloads           │  IP: 10.42.0.1 (FastAPI Port 8000)                     │
│                                              │                                               │                                                        │
│  • 5MP Camera Module (ov5647, 1920x1080)     │  1. POST /api/devices/register                │  • NVIDIA RTX 2050 Laptop GPU (4 GB VRAM)              │
│  • Headless Systemd Service (Autostart)      │  2. POST /api/ingestion/upload (Multipart)    │  • AMD Ryzen 5 7535HS (16 GB RAM)                      │
│  • Local Persistent Retry Queue              │  3. Remote Trigger: POST :8082/capture        │  • OpenCV & PIL Image Enhancement Pipeline             │
│  • rpicam-still (Headless Mode)              │                                               │  • sentence-transformers/all-MiniLM-L6-v2 (384-D)      │
│  • Hardware Capture Button (BCM GPIO 17)     │                                               │  • MobileNetV3-Small Archival Condition Classifier     │
└──────────────────────────────────────────────┘                                               │  • TrOCR Historical Handwriting Line Recognition      │
                                                                                               │  • SQLite FTS5 Full-Text Search Virtual Table          │
                                                                                               │  • Local Grounded RAG Assistant (Qwen2.5 / Extractive) │
                                                                                               │  • Nothing OS Brutalist Archival Intelligence Web UI   │
                                                                                               └────────────────────────────────────────────────────────┘
```

---

## 2. Hardware & Network Configuration

| Component | Specification | Network Configuration | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| **Raspberry Pi** | Raspberry Pi 3B+ (1GB RAM, Debian Trixie 64-bit) | IP: `10.42.0.116` | Physical acquisition station; ov5647 still capture |
| **Laptop Host** | AMD Ryzen 5 7535HS, 16GB RAM, Arch/Linux | IP: `10.42.0.1` (`eno1`) | Main API server, database manager, UI host |
| **Laptop GPU** | NVIDIA GeForce RTX 2050 (4GB GDDR6 VRAM) | CUDA 12.x / 13.x | Model inference acceleration (MiniLM, MobileNet, TrOCR) |
| **Camera Module** | Raspberry Pi Camera Module v1 (5MP ov5647) | CSI Ribbon Cable | Native 1080p still captures via `rpicam-still -n` |
| **Pi Autostart** | Systemd unit (`archivemind-camera.service`) | Auto-runs on boot | Headless daemon listening on port `8082` |

---

## 3. The 8 Integrated AI/ML & Vision Modules

### Module 1: Archival Manuscript Image Enhancement
- **Engine**: OpenCV & Pillow (`backend/services/image_processing.py`)
- **Operations**:
  - Image validation and resolution verification
  - High-precision **Contour / Hough Deskewing** ($\pm 25^\circ$)
  - **Bilateral Filtering** ($d=9, \sigma=75$) removing paper grain noise while preserving sharp ink stroke edges
  - **CLAHE (Contrast Limited Adaptive Histogram Equalization)** enhancing faded iron gall inks
  - **Gaussian Adaptive Thresholding** for binarized legibility
  - Generates side-by-side comparative views without mutating original captures

### Module 2: OCR for Printed Historical Text
- **Engine**: PaddleOCR / Tesseract-v5 with structural layout analysis (`backend/services/ocr_service.py`)
- **Operations**:
  - Detects typographic text blocks
  - Sorts bounding boxes in natural historical reading order
  - Computes character and line-level confidence scores
  - Allows instant manual correction via web dashboard

### Module 3: Historical Handwriting Recognition (HTR)
- **Engine**: Inspired by `microsoft/trocr-base-handwritten` (`backend/services/htr_service.py`)
- **Operations**:
  - Morphological horizontal kernel line segmentation
  - Individual line cropping and vertical coordinate sorting
  - Transcription of historical cursive handwriting (Newton, Da Vinci, Ramanujan)
  - Supports user selection: *Printed*, *Handwritten*, or *Auto-Detect*

### Module 4: Real Semantic Dense Embeddings
- **Engine**: `sentence-transformers/all-MiniLM-L6-v2` (`backend/services/embedding_service.py`)
- **Operations**:
  - Real **384-dimensional** dense vector representations
  - Sliding-window text chunking with 25-word overlap
  - Unit vector normalization: $\|v\|_2 = 1.0$
  - Serialized as binary float32 BLOBs directly in SQLite (`chunk_embedding`)
  - No deterministic fake vectors in the search retrieval pipeline

### Module 5: Hybrid SQL + Vector Search Engine
- **Engine**: Linear Weighted Fusion + Reciprocal Rank Fusion (`backend/services/hybrid_search.py`)
- **Operations**:
  - **Keyword Path**: SQLite FTS5 `bm25(document_fts)` query scoring
  - **Vector Path**: Real-time cosine similarity $\cos(\vec{q}, \vec{d})$ over 384-d embeddings
  - **Score Fusion**:
    $$\text{HybridScore} = \alpha \cdot \text{Norm}(S_{\text{FTS5}}) + (1 - \alpha) \cdot S_{\text{MiniLM}}$$
  - **Reciprocal Rank Fusion (RRF)**:
    $$\text{RRF}(d) = \frac{1}{60 + r_{\text{FTS5}}(d)} + \frac{1}{60 + r_{\text{MiniLM}}(d)}$$
  - Interactive $\alpha$ slider from `0% SQL / 100% VEC` (Pure Semantic) to `100% SQL / 0% VEC` (Pure Keyword)

### Module 6: Physical Condition Classification
- **Engine**: MobileNetV3-Small Archival Assessment (`backend/services/classification_service.py`)
- **Classes**: `Pristine`, `Good`, `Stained`, `Fragile`, `Critical-Decay`
- **Features**: CIELAB $b^*$ yellowing entropy, border tear laplacian variance, surface texture roughness

### Module 7: Handwriting Visual Forensics
- **Engine**: Contour Moments & Stroke Slant Descriptors (`backend/services/handwriting_analysis.py`)
- **Features**: Log Hu Moments, stroke density, dominant stroke angle, 128-d geometric contour moments

### Module 8: Local AI Research Assistant (Grounded RAG)
- **Engine**: Local Retrieval-Augmented Generation (`backend/services/rag_service.py`)
- **Operations**:
  - User asks natural language archival inquiries
  - Hybrid engine retrieves relevant manuscript passages
  - Grounded answers generated via local Ollama (`qwen2.5:3b`) or extractive synthesizer
  - Strict academic citations: `[Doc #ID: Title (Author, Date)]`
  - Hallucination guardrail: Explicitly reports when primary evidence is insufficient

---

## 4. Database Architecture & Relational Schema

Preserves all PES University DBMS capstone deliverables in SQLite and PostgreSQL:

```sql
-- Relational Tables
role                     -- User roles (Head Archivist, Researcher, Tech, Auditor)
app_user                 -- System credentials and RBAC profiles
archive                  -- International repositories (Cambridge, Vatican, etc.)
collection               -- Curated collections across historical eras
access_permission        -- Granular permissions (READ, WRITE, ADMIN)
document                 -- Core manuscript metadata, SHA-256, physical condition
person                   -- Historical authors (Newton, Da Vinci, Ramanujan, Turing)
document_author          -- Attribution type and confidence score
handwriting_sample       -- 128-d geometric forensic vectors
document_view            -- Custody audit trail logging view accesses
citation                 -- Generated citations in Chicago, MLA, APA formats

-- FTS5 Full-Text Search Virtual Table
document_fts             -- SQLite FTS5 table indexing title and OCR text cache

-- AI & Hardware Ingestion Migration Tables
device                   -- Raspberry Pi acquisition units (IP, last seen, status)
ingestion_job            -- Background acquisition jobs and state tracking
document_image           -- Original & enhanced image filesystem pointers, SHA-256
ocr_result               -- Raw text, corrected text, confidence, engine metadata
document_chunk           -- Overlapping text windows for dense semantic retrieval
chunk_embedding          -- 384-dimensional float32 BLOB embeddings (all-MiniLM-L6-v2)
ml_prediction            -- MobileNetV3 physical condition predictions & metrics
model_registry           -- Pretrained models, versions, dimensions, and active state
```

### Database Triggers:
1. `trg_validate_doc_checksum`: Aborts updates if cryptographic SHA-256 checksum is truncated or invalid.
2. `trg_audit_citation_generation`: Automatically logs an entry in `document_view` whenever an archival citation is generated.

---

## 5. Quick Start & Execution

### 1. Prerequisites
- **Python 3.11+**
- **uv** (Extreme-speed package manager)
- **Ethernet Cable** connected between Laptop and Raspberry Pi

### 2. Deploy to Raspberry Pi (One-Time Setup)
To configure the Raspberry Pi 3B+ as a headless auto-running acquisition station:
```bash
chmod +x pi_client/deploy_to_pi.sh
bash pi_client/deploy_to_pi.sh
```
*Installs `archivemind-camera.service` on the Pi so the camera daemon starts automatically on every boot.*

### 3. Start the Central Laptop Server
From the project root:
```bash
chmod +x run.sh
./run.sh
```
Access the dashboard at **`http://localhost:8000`** in your browser.

---

## 6. Live Demonstration Script (For Faculty Viva & Evaluators)

1. **Overview & Hardware Telemetry**: Open dashboard, show live ping to Raspberry Pi (`10.42.0.116`), active NVIDIA RTX 2050 GPU, and 100% SHA-256 validated folios.
2. **Raspberry Pi Physical Acquisition**: Navigate to **Live Pi Scanner**, click **"Capture from Pi Camera"**. Show physical camera shutter, live transfer over Ethernet, and automatic CLAHE deskewing.
3. **OCR & HTR Extraction**: Switch between *Printed* and *Handwritten* modes. Show editable transcription box and click *"Save Correction & Regenerate MiniLM Embeddings"*.
4. **Hybrid Search Demonstrator**:
   - Search for concept: `"gravitation inverse square"`
   - Drag $\alpha$ slider from 0.0 (Pure Semantic) to 1.0 (Pure Keyword)
   - Show how semantic search retrieves Newton even with different vocabulary
   - Show RRF ranking and passage highlights
5. **Local AI Research Assistant**: Ask: *"What are the mock theta functions described in Ramanujan's letter?"* Show grounded synthesis with clickable citations.
6. **Archival Evaluation Queries & RBAC**: Run Query 7.1 (GROUP BY), attempt Mutation 7.2 as Researcher (blocked by RBAC), switch role to Head Archivist, and execute successfully.
7. **Audit Trail**: Open Audit tab to show trigger-generated log rows with client IP and timestamps.

---

## 7. License & Attribution
Developed as part of the **VDUI Project**.  
All historical manuscript transcripts belong to their respective archival repositories (Cambridge Digital Archives, Vatican Apostolic Library, National Archives of India, Bletchley Park Trust).

## Power BI analytics
See [Power BI integration](powerbi/README.md) for CSV export, the three-page Desktop report, and manual refresh. Contribution: Aayaan Kamil.
