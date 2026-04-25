# KeaBuilder ML Service

An intelligent, production-grade ML microservice designed to act as a sidecar to the KeaBuilder Node.js backend. This service provides semantic search, embeddings, and feature-flagged model switching.

## Node.js Integration Flow

KeaBuilder Node.js backend remains the primary API gateway. This Python service is called over REST:

1. Node.js -> `POST /embed` when users create landing pages/prompts.
2. Node.js -> `POST /similarity` for synchronous search and ranking.
3. Node.js -> `POST /predict/async` for heavy jobs that should run in background.
4. Node.js -> `GET /job/{job_id}` to poll async task completion.

This keeps ML compute isolated from the Node.js event loop while preserving simple HTTP integration.

## 🏗 Architecture Overview

This project implements a sidecar architecture, the standard pattern for serving ML models alongside Node.js backends (used by companies like Netflix and Uber).

1. **FastAPI (`app/main.py`)**: High-performance HTTP server handling routing, validation (Pydantic), and structured logging.
2. **Feature Flags**: Managed via Redis, allowing instant, zero-downtime switching between ML models without restarting the server.
3. **Hybrid Search (`app/similarity/`)**: Combines Dense (semantic meaning) and Sparse (exact keywords via TF-IDF) retrieval using Reciprocal Rank Fusion (RRF) for production-grade accuracy.
4. **Async Inference (`app/worker/`)**: Celery workers handle heavy ML workloads in the background, keeping the API responsive.
5. **Vector Database**: Qdrant is used to store and query generated vector embeddings.
6. **Knowledge Distillation**: The default `student_model.pt` (<5MB) was distilled from a large SentenceTransformer teacher model (420MB) to optimize for speed and cost.

> **Note on Training Code:** 
> The Knowledge Distillation training pipeline was executed on a GPU in Google Colab. 
> [Google Colab Notebook](https://colab.research.google.com/drive/1jXZ0yEBCgYpH2bY7Z4tEs2YF0Za55eAH?usp=sharing)
> 
> The resulting artifact (`student_model.pt`) is stored in the `/artifacts` directory. The training scripts are omitted from this production repository to keep it lightweight.

---

## 🚀 How to Start

### Option 1: Docker Compose (Recommended)

This command spins up the entire stack, including FastAPI, Redis, Qdrant, and the Celery workers.

```bash
docker-compose up --build
```
The API will be available at `http://localhost:8000`.

### Option 2: Local Python Environment

If you want to run the code locally, the application will automatically fall back to **in-memory** versions of Redis and Qdrant, making local development frictionless!

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 📡 Endpoints Supported

Once the server is running, visit `http://localhost:8000/docs` to interact with the auto-generated Swagger UI.

### 1. Feature Flags (Model Switching)
**`GET /flags`**
Returns the current configuration and the active model.

**`PUT /flags/active-model/{model_name}`**
Instantly routes all future traffic to the specified model (e.g., `student` or `minilm`).

### 2. Core ML
**`POST /embed`**
Converts input text into a high-dimensional vector and stores it in Qdrant.
```json
{
  "text": "Best landing page builder for e-commerce",
  "payload": {"category": "website"}
}
```

**`POST /similarity`**
Performs a fast, synchronous similarity search.
```json
{
  "query": "How to build an online store",
  "top_k": 5,
  "method": "hybrid" 
}
```
*(Method can be `dense`, `sparse`, or `hybrid`)*

### 3. Async Background Jobs
**`POST /predict/async`**
Enqueues a heavy search job to Celery and immediately returns a `job_id`.

**`GET /job/{job_id}`**
Poll the status of the async job. Returns `pending`, `running`, `done`, or `failed`.

### 4. Ops
**`GET /health`**
Liveness probe. Returns system status, loaded models, and database connectivity.

---

## 🤖 Model Selection Guide

This service ships with three encoder models, each suited for different scenarios. The active model can be switched at runtime via feature flags — no redeployment needed.

| Model | Dimensions | Size | Speed | Accuracy | Best For |
|-------|-----------|------|-------|----------|----------|
| **MiniLM** (`minilm`) | 384 | ~80 MB | ⚡ Fast | ✅ Good | Development, demos, production baseline |
| **Student** (`student`) | 128 | ~5 MB | ⚡⚡ Fastest | ⚠️ Requires teacher | Cost-sensitive production at scale |
| **Teacher** (`teacher`) | 768 | ~420 MB | 🐢 Slow | ✅✅ Best | Offline indexing, training data generation |

### When to use MiniLM (Default)
- **Local development & demos** — works out of the box with real semantic embeddings
- **Production baseline** — strong accuracy-to-speed ratio, no extra setup needed
- **E2E testing** — gives meaningful similarity search results immediately

```bash
# Set MiniLM as default
PUT /flags/active-model/minilm
```

### When to use Student
- **Production at scale** — 3× smaller vectors (128 vs 384) = lower storage & faster search
- **High-throughput inference** — lightweight MLP, no transformer overhead at query time
- **Cost-sensitive deployments** — smaller Qdrant collections, less memory

> ⚠️ **Important:** The student model requires the **teacher model** to be loaded to produce meaningful embeddings. The student pipeline is: `text → teacher.encode() → StudentMLP → 128-dim`.
>
> In **development mode** (without the teacher), the student uses random projections as a fallback — embeddings will be valid but **not semantically meaningful**. This is by design for testing the pipeline.
>
> To use the student model with real embeddings:
> 1. Set `APP_ENV=production` (or `staging`) so the teacher loads at startup
> 2. Ensure `artifacts/student_model.pt` exists (trained via knowledge distillation)
> 3. Switch: `PUT /flags/active-model/student`

### When to use Teacher
- **Offline batch indexing** — highest quality embeddings for building the initial vector index
- **Training the student** — generates the target embeddings for knowledge distillation
- **Quality benchmarking** — compare student/minilm results against the teacher baseline

> The teacher model (`all-mpnet-base-v2`) is **not loaded in development mode** to save memory. Set `APP_ENV=staging` or `production` to enable it.

### Switching Models at Runtime

```bash
# Check current model
GET /flags

# Switch to a different model (takes effect immediately)
PUT /flags/active-model/minilm
PUT /flags/active-model/student

# Per-request override (without changing the global default)
POST /embed  {"text": "...", "model": "minilm"}
POST /similarity  {"query": "...", "model": "student"}
```

> **Note on vector dimensions:** When switching between models with different dimensions (e.g., MiniLM 384-dim → Student 128-dim), the Qdrant collection is automatically recreated. This clears existing vectors since they are incompatible across dimensions. In production, use separate collections per model to avoid this.
