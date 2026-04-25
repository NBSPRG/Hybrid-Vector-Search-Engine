# KeaBuilder ML Service

An intelligent, production-grade ML microservice designed to act as a sidecar to the KeaBuilder Node.js backend. This service provides semantic search, embeddings, and feature-flagged model switching.

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

