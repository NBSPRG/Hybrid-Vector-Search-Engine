
## Additional Architecture Conceptual Questions

**1. Why use a dedicated Vector Database (like Qdrant) instead of a traditional relational database (like PostgreSQL) for this project?**

**Answer:** Traditional databases use B-trees, which are great for exact matches (like finding a specific ID). However, ML models output high-dimensional vectors (arrays of floats). Vector databases are specifically built to perform nearest-neighbor searches (like Cosine Similarity) across millions of these vectors instantly, which a standard SQL database cannot do efficiently.

**2. What is Hybrid Search, and why did we implement it using Reciprocal Rank Fusion (RRF)?**

**Answer:** Dense search (vectors) is great for understanding the meaning of a sentence, but it often misses exact keyword matches (like specific product IDs or names). Sparse search (TF-IDF/BM25) is great for keywords but doesn't understand context. We implemented Hybrid Search to run both simultaneously and used RRF to mathematically combine their scores, giving the user the best of both worlds.

**3. Why did we use Redis for Feature Flags instead of just updating the code to change models?**

**Answer:** If we hardcoded the active model in the Python code, we would have to rebuild the Docker container and restart the server (causing downtime) every time we wanted to switch models. By using Redis to store the active flag, we can instantly switch traffic between the student, minilm, or teacher models in real-time with zero downtime.

**4. Can you explain Knowledge Distillation and why it was necessary for this sidecar?**

**Answer:** Knowledge Distillation is the process of training a smaller, faster "student" model to mimic a massive, slow "teacher" model. We needed this because running a 420MB transformer model on a standard server uses too much CPU and memory. The distilled <5MB student model retains most of the teacher's accuracy but runs fast enough to be used in a real-time API.

**5. How does the Celery + Redis architecture prevent the main API from crashing during heavy tasks?**

**Answer:** It acts as a buffer. When a massive search request comes in, instead of the FastAPI server trying to process it (which would block all other incoming web traffic), it instantly drops a "message" into Redis and returns a `job_id`. The Celery worker picks up that message from Redis and does the heavy ML math in the background, keeping the main API 100% responsive.

---

## Original Assignment Questions

**Q1. In KeaBuilder, we may want to match similar user inputs (e.g., leads, prompts). Create a small system that finds the most similar input to a given query.**
*(See `simple_match.py` in the repository for the exact implementation script using TF-IDF and Cosine Similarity).*

**Q2. KeaBuilder uses Node.js backend. How would you serve an ML model in production?**
Because Node.js is single-threaded, running heavy ML models directly inside it would block the event loop and crash the web server. To solve this, I would use a **Sidecar Microservice Architecture** by isolating the ML models in a fast HTTP API (like FastAPI). The Node.js backend communicates with it via HTTP, and heavy workloads are offloaded to background Celery workers so Node.js never has to wait.

**Q3. Design a simple schema for User inputs and Predictions**
* **`user_inputs`**: `id` (UUID), `user_id` (UUID), `input_text` (Text), `source_type` (String), `created_at` (Timestamp)
* **`predictions`**: `id` (UUID), `input_id` (FK to user_inputs.id), `model_version` (String), `embedding_vector` (Vector), `confidence_score` (Float), `created_at` (Timestamp)

**Q4. If ML responses are slow: What is one way to handle this in UI?**
Use **Asynchronous Polling with a Loading Spinner**. The backend instantly returns a `job_id` and the UI shows a "Loading..." spinner while pinging the server every second with the `job_id` until the result is ready to display.

**Q5. Name 3 challenges when moving ML model from notebook → production**
1. **Model Size vs API Boot Time:** Large models cause APIs to timeout on startup (solved via Knowledge Distillation).
2. **Synchronous vs Asynchronous:** Sequential notebooks freeze web servers under high traffic (solved via Celery workers).
3. **Environment Consistency:** Colab code often breaks on CPU servers due to dependency mismatch (solved via Docker containerization).

**Q6. How would you approach LoRA for face consistency?**
Gather 15–20 high-quality photos of the face, crop and label them with a trigger word, and train a tiny "LoRA" adapter on just those images. During generation, plug the LoRA into the main model and use the trigger word in the prompt to guarantee the specific face.

**Q7. What tools, frameworks, or platforms have you worked with in real projects?**
* **Machine Learning:** PyTorch, SentenceTransformers, Hugging Face, Qdrant Vector DB
* **Backend:** Python, Go, Node.js, FastAPI, Celery, Redis
* **DevOps:** Docker, Docker Compose, GitHub Actions, Linux