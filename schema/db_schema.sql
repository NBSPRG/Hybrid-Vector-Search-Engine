-- KeaBuilder ML Service Database Schema

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Stores raw user inputs with their embedding vectors
CREATE TABLE user_inputs (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID NOT NULL,
    input_text    TEXT NOT NULL,
    input_type    VARCHAR(50),          -- 'lead', 'prompt', 'query'
    embedding     VECTOR(128),
    model_used    VARCHAR(50),
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_user_inputs_embedding ON user_inputs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_user_inputs_session ON user_inputs (session_id);

-- Stores ML inference results and metadata
CREATE TABLE predictions (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    input_id          UUID REFERENCES user_inputs(id) ON DELETE CASCADE,
    job_id            UUID,
    model_version     VARCHAR(50),
    top_matches       JSONB,
    similarity_method VARCHAR(20),      -- 'dense', 'sparse', 'hybrid'
    latency_ms        INTEGER,
    status            VARCHAR(20) DEFAULT 'pending',
    error_message     TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    completed_at      TIMESTAMPTZ
);

CREATE INDEX idx_predictions_job_id ON predictions (job_id) WHERE job_id IS NOT NULL;
CREATE INDEX idx_predictions_model_method ON predictions (model_version, similarity_method);

-- Tracks feature flag changes
CREATE TABLE flag_audit_log (
    id            SERIAL PRIMARY KEY,
    flag_key      VARCHAR(100) NOT NULL,
    old_value     JSONB,
    new_value     JSONB,
    changed_by    VARCHAR(100) DEFAULT 'system',
    changed_at    TIMESTAMPTZ DEFAULT NOW()
);
