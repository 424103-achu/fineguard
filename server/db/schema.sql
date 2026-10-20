CREATE TABLE IF NOT EXISTS analyses (
    id           SERIAL PRIMARY KEY,
    source       TEXT          NOT NULL,
    service_name TEXT,
    risk_score   NUMERIC(5, 1) NOT NULL,
    grade        CHAR(1)       NOT NULL,
    n_sentences  INT           NOT NULL,
    n_flagged    INT           NOT NULL,
    result       JSONB         NOT NULL,
    created_at   TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses (created_at DESC);
