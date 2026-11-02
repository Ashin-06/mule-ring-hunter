-- One row per analysed batch: who sent it, its size, the alert threshold, and what was found.
-- findings holds the flagged accounts (the caller's own IDs), their scores and the ring candidates;
-- the submitted transactions themselves are not stored.
CREATE TABLE analyses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    label TEXT NOT NULL,
    accounts BIGINT NOT NULL,
    transactions BIGINT NOT NULL,
    flagged BIGINT NOT NULL,
    rings BIGINT NOT NULL,
    findings TEXT NOT NULL
);
CREATE INDEX analyses_created ON analyses (created_at);
