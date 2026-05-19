CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    match_date DATE,
    format TEXT NOT NULL CHECK (format IN ('IPL', 'T20I', 'ODI')),
    team1 TEXT NOT NULL,
    team2 TEXT NOT NULL,
    venue TEXT NOT NULL,
    toss_winner TEXT,
    toss_decision TEXT CHECK (toss_decision IN ('bat', 'field')),
    winner TEXT,
    result_margin TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches (match_date);
CREATE INDEX IF NOT EXISTS idx_matches_format ON matches (format);
CREATE INDEX IF NOT EXISTS idx_matches_venue ON matches (venue);

CREATE TABLE IF NOT EXISTS deliveries (
    delivery_id BIGSERIAL PRIMARY KEY,
    match_id TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    innings SMALLINT NOT NULL CHECK (innings >= 1),
    over_number SMALLINT NOT NULL CHECK (over_number >= 0),
    ball_number SMALLINT NOT NULL CHECK (ball_number BETWEEN 0 AND 6),
    batting_team TEXT NOT NULL,
    bowling_team TEXT NOT NULL,
    batter TEXT NOT NULL,
    bowler TEXT NOT NULL,
    runs SMALLINT NOT NULL CHECK (runs >= 0),
    extras SMALLINT NOT NULL DEFAULT 0 CHECK (extras >= 0),
    is_wicket BOOLEAN NOT NULL DEFAULT FALSE,
    wicket_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_deliveries_match ON deliveries (match_id, innings, over_number, ball_number);
CREATE INDEX IF NOT EXISTS idx_deliveries_batter ON deliveries (batter);
CREATE INDEX IF NOT EXISTS idx_deliveries_bowler ON deliveries (bowler);

CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    country TEXT,
    bat_style TEXT,
    bowl_style TEXT,
    role TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS player_match_stats (
    stat_id BIGSERIAL PRIMARY KEY,
    player_id TEXT REFERENCES players(player_id) ON DELETE SET NULL,
    match_id TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    runs SMALLINT NOT NULL DEFAULT 0 CHECK (runs >= 0),
    balls_faced SMALLINT NOT NULL DEFAULT 0 CHECK (balls_faced >= 0),
    wickets SMALLINT NOT NULL DEFAULT 0 CHECK (wickets >= 0),
    overs_bowled NUMERIC(4,1) NOT NULL DEFAULT 0,
    economy NUMERIC(5,2),
    strike_rate NUMERIC(6,2),
    UNIQUE (player_id, match_id)
);

CREATE INDEX IF NOT EXISTS idx_player_match_stats_match ON player_match_stats (match_id);

CREATE TABLE IF NOT EXISTS venues (
    venue_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    city TEXT,
    avg_first_innings_score NUMERIC(6,2) NOT NULL DEFAULT 165.0,
    chase_success_rate NUMERIC(5,2) NOT NULL DEFAULT 50.0,
    dew_factor NUMERIC(5,2) NOT NULL DEFAULT 50.0,
    pitch_type TEXT NOT NULL DEFAULT 'balanced',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS live_match_state (
    match_id TEXT PRIMARY KEY REFERENCES matches(match_id) ON DELETE CASCADE,
    batting_team TEXT NOT NULL,
    bowling_team TEXT NOT NULL,
    innings SMALLINT NOT NULL DEFAULT 1,
    over_number SMALLINT NOT NULL DEFAULT 0,
    ball_number SMALLINT NOT NULL DEFAULT 0,
    score SMALLINT NOT NULL DEFAULT 0,
    wickets SMALLINT NOT NULL DEFAULT 0,
    target SMALLINT,
    current_run_rate NUMERIC(6,2) NOT NULL DEFAULT 0,
    required_run_rate NUMERIC(6,2) NOT NULL DEFAULT 0,
    pressure_index NUMERIC(5,2) NOT NULL DEFAULT 0,
    last_event TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    pred_id UUID PRIMARY KEY,
    match_id TEXT,
    pred_type TEXT NOT NULL,
    predicted_value TEXT NOT NULL,
    confidence NUMERIC(6,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    probability NUMERIC(6,4) CHECK (probability >= 0 AND probability <= 1),
    model_version TEXT NOT NULL DEFAULT 'demo-heuristic-v1',
    explanation JSONB NOT NULL DEFAULT '{}'::jsonb,
    feature_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    actual_value TEXT,
    is_correct BOOLEAN,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions (match_id);
CREATE INDEX IF NOT EXISTS idx_predictions_type_created ON predictions (pred_type, created_at DESC);

