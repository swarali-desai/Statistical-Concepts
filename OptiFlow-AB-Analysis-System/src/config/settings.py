"""Default settings and constants."""

DB_DEFAULT = "sqlite:///data/optiflow.db"
TABLE_DEFAULT = "funnel_logs"

ALPHA = 0.05
POWER = 0.8
BASELINE_CONVERSION = 0.10
TARGET_LIFT = 0.02

# Guardrail: max acceptable page-load regression (ms) before it blocks a
# clean ship decision and forces an explicit trade-off call.
GUARDRAIL_PAGE_LOAD_MS_TOLERANCE = 10.0

# CUPED covariate pairings used throughout the notebooks/dashboard.
CUPED_PAIRS = {
    "revenue": "pre_period_revenue",
    "session_duration": "pre_period_session_duration",
}

# Sequential monitoring defaults.
SEQUENTIAL_N_LOOKS = 14  # one look per day of a 14-day experiment
