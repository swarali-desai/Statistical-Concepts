"""Synthetic data generator for OptiFlow A/B analysis.

Generates funnel logs for a checkout-flow experiment with properties that
mirror what a real product experiment looks like, so the analysis layer has
something non-trivial to work with:

- A pre-experiment covariate (``pre_period_sessions``) correlated with the
  outcome but unaffected by treatment -> lets CUPED demonstrate real variance
  reduction.
- A novelty effect: the treatment lift is largest on day 0 and decays toward
  a steady-state effect -> lets sequential/always-valid testing demonstrate
  why "peeking early and stopping" is dangerous.
- Heterogeneous treatment effects by device type -> lets segmentation /
  heterogeneous-treatment-effect analysis find something real.
- A guardrail metric (``page_load_ms``) that treatment mildly regresses ->
  lets the decision framework demonstrate an OEC-vs-guardrail trade-off.
- An optional injected sample-ratio-mismatch bug (``--srm-bug``) -> lets the
  randomization/SRM check catch a broken experiment on demand.
"""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from faker import Faker
from sqlalchemy import create_engine

DEFAULT_DB_URL = "sqlite:///data/optiflow.db"
DEFAULT_TABLE = "funnel_logs"

DEVICE_TYPES = ["mobile", "desktop", "tablet"]
DEVICE_P = [0.60, 0.35, 0.05]

# Device effect on baseline conversion (logit scale) and on how strongly the
# treatment resonates with that device (multiplier on the treatment effect).
DEVICE_BASE_LOGIT_EFFECT = {"mobile": -0.05, "desktop": 0.10, "tablet": 0.0}
DEVICE_TREATMENT_MULT = {"mobile": 1.35, "desktop": 0.65, "tablet": 1.0}

SEGMENT_BASE_LOGIT_EFFECT = {"new": -0.15, "returning": 0.20}
SEGMENT_P = {"new": 0.55, "returning": 0.45}
SEGMENT_SESSION_LAMBDA = {"new": 1.5, "returning": 6.0}

ENGAGEMENT_COEF = 0.05  # logit shift per pre-period session above the mean


def _ensure_sqlite_dir(db_url: str) -> None:
    """Ensure the directory for a SQLite DB exists when using a file-based URL."""
    if not db_url.startswith("sqlite:///"):
        return
    db_path = Path(db_url.replace("sqlite:///", ""))
    if db_path.parent:
        db_path.parent.mkdir(parents=True, exist_ok=True)


def _sample_timestamps(fake: Faker, n: int, start: datetime, span_days: int) -> Iterable[datetime]:
    """Sample timestamps uniformly over a window ending at start+span."""
    end = start + timedelta(days=span_days)
    for _ in range(n):
        yield fake.date_time_between_dates(datetime_start=start, datetime_end=end)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _novelty_multiplier(day_index: np.ndarray, decay: float, novelty_boost: float, asymptote: float) -> np.ndarray:
    """Multiplier applied to the treatment effect as a function of experiment day.

    Starts at ``novelty_boost`` on day 0 (users notice the new thing) and
    decays exponentially toward ``asymptote`` (the true long-run effect).
    """
    return asymptote + (novelty_boost - asymptote) * np.exp(-decay * day_index)


def _generate_rows(
    n: int,
    baseline: float,
    lift: float,
    seed: int,
    start: datetime,
    span_days: int,
    novelty_decay: float,
    novelty_boost: float,
    novelty_asymptote: float,
    guardrail_ms: float,
    srm_bug: bool,
) -> pd.DataFrame:
    fake = Faker()
    Faker.seed(seed)
    rng = np.random.default_rng(seed)

    variants = rng.choice(["A", "B"], size=n)
    device_types = rng.choice(DEVICE_TYPES, size=n, p=DEVICE_P)
    segments = rng.choice(
        list(SEGMENT_P.keys()), size=n, p=list(SEGMENT_P.values())
    )

    timestamps = list(_sample_timestamps(fake, n, start, span_days))
    day_index = np.array([(ts - start).days for ts in timestamps], dtype=float)
    day_index = np.clip(day_index, 0, span_days)

    # Latent, stable per-user "value" trait (spending/engagement propensity)
    # fixed before the experiment starts. It drives both the pre-period
    # covariate and the post-period outcome, producing the kind of user-level
    # autocorrelation real pre/post experiment metrics have -- this is what
    # makes CUPED's variance reduction non-trivial (a covariate that's only
    # weakly related to the outcome barely helps).
    user_value = rng.lognormal(mean=0.0, sigma=0.5, size=n)

    # Pre-period engagement: unaffected by treatment (it happened before the
    # experiment started), but correlated with the post-period outcome -> the
    # textbook CUPED covariate.
    session_lambda = np.array([SEGMENT_SESSION_LAMBDA[s] for s in segments]) * user_value
    pre_period_sessions = rng.poisson(session_lambda)
    pre_period_revenue = rng.gamma(shape=2.0, scale=user_value * 15.0)
    mean_sessions = pre_period_sessions.mean()

    is_treatment = (variants == "B").astype(float)

    base_logit = _logit(np.array([baseline]))[0]
    device_base_effect = np.array([DEVICE_BASE_LOGIT_EFFECT[d] for d in device_types])
    segment_base_effect = np.array([SEGMENT_BASE_LOGIT_EFFECT[s] for s in segments])
    engagement_effect = ENGAGEMENT_COEF * (pre_period_sessions - mean_sessions)
    user_value_effect = 0.35 * np.log(user_value)

    treatment_logit_lift = _logit(np.array([baseline + lift]))[0] - base_logit
    device_treat_mult = np.array([DEVICE_TREATMENT_MULT[d] for d in device_types])
    novelty_mult = _novelty_multiplier(day_index, novelty_decay, novelty_boost, novelty_asymptote)
    treatment_effect = is_treatment * treatment_logit_lift * device_treat_mult * novelty_mult

    logit_p = (
        base_logit + device_base_effect + segment_base_effect
        + engagement_effect + user_value_effect + treatment_effect
    )
    conversion_probs = _sigmoid(logit_p)
    converted = rng.binomial(1, conversion_probs)

    # Revenue: only realized on conversion. Order value scales with the same
    # latent user_value trait as pre_period_revenue (real high-value
    # customers spend more before *and* during the experiment), plus a
    # small, steady (non-decaying) treatment effect on order value to give
    # the OEC a continuous-metric component alongside binary conversion.
    aov_base = np.where(np.array(device_types) == "desktop", 52.0, 44.0)
    aov_treatment_lift = is_treatment * 1.5
    order_value = rng.gamma(shape=3.0, scale=(aov_base + aov_treatment_lift) * user_value / 3.0)
    revenue = converted * order_value

    # Session duration: a continuous, never-zero-inflated engagement metric
    # driven by the same latent trait as its pre-period counterpart. Unlike
    # revenue (gated behind the rare binary "converted" event, which caps how
    # much correlation any covariate can capture), this pairing gives CUPED
    # a clean, strong-correlation textbook case -- useful to contrast against
    # the more modest, realistic reduction CUPED achieves on revenue.
    user_engagement = rng.lognormal(mean=0.0, sigma=0.6, size=n)
    pre_period_session_duration = rng.gamma(shape=5.0, scale=20.0 * user_engagement)
    session_duration = rng.gamma(shape=5.0, scale=20.0 * user_engagement)

    # Guardrail metric: treatment mildly regresses page-load time (classic
    # "shipping more code slows the page down" trade-off). Mobile is slower
    # at baseline.
    device_load_base = np.where(np.array(device_types) == "mobile", 720.0, 560.0)
    page_load_ms = rng.normal(
        loc=device_load_base + is_treatment * guardrail_ms, scale=90.0
    )
    page_load_ms = np.clip(page_load_ms, 100, None)

    user_ids = np.arange(1, n + 1)

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "timestamp": timestamps,
            "day_index": day_index.astype(int),
            "variant": variants,
            "device_type": device_types,
            "user_segment": segments,
            "pre_period_sessions": pre_period_sessions,
            "pre_period_revenue": np.round(pre_period_revenue, 2),
            "pre_period_session_duration": np.round(pre_period_session_duration, 2),
            "session_duration": np.round(session_duration, 2),
            "converted": converted,
            "revenue": np.round(revenue, 2),
            "page_load_ms": np.round(page_load_ms, 1),
        }
    )

    if srm_bug:
        # Simulate a logging/randomization bug: mobile users assigned to B
        # are disproportionately dropped before logging (e.g. a client-side
        # bug that fails to fire the exposure event on the new variant for
        # the majority device). Because mobile is 60% of traffic, this is
        # large enough to break the overall 50/50 allocation ratio, not just
        # a device-level slice.
        drop_mask = (df["variant"] == "B") & (df["device_type"] == "mobile")
        drop_idx = df[drop_mask].sample(frac=0.25, random_state=seed).index
        df = df.drop(index=drop_idx).reset_index(drop=True)

    return df


def write_to_db(df: pd.DataFrame, db_url: str, table: str, if_exists: str = "append") -> None:
    """Persist the generated data to the target database."""
    engine = create_engine(db_url)
    df.to_sql(table, engine, if_exists=if_exists, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic A/B funnel data with Faker.")
    parser.add_argument("--n", type=int, default=50000, help="Number of rows to generate (default: 50000).")
    parser.add_argument("--baseline", type=float, default=0.10, help="Baseline conversion rate for control (A).")
    parser.add_argument("--lift", type=float, default=0.02, help="Steady-state absolute lift applied to treatment (B).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_URL,
        help=f"Target database URL (default: {DEFAULT_DB_URL}).",
    )
    parser.add_argument("--table", type=str, default=DEFAULT_TABLE, help="Destination table name.")
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="ISO start date for generated timestamps (default: 7 days before now).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Number of days over which to spread timestamps (default: 14).",
    )
    parser.add_argument(
        "--if-exists",
        type=str,
        default="append",
        choices=["fail", "replace", "append"],
        help="Behavior if table exists (append, replace, fail).",
    )
    parser.add_argument(
        "--novelty-decay", type=float, default=0.20,
        help="Exponential decay rate of the novelty effect per day (default: 0.20).",
    )
    parser.add_argument(
        "--novelty-boost", type=float, default=1.6,
        help="Treatment-effect multiplier on day 0, before novelty wears off (default: 1.6).",
    )
    parser.add_argument(
        "--novelty-asymptote", type=float, default=0.7,
        help="Long-run treatment-effect multiplier once novelty has fully worn off (default: 0.7).",
    )
    parser.add_argument(
        "--guardrail-ms", type=float, default=15.0,
        help="Milliseconds of page-load regression the treatment introduces (default: 15.0).",
    )
    parser.add_argument(
        "--srm-bug", action="store_true",
        help="Inject a sample-ratio-mismatch bug (drops some tablet/B rows) to demo SRM detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = (
        datetime.fromisoformat(args.start_date)
        if args.start_date
        else datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=args.days)
    )

    _ensure_sqlite_dir(args.db)

    df = _generate_rows(
        n=args.n,
        baseline=args.baseline,
        lift=args.lift,
        seed=args.seed,
        start=start_date,
        span_days=args.days,
        novelty_decay=args.novelty_decay,
        novelty_boost=args.novelty_boost,
        novelty_asymptote=args.novelty_asymptote,
        guardrail_ms=args.guardrail_ms,
        srm_bug=args.srm_bug,
    )

    write_to_db(df, db_url=args.db, table=args.table, if_exists=args.if_exists)
    print(
        f"Wrote {len(df):,} rows to {args.db} table '{args.table}' "
        f"(steady-state lift={args.lift:+.3f}, novelty_boost={args.novelty_boost}, "
        f"srm_bug={args.srm_bug})."
    )


if __name__ == "__main__":
    main()
