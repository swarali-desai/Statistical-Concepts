"""Synthetic data generator for OptiFlow A/B analysis.

Generates funnel logs with a configurable conversion lift between control (A) and treatment (B).
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from faker import Faker
from sqlalchemy import create_engine

DEFAULT_DB_URL = "sqlite:///data/optiflow.db"
DEFAULT_TABLE = "funnel_logs"


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


def _generate_rows(n: int, baseline: float, lift: float, seed: int, start: datetime, span_days: int) -> pd.DataFrame:
    fake = Faker()
    Faker.seed(seed)
    rng = np.random.default_rng(seed)

    variants = rng.choice(["A", "B"], size=n)
    device_types = rng.choice(["mobile", "desktop", "tablet"], size=n, p=[0.6, 0.35, 0.05])
    session_duration = rng.gamma(shape=2.0, scale=60.0, size=n)  # seconds

    prob_a = max(0.0, min(1.0, baseline))
    prob_b = max(0.0, min(1.0, baseline + lift))
    conversion_probs = np.where(variants == "A", prob_a, prob_b)
    converted = rng.binomial(1, conversion_probs)

    timestamps = list(_sample_timestamps(fake, n, start, span_days))
    user_ids = np.arange(1, n + 1)

    return pd.DataFrame(
        {
            "user_id": user_ids,
            "timestamp": timestamps,
            "variant": variants,
            "device_type": device_types,
            "session_duration": session_duration,
            "converted": converted,
        }
    )


def write_to_db(df: pd.DataFrame, db_url: str, table: str, if_exists: str = "append") -> None:
    """Persist the generated data to the target database."""
    engine = create_engine(db_url)
    df.to_sql(table, engine, if_exists=if_exists, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic A/B funnel data with Faker.")
    parser.add_argument("--n", type=int, default=50000, help="Number of rows to generate (default: 50000).")
    parser.add_argument("--baseline", type=float, default=0.10, help="Baseline conversion rate for control (A).")
    parser.add_argument("--lift", type=float, default=0.02, help="Absolute lift applied to treatment (B).")
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
        default=7,
        help="Number of days over which to spread timestamps (default: 7).",
    )
    parser.add_argument(
        "--if-exists",
        type=str,
        default="append",
        choices=["fail", "replace", "append"],
        help="Behavior if table exists (append, replace, fail).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_date = (
        datetime.fromisoformat(args.start_date)
        if args.start_date
        else datetime.utcnow() - timedelta(days=args.days)
    )

    _ensure_sqlite_dir(args.db)

    df = _generate_rows(
        n=args.n,
        baseline=args.baseline,
        lift=args.lift,
        seed=args.seed,
        start=start_date,
        span_days=args.days,
    )

    write_to_db(df, db_url=args.db, table=args.table, if_exists=args.if_exists)
    print(f"Wrote {len(df):,} rows to {args.db} table '{args.table}' (lift={args.lift:+.3f}).")


if __name__ == "__main__":
    main()
