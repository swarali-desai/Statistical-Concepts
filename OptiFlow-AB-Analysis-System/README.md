# OptiFlow A/B Analysis System

Synthetic A/B funnel data generator and analysis scaffold for experimenting with control (A) vs treatment (B) lift.

## Structure
- `src/simulation/generate.py`: Faker-based data generator with CLI flags.
- `src/analysis/`, `src/stats/`, `src/reporting/`, `src/app/`, `src/config/`: stubs for analysis, stats, plotting, dashboard, and shared settings.
- `data/`, `notebooks/`, `tests/`: storage, exploration, and test scaffolds.

## Setup
1) Use Python 3.10+ and create a virtual environment.
2) Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Generate synthetic data (SQLite default)
From the project root:
```
python -m src.simulation.generate --n 50000 --baseline 0.10 --lift 0.02 --db sqlite:///data/optiflow.db --table funnel_logs --if-exists replace
```

Key flags:
- `--n`: rows to generate (default 50000)
- `--baseline`: control conversion rate (default 0.10)
- `--lift`: absolute lift for treatment (default 0.02, so 12% vs 10%)
- `--days`: spread timestamps over N days (default 7); optional `--start-date` ISO start
- `--db`: SQLAlchemy URL (SQLite by default; you can point to Postgres)
- `--table`: destination table (default `funnel_logs`)
- `--if-exists`: `append|replace|fail` (default `append`)
- `--seed`: RNG seed (default 42)

## Postgres (optional)
If you prefer Postgres, run your own instance and set `--db` to something like:
`postgresql://user:password@localhost:5432/optiflow`

