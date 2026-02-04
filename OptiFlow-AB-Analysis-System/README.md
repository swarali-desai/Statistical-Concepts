# OptiFlow A/B Analysis System

OptiFlow designs and analyzes an end-to-end A/B test for a sign-up funnel, comparing Control (original) vs Treatment (redesign) with a small injected lift.

## Goals
- Generate synthetic funnel logs (Faker) with configurable lift between control and treatment (e.g., 10% vs 12% conversion).
- Store logs in SQLite by default; optionally Postgres via docker-compose.
- Run power/MDE checks up front (alpha=0.05, power=0.8) and randomization/AA checks on device/region splits.
- Provide frequentist z-test with CIs and Bayesian beta-binomial P(Treatment > Control), plus segmentation to guard against Simpson's paradox.
- Surface posterior distributions and cumulative lift over time in a Streamlit/Dash dashboard.

## Repository layout
- src/simulation/: Faker-based data generator and CLI entrypoint to write to SQLite/Postgres.
- src/stats/: power analysis, MDE, randomization/AA utilities.
- src/analysis/: frequentist and Bayesian analysis helpers; segmentation hooks.
- src/reporting/: plotting utilities (posterior distributions, cumulative lift over time).
- src/app/: dashboard entrypoint (Streamlit or Dash) to present results.
- src/config/: experiment/database defaults and shared constants.
- 
otebooks/: exploration and design notes for power, randomization checks, and plots.
- data/: generated data and local DBs (gitignored).
- 	ests/: unit tests for stats and simulation logic.
- docker-compose.yml: optional Postgres service for local runs.
- equirements.txt: Python dependencies.

## Setup
1) Use Python 3.10+ and create a virtual environment.
2) Install deps:
`
pip install -r requirements.txt
`
3) For Postgres: docker-compose up -d then point configs/DSN to postgresql://optiflow:optiflow@localhost:5432/optiflow. SQLite remains default if not provided.

## Planned usage
- Generate synthetic data (placeholders; flags to be implemented):
`
python -m src.simulation.generate --n 50000 --lift 0.02 --baseline 0.10 --db sqlite:///data/optiflow.db --seed 42
`
- Analyze (frequentist/Bayesian) and segment by device/region:
`
python -m src.analysis.frequentist --db sqlite:///data/optiflow.db --segment device_type
python -m src.analysis.bayesian --db sqlite:///data/optiflow.db --segment device_type
`
- Dashboard:
`
streamlit run src/app/main.py
`

## Statistical design notes
- Alpha=0.05, power=0.8; baseline conversion around 10% with target detectable lift to ~12%.
- Randomization/AA: balance checks on device/region; fail fast if allocation skewed.
- Frequentist: two-sample z-test for proportions with 95% CI.
- Bayesian: beta-binomial posterior to report P(Treatment > Control).
- Segmentation: compare mobile vs desktop (extendable) to watch for Simpson's paradox.

## Roadmap
- [ ] Finalize experiment config defaults and data schema
- [ ] Implement Faker-based generator with lift control and SQLite/Postgres writers
- [ ] Add power/MDE utilities and AA checks
- [ ] Implement z-test + Bayesian posterior helpers with segmentation support
- [ ] Add plotting helpers (posteriors, cumulative lift) and integrate dashboard shell
- [ ] Write tests for simulation/stats modules and example notebook for validation
