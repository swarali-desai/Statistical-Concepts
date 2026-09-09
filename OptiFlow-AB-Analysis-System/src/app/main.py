"""Interactive experiment readout dashboard.

Run with:
    streamlit run src/app/main.py

Loads funnel logs from the SQLite DB produced by ``src.simulation.generate``
and walks through the same readout a product analyst would produce: trust
checks, primary-metric result (frequentist + CUPED + Bayesian), guardrails,
sequential-monitoring risk, heterogeneous effects, and a ship/hold decision.
"""
import sys
from pathlib import Path

# Allow `streamlit run src/app/main.py` to resolve `src.*` imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from src.analysis import bayesian, cuped, frequentist, segmentation, sequential
from src.analysis.decision import GuardrailCheck, make_ship_decision
from src.reporting import plots
from src.stats.power import calculate_sample_size, run_randomization_checks

st.set_page_config(page_title="OptiFlow A/B Readout", layout="wide")

DEFAULT_DB = "sqlite:///data/optiflow.db"
DEFAULT_TABLE = "funnel_logs"


@st.cache_data
def load_data(db_url: str, table: str) -> pd.DataFrame:
    engine = create_engine(db_url)
    return pd.read_sql_table(table, engine, parse_dates=["timestamp"])


def main():
    st.title("OptiFlow — Experiment Readout")
    st.caption(
        "Checkout-flow A/B test readout: trust checks -> primary metric -> "
        "guardrails -> sequential risk -> segments -> ship decision."
    )

    with st.sidebar:
        st.header("Data source")
        db_url = st.text_input("Database URL", value=DEFAULT_DB)
        table = st.text_input("Table", value=DEFAULT_TABLE)
        alpha = st.slider("Significance level (alpha)", 0.01, 0.20, 0.05, 0.01)
        load = st.button("Load data", type="primary")

    if "df" not in st.session_state and not load:
        st.info("Generate data first (see README), then click **Load data** in the sidebar.")
        st.code(
            "python -m src.simulation.generate --n 50000 --days 14 "
            "--db sqlite:///data/optiflow.db --if-exists replace",
            language="bash",
        )
        return

    if load:
        try:
            st.session_state["df"] = load_data(db_url, table)
        except Exception as exc:  # noqa: BLE001 - surface any DB error to the user
            st.error(f"Could not load data: {exc}")
            return

    if "df" not in st.session_state:
        return

    df = st.session_state["df"]
    st.success(f"Loaded {len(df):,} rows from `{table}`.")

    tabs = st.tabs(
        ["Trust checks", "Primary metric (OEC)", "Guardrails", "Sequential risk", "Segments", "Decision"]
    )

    # ---- Trust checks -----------------------------------------------------
    with tabs[0]:
        st.subheader("Sample ratio mismatch & covariate balance")
        srm = run_randomization_checks(
            df, covariate_cols=["device_type", "user_segment"], alpha=alpha
        )
        col1, col2 = st.columns(2)
        col1.metric("Variant split p-value", f"{srm['variant_pvalue']:.4f}")
        col2.metric("SRM check", "PASSED" if srm["passed"] else "FAILED")
        st.write("Variant counts:", srm["variant_counts"])
        for cov in srm["covariates"]:
            st.write(f"- `{cov.column}` balance p-value: {cov.pvalue:.4f} ({'OK' if cov.passed else 'IMBALANCED'})")
        if not srm["passed"]:
            st.error(
                "Randomization looks broken. Per Twyman's Law, don't trust the "
                "topline result until this is root-caused — see the Decision tab."
            )

        st.subheader("Sample size sanity check")
        baseline_rate = df.loc[df["variant"] == "A", "converted"].mean()
        sizing = calculate_sample_size(baseline=baseline_rate, mde=0.01, alpha=alpha, power=0.8)
        st.write(
            f"To detect a +0.01 absolute lift on a {baseline_rate:.1%} baseline "
            f"with 80% power, you'd need ~{sizing['total']:,} users "
            f"({sizing['control']:,}/arm). This experiment collected "
            f"{len(df):,} rows."
        )

    # ---- Primary metric -----------------------------------------------------
    with tabs[1]:
        st.subheader("Conversion rate — frequentist")
        conv_result = frequentist.proportion_test(df, metric_col="converted", alpha=alpha)
        st.code(conv_result.summary())
        st.pyplot(plots.plot_cumulative_lift(df, metric_col="converted"))

        st.subheader("Revenue per user — CUPED-adjusted")
        cuped_result = cuped.cuped_test(
            df, metric_col="revenue", covariate_col="pre_period_revenue", alpha=alpha
        )
        st.code(cuped_result.test_result.summary())
        st.write(
            f"theta={cuped_result.theta:.4f}, corr(Y,X)={cuped_result.correlation:.3f}, "
            f"variance reduction={cuped_result.variance_reduction_pct:.1%}"
        )
        st.pyplot(plots.plot_variance_reduction(cuped_result.variance_before, cuped_result.variance_after))

        st.subheader("Conversion rate — Bayesian")
        a_conv = int(df.loc[df.variant == "A", "converted"].sum())
        a_n = int((df.variant == "A").sum())
        b_conv = int(df.loc[df.variant == "B", "converted"].sum())
        b_n = int((df.variant == "B").sum())
        bayes_result = bayesian.beta_binomial_ab_test(a_conv, a_n, b_conv, b_n, alpha=alpha)
        st.write(
            f"P(B > A) = {bayes_result.prob_b_beats_a:.1%} | "
            f"expected loss if you ship B = {bayes_result.expected_loss_choose_b:.5f} | "
            f"95% CI on relative lift: [{bayes_result.rel_lift_ci_low:.1%}, {bayes_result.rel_lift_ci_high:.1%}]"
        )
        st.pyplot(plots.plot_posteriors(bayes_result.posterior_a, bayes_result.posterior_b, "conversion rate"))

    # ---- Guardrails -----------------------------------------------------
    with tabs[2]:
        st.subheader("Page load time (guardrail — lower is better)")
        load_result = frequentist.continuous_test(df, metric_col="page_load_ms", alpha=alpha)
        st.code(load_result.summary())
        guardrail = GuardrailCheck(result=load_result, direction="lower_is_better", max_acceptable_regression=10.0)
        if guardrail.regressed:
            st.warning("GUARDRAIL VIOLATION")
        else:
            st.success("Guardrail within tolerance")

    # ---- Sequential risk -----------------------------------------------------
    with tabs[3]:
        st.subheader("Naive peeking vs. O'Brien-Fleming boundary")
        seq_df = sequential.sequential_monitoring(df, metric_col="converted", alpha=alpha)
        st.pyplot(plots.plot_sequential_boundary(seq_df))
        st.dataframe(seq_df, use_container_width=True)

        st.subheader("Simulated false-positive inflation from naive daily peeking")
        if st.button("Run peeking simulation (500 A/A sims)"):
            with st.spinner("Simulating..."):
                sim = sequential.simulate_peeking_inflation(n_simulations=500, n_looks=10, n_per_look_per_arm=150, alpha=alpha)
            col1, col2 = st.columns(2)
            col1.metric("Naive peeking false-positive rate", f"{sim.naive_false_positive_rate:.1%}")
            col2.metric("O'Brien-Fleming-corrected rate", f"{sim.obf_false_positive_rate:.1%}", delta=f"target ~{alpha:.0%}")

    # ---- Segments -----------------------------------------------------
    with tabs[4]:
        st.subheader("Heterogeneous treatment effect by device")
        subgroup_table = segmentation.subgroup_lift_table(df, group_col="device_type", alpha=alpha)
        st.dataframe(subgroup_table, use_container_width=True)
        st.pyplot(plots.plot_subgroup_lift(subgroup_table, "device_type"))

        st.subheader("Interaction regression (treatment x device)")
        interaction = segmentation.interaction_effect_regression(df, group_col="device_type")
        st.dataframe(interaction, use_container_width=True)

    # ---- Decision -----------------------------------------------------
    with tabs[5]:
        st.subheader("Ship / hold / iterate")
        oec_result = frequentist.proportion_test(df, metric_col="converted", alpha=alpha)
        load_result = frequentist.continuous_test(df, metric_col="page_load_ms", alpha=alpha)
        guardrail = GuardrailCheck(result=load_result, direction="lower_is_better", max_acceptable_regression=10.0)
        decision = make_ship_decision(
            df, oec_result=oec_result, guardrails=[guardrail],
            srm_covariate_cols=["device_type", "user_segment"], alpha=alpha,
        )
        verdict_color = {"SHIP": "success", "HOLD": "warning", "DO_NOT_SHIP": "error"}[decision.verdict]
        getattr(st, verdict_color)(f"**{decision.verdict}**")
        for reason in decision.reasons:
            st.write(f"- {reason}")


if __name__ == "__main__":
    main()
