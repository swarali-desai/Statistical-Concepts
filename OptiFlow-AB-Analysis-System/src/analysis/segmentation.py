"""Heterogeneous treatment effect (HTE) analysis.

An average lift can hide the real story: a feature that helps mobile users
and hurts desktop users can average out to "flat, ship it or don't" -- the
wrong call either way. This module offers two complementary views:

1. ``subgroup_lift_table`` -- run the same test independently within each
   subgroup. Intuitive, but running many subgroup tests inflates the false
   positive rate, so results are corrected with Benjamini-Hochberg FDR
   control rather than read off at raw p < 0.05.
2. ``interaction_effect_regression`` -- a single logistic regression with
   treatment x subgroup interaction terms, which tests for heterogeneity
   directly (does the treatment effect differ by group?) instead of testing
   each group's effect in isolation.
"""
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from src.analysis.frequentist import proportion_test


def subgroup_lift_table(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str = "converted",
    variant_col: str = "variant",
    control: str = "A",
    treatment: str = "B",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-subgroup lift + significance, with BH-FDR correction across groups."""
    rows = []
    for level, group_df in df.groupby(group_col):
        result = proportion_test(
            group_df, metric_col=metric_col, variant_col=variant_col,
            control=control, treatment=treatment, alpha=alpha,
        )
        rows.append(
            {
                group_col: level,
                "n": len(group_df),
                "control_rate": result.control_mean,
                "treatment_rate": result.treatment_mean,
                "abs_lift": result.abs_lift,
                "rel_lift": result.rel_lift,
                "p_value": result.p_value,
            }
        )

    table = pd.DataFrame(rows)
    _, adj_p, _, _ = multipletests(table["p_value"], alpha=alpha, method="fdr_bh")
    table["p_value_bh"] = adj_p
    table["significant_bh"] = table["p_value_bh"] < alpha
    return table.sort_values("abs_lift", ascending=False).reset_index(drop=True)


def interaction_effect_regression(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str = "converted",
    variant_col: str = "variant",
    covariate_cols: Optional[list] = None,
) -> pd.DataFrame:
    """Logistic regression with a treatment x group_col interaction.

    A significant interaction coefficient means the treatment effect
    genuinely differs across that subgroup's levels -- a more direct test
    for heterogeneity than eyeballing per-group lifts, and it only costs one
    multiple-comparisons correction (across interaction terms) instead of
    one per subgroup test.
    """
    model_df = df.copy()
    model_df["treatment"] = (model_df[variant_col] == "B").astype(int)
    model_df[group_col] = model_df[group_col].astype("category")

    formula = f"{metric_col} ~ treatment * C({group_col})"
    if covariate_cols:
        formula += " + " + " + ".join(covariate_cols)

    model = smf.logit(formula, data=model_df).fit(disp=0)

    summary = model.summary2().tables[1].reset_index()
    summary.columns = ["term", "coef", "std_err", "z", "p_value", "ci_low", "ci_high"]
    interaction_rows = summary[summary["term"].str.contains("treatment:")].copy()
    interaction_rows["odds_ratio"] = np.exp(interaction_rows["coef"])
    return interaction_rows.reset_index(drop=True)
