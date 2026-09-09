"""Ship / hold / iterate decision framework.

Mirrors how experimentation platforms like Microsoft's ExP operationalize a
readout: don't trust any effect estimate until the experiment itself is
validated (Twyman's Law -- "any figure that looks interesting or unusual is
usually wrong"), then weigh the primary metric (the OEC, Overall Evaluation
Criterion) against guardrail metrics that must not regress, rather than
optimizing the OEC in isolation.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from src.analysis.frequentist import TestResult
from src.stats.power import run_randomization_checks


@dataclass
class GuardrailCheck:
    result: TestResult
    # "lower_is_better" (e.g. page load time) or "higher_is_better"
    direction: str
    max_acceptable_regression: float  # in absolute metric units, always positive

    @property
    def regressed(self) -> bool:
        signed_change = self.result.abs_lift if self.direction == "higher_is_better" else -self.result.abs_lift
        return self.result.significant and signed_change < -self.max_acceptable_regression


@dataclass
class DecisionResult:
    verdict: str  # "SHIP", "HOLD", "DO_NOT_SHIP"
    reasons: List[str] = field(default_factory=list)
    srm_passed: bool = True
    oec_significant: bool = False
    oec_positive: bool = False
    guardrail_violations: List[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"Verdict: {self.verdict}"]
        lines.extend(f"  - {r}" for r in self.reasons)
        return "\n".join(lines)


def make_ship_decision(
    df,
    oec_result: TestResult,
    guardrails: Optional[List[GuardrailCheck]] = None,
    variant_col: str = "variant",
    srm_covariate_cols: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> DecisionResult:
    """Apply a standard trustworthiness-then-trade-off decision rule.

    1. SRM / randomization check first -- an untrustworthy experiment can't
       be used to justify anything, no matter how good the topline looks.
    2. OEC must be significant and positive to be eligible to ship.
    3. Any guardrail that regressed beyond its tolerance blocks a clean ship
       and forces an explicit trade-off call (HOLD), even if the OEC is
       great.
    """
    guardrails = guardrails or []
    reasons: List[str] = []

    srm = run_randomization_checks(
        df, variant_col=variant_col, covariate_cols=srm_covariate_cols, alpha=alpha
    )
    srm_passed = bool(srm["passed"])
    if not srm_passed:
        reasons.append(
            "SRM / randomization check FAILED (variant p="
            f"{srm['variant_pvalue']:.4f}) -- do not trust the topline result "
            "until the allocation bug is found and fixed."
        )
        return DecisionResult(
            verdict="DO_NOT_SHIP", reasons=reasons, srm_passed=False,
        )

    reasons.append("SRM / randomization check passed -- allocation looks trustworthy.")

    oec_significant = oec_result.significant
    oec_positive = oec_result.abs_lift > 0
    reasons.append(f"OEC ({oec_result.metric}): {oec_result.summary()}")

    violations = []
    for g in guardrails:
        if g.regressed:
            violations.append(g.result.metric)
            reasons.append(
                f"GUARDRAIL VIOLATION: {g.result.metric} regressed "
                f"{g.result.abs_lift:+.4f} (tolerance: {g.max_acceptable_regression})."
            )
        else:
            reasons.append(f"Guardrail OK: {g.result.metric} {g.result.summary()}")

    if not oec_significant:
        verdict = "DO_NOT_SHIP"
        reasons.append("OEC is not statistically significant -- no evidence of a real effect.")
    elif not oec_positive:
        verdict = "DO_NOT_SHIP"
        reasons.append("OEC moved significantly in the wrong direction.")
    elif violations:
        verdict = "HOLD"
        reasons.append(
            "OEC is significantly positive, but guardrail regressions require an "
            "explicit trade-off decision before shipping."
        )
    else:
        verdict = "SHIP"
        reasons.append("OEC is significantly positive and all guardrails are clean.")

    return DecisionResult(
        verdict=verdict,
        reasons=reasons,
        srm_passed=srm_passed,
        oec_significant=oec_significant,
        oec_positive=oec_positive,
        guardrail_violations=violations,
    )
