"""Streamlit portfolio demo: reproducible analysis of invented assignments."""

import math

import altair as alt
import pandas as pd
import streamlit as st

from court_assignment_audit.reporting import (
    MC_SEED,
    MC_SIMULATIONS,
    assignments_csv,
    build_run,
    manifest_json,
)
from court_assignment_audit.synthetic import COURTS, SimulationConfig

st.set_page_config(page_title="Court Assignment Audit", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def simulate(n_cases: int, seed: int, scenario: str) -> dict:
    return build_run(SimulationConfig(n_cases=n_cases, seed=seed, scenario=scenario))


def p_text(value: float, *, underflow: bool = False, log_value: float = 0) -> str:
    if underflow:
        return f"10^({log_value / math.log(10):.1f})"
    return f"{value:.3g}" if value >= 0.001 else f"{value:.2e}"


with st.sidebar:
    st.title("Simulation studio")
    scenario_label = st.selectbox(
        "Assignment scenario", ["Null mechanism", "Illustrative anomaly"], key="scenario"
    )
    n_cases = st.number_input("Number of assignments", 1, 5000, 1000, key="n_cases")
    seed = st.number_input("Dataset seed", 0, 2**32 - 1, 20261011, key="seed")
    st.caption("Every case is one independent draw. Totals are never forced to match expectations.")
    st.divider()
    st.markdown("**Preselected question**")
    st.caption("Does Court A receive more assignments than its defined 40% baseline?")
    st.caption(f"Global calibration: {MC_SIMULATIONS:,} simulations · seed {MC_SEED}")

scenario = "null" if scenario_label == "Null mechanism" else "anomaly"
run = simulate(int(n_cases), int(seed), scenario)
dataset, global_result, target = run["dataset"], run["global"], run["target"]
config = SimulationConfig(n_cases=int(n_cases), seed=int(seed), scenario=scenario)

st.caption("STATISTICAL AUDIT LAB / 100% SYNTHETIC DATA")
st.title("Court Assignment Audit")
st.markdown("Explore how random variation differs from an illustrative shift in assignments.")
st.info(
    "This demo uses invented cases and generic courts. Results describe compatibility with a "
    "defined random mechanism; they do not estimate fraud or guilt and are not legal evidence."
)

overview, methodology, exports = st.tabs(["Analysis", "Method & assumptions", "Data & run log"])

with overview:
    columns = st.columns(3)
    columns[0].metric("Assignments", f"{n_cases:,}")
    columns[1].metric("Global Monte Carlo p-value", p_text(global_result.p_value))
    columns[2].metric("Distribution distance (TV)", f"{global_result.total_variation:.1%}")
    st.caption(
        "The global p-value estimates an upper-tail probability under the known null. "
        "Total variation (TV) is half the sum of absolute share differences; "
        "it measures effect size."
    )

    st.subheader("Observed vs. expected assignments")
    summary = pd.DataFrame(
        {
            "Court": COURTS,
            "Observed": [dataset.counts[court] for court in COURTS],
            "Expected": [n_cases * p for p in config.null_probabilities],
            "Null share": config.null_probabilities,
            "Generating share": (
                config.null_probabilities if scenario == "null" else config.anomaly_probabilities
            ),
        }
    )
    summary["Observed share"] = summary["Observed"] / n_cases
    summary["Lift (pp)"] = 100 * (summary["Observed share"] - summary["Null share"])
    summary["Observed / expected"] = summary["Observed"] / summary["Expected"]
    chart_data = summary.melt(
        id_vars="Court",
        value_vars=["Expected", "Observed"],
        var_name="Series",
        value_name="Assignments",
    )
    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Court:N", sort=list(COURTS), title=None, axis=alt.Axis(labelAngle=0)),
            xOffset="Series:N",
            y=alt.Y("Assignments:Q", title="Number of assignments", scale=alt.Scale(zero=True)),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=["Expected", "Observed"], range=["#94A3B8", "#167D9A"]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=["Court:N", "Series:N", alt.Tooltip("Assignments:Q", format=",.1f")],
        )
        .properties(height=320)
    )
    st.altair_chart(
        chart,
        width="stretch",
        alt="Grouped bars compare observed and expected assignments for Courts A through D.",
    )
    st.dataframe(summary, hide_index=True, width="stretch")
    st.caption(
        "Null: 40% / 30% / 20% / 10%. Illustrative anomaly: 58% / 21% / 14% / 7%. "
        "The comparison baseline stays fixed in both scenarios. A null sample can look unusual."
    )

    st.subheader("Preselected target · Court A")
    target_columns = st.columns(3)
    target_columns[0].metric("Observed share", f"{target.observed_proportion:.1%}")
    target_columns[1].metric("Lift above 40% baseline", f"{target.lift_pp:+.1f} pp")
    target_columns[2].metric(
        "Exact enrichment p-value",
        p_text(target.p_value, underflow=target.p_value_underflow, log_value=target.log_p_value),
    )
    st.caption(
        "Court A and the upper-tail direction are fixed before generating data. The exact "
        "binomial p-value is P(X ≥ observed | n, p₀ = 0.40). "
        "A large p-value does not establish fairness."
    )

    with st.expander("Exploratory court comparisons · Holm adjustment"):
        st.write(
            "Inspecting all four courts creates a family of four enrichment tests. "
            "Holm-adjusted values account for this family, "
            "including courts that do not look unusual."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Court": court,
                        "Lift (pp)": result.lift_pp,
                        "Raw enrichment p-value": p_text(
                            result.p_value,
                            underflow=result.p_value_underflow,
                            log_value=result.log_p_value,
                        ),
                        "Holm-adjusted p-value": p_text(
                            result.adjusted_p_value,
                            underflow=result.adjusted_p_value_underflow,
                            log_value=result.adjusted_log_p_value,
                        ),
                    }
                    for court, result in zip(COURTS, run["exploratory"], strict=True)
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "This adjustment covers one four-court analysis. Trying many seeds, sample sizes or "
            "scenarios and keeping an interesting result adds selection that is not adjusted here."
        )

with methodology:
    st.subheader("A known assignment mechanism")
    st.write(
        "Under the null, every independent case is assigned once to Court A, B, C or D with "
        "probabilities 0.40, 0.30, 0.20 and 0.10. Eligibility and probabilities do not change. "
        "These are specified probabilities, not estimated from the same sample."
    )
    st.markdown(
        "1. **Global question (primary):** compare all observed counts with the known multinomial "
        "baseline using Pearson's discrepancy statistic.\n"
        "2. **Calibration:** simulate count vectors under that baseline; "
        "count simulated statistics "
        "at least as large as observed, including ties.\n"
        "3. **Target question (planned secondary):** calculate the exact, one-sided binomial tail "
        "for Court A.\n"
        "4. **Exploration:** adjust the complete four-court family with Holm's method."
    )
    st.latex(r"p_{MC}=\frac{b+1}{B+1}")
    st.write(
        f"Global statistic: {global_result.statistic:.4f}. At least as extreme: "
        f"{global_result.exceedances:,} of {global_result.simulations:,} simulations. "
        f"Smallest reportable Monte Carlo p-value: {global_result.min_attainable_p_value:.2e}."
    )
    se_label = (
        "boundary estimate; use the interval"
        if global_result.exceedances in (0, global_result.simulations)
        else f"{global_result.mc_standard_error:.2e}"
    )
    st.write(
        f"Approximate Monte Carlo standard error: {se_label}. "
        f"Exact 95% interval for the underlying simulated tail probability: "
        f"[{global_result.mc_tail_ci_low:.2e}, {global_result.mc_tail_ci_high:.2e}]."
    )
    st.caption(
        "The SE uses the simulated exceedance fraction b/B and can be zero at a boundary; "
        "this does not mean there is no simulation uncertainty. The interval refers to simulation "
        "error, not uncertainty about fraud, fairness or the data-generating probabilities."
    )
    st.subheader("Interpretation boundaries")
    st.write(
        "Monte Carlo calibration supports sparse counts without a large-sample chi-square "
        "approximation. Its validity still depends on the defined independent mechanism. Real "
        "eligibility rules, workloads, related cases or changing probabilities would require a "
        "different model. Judicial outcomes and resolved-case subsets are not analyzed."
    )
    st.write(
        "The global and planned target p-values answer different questions. No combined "
        "significance decision is made from these two values. Smaller p-values describe "
        "incompatibility with an assumption; effect size describes the magnitude of a departure."
    )

with exports:
    st.subheader("Reproduce this run")
    st.code(
        "python -m court_assignment_audit.cli "
        f"--scenario {scenario} --seed {seed} --n-cases {n_cases} --output-dir artifacts/demo"
    )
    st.write(
        "The run log records the mechanism, both seeds, sample size, all results, software "
        "versions and a SHA-256 checksum of the synthetic CSV. There is no data upload."
    )
    st.download_button(
        "Download synthetic assignments",
        assignments_csv(run),
        file_name="synthetic_assignments.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download run log",
        manifest_json(run),
        file_name="run_manifest.json",
        mime="application/json",
    )
    st.dataframe(pd.DataFrame(dataset.to_records()).head(25), hide_index=True)
    st.caption("Preview shows the first 25 generated cases. The CSV contains every generated case.")
    with st.expander("Inspect generation metadata"):
        st.json(dataset.metadata)

st.divider()
st.caption("Court Assignment Audit · Reproducible statistics with entirely synthetic data")
