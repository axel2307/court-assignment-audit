"""An explainable Streamlit case study using entirely invented assignments."""

import math

import altair as alt
import pandas as pd
import streamlit as st

from court_assignment_audit.presentation import (
    global_reading,
    reference_allocations,
    target_reading,
)
from court_assignment_audit.reporting import (
    MC_SEED,
    MC_SIMULATIONS,
    assignments_csv,
    build_run,
    manifest_json,
)
from court_assignment_audit.synthetic import COURTS, SimulationConfig

st.set_page_config(page_title="Court Assignment Audit", page_icon="⚖️", layout="wide")


@st.cache_data(show_spinner=False)
def simulate(n_cases: int, seed: int, scenario: str) -> dict:
    return build_run(SimulationConfig(n_cases=n_cases, seed=seed, scenario=scenario))


def p_text(value: float, *, underflow: bool = False, log_value: float = 0) -> str:
    if underflow:
        return f"10^({log_value / math.log(10):.1f})"
    return f"{value:.3g}" if value >= 0.001 else f"{value:.2e}"


def count_text(value: float) -> str:
    return f"{value:,.1f}".rstrip("0").rstrip(".")


with st.sidebar:
    st.title("Try a different allocation")
    scenario_label = st.selectbox(
        "Assignment scenario", ["Null mechanism", "Illustrative anomaly"], key="scenario"
    )
    n_cases = st.number_input("Number of assignments", 1, 5000, 1000, key="n_cases")
    seed = st.number_input("Dataset seed", 0, 2**32 - 1, 20261011, key="seed")
    st.caption("Same settings, same invented dataset. Each case is one independent draw.")
    with st.expander("What do the scenarios change?"):
        st.write(
            "Null mechanism generates cases with the stated 40% / 30% / 20% / 10% rule. "
            "Illustrative anomaly changes the generating shares to 58% / 21% / 14% / 7%. "
            "Both are compared with the original rule."
        )
        st.caption("A scenario name does not determine the conclusion; the generated counts do.")

scenario = "null" if scenario_label == "Null mechanism" else "anomaly"
config = SimulationConfig(n_cases=int(n_cases), seed=int(seed), scenario=scenario)
run = simulate(config.n_cases, config.seed, config.scenario)
dataset, global_result, target = run["dataset"], run["global"], run["target"]
reading = global_reading(global_result, config.n_cases)
court_a_reading = target_reading(target)

summary = pd.DataFrame(
    {
        "Court": COURTS,
        "Observed": [dataset.counts[court] for court in COURTS],
        "Expected": [n_cases * p for p in config.null_probabilities],
        "Expected share": config.null_probabilities,
    }
)
summary["Observed share"] = summary["Observed"] / n_cases
summary["Difference (pp)"] = 100 * (summary["Observed share"] - summary["Expected share"])
summary["Observed / expected"] = summary["Observed"] / summary["Expected"]
largest = summary.loc[summary["Difference (pp)"].abs().idxmax()]
gap = largest["Observed"] - largest["Expected"]
direction = "more" if gap >= 0 else "fewer"

st.caption("A FICTIONAL CASE STUDY · 100% SYNTHETIC DATA")
st.title("Court Assignment Audit")
st.write(
    "A fictional registry assigns each new case independently to one of four courts. "
    "The stated chances are 40% for Court A, 30% for Court B, 20% for Court C and 10% for Court D."
)

overview, methodology, exports = st.tabs(["Analysis", "How to read this", "Data & run log"])

with overview:
    with st.container(border=True):
        st.caption("THE MAIN QUESTION · DOES THE WHOLE ALLOCATION FIT THE STATED RULE?")
        st.markdown(f"### {reading.headline}")
        st.write(reading.explanation)
        st.caption(
            "Synthetic example. No clear departure does not prove randomness or fairness; "
            "an unusual result does not establish misconduct."
        )

    if gap == 0:
        st.subheader("All four counts match the model averages")
        st.write(
            f"Across **{n_cases:,} assignments**, every observed count equals its expectation."
        )
    else:
        st.subheader(f"The largest count gap is in {largest['Court']}")
        st.write(
            f"Across **{n_cases:,} assignments**, **{largest['Court']}** received "
            f"**{largest['Observed']:,} assignments** versus "
            f"**{count_text(largest['Expected'])} expected** on average. That is "
            f"**{count_text(abs(gap))} {direction} assignments** "
            f"and a **{abs(largest['Difference (pp)']):.1f} percentage-point** gap."
        )
    st.caption("This is a descriptive comparison of counts, not a separate test of that court.")
    chart_data = summary.melt(
        id_vars="Court",
        value_vars=["Expected", "Observed"],
        var_name="Series",
        value_name="Assignments",
    )
    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("Court:N", sort=list(COURTS), title=None),
            yOffset=alt.YOffset("Series:N", sort=["Expected", "Observed"]),
            x=alt.X("Assignments:Q", title="Number of assignments", scale=alt.Scale(zero=True)),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=["Expected", "Observed"], range=["#94A3B8", "#167D9A"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["Court:N", "Series:N", alt.Tooltip("Assignments:Q", format=",.1f")],
        )
        .properties(height=240)
    )
    st.altair_chart(
        chart,
        width="stretch",
        alt="Four pairs of horizontal bars compare observed assignments with expected averages.",
    )
    st.caption(
        "Grey is the model's average; teal is this run. Expected counts are averages over "
        "many repetitions, not quotas that each run must meet."
    )
    with st.expander("See the counts and share differences"):
        st.dataframe(summary, hide_index=True, width="stretch")
        st.write(
            f"Total variation distance: {global_result.total_variation:.1%}. "
            "This is half the sum of absolute share differences, a measure of effect size."
        )

    st.subheader("The same rule can produce different allocations")
    st.write(
        "Compare this run with twelve examples generated by the stated rule. "
        "Each coloured segment is one court's share of the same total number of cases."
    )
    reference = reference_allocations(config.n_cases, config.null_probabilities)
    labels = ["This run", *[f"Example {i + 1:02d}" for i in range(len(reference))]]
    vectors = [[dataset.counts[court] for court in COURTS], *reference.tolist()]
    comparison_data = pd.DataFrame(
        [
            {
                "Allocation": label,
                "Court": court,
                "Court order": index,
                "Assignments": counts[index],
                "Share": counts[index] / n_cases,
            }
            for label, counts in zip(labels, vectors, strict=True)
            for index, court in enumerate(COURTS)
        ]
    )
    comparison = (
        alt.Chart(comparison_data)
        .mark_bar()
        .encode(
            y=alt.Y("Allocation:N", sort=labels, title=None, axis=alt.Axis(labelOverlap=False)),
            x=alt.X(
                "Share:Q",
                stack="zero",
                title="Share of assignments",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "Court:N",
                sort=list(COURTS),
                scale=alt.Scale(
                    domain=list(COURTS), range=["#167D9A", "#8262AB", "#C58C31", "#A6B6C9"]
                ),
                legend=alt.Legend(orient="top", title=None, columns=2),
            ),
            order=alt.Order("Court order:Q"),
            stroke=alt.condition(
                alt.datum.Allocation == "This run", alt.value("#132A43"), alt.value("transparent")
            ),
            strokeWidth=alt.value(1),
            tooltip=[
                "Allocation:N",
                "Court:N",
                "Assignments:Q",
                alt.Tooltip("Share:Q", format=".1%"),
            ],
        )
        .properties(height=440)
    )
    st.altair_chart(
        comparison,
        width="stretch",
        alt=(
            "One highlighted stacked bar shows this run; twelve stacked bars show complete "
            "random allocations under the same expected probabilities and sample size."
        ),
    )
    st.caption(
        f"Every row totals {n_cases:,} cases. These fixed examples illustrate variation; "
        f"the global check uses all {MC_SIMULATIONS:,} simulated allocations."
    )

    with st.expander("Show the global test details"):
        st.write(
            f"**{global_result.exceedances:,} of {global_result.simulations:,}** simulated "
            "allocations had an overall discrepancy at least as large as this run, including ties."
        )
        if global_result.exceedances == 0:
            st.write(
                "Zero exceedances in a finite simulation does not mean the pattern is impossible."
            )
        st.metric("Global Monte Carlo p-value", p_text(global_result.p_value))
        st.write(
            "This p-value is a conditional tail probability under the stated model, "
            "not the probability of fraud or of the model being true."
        )
        st.latex(r"p_{MC}=\frac{b+1}{B+1}")
        st.write(
            f"Pearson discrepancy: {global_result.statistic:.4f}. "
            f"Smallest reportable Monte Carlo p-value: {global_result.min_attainable_p_value:.2e}."
        )
        se_label = (
            "boundary estimate; use the interval"
            if global_result.exceedances in (0, global_result.simulations)
            else f"{global_result.mc_standard_error:.2e}"
        )
        st.write(
            f"Approximate Monte Carlo standard error: {se_label}. "
            f"Exact 95% Monte Carlo uncertainty interval for the null tail probability: "
            f"[{global_result.mc_tail_ci_low:.2e}, {global_result.mc_tail_ci_high:.2e}]."
        )
        st.caption(
            "The SE uses b/B and may be zero at a boundary. The interval describes Monte Carlo "
            "uncertainty in the p-value estimate, not the probability that the model is true "
            "or uncertainty about wrongdoing or the assignment probabilities."
        )
        st.write(
            "**How the headline is chosen:** 5% is a reading guide, not a boundary between fair "
            "and unfair. We call a pattern unusual only when the corrected p-value and the "
            "interval's upper limit are below 5%. When the whole interval is above 5%, we report "
            "no clear departure. Otherwise the headline stays neutral because simulation "
            "uncertainty affects the comparison. One case receives a small-sample reminder."
        )
        st.caption(
            f"Comparison examples are the first 12 joint multinomial draws from seed {MC_SEED}, "
            "using this run's sample size and the stated probabilities. Their selection is fixed."
        )

    st.divider()
    st.subheader("A separate question, chosen in advance: Court A")
    st.write(
        "Before generating data, we selected Court A and asked only whether it receives "
        "unusually many assignments relative to its 40% chance."
    )
    target_columns = st.columns(2)
    target_columns[0].metric(
        "Court A observed / expected",
        f"{dataset.counts['Court A']:,} / {count_text(n_cases * 0.4)}",
    )
    target_columns[1].metric("Difference from its 40% share", f"{target.lift_pp:+.1f} pp")
    st.markdown(f"**{court_a_reading.headline}**")
    st.write(court_a_reading.explanation)
    st.caption(
        "This planned question is separate from the global check. Neither result identifies "
        "a cause or proves that the allocation process is fair."
    )
    with st.expander("Show the Court A test details"):
        st.metric(
            "Exact enrichment p-value",
            p_text(
                target.p_value, underflow=target.p_value_underflow, log_value=target.log_p_value
            ),
        )
        st.write(
            "The exact one-sided binomial calculation is P(X ≥ observed | n, p₀ = 0.40). "
            "The reading guide uses 5% for this preselected excess question only. "
            "It is not a test for a shortfall and the p-value is not a probability of fraud."
        )

    with st.expander("Explore all four courts — adjusted comparisons"):
        st.write(
            "This is a different question: which courts show an excess when all four are "
            "examined? Holm adjusts the complete family of four one-sided tests. "
            "These results need not agree with the global check or the planned Court A check."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Court": court,
                        "Difference (pp)": result.lift_pp,
                        "Holm-adjusted p-value": p_text(
                            result.adjusted_p_value,
                            underflow=result.adjusted_p_value_underflow,
                            log_value=result.adjusted_log_p_value,
                        ),
                        "Raw enrichment p-value": p_text(
                            result.p_value,
                            underflow=result.p_value_underflow,
                            log_value=result.log_p_value,
                        ),
                    }
                    for court, result in zip(COURTS, run["exploratory"], strict=True)
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Holm covers these four courts in one analysis. Repeatedly trying seeds, sample "
            "sizes or scenarios and retaining interesting results adds selection not covered here."
        )

with methodology:
    st.subheader("Three questions, three different readings")
    st.write(
        "The main summary checks the whole four-court allocation. The planned Court A question "
        "looks only for an excess in that court. The optional exploration checks all four courts "
        "with a multiple-testing adjustment. These are separate questions, not a combined verdict."
    )
    with st.expander("What the model assumes"):
        st.write(
            "Every independent case is assigned exactly once to Court A, B, C or D with stated "
            "probabilities 0.40, 0.30, 0.20 and 0.10. These probabilities are known inputs, "
            "not estimates from the sample. Eligibility and probabilities do not change."
        )
        st.write(
            "Real eligibility rules, workloads, related cases or changing probabilities could "
            "require a different model. Judicial outcomes and resolved-case subsets "
            "are not analyzed."
        )
    with st.expander("How the calculations work"):
        st.markdown(
            "1. **Global:** compare all counts using Pearson's discrepancy from the specified "
            "multinomial probabilities.\n"
            "2. **Calibration:** simulate complete allocations under that model and count "
            "discrepancies at least as large, including ties. This supports sparse counts without "
            "a large-sample chi-square approximation.\n"
            "3. **Planned Court A question:** calculate the exact upper-tail "
            "binomial probability.\n"
            "4. **Exploration:** adjust the complete four-court enrichment family with Holm."
        )
        st.caption(
            "A p-value measures compatibility with assumptions; effect size measures the size "
            "of a difference. No combined significance decision is made from the separate checks."
        )
    with st.expander("What you can and cannot conclude"):
        st.write(
            "A pattern can be unusual under a model without identifying why it occurred. "
            "A pattern without a clear departure does not establish that the real process was "
            "random or fair. These synthetic examples are not legal evidence, and no p-value "
            "is the probability of fraud, guilt or a hypothesis being true."
        )

with exports:
    st.subheader("Reproduce or download this run")
    st.code(
        "python -m court_assignment_audit.cli "
        f"--scenario {scenario} --seed {seed} --n-cases {n_cases} --output-dir artifacts/demo"
    )
    st.write(
        "The run log records the mechanism, both seeds, sample size, all results, software "
        "versions and a SHA-256 checksum of the synthetic CSV. Every row is invented."
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
