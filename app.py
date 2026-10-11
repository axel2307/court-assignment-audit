"""An explainable Streamlit case study using entirely invented assignments."""

import math

import altair as alt
import pandas as pd
import streamlit as st

from court_assignment_audit.binomial_view import build_binomial_view
from court_assignment_audit.charts import build_binomial_chart
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
binomial_view = build_binomial_view(target)

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
    st.subheader("Court A's count under the stated 40% rule")
    st.write("We chose Court A in advance and asked only about receiving extra assignments.")
    distribution_column, result_column = st.columns([2.2, 1], gap="large")
    with distribution_column:
        st.altair_chart(
            build_binomial_chart(binomial_view),
            width="stretch",
            alt=(
                "Exact binomial probabilities for each possible Court A count. "
                "Orange bars mark the observed count or more; a dashed line marks the "
                "expected count and a solid line marks the observed count."
            ),
        )
        if binomial_view.lower_count == 0 and binomial_view.upper_count == n_cases:
            st.caption(f"All possible counts from 0 to {n_cases:,} are shown.")
        else:
            st.caption(
                f"Showing counts {binomial_view.lower_count:,} to {binomial_view.upper_count:,}. "
                "At most 0.0001% of the total probability lies outside this range. "
                "The reported probability uses the full tail, even outside the plotted range."
            )
        if math.isclose(target.observed_count, target.expected_count, rel_tol=0, abs_tol=1e-10):
            st.caption(
                "Observed matches expected. Both markers share the same position: "
                "the dashed expected line sits over the solid observed line."
            )
        tallest_bar = max(point.pmf for point in binomial_view.points)
        tallest_tail_bar = max(
            (point.pmf for point in binomial_view.points if point.in_tail), default=0
        )
        if tallest_tail_bar < tallest_bar * 0.003:
            st.caption(
                "The orange tail is too small to see clearly on this linear scale. "
                "Its probability is still calculated over every count through n."
            )
    with result_column:
        with st.container(border=True):
            st.caption("COURT A: OBSERVED / EXPECTED")
            st.markdown(f"### {target.observed_count:,} / {count_text(target.expected_count)}")
            st.caption(f"Share difference: {target.lift_pp:+.1f} percentage points")
            st.divider()
            st.caption(f"PROBABILITY OF {target.observed_count:,} OR MORE")
            probability_display = binomial_view.probability_label.split(" (", 1)[0]
            probability_display = probability_display.removesuffix(" probability")
            st.markdown(f"### {probability_display}")
            if target.p_value_underflow:
                st.caption("Positive probability, displayed using its logarithm.")
            st.write(
                f"Under the stated rule, this is the probability of **{target.observed_count:,} "
                f"or more** assignments to Court A in another **{n_cases:,}-case allocation**."
            )
            if binomial_view.frequency_label:
                st.write(
                    f"{binomial_view.frequency_label.capitalize()} would give Court A "
                    "at least this many cases."
                )
            if n_cases == 1:
                st.caption("One assignment cannot establish a stable allocation pattern.")
            st.markdown(f"**{court_a_reading.headline}**")
    st.caption(
        "Entirely synthetic data. This is not the probability of misconduct. "
        "Neither a common nor an unusual result proves whether the process was random or fair."
    )
    with st.expander("Show the Court A calculation"):
        st.write(court_a_reading.explanation)
        st.caption(binomial_view.range_note)
        st.write(
            "Every bar is the exact binomial probability of one integer count, not a normal "
            "approximation. Orange marks counts at least as large as observed, including ties. "
            "The reported tail sums over the complete support from the observed count to n; "
            "cropping the chart does not change or renormalize it."
        )
        st.latex(
            r"X_A \sim \operatorname{Binomial}(n, 0.40), \qquad "
            r"p = P(X_A \geq k)"
        )
        st.metric(
            "Exact enrichment p-value",
            p_text(
                target.p_value, underflow=target.p_value_underflow, log_value=target.log_p_value
            ),
        )
        st.write(f"Natural log of the tail probability: {target.log_p_value:.12g}.")
        if target.p_value_underflow:
            st.write(
                "The probability is positive but too small for an ordinary floating-point "
                "number. The displayed power of ten uses the log tail; it is not zero."
            )
        st.write(
            "Displayed probabilities are rounded. Expected count means an average over "
            "repetitions, not a quota. The 5% reading guide applies only to this preselected "
            "upper-tail question; it does not test for a shortfall."
        )
    st.divider()
    st.subheader("The whole allocation: a separate model check")
    with st.container(border=True):
        st.caption("DO THE FOUR COURTS TOGETHER FIT THE STATED RULE?")
        st.markdown(f"### {reading.headline}")
        st.write(reading.explanation)
        st.caption(
            "This global check and Court A use the same allocation. They answer different "
            "questions and need not agree."
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

    with st.expander("Compare twelve complete random allocations"):
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
                    alt.datum.Allocation == "This run",
                    alt.value("#132A43"),
                    alt.value("transparent"),
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
        "The first graphic shows the planned Court A question, which looks only for an excess. "
        "The global check below it assesses the whole four-court allocation. "
        "The optional exploration checks all four courts "
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
