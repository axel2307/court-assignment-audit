"""Reusable charts for the exact, preselected binomial comparison."""

from json import dumps
from math import isclose, isfinite, log

import altair as alt

from court_assignment_audit.binomial_view import BinomialView

BASELINE_COLOR = "#167D9A"
TAIL_COLOR = "#D97706"
EXPECTED_COLOR = "#173B64"
OBSERVED_COLOR = "#C64B13"


def _point_probability_label(pmf: float, log_pmf: float) -> str:
    """Preserve a tiny positive probability when its float has underflowed."""
    if pmf == 0 and isfinite(log_pmf):
        return f"approximately 10^({log_pmf / log(10):.3f}); below floating-point range"
    if pmf == 0:
        return "0 (impossible under this model)"
    return f"{pmf:.5g}"


def build_binomial_chart(view: BinomialView) -> alt.LayerChart:
    """Plot the view's unmodified PMF, inclusive tail and two reference lines.

    Every bar represents one integer count; no binning, aggregation, probability
    normalization, or approximation occurs here. A small omitted mass and bars
    too short to see must be explained by the app using the view's range note.
    """
    below_label = "Below observed count"
    tail_label = "Observed count or more (tail)"
    same_position = isclose(
        view.observed_count, view.expected_count, rel_tol=0.0, abs_tol=1e-10
    )
    observed_label = (
        "Observed (same position, solid)" if same_position else "Observed count (solid)"
    )
    expected_label = (
        "Expected (same position, dashed)" if same_position else "Expected count (dashed)"
    )
    reference_labels = [observed_label, expected_label]
    legend_labels = [below_label, tail_label, observed_label, expected_label]
    legend_colors = [BASELINE_COLOR, TAIL_COLOR, OBSERVED_COLOR, EXPECTED_COLOR]
    is_reference = (
        f"datum.value === {dumps(observed_label)} || datum.value === {dumps(expected_label)}"
    )
    # One shared legend avoids horizontal stacking of multiple legend groups
    # on narrow screens. Each entry retains its own square or line symbol.
    chart_legend = alt.Legend(
        title="Bars and reference lines",
        orient="top",
        direction="vertical",
        columns=1,
        symbolType=alt.ExprRef(expr=f"{is_reference} ? 'stroke' : 'square'"),
        symbolSize=alt.ExprRef(expr=f"{is_reference} ? 500 : 100"),
        symbolDash=alt.ExprRef(
            expr=f"datum.value === {dumps(expected_label)} ? [6, 4] : [1, 0]"
        ),
        symbolStrokeWidth=2,
        labelLimit=260,
    )
    points = [
        {
            "count": point.count,
            "pmf": point.pmf,
            "probability_label": _point_probability_label(point.pmf, point.log_pmf),
            "region": tail_label if point.in_tail else below_label,
            "legend_label": tail_label if point.in_tail else below_label,
            "observed_count": view.observed_count,
            "expected_count": view.expected_count,
        }
        for point in view.points
    ]
    # Literal values avoid Altair's 5,000-row dataframe limit for a valid view
    # spanning all 5,001 possible counts at the application's largest sample.
    data = alt.Data(values=points)
    lower = min(view.lower_count, view.observed_count, view.expected_count) - 0.5
    upper = max(view.upper_count, view.observed_count, view.expected_count) + 0.5
    x_scale = alt.Scale(type="linear", domain=[lower, upper], nice=False, zero=False)
    x_axis = alt.Axis(format=",d", tickMinStep=1, tickCount=5, labelOverlap=True)

    bars = (
        alt.Chart(data)
        .mark_bar(
            # Width follows integer spacing at every container size, retaining
            # gaps between adjacent counts instead of overlapping fixed bars.
            size=alt.ExprRef(expr=f"0.85 * width / {upper - lower}"),
            clip=True,
        )
        .encode(
            x=alt.X("count:Q", title="Number of assignments", scale=x_scale, axis=x_axis),
            y=alt.Y(
                "pmf:Q",
                title="Probability of this exact count",
                scale=alt.Scale(type="linear", zero=True),
                axis=alt.Axis(format=".1%"),
                stack=None,
            ),
            color=alt.Color(
                "legend_label:N",
                scale=alt.Scale(domain=legend_labels, range=legend_colors),
                legend=chart_legend,
            ),
            tooltip=[
                alt.Tooltip("count:Q", title="Assignment count", format=",d"),
                alt.Tooltip("probability_label:N", title="P(X = this count)"),
                alt.Tooltip("region:N", title="Tail membership"),
                alt.Tooltip("expected_count:Q", title="Expected count (mean)", format=",.2f"),
                alt.Tooltip("observed_count:Q", title="Observed count", format=",d"),
            ],
        )
    )

    observed_point = next(point for point in view.points if point.count == view.observed_count)

    def reference_line(position: float, label: str, width: int) -> alt.Chart:
        return (
            alt.Chart(
                alt.Data(
                    values=[
                        {
                            "position": position,
                            "reference": label,
                            "legend_label": label,
                            "observed_count": view.observed_count,
                            "expected_count": view.expected_count,
                            "observed_probability": _point_probability_label(
                                observed_point.pmf, observed_point.log_pmf
                            ),
                        }
                    ]
                )
            )
            .mark_rule(strokeWidth=width, clip=True)
            .encode(
                x=alt.X("position:Q", title="Number of assignments", scale=x_scale, axis=x_axis),
                color=alt.Color(
                    "legend_label:N",
                    scale=alt.Scale(domain=legend_labels, range=legend_colors),
                    legend=chart_legend,
                ),
                strokeDash=alt.StrokeDash(
                    "reference:N",
                    scale=alt.Scale(domain=reference_labels, range=[[1, 0], [6, 4]]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("reference:N", title="Reference line"),
                    alt.Tooltip("expected_count:Q", title="Expected count (mean)", format=",.2f"),
                    alt.Tooltip("observed_count:Q", title="Observed count", format=",d"),
                    alt.Tooltip("observed_probability:N", title="P(X = observed count)"),
                ],
            )
        )

    # Paint the wider solid line first, then the narrower dashed line. When
    # expected and observed coincide, both styles and colors remain visible.
    references = alt.layer(
        reference_line(view.observed_count, observed_label, 4),
        reference_line(view.expected_count, expected_label, 2),
    )
    return (
        alt.layer(bars, references)
        .resolve_scale(color="shared")
        .properties(height=380, width="container")
        .configure_legend(orient="top", direction="vertical", columns=1)
        .configure_view(stroke=None)
    )
