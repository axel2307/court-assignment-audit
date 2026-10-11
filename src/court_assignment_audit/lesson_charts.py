"""Spanish teaching charts that distinguish exact counts from inclusive tails."""

from collections import Counter
from json import dumps
from math import isclose, isfinite, log
from numbers import Integral
from typing import Literal, Sequence

import altair as alt
from scipy.stats import binom

from court_assignment_audit.binomial_view import BinomialView

LESSON_CASES = 60
LESSON_PLOT_MAX = 23
BASE_COLOR = "#6298A5"
EXACT_COLOR = "#D97706"
MORE_COLOR = "#F2BA70"
EXPECTED_COLOR = "#173B64"
OBSERVED_COLOR = "#C64B13"
DistributionMode = Literal["breakdown", "tail", "exact", "neutral"]


def plot_upper_count(view: BinomialView) -> int:
    """Return the actual last PMF count displayed, always within the support."""
    return min(view.n_cases, max(LESSON_PLOT_MAX, view.upper_count))


def _number_label(value: float) -> str:
    if isclose(value, round(value), rel_tol=0, abs_tol=1e-10):
        return str(round(value))
    return f"{value:.4g}".replace(".", ",")


def _probability_label(probability: float, log_probability: float) -> str:
    if probability == 0 and isfinite(log_probability):
        exponent = log_probability / log(10) + 2
        return f"aprox. 10^({_number_label(exponent)})% (menor que la precisión numérica)"
    if probability == 0:
        return "0% (imposible bajo esta regla)"
    return f"{100 * probability:.10g}%".replace(".", ",")


def _x(field: str, upper: int) -> alt.X:
    return alt.X(
        f"{field}:Q",
        title="Cantidad de expedientes en A",
        scale=alt.Scale(type="linear", domain=[-0.5, upper + 0.5], zero=False, nice=False),
        axis=alt.Axis(format="d", tickMinStep=1, tickCount=6, labelOverlap=True),
    )


def _color(
    labels: list[str],
    colors: list[str],
    reference_labels: list[str],
    dashed_labels: list[str],
) -> alt.Color:
    # A single shared legend keeps its entries in one column at mobile widths.
    # Expressions preserve both line styles without a second dash legend.
    is_reference = f"indexof({dumps(reference_labels)}, datum.value) >= 0"
    is_dashed = f"indexof({dumps(dashed_labels)}, datum.value) >= 0"
    return alt.Color(
        "legend_label:N",
        scale=alt.Scale(domain=labels, range=colors),
        legend=alt.Legend(
            title=None,
            orient="top",
            direction="vertical",
            columns=1,
            symbolType=alt.ExprRef(expr=f"{is_reference} ? 'stroke' : 'square'"),
            symbolSize=alt.ExprRef(expr=f"{is_reference} ? 500 : 100"),
            symbolDash=alt.ExprRef(expr=f"{is_dashed} ? [6, 4] : [1, 0]"),
            symbolStrokeWidth=2,
            labelLimit=250,
        ),
    )


def _rule(
    position: float,
    label: str,
    upper: int,
    color: alt.Color,
    *,
    dashed: bool = False,
    probability_label: str | None = None,
) -> alt.Chart:
    row = {"position": position, "legend_label": label}
    tooltip = [
        alt.Tooltip("legend_label:N", title="Referencia"),
        alt.Tooltip("position:Q", title="Cantidad de expedientes", format=".6g"),
    ]
    if probability_label is not None:
        row["probability_label"] = probability_label
        tooltip.append(alt.Tooltip("probability_label:N", title="Probabilidad de ese conteo"))
    return (
        alt.Chart(alt.Data(values=[row]))
        .mark_rule(strokeWidth=2 if dashed else 4, strokeDash=[6, 4] if dashed else [1, 0])
        .encode(x=_x("position", upper), color=color, tooltip=tooltip)
    )


def _finish(*layers: alt.Chart) -> alt.LayerChart:
    return (
        alt.layer(*layers)
        .resolve_scale(color="shared")
        .properties(width="container", height=360)
        .configure_legend(orient="top", direction="vertical", columns=1)
        .configure_view(stroke=None)
    )


def build_lesson_distribution(
    view: BinomialView,
    mode: DistributionMode = "tail",
    hypothetical: bool = False,
) -> alt.LayerChart:
    """Plot true binomial probabilities, with distinct exact-count/tail modes.

    The lesson normally uses 60 assignments and probability 1/12. Its default
    0..23 axis includes every true PMF in that interval, including counts beyond
    the helper's narrower display. It expands if the view needs more support.
    Full-tail results remain those of the view, never a sum of visible bars.
    """
    if mode not in ("breakdown", "tail", "exact", "neutral"):
        raise ValueError("mode must be breakdown, tail, exact, or neutral.")
    upper = plot_upper_count(view)
    observed = view.observed_count
    if mode == "breakdown":
        bar_labels = [f"Menos de {observed}", f"Exactamente {observed}", f"{observed + 1} o más"]
        bar_colors = [BASE_COLOR, EXACT_COLOR, MORE_COLOR]
    elif mode == "tail":
        bar_labels = [f"Menos de {observed}", f"{observed} o más"]
        bar_colors = [BASE_COLOR, EXACT_COLOR]
    elif mode == "exact":
        bar_labels = ["Otros conteos", f"Exactamente {observed}"]
        bar_colors = [BASE_COLOR, EXACT_COLOR]
    else:
        bar_labels = ["Todos los conteos"]
        bar_colors = [BASE_COLOR]

    expected_label = f"Esperado: {_number_label(view.expected_count)}"
    observed_label = f"{'Hipotético' if hypothetical else 'Observado'}: {observed}"
    if isclose(observed, view.expected_count, rel_tol=0, abs_tol=1e-10):
        expected_label += " (coincide)"
        observed_label += " (coincide)"
    references = [observed_label, expected_label]
    color = _color(
        bar_labels + references,
        bar_colors + [OBSERVED_COLOR, EXPECTED_COLOR],
        references,
        [expected_label],
    )

    counts = list(range(upper + 1))
    probabilities = binom.pmf(counts, view.n_cases, view.probability)
    log_probabilities = binom.logpmf(counts, view.n_cases, view.probability)
    rows = []
    for count, probability, log_probability in zip(
        counts, probabilities, log_probabilities, strict=True
    ):
        if mode == "breakdown":
            label = bar_labels[0 if count < observed else 1 if count == observed else 2]
        elif mode == "tail":
            label = bar_labels[int(count >= observed)]
        elif mode == "exact":
            label = bar_labels[int(count == observed)]
        else:
            label = bar_labels[0]
        rows.append(
            {
                "count": count,
                "pmf": float(probability),
                "probability_label": _probability_label(float(probability), float(log_probability)),
                "legend_label": label,
            }
        )

    bars = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(size=alt.ExprRef(expr=f"0.82 * width / {upper + 1}"), clip=True)
        .encode(
            x=_x("count", upper),
            y=alt.Y(
                "pmf:Q",
                title="Probabilidad de ese conteo",
                scale=alt.Scale(type="linear", zero=True),
                axis=alt.Axis(format=".0%"),
                stack=None,
            ),
            color=color,
            tooltip=[
                alt.Tooltip("count:Q", title="Expedientes en A", format="d"),
                alt.Tooltip("probability_label:N", title="Probabilidad de ese conteo"),
                alt.Tooltip("legend_label:N", title="Qué muestra"),
            ],
        )
    )
    # The wider observed/hypothetical line sits behind the dashed expectation,
    # keeping both references visible when their counts coincide.
    return _finish(
        bars,
        _rule(
            observed,
            observed_label,
            upper,
            color,
            probability_label=rows[observed]["probability_label"],
        ),
        _rule(view.expected_count, expected_label, upper, color, dashed=True),
    )


def _validated_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= LESSON_CASES:
        raise ValueError(f"{name} must be an integer between 0 and {LESSON_CASES}.")
    return int(value)


def build_simulation_chart(counts: Sequence[int], threshold: int = 10) -> alt.LayerChart:
    """Count whole simulated 60-assignment allocations, keeping empty bins.

    These are empirical frequencies, not probabilities from the exact PMF.
    A lesson run supplies 1,000 counts; smaller sequences remain useful for
    tests and demonstrations and are labelled with their actual sample size.
    """
    threshold = _validated_count(threshold, "threshold")
    values = [_validated_count(value, "simulated count") for value in counts]
    if not values:
        raise ValueError("counts must contain at least one simulated allocation.")
    upper = max(LESSON_PLOT_MAX, threshold, max(values))
    frequencies = Counter(values)
    below_label, tail_label = f"Menos de {threshold}", f"{threshold} o más"
    reference_label = f"Observado: {threshold}"
    color = _color(
        [below_label, tail_label, reference_label],
        [BASE_COLOR, EXACT_COLOR, OBSERVED_COLOR],
        [reference_label],
        [],
    )
    rows = [
        {
            "count": count,
            "frequency": frequencies[count],
            "total_simulations": len(values),
            "legend_label": tail_label if count >= threshold else below_label,
        }
        for count in range(upper + 1)
    ]
    bars = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(size=alt.ExprRef(expr=f"0.82 * width / {upper + 1}"), clip=True)
        .encode(
            x=_x("count", upper),
            y=alt.Y(
                "frequency:Q",
                title="Cantidad de repartos",
                scale=alt.Scale(type="linear", zero=True),
                axis=alt.Axis(format="d", tickMinStep=1),
                stack=None,
            ),
            color=color,
            tooltip=[
                alt.Tooltip("count:Q", title="Expedientes en A", format="d"),
                alt.Tooltip("frequency:Q", title="Simulaciones con ese conteo", format="d"),
                alt.Tooltip("total_simulations:Q", title="Total de simulaciones", format="d"),
                alt.Tooltip("legend_label:N", title="Qué muestra"),
            ],
        )
    )
    return _finish(bars, _rule(threshold, reference_label, upper, color))
