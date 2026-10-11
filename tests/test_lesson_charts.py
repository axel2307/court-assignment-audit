"""The lesson's exact probabilities and empirical frequencies stay distinct."""

import json
from math import comb

import pytest

from court_assignment_audit.binomial_view import build_binomial_view
from court_assignment_audit.lesson_charts import (
    BASE_COLOR,
    EXACT_COLOR,
    MORE_COLOR,
    build_lesson_distribution,
    build_simulation_chart,
    plot_upper_count,
)
from court_assignment_audit.statistics import exact_target_test


def view_for(observed=10):
    return build_binomial_view(exact_target_test([observed, 60 - observed], [1 / 12, 11 / 12]))


def color_map(layer):
    scale = layer["encoding"]["color"]["scale"]
    return dict(zip(scale["domain"], scale["range"], strict=True))


def test_default_display_adds_true_pmf_bars_through_23_instead_of_empty_padding():
    view = view_for(0)
    assert view.upper_count < 23
    assert plot_upper_count(view) == 23
    spec = build_lesson_distribution(view).to_dict()
    bar = spec["layer"][0]
    rows = bar["data"]["values"]
    assert [row["count"] for row in rows] == list(range(24))
    for row in rows:
        count = row["count"]
        independent_probability = comb(60, count) * (1 / 12) ** count * (11 / 12) ** (60 - count)
        assert row["pmf"] == pytest.approx(independent_probability, rel=2e-12, abs=0)
        assert row["pmf"] > 0
    assert bar["encoding"]["x"]["scale"]["domain"] == [-0.5, 23.5]


def test_breakdown_distinguishes_exactly_10_from_11_or_more_without_changing_heights():
    view = view_for()
    spec = build_lesson_distribution(view, mode="breakdown").to_dict()
    bar = spec["layer"][0]
    rows = bar["data"]["values"]
    colors = color_map(bar)
    for row in rows:
        count = row["count"]
        assert row["legend_label"] == (
            "Menos de 10" if count < 10 else "Exactamente 10" if count == 10 else "11 o más"
        )
        assert colors[row["legend_label"]] == (
            BASE_COLOR if count < 10 else EXACT_COLOR if count == 10 else MORE_COLOR
        )
    exact_probability = comb(60, 10) * 11**50 / 12**60
    assert rows[10]["pmf"] == pytest.approx(exact_probability, rel=2e-12, abs=0)
    shown_label_probability = float(rows[10]["probability_label"].replace(",", ".")[:-1]) / 100
    assert shown_label_probability == pytest.approx(exact_probability, rel=1e-9)
    for mode in ["tail", "exact", "neutral"]:
        other = build_lesson_distribution(view, mode=mode).to_dict()["layer"][0]["data"]["values"]
        assert [row["pmf"] for row in other] == [row["pmf"] for row in rows]


@pytest.mark.parametrize("observed", [0, 10, 15, 60])
def test_tail_mode_is_inclusive_at_the_threshold_and_support_ends(observed):
    bar = build_lesson_distribution(view_for(observed), mode="tail").to_dict()["layer"][0]
    colors = color_map(bar)
    for row in bar["data"]["values"]:
        assert (colors[row["legend_label"]] == EXACT_COLOR) == (row["count"] >= observed)


def test_exact_mode_highlights_one_count_while_neutral_has_no_highlight():
    view = view_for()
    exact = build_lesson_distribution(view, mode="exact").to_dict()["layer"][0]
    colors = color_map(exact)
    highlighted = [
        row["count"]
        for row in exact["data"]["values"]
        if colors[row["legend_label"]] == EXACT_COLOR
    ]
    assert highlighted == [10]
    neutral = build_lesson_distribution(view, mode="neutral").to_dict()["layer"][0]
    neutral_colors = color_map(neutral)
    assert {neutral_colors[row["legend_label"]] for row in neutral["data"]["values"]} == {
        BASE_COLOR
    }


def test_reference_mean_and_observation_have_distinct_spanish_labels_and_styles():
    _, observed, expected = build_lesson_distribution(view_for()).to_dict()["layer"]
    assert observed["data"]["values"][0]["position"] == 10
    assert observed["data"]["values"][0]["legend_label"] == "Observado: 10"
    assert observed["mark"]["strokeDash"] == [1, 0]
    assert expected["data"]["values"][0]["position"] == pytest.approx(5)
    assert expected["data"]["values"][0]["legend_label"] == "Esperado: 5"
    assert expected["mark"]["strokeDash"] == [6, 4]


def test_practice_threshold_is_labelled_hypothetical_and_coincident_lines_stay_visible():
    spec = build_lesson_distribution(view_for(5), hypothetical=True).to_dict()
    _, hypothetical, expected = spec["layer"]
    assert hypothetical["data"]["values"][0]["legend_label"] == "Hipotético: 5 (coincide)"
    assert expected["data"]["values"][0]["legend_label"] == "Esperado: 5 (coincide)"
    assert hypothetical["mark"]["strokeWidth"] > expected["mark"]["strokeWidth"]
    assert hypothetical["data"]["values"][0]["position"] == 5
    assert expected["data"]["values"][0]["position"] == pytest.approx(5)


@pytest.mark.parametrize("observed", [0, 60])
def test_full_support_endpoints_remain_serializable_and_observed_is_never_outside_domain(observed):
    view = view_for(observed)
    spec = build_lesson_distribution(view, mode="breakdown").to_dict()
    bar = spec["layer"][0]
    rows = bar["data"]["values"]
    assert rows[observed]["count"] == observed
    assert rows[observed]["pmf"] > 0
    assert rows[-1]["count"] == plot_upper_count(view)
    assert bar["encoding"]["x"]["scale"]["domain"][1] > observed
    json.dumps(spec, allow_nan=False)


def test_probability_chart_uses_discrete_counts_linear_probability_and_one_compact_legend():
    spec = build_lesson_distribution(view_for(), mode="breakdown").to_dict()
    bar = spec["layer"][0]
    assert bar["mark"]["type"] == "bar"
    assert bar["encoding"]["x"]["field"] == "count"
    assert bar["encoding"]["x"]["type"] == "quantitative"
    assert bar["encoding"]["x"]["title"] == "Cantidad de expedientes en A"
    assert bar["encoding"]["x"]["axis"]["tickMinStep"] == 1
    assert bar["encoding"]["y"]["field"] == "pmf"
    assert bar["encoding"]["y"]["title"] == "Probabilidad de ese conteo"
    assert bar["encoding"]["y"]["axis"]["format"] == ".0%"
    assert bar["encoding"]["y"]["scale"] == {"type": "linear", "zero": True}
    assert spec["width"] == "container"
    assert spec["resolve"]["scale"]["color"] == "shared"
    color = bar["encoding"]["color"]
    assert len(color["scale"]["domain"]) == 5
    assert color["legend"]["columns"] == 1
    assert color["legend"]["direction"] == "vertical"
    for layer in spec["layer"]:
        assert layer["encoding"]["color"] == color
        assert "transform" not in layer
        assert "aggregate" not in layer["encoding"]["x"]
        assert "bin" not in layer["encoding"]["x"]
        assert "strokeDash" not in layer["encoding"]


def test_invalid_distribution_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        build_lesson_distribution(view_for(), mode="density")


def test_simulation_histogram_preserves_frequencies_and_empty_integer_bins():
    values = [0, 5, 5, 9, 10, 10, 11, 23]
    spec = build_simulation_chart(values).to_dict()
    bar, reference = spec["layer"]
    rows = bar["data"]["values"]
    assert [row["count"] for row in rows] == list(range(24))
    assert sum(row["frequency"] for row in rows) == len(values)
    assert rows[5]["frequency"] == 2
    assert rows[10]["frequency"] == 2
    assert rows[6]["frequency"] == 0
    assert all(row["total_simulations"] == len(values) for row in rows)
    colors = color_map(bar)
    assert all(
        (colors[row["legend_label"]] == EXACT_COLOR) == (row["count"] >= 10) for row in rows
    )
    assert bar["encoding"]["y"]["field"] == "frequency"
    assert bar["encoding"]["y"]["title"] == "Cantidad de repartos"
    assert bar["encoding"]["y"]["axis"]["tickMinStep"] == 1
    assert bar["encoding"]["x"]["scale"]["domain"] == [-0.5, 23.5]
    assert reference["data"]["values"][0]["position"] == 10
    assert reference["data"]["values"][0]["legend_label"] == "Observado: 10"
    assert "frequency" in {item["field"] for item in bar["encoding"]["tooltip"]}


def test_simulation_domain_extends_to_include_every_count_up_to_60():
    spec = build_simulation_chart([0, 24, 60], threshold=60).to_dict()
    bar = spec["layer"][0]
    rows = bar["data"]["values"]
    assert len(rows) == 61
    assert rows[0]["frequency"] == rows[24]["frequency"] == rows[60]["frequency"] == 1
    assert bar["encoding"]["x"]["scale"]["domain"] == [-0.5, 60.5]
    assert spec["layer"][1]["data"]["values"][0]["position"] == 60
    json.dumps(spec, allow_nan=False)


@pytest.mark.parametrize("values", [[], [True], [-1], [61], [1.5], ["5"], [float("nan")]])
def test_invalid_simulated_counts_are_rejected(values):
    with pytest.raises(ValueError):
        build_simulation_chart(values)


@pytest.mark.parametrize("threshold", [True, -1, 61, 1.5])
def test_invalid_simulation_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        build_simulation_chart([5], threshold=threshold)
