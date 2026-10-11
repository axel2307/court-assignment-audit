"""Chart contracts preserve discrete probability and reference-line meaning."""

import json
from dataclasses import replace
from math import log10

import pytest

from court_assignment_audit.binomial_view import build_binomial_view
from court_assignment_audit.charts import (
    BASELINE_COLOR,
    EXPECTED_COLOR,
    OBSERVED_COLOR,
    TAIL_COLOR,
    build_binomial_chart,
)
from court_assignment_audit.statistics import exact_target_test


def make_view(n=10, observed=7, probability=0.4):
    result = exact_target_test([observed, n - observed], [probability, 1 - probability])
    return build_binomial_view(result)


def leaves(spec):
    if "layer" in spec:
        return [leaf for layer in spec["layer"] for leaf in leaves(layer)]
    return [spec]


def test_exact_count_axis_and_linear_probability_bars_have_no_aggregation():
    spec = build_binomial_chart(make_view()).to_dict()
    bar, *rules = leaves(spec)
    assert bar["mark"]["type"] == "bar"
    assert all(rule["mark"]["type"] == "rule" for rule in rules)
    assert len(rules) == 2
    assert bar["encoding"]["x"]["field"] == "count"
    assert bar["encoding"]["x"]["type"] == "quantitative"
    assert bar["encoding"]["x"]["axis"]["tickMinStep"] == 1
    assert bar["encoding"]["x"]["axis"]["format"] == ",d"
    assert bar["encoding"]["y"]["field"] == "pmf"
    assert bar["encoding"]["y"]["axis"]["format"] == ".1%"
    assert bar["encoding"]["y"]["scale"] == {"type": "linear", "zero": True}
    assert bar["encoding"]["y"]["stack"] is None
    for layer in leaves(spec):
        assert "transform" not in layer
        for encoding in layer["encoding"].values():
            if isinstance(encoding, dict):
                assert "aggregate" not in encoding
                assert "bin" not in encoding
    assert spec["width"] == "container"
    assert 360 <= spec["height"] <= 400


def test_small_exact_distribution_retains_each_probability_and_inclusive_tail():
    spec = build_binomial_chart(make_view(n=3, observed=2, probability=0.5)).to_dict()
    bar = leaves(spec)[0]
    rows = bar["data"]["values"]
    assert [row["count"] for row in rows] == [0, 1, 2, 3]
    assert [row["pmf"] for row in rows] == pytest.approx([1 / 8, 3 / 8, 3 / 8, 1 / 8])
    scale = bar["encoding"]["color"]["scale"]
    colors = dict(zip(scale["domain"], scale["range"], strict=True))
    assert [colors[row["region"]] for row in rows] == [
        BASELINE_COLOR,
        BASELINE_COLOR,
        TAIL_COLOR,
        TAIL_COLOR,
    ]
    assert "or more" in rows[2]["region"]
    assert "tail" in rows[2]["region"]


def test_one_shared_four_entry_legend_prevents_duplicate_and_side_by_side_groups():
    spec = build_binomial_chart(make_view()).to_dict()
    bar, *rules = leaves(spec)
    color = bar["encoding"]["color"]
    assert spec["resolve"]["scale"]["color"] == "shared"
    assert color["field"] == "legend_label"
    assert len(color["scale"]["domain"]) == 4
    assert color["legend"]["orient"] == "top"
    assert color["legend"]["direction"] == "vertical"
    assert color["legend"]["columns"] == 1
    for rule in rules:
        assert rule["encoding"]["color"] == color
        assert rule["encoding"]["strokeDash"]["legend"] is None
    assert "'stroke' : 'square'" in color["legend"]["symbolType"]["expr"]
    assert "[6, 4] : [1, 0]" in color["legend"]["symbolDash"]["expr"]


@pytest.mark.parametrize("observed", [0, 3])
def test_tail_includes_observation_at_both_support_endpoints(observed):
    bar = leaves(build_binomial_chart(make_view(3, observed, 0.5)).to_dict())[0]
    scale = bar["encoding"]["color"]["scale"]
    colors = dict(zip(scale["domain"], scale["range"], strict=True))
    tail_counts = [
        row["count"]
        for row in bar["data"]["values"]
        if colors[row["region"]] == TAIL_COLOR
    ]
    assert tail_counts == list(range(observed, 4))


def test_trimmed_display_keeps_original_pmf_and_domain_includes_both_references():
    view = make_view(n=1000, observed=600)
    bar, observed, expected = leaves(build_binomial_chart(view).to_dict())
    rows = bar["data"]["values"]
    assert [(row["count"], row["pmf"]) for row in rows] == [
        (point.count, point.pmf) for point in view.points
    ]
    assert view.omitted_mass > 0
    assert sum(row["pmf"] for row in rows) < 1
    domain = bar["encoding"]["x"]["scale"]["domain"]
    assert domain[0] < view.expected_count < domain[1]
    assert domain[0] < view.observed_count < domain[1]
    assert observed["data"]["values"][0]["position"] == 600
    assert expected["data"]["values"][0]["position"] == 400


@pytest.mark.parametrize("observed_count", [4, 7])
def test_reference_positions_styles_and_paint_order_remain_distinguishable(observed_count):
    spec = build_binomial_chart(make_view(observed=observed_count)).to_dict()
    _, observed, expected = leaves(spec)
    assert observed["data"]["values"][0]["position"] == observed_count
    assert expected["data"]["values"][0]["position"] == 4
    assert observed["mark"]["strokeWidth"] > expected["mark"]["strokeWidth"]
    for rule, intended_color, intended_dash in [
        (observed, OBSERVED_COLOR, [1, 0]),
        (expected, EXPECTED_COLOR, [6, 4]),
    ]:
        label = rule["data"]["values"][0]["reference"]
        for channel, intended in [("color", intended_color), ("strokeDash", intended_dash)]:
            scale = rule["encoding"][channel]["scale"]
            assert dict(zip(scale["domain"], scale["range"], strict=True))[label] == intended
        if observed_count == 4:
            assert "same position" in label
    assert "Observed" in observed["data"]["values"][0]["reference"]
    assert "Expected" in expected["data"]["values"][0]["reference"]


def test_underflow_tooltip_uses_log_pmf_on_the_observed_rule_and_count():
    view = make_view(n=5000, observed=5000)
    spec = build_binomial_chart(view).to_dict()
    bar, observed, _ = leaves(spec)
    row = next(row for row in bar["data"]["values"] if row["count"] == 5000)
    assert row["pmf"] == 0  # Numeric underflow; the mathematical probability is positive.
    expected_exponent = f"{5000 * log10(0.4):.3f}"
    assert expected_exponent in row["probability_label"]
    assert "10^(" in row["probability_label"]
    assert "below floating-point range" in row["probability_label"]
    assert "impossible" not in row["probability_label"]
    assert observed["data"]["values"][0]["observed_probability"] == row["probability_label"]
    tooltip_fields = {tooltip["field"] for tooltip in bar["encoding"]["tooltip"]}
    assert {"count", "probability_label", "expected_count", "observed_count"} <= tooltip_fields
    assert any(
        tooltip["field"] == "observed_probability"
        for tooltip in observed["encoding"]["tooltip"]
    )
    json.dumps(spec, allow_nan=False)


def test_coincidence_label_tolerates_roundoff_without_moving_either_reference():
    view = replace(make_view(n=10, observed=2, probability=0.2), expected_count=2.0000000000000004)
    _, observed, expected = leaves(build_binomial_chart(view).to_dict())
    observed_row = observed["data"]["values"][0]
    expected_row = expected["data"]["values"][0]
    assert "same position" in observed_row["reference"]
    assert "same position" in expected_row["reference"]
    assert observed_row["position"] == 2
    assert expected_row["position"] == 2.0000000000000004


def test_full_supported_range_serializes_5001_counts_and_true_impossible_probability():
    view = make_view(n=5000, observed=5000, probability=0)
    spec = build_binomial_chart(view).to_dict()
    bar = leaves(spec)[0]
    rows = bar["data"]["values"]
    assert len(rows) == 5001
    assert [row["count"] for row in rows] == list(range(5001))
    assert rows[0]["pmf"] == 1
    assert rows[-1]["probability_label"] == "0 (impossible under this model)"
    json.dumps(spec, allow_nan=False)
