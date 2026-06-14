"""Tests for homogene.estimator"""

import math

import pytest

from homogene.estimator import StatsEstimator


def make_stats(**overrides) -> dict:
    base = {
        "total_succeeded": 2,
        "total_failed": 0,
        "total_duration": 1.0,
        "total_cost": 0.02,
        "total_input_tokens": 100,
        "total_output_tokens": 50,
    }
    return {**base, **overrides}


def make_estimator(**overrides) -> StatsEstimator:
    defaults = dict(
        sample_stats=make_stats(),
        sample_durations=[0.5, 0.5],
        sample_prompt_lengths=[100, 100],
        full_prompt_lengths=[100, 100, 100, 100],
    )
    return StatsEstimator(**{**defaults, **overrides})


class TestStatsEstimatorEstimate:
    def test_returns_dict(self):
        assert isinstance(make_estimator().estimate(), dict)

    def test_has_all_keys(self):
        result = make_estimator().estimate()
        assert set(result.keys()) == {"total_succeeded", "total_failed", "total_duration", "total_cost", "total_input_tokens", "total_output_tokens"}

    def test_total_succeeded_scales_with_rows(self):
        # 2/2 succeeded → 4/4 estimated
        assert make_estimator().estimate()["total_succeeded"] == 4

    def test_total_failed_scales_with_rows(self):
        assert make_estimator().estimate()["total_failed"] == 0

    def test_total_failed_with_partial_errors(self):
        # 1/2 failed → 2/4 estimated
        e = make_estimator(sample_stats=make_stats(total_succeeded=1, total_failed=1))
        assert e.estimate()["total_failed"] == 2
        assert e.estimate()["total_succeeded"] == 2

    def test_total_failed_all_errors(self):
        e = make_estimator(sample_stats=make_stats(total_succeeded=0, total_failed=2))
        assert e.estimate()["total_failed"] == 4
        assert e.estimate()["total_succeeded"] == 0

    def test_duration_sequential(self):
        # mean row duration = 0.5s, 4 rows, 1 worker → 4 * 0.5 = 2.0s
        assert make_estimator().estimate(max_workers=1)["total_duration"] == pytest.approx(2.0)

    def test_duration_scales_with_workers(self):
        # mean row duration = 0.5s, 4 rows, 2 workers → ceil(4/2) * 0.5 = 1.0s
        assert make_estimator().estimate(max_workers=2)["total_duration"] == pytest.approx(1.0)

    def test_duration_more_workers_than_rows(self):
        # mean row duration = 0.5s, 4 rows, 10 workers → ceil(4/10) * 0.5 = 0.5s
        assert make_estimator().estimate(max_workers=10)["total_duration"] == pytest.approx(0.5)

    def test_cost_scales_by_prompt_length(self):
        # sample: 200 chars total → 0.02 cost; full: 400 chars total → 0.04 cost
        assert make_estimator().estimate()["total_cost"] == pytest.approx(0.04)

    def test_cost_uneven_prompt_lengths(self):
        # sample: 50+150=200 chars → 0.02; full: 50+150+200+400=800 chars → 0.08
        e = make_estimator(
            sample_prompt_lengths=[50, 150],
            full_prompt_lengths=[50, 150, 200, 400],
        )
        assert e.estimate()["total_cost"] == pytest.approx(0.08)

    def test_cost_none_when_sample_cost_is_none(self):
        e = make_estimator(sample_stats=make_stats(total_cost=None))
        assert e.estimate()["total_cost"] is None

    def test_cost_none_when_sample_lengths_are_zero(self):
        e = make_estimator(sample_prompt_lengths=[0, 0])
        assert e.estimate()["total_cost"] is None

    def test_input_tokens_scales_by_prompt_length(self):
        # 100 tokens in 200 chars → 200 tokens in 400 chars
        assert make_estimator().estimate()["total_input_tokens"] == 200

    def test_output_tokens_scales_by_prompt_length(self):
        # 50 tokens in 200 chars → 100 tokens in 400 chars
        assert make_estimator().estimate()["total_output_tokens"] == 100

    def test_input_tokens_none_when_sample_is_none(self):
        e = make_estimator(sample_stats=make_stats(total_input_tokens=None))
        assert e.estimate()["total_input_tokens"] is None

    def test_output_tokens_none_when_sample_is_none(self):
        e = make_estimator(sample_stats=make_stats(total_output_tokens=None))
        assert e.estimate()["total_output_tokens"] is None

    def test_empty_durations_returns_zero_duration(self):
        e = make_estimator(sample_durations=[])
        assert e.estimate()["total_duration"] == 0.0
