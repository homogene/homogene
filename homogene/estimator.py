"""Extrapolates run stats from a sample to a full dataset."""

from __future__ import annotations

import math


class StatsEstimator:
    """Extrapolates run stats from a sample to a full dataset.

    Uses prompt length in characters as the scaling variable for cost and token estimates,
    and mean per-row duration for wall-clock time estimation.

    Args:
        sample_stats: Stats dict returned by `Processor.stats` after a partial run.
        sample_durations: Per-row wall-clock durations from the sample run, in seconds.
        sample_prompt_lengths: Prompt length in characters for each sampled row.
        full_prompt_lengths: Prompt length in characters for every row in the full dataset.
    """

    def __init__(
        self,
        sample_stats: dict,
        sample_durations: list[float],
        sample_prompt_lengths: list[int],
        full_prompt_lengths: list[int],
    ) -> None:
        self._stats = sample_stats
        self._sample_durations = sample_durations
        self._sample_prompt_lengths = sample_prompt_lengths
        self._full_prompt_lengths = full_prompt_lengths

    def _scale_by_length(self, value: float | None) -> float | None:
        sample_total = sum(self._sample_prompt_lengths)
        if value is None or sample_total == 0:
            return None
        return value * sum(self._full_prompt_lengths) / sample_total

    def estimate(self, max_workers: int = 1) -> dict:
        """Return estimated stats for the full dataset.

        Args:
            max_workers: Number of parallel workers to assume for duration estimation.

        Returns:
            Dict with the same keys as `Processor.stats`, estimated for the full dataset.
        """
        n_sample = self._stats["total_succeeded"] + self._stats["total_failed"]
        n_full = len(self._full_prompt_lengths)
        error_rate = self._stats["total_failed"] / n_sample if n_sample > 0 else 0.0
        mean_duration = sum(self._sample_durations) / len(self._sample_durations) if self._sample_durations else 0.0

        scaled_cost = self._scale_by_length(self._stats["total_cost"])
        scaled_input = self._scale_by_length(self._stats["total_input_tokens"])
        scaled_output = self._scale_by_length(self._stats["total_output_tokens"])

        return {
            "total_succeeded": round(n_full * (1 - error_rate)),
            "total_failed": round(n_full * error_rate),
            "total_duration": mean_duration * math.ceil(n_full / max_workers),
            "total_cost": scaled_cost,
            "total_input_tokens": round(scaled_input) if scaled_input is not None else None,
            "total_output_tokens": round(scaled_output) if scaled_output is not None else None,
        }
