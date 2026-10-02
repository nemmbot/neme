"""Comment formatting shared by TemplateForecaster's section overrides."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from forecasting_tools import MetaculusQuestion, ReasonedPrediction

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.comment.markers import (
    STACKED_MARKER_FALSE,
    STACKED_MARKER_TRUE,
    STACKER_OUTCOME_FALLBACK_LLM,
    STACKER_OUTCOME_FALLBACK_MEAN,
    STACKER_OUTCOME_FALLBACK_MEDIAN,
    STACKER_OUTCOME_PRIMARY,
    STACKER_OUTCOME_SKIPPED,
    STACKER_OUTCOME_SKIPPED_CONFIG_OFF,
    TOOLS_USED_MARKER_FALSE,
    TOOLS_USED_MARKER_TRUE,
    format_forecasters_used_marker,
    format_stacker_skip_reason_marker,
)
from metaculus_bot.comment.trimming import trim_comment, trim_section
from metaculus_bot.performance_analysis.parsing import (
    annotate_forecaster_bullets_with_models,
    extract_model_display_name_from_reasoning,
)
from metaculus_bot.question_types import question_type_of
from metaculus_bot.tool_runner import _feature_enabled as _tool_runner_feature_enabled

logger = logging.getLogger(__name__)


def format_research_summary_with_models(
    base_text: str,
    predictions: Sequence[ReasonedPrediction],
    report_number: int,
) -> str:
    """Inject model display names into summary bullets, then trim to section limit."""
    model_names_by_index: dict[int, str] = {}
    for forecaster_number, forecast in enumerate(predictions, start=1):
        model_name = extract_model_display_name_from_reasoning(forecast.reasoning)
        if model_name is not None:
            model_names_by_index[forecaster_number] = model_name
    text = annotate_forecaster_bullets_with_models(base_text, model_names_by_index)
    return trim_section(text, f"report_{report_number}_summary")


def format_main_research_section(base_text: str, report_number: int) -> str:
    """Trim the main research section to the configured section limit."""
    return trim_section(base_text, f"report_{report_number}_research")


def format_forecast_metadata_summary(
    question: MetaculusQuestion,
    prediction: object,
    research_text: str,
    *,
    n_used: int | None,
) -> str:
    """Render three compact comment points for the forecast, ensemble, and search provenance."""
    question_text = " ".join(question.question_text.split())
    if len(question_text) > 240:
        question_text = f"{question_text[:237]}..."

    if isinstance(prediction, (float, int)):
        value_text = f"P(Yes)={float(prediction):.1%}"
    else:
        predicted_options = getattr(prediction, "predicted_options", None)
        if isinstance(predicted_options, list) and predicted_options:
            ranked = sorted(predicted_options, key=lambda option: float(option.probability), reverse=True)[:3]
            value_text = ", ".join(f"{option.option_name}={option.probability:.1%}" for option in ranked)
        else:
            percentiles = getattr(prediction, "declared_percentiles", None)
            median = percentiles.get(0.5) if isinstance(percentiles, dict) else None
            value_text = f"median={median}" if median is not None else str(prediction)
    if len(value_text) > 200:
        value_text = f"{value_text[:197]}..."

    search_sources = list(dict.fromkeys(re.findall(r"(?m)^Search source: ([^\r\n]+)", research_text)))
    source_urls = set(re.findall(r"(?m)^URL: (https?://\S+)", research_text))
    sources_text = ", ".join(search_sources) if search_sources else "no web search returned usable results"
    search_line = f"{sources_text}; {len(source_urls)} cited URL(s) in Research."
    ensemble_count = str(n_used) if n_used is not None else "the available"

    return "\n".join(
        (
            f"- Forecast: {question_text} | {value_text}.",
            f"- Basis: combined {ensemble_count} surviving model forecast(s); per-model estimates and rationales follow.",
            f"- Search used: {search_line}",
        )
    )


def format_forecaster_rationales_section(base_text: str, report_number: int) -> str:
    """Trim the forecaster rationales section to the configured section limit."""
    return trim_section(base_text, f"report_{report_number}_rationales")


def _forecasters_used_suffix(n_used: int | None, n_configured: int | None) -> str:
    """The FORECASTERS_USED marker line (with a leading newline) when both counts
    are known, else ``""``.

    Emitted on every comment whose caller knows the ensemble size (the production
    ``_create_unified_explanation`` always does), so a comment with fewer than N
    bullets is self-describing — a dropped model is distinguishable from a roster
    change. Absent (rather than a fake ``0/0``) when a caller doesn't supply the
    counts, matching how pre-marker comments read as "unknown".
    """
    if n_used is None or n_configured is None:
        return ""
    return f"\n{format_forecasters_used_marker(n_used, n_configured)}"


def _stacker_outcome_markers(stacker_outcome: str) -> tuple[str, str]:
    """The (STACKER_OUTCOME, legacy STACKED) marker pair for one outcome.

    Raises on an unknown outcome rather than defaulting: a new outcome that silently
    published as ``STACKED: false`` would misreport whether the stacker ran.
    """
    match stacker_outcome:
        case "primary":
            return STACKER_OUTCOME_PRIMARY, STACKED_MARKER_TRUE
        case "fallback_llm":
            return STACKER_OUTCOME_FALLBACK_LLM, STACKED_MARKER_TRUE
        case "fallback_median":
            return STACKER_OUTCOME_FALLBACK_MEDIAN, STACKED_MARKER_FALSE
        case "fallback_mean":
            return STACKER_OUTCOME_FALLBACK_MEAN, STACKED_MARKER_FALSE
        case "skipped":
            return STACKER_OUTCOME_SKIPPED, STACKED_MARKER_FALSE
        case "skipped_config_off":
            return STACKER_OUTCOME_SKIPPED_CONFIG_OFF, STACKED_MARKER_FALSE
        case other:
            raise ValueError(f"Unknown stacker outcome {other!r}")


def build_unified_explanation(
    base_text: str,
    question: MetaculusQuestion,
    aggregation_strategy: AggregationStrategy,
    stacker_outcome: str | None,
    *,
    skip_reason: str | None = None,
    n_used: int | None = None,
    n_configured: int | None = None,
    forecast_summary: str | None = None,
) -> str:
    """Build the final Metaculus comment with stacker/tools/ensemble markers appended.

    For non-stacking strategies, trims and returns (plus the ensemble marker). For
    STACKING / CONDITIONAL_STACKING, also appends STACKER_OUTCOME, legacy STACKED,
    and TOOLS_USED markers. ``skip_reason`` is additive: the skip paths in
    stacking_route supply it, and a STACKER_SKIP_REASON marker then rides directly
    under STACKER_OUTCOME so a plain ``skipped`` no longer conflates
    spread-below-threshold with the single-forecaster short-circuit; when ``None``
    (every non-skip outcome, and comments published before the field) the comment
    is unchanged. ``n_used`` / ``n_configured`` (contributed / configured
    forecasters) are keyword-only and additive: when both are supplied a
    FORECASTERS_USED marker rides the comment tail; when omitted the comment is
    unchanged (back-compat with callers that don't track ensemble size).
    """
    if forecast_summary:
        base_text = base_text.replace("_Full research in the RESEARCH section below._", forecast_summary, 1)
    ensemble_suffix = _forecasters_used_suffix(n_used, n_configured)
    if aggregation_strategy not in (AggregationStrategy.STACKING, AggregationStrategy.CONDITIONAL_STACKING):
        return trim_comment(f"{base_text}{ensemble_suffix}")

    assert stacker_outcome is not None, (
        "stacker_outcome must be provided for STACKING/CONDITIONAL_STACKING strategies; "
        "every reachable code path in _aggregate_predictions sets it. Missing entry = real bug."
    )

    outcome_marker, legacy_marker = _stacker_outcome_markers(stacker_outcome)
    qtype = question_type_of(question)

    skip_reason_suffix = "" if skip_reason is None else f"\n{format_stacker_skip_reason_marker(skip_reason)}"
    tools_marker = TOOLS_USED_MARKER_TRUE if _tool_runner_feature_enabled(qtype) else TOOLS_USED_MARKER_FALSE
    return trim_comment(
        f"{base_text}\n{outcome_marker}{skip_reason_suffix}\n{legacy_marker}\n{tools_marker}{ensemble_suffix}\n"
    )


__all__ = [
    "build_unified_explanation",
    "format_forecaster_rationales_section",
    "format_main_research_section",
    "format_research_summary_with_models",
]
