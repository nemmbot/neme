"""Centralised model configuration for TemplateForecaster.

Keeping these objects in a single module avoids merge-conflicts and makes it
possible to tweak/benchmark models without touching application code.
"""

from typing import Any

from forecasting_tools import GeneralLlm

from metaculus_bot.constants import OPENROUTER_FREE_MODEL
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback

__all__ = [
    "DISAGREEMENT_ANALYZER_LLM",
    "FORECASTER_LLMS",
    "FORECASTER_MODEL_NAMES",
    "MANTIC_FORECASTER_LLMS",
    "MANTIC_FORECASTER_MODELS",
    "MARKET_QUERY_AUTHOR_LLM_CONFIG",
    "MARKET_RANKER_LLM_CONFIG",
    "OPENROUTER_FREE_MODEL_FALLBACKS",
    "PARSER_LLM",
    "RESEARCHER_LLM",
    "STACKER_FALLBACK_LLM",
    "STACKER_LLM",
    "SUMMARIZER_LLM",
]
# Reasoning models ignore (or degrade under) explicit sampling params, so we
# defer to provider defaults. temperature=None is explicit but redundant on
# ft 0.2.92, whose GeneralLlm ctor already defaults temperature to None (0.2.54
# injected 0 when the arg was omitted); top_p flows via **kwargs and is never set.
REASONING_MODEL_CONFIG: dict[str, Any] = {
    "temperature": None,
    "max_tokens": 64_000,  # Prevent truncation; all current forecasters/stackers support 64k output
    "stream": False,
    "timeout": 480,
    "allowed_tries": 3,
}
# Low-effort utility slots (parser, summarizer, analyzer). Same sampling-param
# rationale as REASONING_MODEL_CONFIG: temperature=None defers to provider
# defaults (redundant on ft 0.2.92, whose ctor default is already None); top_p
# left unset.
UTILITY_MODEL_CONFIG: dict[str, Any] = {
    "temperature": None,
    "max_tokens": 32_000,
    "stream": False,
    "timeout": 300,
    "allowed_tries": 3,
}
ACCEPTABLE_QUANTS = [
    "fp8",
    "fp16",
    "bf16",
    "fp32",
    "unknown",
]

# Per-instance allowed_tries=1 override (Round-2): forecaster .invoke is wrapped
# in the broad retry gated on TRANSIENT_RETRY_MAX_ELAPSED_S (forecaster_runners.py)
# so we can impose the universal "never retry a slow failure" deadline-safety rule
# that forecasting-tools' un-gated tenacity cannot. Spread per-instance (NOT by mutating
# REASONING_MODEL_CONFIG) so PARSER_LLM / STACKER configs are untouched.
_FORECASTER_CONFIG = {**REASONING_MODEL_CONFIG, "allowed_tries": 1}
_MANTIC_FORECASTER_CONFIG = {**_FORECASTER_CONFIG, "max_tokens": 32_768}

OPENROUTER_FREE_MODEL_FALLBACKS: tuple[str, ...] = (
    "nvidia/nemotron-3-super-120b-a12b:free",
    "thinkingmachines/inkling:free",
)


def forecaster_role(model: str) -> str:
    """``forecaster:<vendor>`` for an ``openrouter/<vendor>/<model>`` roster slug.

    The CREDIT_ROLE_SPEND spend line every roster slot books under. The roster is
    latest-per-vendor, one slot each, so the VENDOR is the stable identity of a slot
    across model rotations — a per-model role would start a new time series at every swap
    and defeat the era-over-era cost comparison this exists for.
    """
    if model == OPENROUTER_FREE_MODEL:
        return "forecaster:free"
    parts = model.split("/")
    if len(parts) < 3 or parts[0] != "openrouter":
        raise ValueError(f"forecaster_role expects an openrouter/<vendor>/<model> slug, got {model!r}")
    return f"forecaster:{parts[1]}"


def _forecaster_slot(model: str, **kwargs: Any) -> GeneralLlm:
    """One roster member, booked in the CREDIT_ROLE_SPEND ledger under ``forecaster:<vendor>``.

    The role is derived from the slug rather than written beside it so a roster swap cannot
    leave a slot mislabeled.
    """
    return build_llm_with_openrouter_fallback(
        model=model,
        role=forecaster_role(model),
        extra_body={"models": list(OPENROUTER_FREE_MODEL_FALLBACKS)},
        **_FORECASTER_CONFIG,
        **kwargs,
    )


MANTIC_FORECASTER_MODELS: tuple[str, ...] = (
    "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/qwen/qwen3.8-27b:free",
)
MANTIC_FORECASTER_LLMS: list[GeneralLlm] = [
    build_llm_with_openrouter_fallback(
        model=model,
        role=forecaster_role(model),
        extra_body={"models": list(OPENROUTER_FREE_MODEL_FALLBACKS)},
        **_MANTIC_FORECASTER_CONFIG,
    )
    for model in MANTIC_FORECASTER_MODELS
]


# Free-tier roster policy: keep the live model route pinned to the generic
# OpenRouter free model and avoid vendor-specific assumptions in the active
# configuration. The roster is intentionally provider-agnostic so the same
# prompt and fallback logic stays stable regardless of upstream model churn.
FORECASTER_LLMS: list[GeneralLlm] = [
    # The free-tier path stays provider-agnostic by design. The reasoning config
    # keeps the same effort profile across all slots, and any upstream provider
    # change is handled in the shared free-model alias instead of in the per-slot
    # configuration.
    _forecaster_slot(
        OPENROUTER_FREE_MODEL,
        reasoning={"effort": "xhigh"},
    ),
    # Secondary free-tier slot. The configuration avoids vendor-specific effort
    # tuning and keeps the same xhigh safety profile across the roster.
    _forecaster_slot(
        OPENROUTER_FREE_MODEL,
        reasoning={"effort": "xhigh"},
    ),
    # Final free-tier slot. Keeps the same generic free-model route and avoids
    # any vendor-specific tuning or assumptions.
    _forecaster_slot(OPENROUTER_FREE_MODEL),
]


def _forecaster_display_name(llm: GeneralLlm) -> str:
    """Short label for a forecaster model slug.

    Used by performance_analysis.parsing to map 'Forecaster N' labels in bot comments
    back to a model name without having to hand-maintain a parallel list.
    """
    return llm.model.rsplit("/", 1)[-1]


FORECASTER_MODEL_NAMES: list[str] = [_forecaster_display_name(llm) for llm in FORECASTER_LLMS]

# Summarizer: compresses raw AskNews article markdown into an analyst briefing
# (AskNews-only; all other providers already emit LLM prose). sol → terra
# 2026-07-18 operator decision: AskNews is an auxiliary/augmenting source
# (content audit: 16% unique content vs native-search 54% / gap-fill 59%), so
# the absolute-frontier tier isn't warranted. The role audit
# (scratch/research_role_audit_2026-07-17/) had sol 1st but verdict "MARGINAL
# EDGE" with terra 2nd (one attribution blur, no fabrications), and 4/5 briefing
# failures in the AskNews quality audit (scratch/asknews_quality_audit_2026-07-18/)
# were prompt-era (mini summarizer + missing no-forecast rule), not model-tier.
# Terra: -43% cost, ~50s vs ~118s wall. Effort stays low (latency).
# allowed_tries=1 (Round-2): the summarizer invoke is wrapped in the broad,
# elapsed-gated retry (orchestrator._summarize_asknews) to impose the universal
# "never retry a slow failure" deadline rule. Per-instance override so PARSER_LLM (which
# also uses UTILITY_MODEL_CONFIG) keeps its allowed_tries=3.
# Summarizer role is intentionally kept on the generic free-tier route and avoids
# provider-specific assumptions in the active config.
SUMMARIZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    OPENROUTER_FREE_MODEL,
    role="summarizer",
    reasoning={"effort": "low"},
    **{**UTILITY_MODEL_CONFIG, "allowed_tries": 1},
)
# Parser: deterministic extraction of percentiles/JSON from rationales — a
# capability-saturated task, so it rides the cheapest tier that saturates it and
# keeps allowed_tries=3 for robustness. mini → luna 2026-08-03: the per-token
# comparison that used to favor mini inverted. Luna was $0.20/$1.20 vs mini's
# $0.75/$4.50 per 1M, so the newer model was also the ~3.75x cheaper one. (The
# models API showed $0.10/$0.60 behind a "50% off" badge on 2026-08-03; a live
# call on 2026-08-04 billed at double that, so the promo does not apply on this
# route — see the ranker cost comment below. The swap still won, by less.)
# Free-tier parser stays on the generic route with the low-effort profile.
PARSER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    OPENROUTER_FREE_MODEL,
    role="parser",
    reasoning={"effort": "low"},
    **UTILITY_MODEL_CONFIG,
)
# Researcher slot in the forecasting-tools LLM config dict. Effectively dead
# code in our pipeline — we use research providers (AskNews/Gemini/native_search)
# rather than the framework's researcher path — but the slot must be populated
# to avoid silent framework defaults. Aliasing to SUMMARIZER_LLM rather than
# constructing a duplicate config: same model, same effort, same job tier, no
# reason to maintain two parallel definitions.
RESEARCHER_LLM = SUMMARIZER_LLM

# Stacker meta-model for conditional stacking (invoked only on high-disagreement questions).
#
# allowed_tries=1: a single attempt at REASONING_MODEL_CONFIG's timeout, no
# retries. The outer STACKER_SOFT_DEADLINE catches wholly stuck calls; on failure
# we fall back to STACKER_FALLBACK_LLM instead of burning another full timeout on
# the same route.
STACKER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    # Free-tier stacker keeps the same generic route and xhigh effort profile.
    OPENROUTER_FREE_MODEL,
    role="stacker",
    reasoning={"effort": "xhigh"},
    **{**REASONING_MODEL_CONFIG, "allowed_tries": 1},
)

# Fallback stacker used when the primary stacker times out or errors.
# The active path stays on the generic free-tier route to avoid vendor-specific
# behavior in the critical path. Tighter timeout and single try since we're
# already running late on the critical path by the time this fires.
STACKER_FALLBACK_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    OPENROUTER_FREE_MODEL,
    role="stacker_fallback",
    reasoning={"effort": "xhigh"},
    **{**REASONING_MODEL_CONFIG, "allowed_tries": 1, "timeout": 300},
)

# --- The prediction-market provider's two LLM stages ---
#
# Both are RAW DICTS rather than built GeneralLlm singletons, unlike PARSER_LLM and friends:
# the provider is gated OFF by default, so paying construction cost at import would be waste,
# and the tests patch `build_llm_with_openrouter_fallback` at the provider's one invocation
# helper. The active route remains free-tier and provider-neutral.
#
# `allowed_tries=1` is required, not decorative: the repo's elapsed-gated `llm_retry` wrapper
# (prediction_market._invoke_market_llm) is the SOLE retry layer, and leaving this unpinned
# inherits forecasting-tools' default of 2 with an UN-GATED `random.uniform(5, 10)` tenacity
# sleep — a large slice of PREDICTION_MARKET_TIMEOUT spent sleeping blind, which is exactly
# what llm_retry exists to eliminate. `temperature=None` defers reasoning models to provider
# defaults (redundant on ft 0.2.92, whose ctor default is already None); top_p left unset. Each
# litellm `timeout` sits ABOVE its elapsed-gated wall cap in constants.py, so the wall is the
# binding bound.
#
# Luna is the cheapest tier that saturates both tasks. The measured rate on this route was
# $0.20/M in and $1.20/M out — TWICE the $0.10/$0.60 the bake-off read off the models API on
# 2026-08-03, where a "50% off" badge was displayed that has since lapsed or never applied here. A
# live ranking call reconciled the true rates to 7 significant figures against OpenRouter's own
# `upstream_inference_cost` (26,250 in / 685 out / a 25% cache-WRITE surcharge on the input,
# `scratch/market_port_2026-08-04/QA_DRY_RUN.md`), so this is measured rather than quoted.
# Free-tier benchmark cost notes stay generic and do not assume a specific
# upstream provider; the figures below are receipts for the active route.
#
# MEASURED cost per question: ranker $0.0074 (26k in at the median post-enrichment,
# full-PredictIt shape + ~685 out, cache write included); author ~1.4k in + ~300 out ≈ $0.0005.
# The two keyword calls they replace measured ~170 tok in / ~50 out ≈ $0.0001, so net new is
# ≈ +$0.008 per question — under a cent per run at the prod shape of 1-2 questions, and ~$0.24
# of ranker spend across a 30-question tournament run. The earlier ~$0.003-0.004 arithmetic in
# the port plan understated by 2.4x purely because of the promo price; the token shapes were
# right. This traffic is ~97% input, so the input rate is the whole cost.

# Prediction-market RANKER: one call per question over the whole ~380-440-candidate pool,
# emitting up to 8 ranked rows with a relation tier and a one-phrase label. Measured completion
# averages 589 tokens including reasoning, max 1,042 (scratch/bakeoff_run_2026-08-03/results/
# RANKED_ARM_RESULTS.md). No max_tokens since 2026-09-22 (operator): a TRUNCATED ranking is a
# fail-open that loses the whole ranking, and MARKET_RANKER_WALL_TIMEOUT already bounds a runaway.
MARKET_RANKER_LLM_CONFIG: dict = {
    "model": OPENROUTER_FREE_MODEL,
    "role": "market_ranker",
    "temperature": None,
    "reasoning_effort": "low",
    "timeout": 90,
    "allowed_tries": 1,
}

# Prediction-market QUERY AUTHOR: one call per question emitting the domain vocabulary the
# question's own tokens cannot reach (up to 8 synonyms + 3 framings). Its output is ADDITIVE to
# a deterministic query set, so its failure costs recall nothing. Measured completion max 588
# tokens including reasoning. No max_tokens since 2026-09-22: MARKET_QUERY_AUTHOR_WALL_TIMEOUT bounds it.
MARKET_QUERY_AUTHOR_LLM_CONFIG: dict = {
    "model": OPENROUTER_FREE_MODEL,
    "role": "market_query_author",
    "temperature": None,
    "reasoning_effort": "low",
    "timeout": 45,
    "allowed_tries": 1,
}


# Tier-B auxiliary: read-and-synthesize work that needs taste but not deep
# reasoning. Identifies the crux of forecaster disagreement; output text seeds
# the targeted-search query downstream. Runs under CRUX_SOFT_DEADLINE;
# effort deliberately low since 2026-05-20 for latency — the tier was upgraded
# instead (smarter-model-at-lower-effort beats more effort on a smaller model).
# 2026-07-17: sol→terra per the role audit; terra 2nd (sol 3rd) at -49% cost;
# the role fires rarely (stacking disabled in prod).
# allowed_tries=1 (Round-2): the crux-analyzer invoke is wrapped in the broad,
# elapsed-gated retry (targeted.extract_disagreement_crux) to impose the universal
# "never retry a slow failure" deadline rule on the conditional-stacking critical path.
# Per-instance override so PARSER_LLM keeps its allowed_tries=3.
# The multi-role free-tier route stays generic and provider-neutral; updates are
# tracked in the shared free-model config rather than vendor-specific names.
DISAGREEMENT_ANALYZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    OPENROUTER_FREE_MODEL,
    role="crux_analyzer",
    reasoning={"effort": "low"},
    **{**UTILITY_MODEL_CONFIG, "allowed_tries": 1},
)
