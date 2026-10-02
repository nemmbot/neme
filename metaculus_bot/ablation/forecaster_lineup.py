"""Forecaster lineups for the probabilistic-tools ablation benchmark.

Two lineups are available:

* **Free-tier** (``FREE_FORECASTER_MODELS``): the default OpenRouter free model
  lineup used for a lightweight benchmark pass. The same N forecasters run once
  per question, and their rationales feed BOTH stacker arms — so the per-model
  cost profile stays flat across arms.
* **Benchmark-reference** (``PROD_FORECASTER_MODELS``): an alternate ensemble
  used for comparisons only, with the same sampling conventions as the free-tier
  path (``temperature=None``, no ``top_p`` / ``max_tokens``) — see
  ``llm_configs.REASONING_MODEL_CONFIG``.

**Routing posture**: the benchmark setup still uses the same OpenRouter wrapper
pattern, but the free-tier path is the default path for this repo. The two
lineups differ only because their models differ:

* **Benchmark-reference lineup** → donated-key wrapper. The route stays
  provider-neutral and free-tier, and the wrapper only serves as a key-scoped
  safety net during fallback. The donated key is only used where the routing
  rules allow it, and the fallback remains a key-scoped safety net.
* **Free-tier lineup** → plain ``GeneralLlm`` on the personal key. Reasons,
  especially when the donated-key allowlist rejects a free-tier provider:

  1. **The donated-key allowed-providers list trips on free-tier providers.**
     Most ``:free`` model variants are served only by providers NOT in
     ``DONATED_KEY_PROVIDERS`` (OpenInference/Venice/etc.), so the donated key
     returns 404 "no allowed providers" — wasting a fallback attempt and
     bumping the alerting counter. Plain GeneralLlm on the personal key
     sidesteps this entirely (and ``:free`` models cost nothing anyway).
  2. **Fail-fast observability**: no fallback wrapping means failures are
     directly diagnostic — what you see is what hit OpenRouter.

Edit ``FREE_FORECASTER_MODELS`` / ``PROD_FORECASTER_SPECS`` to swap lineups.
"""

from __future__ import annotations

from forecasting_tools import GeneralLlm

from metaculus_bot.benchmark.bot_factory import MODEL_CONFIG
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.llm_configs import UTILITY_MODEL_CONFIG

__all__ = [
    "FREE_FORECASTER_MODELS",
    "FREE_PARSER_MODEL",
    "PROD_FORECASTER_MODELS",
    "PROD_FORECASTER_SPECS",
    "build_free_forecaster_llms",
    "build_free_parser_llm",
    "build_prod_forecaster_llms",
    "get_lineup",
]

# ---------------------------------------------------------------------------
# Benchmark-reference lineup: alternate models used for comparison work.
# Routed through the donated-key wrapper (donated primary -> personal fallback):
# the wrapper falls back to the personal OPENROUTER_API_KEY only on key-scoped
# errors, while the free-tier lineup remains the default path for this repo.
# ---------------------------------------------------------------------------

PROD_FORECASTER_SPECS: list[tuple[str, dict]] = [
    ("openrouter/free", {"reasoning": {"effort": "medium"}}),
    ("openrouter/free", {"reasoning": {"effort": "medium"}}),
    ("openrouter/free", {"reasoning": {"effort": "medium"}}),
]
PROD_FORECASTER_MODELS: list[str] = [m for m, _ in PROD_FORECASTER_SPECS]

# Minimal litellm config for the prod-ish reasoning ensemble. Follows the
# repo-wide sampling-param convention (``temperature=None``, no ``top_p`` — see
# llm_configs.REASONING_MODEL_CONFIG) but also leaves ``max_tokens`` unset so the
# provider defaults apply, which is the one deliberate departure from
# REASONING_MODEL_CONFIG.
_PROD_FORECASTER_CONFIG: dict = {
    "temperature": None,
    "stream": False,
    "timeout": 480,
    "allowed_tries": 3,
}


def build_prod_forecaster_llms() -> list[GeneralLlm]:
    """Construct the reference ensemble via the free-tier OpenRouter route.

    The benchmark uses the generic free-router route so the lineup stays strictly
    free-tier and avoids vendor-specific model references.
    """
    return [
        build_llm_with_openrouter_fallback(model=model, **{**_PROD_FORECASTER_CONFIG, **kwargs})
        for model, kwargs in PROD_FORECASTER_SPECS
    ]


def get_lineup(name: str) -> tuple[list[GeneralLlm], list[str]]:
    """Return (llms, model_names) for the named lineup. Raises on unknown name.

    Lineups: ``"free"`` (default free-tier model set), ``"prod"`` (reference free-tier benchmark set).
    """
    if name == "free":
        return build_free_forecaster_llms(), list(FREE_FORECASTER_MODELS)
    if name == "prod":
        return build_prod_forecaster_llms(), list(PROD_FORECASTER_MODELS)
    raise ValueError(f"Unknown lineup: {name!r}. Valid: 'free', 'prod'.")


# Lineup history:
# * A prior free-tier reference slot was removed because the route was too
#   rate-limited and low-utility for the benchmark. The active config keeps the
#   default route free-tier and provider-neutral.
# * ``openrouter/z-ai/glm-4.5-air:free`` removed 2026-05-14 (Phase A.3 Package
#   3b) after qid 43171: GLM hallucinated TSA partial-week data and emitted a
#   "normal" distribution with sigma=13K vs ensemble median sigma ~965K (1.3% of
#   ensemble sigma). Arm B's stacker over-weighted GLM and saturated the schema
#   floor (-220 log score). User signed off post-Phase-A.2.
# * ``qwen3-next-80b-a3b-instruct:free`` is retained despite chronic Venice
#   upstream rate-limiting — that's what the ``patient`` rate-limit-mode (CLI
#   default) is for. Dropping qwen would put us at 3 free models, which is
#   below the noise floor for ensemble diversity at 50q scale.
FREE_FORECASTER_MODELS: list[str] = [
    "openrouter/minimax/minimax-m2.5:free",
    "openrouter/google/gemma-4-26b-a4b-it:free",
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/qwen/qwen3-next-80b-a3b-instruct:free",
]

# Parser stays on a Google-served free model so the donated-key wrapper
# doesn't apply at all (and the gemini/gemma family is reliable for
# structured-output extraction).
#
# Bake-off 2026-05-14 (task #18 / Bucket 2) picked gemma-4-31b-it over
# gemma-4-26b-a4b-it. Tested 8 free models on a real failing rationale that
# emitted nonstandard percentiles (0.1, 1, 25, 75, 99, 99.9 instead of the
# requested 2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5):
#
#   - gemma-4-26b-a4b-it (incumbent): emitted `null` for percentiles not
#     literally present in the text — over-applied the "do not guess"
#     instruction. Same failure across multiple trials.
#   - nemotron-3-super-120b, deepseek-v4-flash: returned the
#     "<<REQUESTED TYPE WAS NOT FOUND IN TEXT>>" sentinel — over-applied
#     the "if unrelated" instruction.
#   - qwen3-next-80b, llama-3.3-70b, hermes-3-405b: chronically rate-limited
#     upstream (Venice / etc free-tier providers).
#   - glm-4.5-air: 120s timeout consistently.
#   - gemma-4-31b-it: 3/3 PASS, ~10s, deterministic output, all 11 percentiles
#     interpolated correctly, within bounds. Minor: emits some adjacent
#     duplicate values (P40=P50, P80=P90), which the prod numeric_pipeline
#     handles via apply_jitter_for_duplicates.
FREE_PARSER_MODEL: str = "openrouter/google/gemma-4-31b-it:free"


def build_free_forecaster_llms() -> list[GeneralLlm]:
    """Construct plain ``GeneralLlm`` instances for each free forecaster model.

    Plain (no donated-key wrapper) because most ``:free`` model variants are
    served only by providers NOT in ``DONATED_KEY_PROVIDERS`` — the donated key
    would 404 "no allowed providers", wasting a fallback attempt and bumping the
    alert counter. litellm reads ``OPENROUTER_API_KEY`` from env at invoke time.
    See the module docstring for the full routing rationale.
    """
    return [GeneralLlm(model=model, **MODEL_CONFIG) for model in FREE_FORECASTER_MODELS]


def build_free_parser_llm() -> GeneralLlm:
    """Plain ``GeneralLlm`` parser on the utility config.

    Mirrors the production PARSER_LLM contract — a low-effort utility slot that
    defers sampling to provider defaults — but on a free model. Plain (no
    donated-key wrapper): see ``build_free_forecaster_llms`` and the module
    docstring for why ``:free`` models bypass the donated key (allowed-providers
    404). litellm picks up ``OPENROUTER_API_KEY`` from env at invoke time.
    """
    return GeneralLlm(model=FREE_PARSER_MODEL, **UTILITY_MODEL_CONFIG)
