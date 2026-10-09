from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any


BRANCHES = ("episodic", "latent", "hybrid")
OBJECTIVE_TOLERANCE = 1e-10


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _unit_interval(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


@dataclass(frozen=True)
class RouterThresholds:
    """Shared thresholds; -1 is the calibration sentinel for always Hybrid."""

    tau_jsd_normalized: float = 0.1
    tau_confidence: float = 0.95

    def __post_init__(self) -> None:
        jsd = _finite_number(self.tau_jsd_normalized, "tau_jsd_normalized")
        if jsd != -1.0 and not 0.0 <= jsd <= 1.0:
            raise ValueError("tau_jsd_normalized must be in [0, 1] or equal -1")
        confidence = _unit_interval(self.tau_confidence, "tau_confidence")
        object.__setattr__(self, "tau_jsd_normalized", jsd)
        object.__setattr__(self, "tau_confidence", confidence)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def load_thresholds(path: Path | None = None) -> RouterThresholds:
    """Load a flat threshold object or the original shared_config wrapper."""
    if path is None:
        return RouterThresholds()
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Router configuration must be a JSON object")
    values = config.get("thresholds", config)
    if not isinstance(values, dict):
        raise ValueError("Router thresholds must be a JSON object")
    required = ("tau_jsd_normalized", "tau_confidence")
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Missing router thresholds: {', '.join(missing)}")
    return RouterThresholds(**{name: values[name] for name in required})


def route(
    jsd_normalized: float,
    confidence: float,
    thresholds: RouterThresholds = RouterThresholds(),
) -> str:
    """Choose Hybrid, Latent, or Episodic with strict > comparisons.

    Empty latent answers must be represented by confidence=0.0; they still
    select Hybrid when JSD exceeds its threshold. Equality falls through.
    """
    jsd = _unit_interval(jsd_normalized, "jsd_normalized")
    confidence = _unit_interval(confidence, "confidence")
    if jsd > thresholds.tau_jsd_normalized:
        return "hybrid"
    if confidence > thresholds.tau_confidence:
        return "latent"
    return "episodic"


def _validate_logits(torch: Any, logits: Any) -> None:
    if logits.ndim != 1 or logits.numel() < 2 or logits.device.type != "cpu":
        raise ValueError("Router logits must be one-dimensional CPU vocabulary vectors")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Router logits contain nonfinite values")


def first_token_jsd(torch: Any, left: Any, right: Any) -> float:
    """Return symmetric Jensen-Shannon divergence in natural-log units."""

    _validate_logits(torch, left)
    _validate_logits(torch, right)
    if left.shape != right.shape:
        raise ValueError("Latent/Base vocabulary shapes differ")
    log_left = torch.log_softmax(left.float(), dim=-1)
    log_right = torch.log_softmax(right.float(), dim=-1)
    log_mixture = torch.logaddexp(log_left, log_right) - math.log(2.0)
    value = float(0.5 * (
        (log_left.exp() * (log_left - log_mixture)).sum()
        + (log_right.exp() * (log_right - log_mixture)).sum()
    ))
    bound = math.log(2.0)
    if not math.isfinite(value) or value < -2e-6 or value > bound + 2e-6:
        raise ValueError(f"JSD is outside its mathematical bounds: {value}")
    return max(0.0, min(bound, value))


def prefix_confidence(
    torch: Any,
    logits: Sequence[Any],
    token_ids: Sequence[int],
    special_ids: Iterable[int],
    eos_ids: Iterable[int],
) -> dict[str, Any]:
    """Measure the first up to eight generated non-special/non-EOS tokens."""

    if len(logits) != len(token_ids):
        raise ValueError("Native generation hook count does not match generated token count")
    excluded = set(special_ids) | set(eos_ids)
    selected_logps = []
    for vector, token_id in zip(logits, token_ids):
        _validate_logits(torch, vector)
        if not isinstance(token_id, int) or isinstance(token_id, bool) or not 0 <= token_id < vector.numel():
            raise ValueError(f"Generated token ID is outside the vocabulary: {token_id}")
        if token_id in excluded:
            continue
        value = float(torch.log_softmax(vector.float(), dim=-1)[token_id])
        if not math.isfinite(value) or value > 0:
            raise ValueError(f"Generated token has invalid raw log probability: {value}")
        selected_logps.append(value)
        if len(selected_logps) == 8:
            break
    mean_logp = sum(selected_logps) / len(selected_logps) if selected_logps else None
    return {
        "confidence": math.exp(mean_logp) if mean_logp is not None else 0.0,
        "mean_logp8": mean_logp,
        "confidence_token_count": len(selected_logps),
    }


def _eos_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    return {int(value)}


def probe_router(
    model: Any,
    tokenizer: Any,
    generate: Callable[[int], list[int]],
    max_new_tokens: int,
    torch: Any,
) -> tuple[list[int], dict[str, Any]]:
    """Generate a native Latent answer and observe an identical-input Base probe.

    ``generate(limit)`` must always use the same encoded direct-question input,
    run native greedy generation, and return only its actual newly generated
    token IDs. The temporary hook observes model outputs and never replaces
    them. The Base intervention disables the loaded PEFT adapter for one token.
    """

    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    head = model.get_output_embeddings()
    if head is None or not callable(getattr(head, "register_forward_hook", None)):
        raise ValueError("Adaptive routing requires an observable output embedding module")
    if not callable(getattr(model, "disable_adapter", None)):
        raise ValueError("Adaptive routing requires a loaded adapter that can be disabled")
    captured = []

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] < 1:
            raise ValueError("Unexpected native output-embedding shape; require one sequence")
        # A CPU clone also keeps observations independent when the model runs
        # on CPU or reuses its output storage. Retain no GPU logits or graphs.
        captured.append(output[0, -1].detach().float().cpu().clone())

    model.eval().requires_grad_(False)
    handle = head.register_forward_hook(capture)
    try:
        with torch.no_grad():
            latent_ids = list(generate(max_new_tokens))
            latent_logits = captured
            if not latent_ids or len(latent_ids) > max_new_tokens or len(latent_logits) != len(latent_ids):
                raise ValueError("Latent native generation hook/token alignment failed")
            captured = []
            with model.disable_adapter():
                base_ids = list(generate(1))
            base_logits = captured
            if len(base_ids) != 1 or len(base_logits) != 1:
                raise ValueError("Base probe must produce exactly one token and raw distribution")
    finally:
        try:
            # Some PEFT versions re-enable parameter gradients when restoring
            # adapters, including when the protected generation raises.
            model.eval().requires_grad_(False)
        finally:
            handle.remove()

    divergence = first_token_jsd(torch, latent_logits[0], base_logits[0])
    eos_ids = _eos_ids(getattr(tokenizer, "eos_token_id", None))
    eos_ids.update(_eos_ids(getattr(getattr(model, "generation_config", None), "eos_token_id", None)))
    confidence = prefix_confidence(
        torch, latent_logits, latent_ids, getattr(tokenizer, "all_special_ids", ()), eos_ids,
    )
    return latent_ids, {
        "jsd_nats": divergence,
        "jsd_normalized": divergence / math.log(2.0),
        **confidence,
        "latent_generated_token_count": len(latent_ids),
    }


def _validated_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Calibration data must contain at least one row")
    validated = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"Calibration row {index} must be an object")
        try:
            jsd = _unit_interval(row["jsd_normalized"], "jsd_normalized")
            confidence = _unit_interval(row["confidence"], "confidence")
            raw_scores = row["scores"]
            if not isinstance(raw_scores, Mapping):
                raise ValueError("scores must be an object")
            scores = {
                branch: _unit_interval(raw_scores[branch], f"scores.{branch}")
                for branch in BRANCHES
            }
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid calibration row {index}: {exc}") from exc
        validated.append({"jsd_normalized": jsd, "confidence": confidence, "scores": scores})
    return validated


def _candidate_totals(
    rows: Sequence[Mapping[str, Any]],
    tau_jsd: float,
    confidence_grid: Sequence[float],
    fixed_routes: Sequence[str],
) -> tuple[list[float], list[int]]:
    """Sweep strict confidence boundaries instead of replaying every pair."""
    # All low-JSD examples initially choose Latent. Passing a confidence
    # boundary switches examples equal to that boundary to Episodic.
    deltas = {}
    change_deltas = {}
    initial_scores = []
    changed = 0
    for row, fixed in zip(rows, fixed_routes):
        if row["jsd_normalized"] > tau_jsd:
            initial_scores.append(row["scores"]["hybrid"])
            changed += fixed != "hybrid"
        else:
            initial_scores.append(row["scores"]["latent"])
            changed += fixed != "latent"
            confidence = row["confidence"]
            deltas.setdefault(confidence, []).append(
                row["scores"]["episodic"] - row["scores"]["latent"]
            )
            change_deltas[confidence] = change_deltas.get(confidence, 0) + (
                (fixed != "episodic") - (fixed != "latent")
            )
    total = math.fsum(initial_scores)
    totals, changes = [], []
    for threshold in confidence_grid:
        total += math.fsum(deltas.get(threshold, ()))
        changed += change_deltas.get(threshold, 0)
        totals.append(total)
        changes.append(changed)
    return totals, changes


def fit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Maximize mean calibration score on the original research grid.

    JSD candidates are observed values plus {-1, 0, .1, 1}; confidence
    candidates are observed values plus {0, .95, 1}. Objective ties use the
    original total-score tolerance, then minimize changed routes from .1/.95,
    L1 distance to .1/.95, JSD threshold, and confidence threshold, in order.
    """
    rows = _validated_rows(rows)
    defaults = RouterThresholds()
    jsd_grid = sorted({-1.0, 0.0, 0.1, 1.0, *(row["jsd_normalized"] for row in rows)})
    confidence_grid = sorted({0.0, 0.95, 1.0, *(row["confidence"] for row in rows)})
    fixed_routes = [route(row["jsd_normalized"], row["confidence"], defaults) for row in rows]
    best_key = None
    best_thresholds = None
    objective_max = -math.inf
    optimal_pairs = 0
    for tau_jsd in jsd_grid:
        totals, changes = _candidate_totals(rows, tau_jsd, confidence_grid, fixed_routes)
        max_here = max(totals)
        if max_here > objective_max + OBJECTIVE_TOLERANCE:
            objective_max, optimal_pairs, best_key = max_here, 0, None
        if max_here < objective_max - OBJECTIVE_TOLERANCE:
            continue
        for tau_confidence, total, changed in zip(confidence_grid, totals, changes):
            if abs(total - objective_max) >= OBJECTIVE_TOLERANCE:
                continue
            optimal_pairs += 1
            key = (
                changed,
                abs(tau_jsd - defaults.tau_jsd_normalized)
                + abs(tau_confidence - defaults.tau_confidence),
                tau_jsd,
                tau_confidence,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_thresholds = RouterThresholds(tau_jsd, tau_confidence)
    if best_thresholds is None:
        raise RuntimeError("No valid threshold candidate found")
    selected_routes = [route(row["jsd_normalized"], row["confidence"], best_thresholds) for row in rows]
    measured = math.fsum(row["scores"][chosen] for row, chosen in zip(rows, selected_routes))
    fixed_score = math.fsum(row["scores"][chosen] for row, chosen in zip(rows, fixed_routes))
    return {
        "schema": "adaptive_router_thresholds_v1",
        "status": "user_calibrated_configuration",
        "thresholds": best_thresholds.to_dict(),
        "calibration": {
            "n": len(rows),
            "objective": "mean supplied calibration branch score",
            "mean_score": measured / len(rows),
            "default_mean_score": fixed_score / len(rows),
            "changed_from_fixed_n": best_key[0],
            "grid_size": [len(jsd_grid), len(confidence_grid)],
            "candidate_pairs_n": len(jsd_grid) * len(confidence_grid),
            "optimal_grid_pairs_n": optimal_pairs,
            "total_score_tie_tolerance": OBJECTIVE_TOLERANCE,
        },
        "rule": "JSD_normalized > tau_jsd_normalized -> hybrid; else confidence > tau_confidence -> latent; else episodic",
        "evaluation_interpretation": "Fitted on all supplied calibration rows; evaluate on a separate held-out split.",
    }


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Calibration line {line_number} must contain an object")
            rows.append(row)
    return rows
