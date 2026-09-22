"""Transcription-assisted prompt mapping (STT-1B workflow, productized).

Maps ASR transcripts of raw recordings onto a known reference prompt list
using normalized character error rate, with unique global assignment,
confidence grading, and ambiguity gating. Pure functions - no audio I/O -
so the whole policy is unit-testable. ``scripts/ingest_speaker_batch.py``
drives it with a real model.
"""

from __future__ import annotations

from dataclasses import dataclass

from morves_stt.metrics import cer
from morves_stt.normalize import normalize_for_metric

# Confidence thresholds (CER cost + margin over the runner-up prompt).
HIGH_MAX_CER = 0.30
HIGH_MIN_MARGIN = 0.10
MEDIUM_MAX_CER = 0.50
MEDIUM_MIN_MARGIN = 0.05
AMBIGUOUS_MAX_GAP = 0.05
AMBIGUOUS_MAX_RUNNERUP_CER = 0.55


@dataclass(frozen=True)
class MappingDecision:
    original_filename: str
    prompt_id: str
    cost: float
    runner_up_prompt_id: str | None
    runner_up_cost: float
    confidence: str  # HIGH | MEDIUM | LOW | AMBIGUOUS
    reason: str


def build_cost_matrix(
    references: dict[str, str],  # prompt_id -> reference text
    hypotheses: dict[str, str],  # original_filename -> transcript
) -> dict[str, dict[str, float]]:
    """cost[filename][prompt_id] = normalized CER(reference, transcript)."""
    norm_ref = {pid: normalize_for_metric(t) for pid, t in references.items()}
    norm_hyp = {fn: normalize_for_metric(t) for fn, t in hypotheses.items()}
    return {
        fn: {pid: cer(ref, hyp) for pid, ref in norm_ref.items()} for fn, hyp in norm_hyp.items()
    }


def assign_prompts(cost: dict[str, dict[str, float]]) -> dict[str, str]:
    """Greedy global unique assignment: lowest total cost, one prompt per file.

    Ties broken deterministically by (cost, prompt_id, filename).
    """
    pairs = sorted((c, pid, fn) for fn, row in cost.items() for pid, c in row.items())
    used_files: set[str] = set()
    used_prompts: set[str] = set()
    out: dict[str, str] = {}
    for _, pid, fn in pairs:
        if fn in used_files or pid in used_prompts:
            continue
        out[fn] = pid
        used_files.add(fn)
        used_prompts.add(pid)
    return out


def grade(
    filename: str,
    prompt_id: str,
    cost_row: dict[str, float],
    assigned_prompt: str,
) -> MappingDecision:
    """Confidence grade for one file given the full cost row and assignment."""
    best = cost_row[assigned_prompt]
    runners = sorted((c, pid) for pid, c in cost_row.items() if pid != assigned_prompt)
    runner_cost, runner_pid = runners[0] if runners else (1.0, None)
    margin = runner_cost - best

    if assigned_prompt != prompt_id:
        return MappingDecision(
            filename,
            prompt_id,
            best,
            runner_pid,
            runner_cost,
            "LOW",
            f"assigned elsewhere; this pair cost {best:.3f}",
        )
    if margin < AMBIGUOUS_MAX_GAP and runner_cost < AMBIGUOUS_MAX_RUNNERUP_CER:
        return MappingDecision(
            filename,
            assigned_prompt,
            best,
            runner_pid,
            runner_cost,
            "AMBIGUOUS",
            f"margin {margin:.3f} vs {runner_pid}",
        )
    if best <= HIGH_MAX_CER and margin >= HIGH_MIN_MARGIN:
        conf = "HIGH"
    elif best <= MEDIUM_MAX_CER and margin >= MEDIUM_MIN_MARGIN:
        conf = "MEDIUM"
    else:
        conf = "LOW"
    return MappingDecision(
        filename,
        assigned_prompt,
        best,
        runner_pid,
        runner_cost,
        conf,
        f"cer {best:.3f}, margin {margin:.3f}",
    )


def map_batch(
    references: dict[str, str],
    hypotheses: dict[str, str],
) -> list[MappingDecision]:
    """Full policy: assign + grade every file. AMBIGUOUS/LOW files must not be
    canonically renamed without review; callers decide what to do with them."""
    cost = build_cost_matrix(references, hypotheses)
    assignment = assign_prompts(cost)
    decisions: list[MappingDecision] = []
    for fn in sorted(hypotheses):
        row = cost[fn]
        # Grade against the file's OWN best prompt (not the forced assignment)
        # so leftover assignments surface as LOW instead of silently passing.
        own_best = min(row, key=lambda pid: (row[pid], pid))
        decisions.append(grade(fn, own_best, row, assignment.get(fn, own_best)))
    return decisions
