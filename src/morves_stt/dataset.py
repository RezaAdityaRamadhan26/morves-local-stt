"""Dataset manifest schema and speaker-independent splitting logic."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SPEAKER_ID_PATTERN = re.compile(r"^spk_[0-9A-Za-z_-]+$")
VALID_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
DOMAIN_TAXONOMY: tuple[str, ...] = (
    "general",
    "navigation",
    "cash_flow",
    "transaction",
    "receivable",
    "payable",
    "intercompany",
    "reporting",
    "project",
    "counterparty",
    "currency_fx",
    "amount",
    "date_time",
    "management",
    "assistant_query",
)
VALID_SOURCES = {"recorded", "synthetic"}
VALID_LANGUAGES = {"id", "en", "mixed"}

# Fields required by the manifest schema
REQUIRED_FIELDS = ("audio", "text", "language", "speaker_id", "duration_sec", "domain", "source")


class ManifestValidationError(ValueError):
    """Raised when a manifest entry violates the schema."""


class SpeakerLeakageError(ValueError):
    """Raised when the same speaker appears in multiple splits."""


@dataclass
class ManifestEntry:
    audio: str
    text: str
    language: str
    speaker_id: str
    duration_sec: float
    domain: str = "general"
    source: str = "recorded"
    recording_profile: str | None = None
    noise_profile: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


def parse_entry(raw: dict[str, Any], index: int = 0) -> ManifestEntry:
    """Parse and validate a single JSON manifest entry."""
    label = f"entry[{index}]"
    if not isinstance(raw, dict):
        raise ManifestValidationError(f"{label}: not a JSON object")

    for req in REQUIRED_FIELDS:
        if req not in raw:
            raise ManifestValidationError(f"{label}: missing required field '{req}'")

    audio = raw["audio"]
    if not isinstance(audio, str) or not audio.strip():
        raise ManifestValidationError(f"{label}: 'audio' must be a non-empty string")
    ext = Path(audio).suffix.lower()
    if ext and ext not in VALID_AUDIO_EXTENSIONS:
        raise ManifestValidationError(
            f"{label}: unsupported audio extension '{ext}' for '{audio}' "
            f"(allowed: {sorted(VALID_AUDIO_EXTENSIONS)})"
        )

    text = raw["text"]
    if not isinstance(text, str) or not text.strip():
        raise ManifestValidationError(f"{label}: 'text' must be a non-empty string")

    language = raw["language"]
    if language not in VALID_LANGUAGES:
        raise ManifestValidationError(
            f"{label}: 'language' must be one of {sorted(VALID_LANGUAGES)}, got '{language}'"
        )

    speaker_id = raw["speaker_id"]
    if not isinstance(speaker_id, str) or not SPEAKER_ID_PATTERN.match(speaker_id):
        raise ManifestValidationError(
            f"{label}: 'speaker_id' must match ^spk_[0-9A-Za-z_-]+$ "
            f"(anonymized IDs like spk_001); got '{speaker_id}'"
        )

    duration = raw["duration_sec"]
    if not isinstance(duration, int | float) or isinstance(duration, bool):
        raise ManifestValidationError(f"{label}: 'duration_sec' must be numeric")
    if duration <= 0 or duration > 600:
        raise ManifestValidationError(
            f"{label}: 'duration_sec' must be in (0, 600] seconds, got {duration}"
        )

    domain = raw["domain"]
    if domain not in DOMAIN_TAXONOMY:
        raise ManifestValidationError(
            f"{label}: 'domain' '{domain}' not in taxonomy {list(DOMAIN_TAXONOMY)}"
        )

    source = raw["source"]
    if source not in VALID_SOURCES:
        raise ManifestValidationError(
            f"{label}: 'source' must be one of {sorted(VALID_SOURCES)}, got '{source}'"
        )

    return ManifestEntry(
        audio=audio,
        text=text,
        language=language,
        speaker_id=speaker_id,
        duration_sec=float(duration),
        domain=domain,
        source=source,
        recording_profile=raw.get("recording_profile"),
        noise_profile=raw.get("noise_profile"),
    )


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    """Load a JSONL manifest file into validated entries."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")

    entries: list[ManifestEntry] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ManifestValidationError(
                    f"{p.name} line {i + 1}: invalid JSON: {exc}"
                ) from exc
            entries.append(parse_entry(raw, index=i))
    return entries


def save_manifest(entries: list[ManifestEntry], path: str | Path) -> None:
    """Write entries to a JSONL manifest file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write("# morves-stt manifest (schema: datasets/README.md)\n")
        for entry in entries:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


def speaker_ids(entries: list[ManifestEntry]) -> set[str]:
    return {e.speaker_id for e in entries}


def speaker_overlap(a: list[ManifestEntry], b: list[ManifestEntry]) -> set[str]:
    return speaker_ids(a) & speaker_ids(b)


def assert_no_speaker_leakage(
    train: list[ManifestEntry],
    validation: list[ManifestEntry],
    test: list[ManifestEntry],
) -> None:
    """Raise SpeakerLeakageError if any speaker appears in multiple splits."""
    overlaps = {
        "train∩validation": speaker_overlap(train, validation),
        "train∩test": speaker_overlap(train, test),
        "validation∩test": speaker_overlap(validation, test),
    }
    problems = {k: sorted(v) for k, v in overlaps.items() if v}
    if problems:
        raise SpeakerLeakageError(
            "Speaker leakage detected - the same speaker appears in multiple splits: "
            + "; ".join(f"{k}: {v}" for k, v in problems.items())
        )


def split_by_speaker(
    entries: list[ManifestEntry],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[ManifestEntry]]:
    """Split entries into train/validation/test such that speakers are independent.

    Groups are speaker blocks; the split operates on speakers, not utterances.
    Requires at least 3 distinct speakers (one per non-empty split).
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    by_speaker: dict[str, list[ManifestEntry]] = {}
    for e in entries:
        by_speaker.setdefault(e.speaker_id, []).append(e)

    if len(by_speaker) < 3:
        raise ManifestValidationError(
            f"Speaker-independent 3-way split requires at least 3 speakers, got {len(by_speaker)}. "
            "Collect more speakers or merge splits manually."
        )

    rng = random.Random(seed)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)

    total = len(entries)
    targets = {"train": ratios[0], "validation": ratios[1], "test": ratios[2]}
    order = ("train", "validation", "test")

    groups: dict[str, list[ManifestEntry]] = {k: [] for k in order}
    assigned = {k: 0 for k in order}
    speaker_assignment: dict[str, str] = {}

    # Greedy deficit fill: each speaker goes to the split most under its target ratio.
    for spk in speakers:
        deficits = {k: targets[k] * total - assigned[k] for k in order}
        best = max(order, key=lambda k: deficits[k])
        groups[best].extend(by_speaker[spk])
        assigned[best] += len(by_speaker[spk])
        speaker_assignment[spk] = best

    # Guarantee no empty split: donate a speaker from the largest split if needed.
    for empty_split in order:
        if groups[empty_split]:
            continue
        donor = max((k for k in order if k != empty_split), key=lambda k: len(groups[k]))
        if not groups[donor]:
            raise ManifestValidationError(
                "Cannot guarantee non-empty speaker-independent splits with so few speakers."
            )
        moved_spk = next(s for s in speaker_assignment if speaker_assignment[s] == donor)
        groups[donor] = [e for e in groups[donor] if e.speaker_id != moved_spk]
        groups[empty_split].extend(by_speaker[moved_spk])
        speaker_assignment[moved_spk] = empty_split

    for split_name in order:
        if not groups[split_name]:
            raise ManifestValidationError(
                f"Split '{split_name}' ended up empty; adjust ratios or add speakers."
            )

    assert_no_speaker_leakage(groups["train"], groups["validation"], groups["test"])
    return groups
