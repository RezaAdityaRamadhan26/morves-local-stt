# Datasets

This directory holds the metadata, manifests, and local audio for Morves Local STT.

## Strict Privacy Rules

Voice recordings are biometric data. The following rules are absolute across all branches and contributors:

1. **NEVER commit raw or processed audio files.**
   - `datasets/raw/**` and `datasets/processed/**` are `.gitignore`d.
   - Any audio file matching `*.wav`, `*.mp3`, `*.flac`, `*.m4a`, `*.ogg` is automatically ignored.
   - Run `git status` before committing. If any audio file appears staged, unstage it immediately.

2. **NEVER commit production or customer audio.**
   - Only audio recorded by explicitly consenting internal participants under an approved recording plan may be placed in this directory.

3. **Anonymize speakers.**
   - Never use real names or identifying labels as speaker IDs or directory names.
   - Canonical format: `spk_001`, `spk_002`, `spk_003`, etc.
   - Keep any key linking `spk_XXX` to a human identity stored offline in a secure, non-repository location.

4. **Speaker-Independent Splits.**
   - A single speaker's utterances MUST NOT be divided across train, validation, and test splits.
   - If `spk_001` is in `train.jsonl`, `spk_001` MUST NOT appear in `validation.jsonl` or `test.jsonl`.
   - The validation script `scripts/validate_dataset.py` checks this and fails on any leakage.

5. **No remote uploads without explicit approval.**
   - Data in this directory stays local to the machine unless an encrypted, authorized storage transfer is explicitly approved.

6. **Inference privacy.**
   - The inference service never logs raw or base64 audio.
   - Transcript logging is disabled by default (`log_transcripts: false`).

## Directory Layout

```
datasets/
├── README.md               <- This document (privacy policy and directory guide)
├── lexicon/
│   └── morves-finance.txt  <- Canonical domain terms and vocabulary
├── manifests/
│   ├── train.jsonl         <- Speaker-independent training split (manifest only)
│   ├── validation.jsonl    <- Validation split
│   └── test.jsonl          <- Evaluation split (REAL human speech only)
├── raw/                    <- [GITIGNORED] Ingest folder for multi-format recordings
└── processed/              <- [GITIGNORED] Normalized WAV (16 kHz, mono, 16-bit PCM)
```

## Canonical Audio Profile

Every processed audio file in `datasets/processed/` must match:

| Property | Canonical Value |
|---|---|
| Container | WAV |
| Encoding | Linear PCM (16-bit little-endian) |
| Channels | 1 (mono) |
| Sample rate | 16,000 Hz |
| Peak amplitude | Bounded below 0 dBFS to prevent clipping |

Convert incoming recordings using `scripts/normalize_audio.py`.

## Manifest Schema

Manifest files are newline-delimited JSON (`.jsonl`). Each line must be a valid JSON object matching:

```json
{
  "audio": "processed/spk_001/000001.wav",
  "text": "Tampilkan piutang Medita bulan ini.",
  "language": "id",
  "speaker_id": "spk_001",
  "duration_sec": 3.82,
  "domain": "receivable",
  "source": "recorded"
}
```

### Required Fields
- `audio` (string): path to canonical WAV file, relative to `datasets/` (or absolute).
- `text` (string): reference transcript. What was spoken, as words (e.g. "dua puluh dua", not "22").
- `language` (string): BCP-47 / ISO-639-1 code. Default `"id"`.
- `speaker_id` (string): anonymized identifier matching `^spk_[0-9A-Za-z_-]+$`.
- `duration_sec` (float): duration in seconds (must be > 0.0 and <= 600.0).
- `domain` (string): one of the approved taxonomy categories.
- `source` (string): `"recorded"` for human speech, `"synthetic"` for pipeline test fixtures only.

### Optional Fields
- `recording_profile` (string): e.g. `"office_quiet"`, `"office_laptop_mic"`, `"mobile_quiet"`.
- `noise_profile` (string): e.g. `"clean"`, `"light_background"`, `"hvac"`.
- `notes` (string): non-identifying operational notes.

### Test Set Rule
The `test.jsonl` split MUST contain ONLY `source: "recorded"` human utterances. Synthetic voices are strictly forbidden in evaluation.
