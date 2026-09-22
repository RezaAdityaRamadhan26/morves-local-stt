# Dataset Guide — Collection, Privacy, and Preparation

Scope: how speech data is collected, anonymized, normalized, split, and
validated for Morves Local STT. The privacy policy lives in
[datasets/README.md](../datasets/README.md) and is binding.

## Privacy Rules (summary — full policy in datasets/README.md)

- Raw and processed audio are **never committed** (gitignored, both levels).
- Production/customer audio is **never used** without explicit authorization.
- Speakers are pseudonymous (`spk_001`), never names; the identity key stays
  offline outside the repository.
- Test data must be **real human speech** (see Synthetic Audio Policy).
- The inference service never logs audio; transcript logging is off by default.

## Canonical Audio Contract

Target profile (matches the future Morves browser pipeline):

| Property | Value |
|---|---|
| Container | WAV |
| Encoding | Linear PCM, 16-bit |
| Channels | mono |
| Sample rate | 16,000 Hz |

Dataset preparation may accept WAV/FLAC/MP3/M4A/OGG and **decodes and converts**
them (`scripts/normalize_audio.py`). Never rename a compressed file to `.wav`.
Note: M4A/AAC decode requires ffmpeg on PATH (not currently installed on the
audit machine).

## Number / Date / Currency Policy (binding for reference transcripts)

The ASR layer is verbatim: references record **what was spoken**.

- "dua ratus lima puluh juta rupiah" stays words — never "Rp250.000.000"
- Number interpretation (words → values) is downstream Assistant logic in
  morves-finance-core, not STT.
- Evaluation must therefore include phrases like:
  `seratus ribu`, `satu juta`, `dua ratus lima puluh juta`, `satu koma lima miliar`,
  `tanggal dua belas September`, `dua puluh dua September dua ribu dua puluh enam`,
  `IDR`, `USD`, `rupiah`, `dolar`.

## Recording Guide

- **Environment**: quiet office first; add light background noise (fan/AC) and
  normal office ambience in later sessions — label via `noise_profile`.
- **Devices**: vary microphones — laptop built-in, headset, phone mic — recorded
  in `recording_profile`; the service will receive browser mic audio.
- **Pace**: natural pace primarily; include deliberately fast and slow variants.
- **Register**: formal and casual Indonesian; mixed Indonesian/English finance
  vocabulary ("invoice", "overdue", "project") is in-domain, not an error.
- **Speakers**: male/female/diverse voices where available; never fewer than the
  stage's speaker floor. More speakers beats more utterances per speaker.
- **Distance**: mostly 20–50 cm; include a few at arm's length.
- **Quality**: no intentional clipping; record at source quality (e.g. 44.1/48
  kHz) and let `scripts/normalize_audio.py` convert to canonical 16 kHz mono.

## Domain Taxonomy

Fixed vocabulary for manifest `domain` fields (see `configs/base.yaml`):

`general, navigation, cash_flow, transaction, receivable, payable, intercompany,
reporting, project, counterparty, currency_fx, amount, date_time, management,
assistant_query`

Sentence prompts: [evaluation/fixtures/sentence_bank.jsonl](../evaluation/fixtures/sentence_bank.jsonl)
(cross-domain, numbers/dates/currencies/entities included). Domain lexicon:
[datasets/lexicon/morves-finance.txt](../datasets/lexicon/morves-finance.txt) —
vocabulary planning list, not pronunciation rules.

## Preparation Workflow

1. Drop recordings into `datasets/raw/` (any supported format).
2. `python scripts/normalize_audio.py datasets/raw --out-dir datasets/processed`
   → canonical WAVs, directory structure preserved.
3. Create `datasets/transcripts/records.csv` with columns
   `audio,text,speaker_id,language,duration_sec,domain,source` (duration may be
   blank — it is probed from audio).
4. `python scripts/prepare_manifest.py --csv datasets/transcripts/records.csv`
   → speaker-disjoint `train/validation/test.jsonl`.
5. `python scripts/validate_dataset.py` → schema + leakage + policy checks
   (non-zero exit on any failure).

## Speaker-Independent Splits (hard requirement)

- Splits are produced by assigning **whole speakers** (greedy deficit fill,
  seeded), never by shuffling utterances.
- `scripts/validate_dataset.py` fails if any speaker appears in two splits.
- Minimum 3 speakers for a 3-way split to exist at all.

## Staged Quantity Targets

| Stage | Speakers | Hours | Coverage | Why |
|---|---|---|---|---|
| Pilot | 4–6 | 1–2 | all 15 domains, 1–2 devices, quiet | enough to benchmark pretrained baselines honestly (STT-2) |
| MVP domain | ≥20 | 10–20 | all domains, ≥3 devices, +light noise | speaker-disjoint LoRA fine-tune without overfitting to voices (STT-3/4) |
| Production-quality | ≥50 | 40+ | + distance/pace/register variation | generalization claims across real usage (STT-5+) |

Speaker diversity and recording-condition diversity matter more than hour
count. Do not pad hours with synthetic audio.

## Synthetic Audio Policy

- Synthetic TTS audio is allowed **only** for pipeline debugging and rare-phrase
  augmentation **experiments** during training.
- It must **never** appear in `test.jsonl` — evaluation requires real human
  speech (enforced by `validate_dataset.py` policy check).
- Synthetic voices never substitute for real speaker diversity; a model
  benchmarked mostly on synthetic speech has not been benchmarked.
