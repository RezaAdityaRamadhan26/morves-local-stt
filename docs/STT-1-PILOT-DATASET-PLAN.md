# STT-1 Pilot Dataset Plan

Date: 2026-09-22
Status: **STT-1B SANITY BATCH INGESTED (1 speaker, 29/30) — WAITING FOR MORE HUMAN RECORDINGS**

This plan prepares (and only prepares) the STT-1 pilot dataset. **No training
happens in STT-1**: no LoRA, no PEFT, no full fine-tune, no STT-4. The stage
ends when `python scripts/validate_dataset.py` prints
`PILOT DATASET READY FOR STT-2`.

## Objective

| Floor | Value |
|---|---|
| Speakers | 4–6 consenting adults, pseudonymous (`spk_001`…) |
| Total speech | ≥ 1.0 hour (target 1–2 h) |
| Per speaker | ≥ 10 minutes each |
| Domains | all 15 taxonomy domains present |
| Splits | speaker-independent (train/validation/test, zero overlap) |
| Test split | real human speech only, no synthetic |

Synthetic audio may exercise the pipeline but **never counts** toward
speakers, hours, or any quality claim.

## Prompt bank

`datasets/recording-pack/pilot-prompts.csv` — 180 prompts (15 domains × 12),
newly authored for STT-1 (not a duplication of the 20-entry sentence bank in
`evaluation/fixtures/sentence_bank.jsonl`, which stays a metric fixture).

Coverage guarantees:

- **Entities**: Morves, Morves Finance, Medita, Medita Solusi Digital, Creatifyl
  — including near-name discrimination ("Eh, yang Medita Solusi Digital ya,
  bukan Medita biasa."; "Kreativitas tim Creatifyl bagus…").
- **Finance vocabulary**: piutang, utang, pendapatan, pengeluaran, aset,
  liabilitas, ekuitas, neraca, neraca saldo, laba rugi, arus kas, jurnal,
  debit, kredit, margin, proyek, invoice, overdue, jatuh tempo, rekening,
  BCA, Mandiri, intercompany, antarperusahaan, konsolidasi, eliminasi.
- **Critical confusion pairs** (mirror of
  `evaluation/fixtures/terms.yaml → critical_confusions`): Medita↔media,
  Creatifyl↔kreatif, piutang↔utang, debit↔kredit, ribu↔juta↔miliar↔triliun,
  IDR↔USD, rupiah↔dolar, plus date-month corrections (September↔Oktober).
- **Spoken-form numbers**: "dua ratus lima puluh juta rupiah", "satu koma
  lima miliar", "nol koma nol lima", "dua koma lima persen", "tiga belas
  juta lima ratus ribu", dates as words, years as words.
- **Mixed Indonesian/English**: "Transaction terakhir apa?", "Jurnal umum
  bulan ini ada berapa entry?", "Rate EUR juga bisa dilihat?".
- **Register variation** (`style` column): formal ("Tampilkan piutang Medita
  bulan ini."), casual ("Tampilin piutang Medita sekarang berapa?", "Coba
  lihat invoice yang overdue.", "Kas kita aman nggak bulan ini?"), short
  commands ("Buka neraca."-class), long questions, hesitations and
  self-corrections ("Eh, bukan ribu, juta."). Speakers must keep their
  natural register — no normalization to one style.

## Session design

Per-session metadata goes in
`datasets/recording-pack/speaker-session-template.csv`
(`speaker_id, session_id, session_date, language, recording_profile,
noise_profile, device, prompts_range, consent_confirmed, notes`).

- **Assignment** (~120 prompts/speaker ≈ 15–20 min speech):
  shared core `pp_001–pp_060` for every speaker; rotation blocks
  `pp_061–pp_090 / pp_091–pp_120 / pp_121–pp_150 / pp_151–pp_180`
  distributed so all 180 prompts are covered collectively.
- **Recording profiles** (mix across speakers, don't standardize):
  `laptop_quiet`, `phone_quiet`, `office_normal`, `headset_mic`.
- **Noise profiles**: `quiet`, `ambient`, `mild_noise` (no artificial
  degradation; record the environment as it naturally is).
- **Language**: `id` expected; mixed-code utterances keep `id` with English
  terms inline.
- Speaking pace: normal. No over-articulation, no slowdown.

## Shared anchor set (STT-1C)

`datasets/recording-pack/anchor-prompts.csv` — 36 prompts (~30–40 target)
that **every speaker records**, whatever else their assignment covers. The
`covers` column lists the 25 critical terms each anchor protects:
Morves, Medita, Creatifyl, piutang, utang, debit, kredit, ribu, juta, miliar,
IDR, USD, BCA, Mandiri, intercompany, antarperusahaan, neraca saldo,
laba rugi, arus kas, invoice, outstanding, overdue (plus rupiah, ekuitas,
jurnal, and near-name contrasts). The readiness validator reports per-term
speaker coverage and blocks STT-2 while any anchor term is spoken by fewer
than the minimum speaker count.

## Speaker-disjoint split proposal (STT-1C)

| Split | Speakers |
|---|---|
| TRAIN | spk_001–spk_004 |
| VALIDATION | spk_005 |
| TEST | spk_006 |

Speaker overlap across splits is a hard deficit: `validate_dataset.py`
reports leakage and the verdict flips to NOT READY. The split is a proposal —
final assignment happens when all 6 speakers exist, but speakers may never
appear in two splits.

## Semantic diagnostics (STT-1C) — STRICT metrics stay authoritative

`src/morves_stt/semantic.py` adds *separate* diagnostic metrics:
amount/date/currency semantic accuracy, where "dua ratus lima puluh juta" ≡
"250 juta" and "lima juta" ≢ "lima miliar". These never replace strict
WER/CER/term accuracies and never hide transcript errors: reports always
carry both STRICT and SEMANTIC numbers. **Entity metrics are never
semantically normalized** — Medita → media, Morves → marfes,
Creatifyl → kreatif remain strict errors, always.

## Ingesting speaker batches (STT-1C)

```bash
# opaque filenames (e.g. Telegram dumps): transcribe-assisted mapping,
# dry-run first (prints the mapping table), then --apply to canonicalize:
python scripts/ingest_speaker_batch.py --speaker spk_002 --input-dir <dir> \
    [--recording-profile phone_quiet] [--noise-profile quiet]

# files already named <prompt_id>.<ext> (e.g. the spk_001 p017 recovery take):
python scripts/ingest_speaker_batch.py --speaker spk_001 \
    --input-dir datasets/raw/spk_001/recovery --direct --apply
```

Mapping policy (unit-tested in `tests/test_mapping.py`): normalized-CER cost
matrix, greedy unique assignment, confidence HIGH/MEDIUM accepted, LOW/AMBIGUOUS
blocked from canonical rename. Private per-speaker quality reports:

```bash
python scripts/data_quality.py --extra-manifest evaluation/private-manifest.jsonl \
    --references datasets/recording-pack/sanity-prompts.csv
```

The readiness command now accepts extra manifests (private sanity batches
never become production splits):

```bash
python scripts/validate_dataset.py --extra-manifest evaluation/private-manifest.jsonl
```

## Directories and schema

```
datasets/raw/         # original recordings (gitignored, NEVER committed)
datasets/processed/   # canonical 16 kHz mono PCM16 (gitignored, NEVER committed)
datasets/transcripts/ # transcripts CSV (*.private.* gitignored; real
                      #   transcripts are private — do not commit them)
datasets/manifests/   # split manifests (*.private.* gitignored)
```

Transcripts CSV schema (see `datasets/recording-pack/transcripts-template.csv`):

```
audio,text,speaker_id,language,duration_sec,domain,source,recording_profile,noise_profile
processed/spk_001/000001.wav,"Tampilkan piutang Medita bulan ini.",spk_001,id,,receivable,recorded,laptop_quiet,quiet
```

`audio` is relative to `datasets/`; `duration_sec` may be blank (probed from
the file). **No real speaker names in any column.**

## Pipeline

```bash
# per speaker, after recording:
python scripts/normalize_audio.py datasets/raw/spk_001 --out-dir datasets/processed/spk_001
# transcribe manually into datasets/transcripts/records.csv, then:
python scripts/prepare_manifest.py --csv datasets/transcripts/records.csv
python scripts/validate_dataset.py        # THE readiness command
```

`validate_dataset.py` is the single readiness entry point. It reports:
speaker count, total and per-speaker duration, per-domain counts, recording
and noise profile distribution, language distribution, speaker overlap,
synthetic utterances in test, missing audio files, and schema errors — then
the verdict. Readiness floors live in `src/morves_stt/readiness.py`
(single source of truth, unit-tested in `tests/test_readiness.py`) and can be
inspected/overridden via CLI flags for experimentation, never to fake a pass.

## Privacy hard gate

1. Raw and processed audio are **never committed** (gitignore walls:
   `datasets/raw/**`, `datasets/processed/**`, plus every audio extension).
2. Private manifests and transcript-bearing reports are **never committed**
   (`datasets/manifests/*.private.*`, `evaluation/private-manifest.jsonl`,
   `evaluation/reports/*`).
3. Speaker identities never enter the repository — `spk_NNN` pseudonyms only,
   enforced by the manifest schema.
4. No customer or production audio, ever. Prompt read-alouds and ad-libs by
   consenting adults only.
5. Explicit consent before recording; consent records live outside the repo
   (`consent_confirmed` tracks it in the session template).
6. The service never logs audio or transcripts (`log_transcripts: false`).

## Honest status reporting

Until enough real recordings exist, the validator verdict is
`PILOT DATASET NOT READY` with a `WAITING FOR HUMAN RECORDINGS` deficit.
Current reality: one speaker (spk_001) has ingested the 30-prompt sanity
batch (29/30 prompts; **p017 is missing and must be re-recorded by the
speaker** into `datasets/raw/spk_001/recovery/p017.ogg` — never fabricated).
No speaker/hour/quality claim beyond that is made. The final quality
benchmark happens at STT-2 (faster-whisper `base` + `small` only) once the
verdict flips to `PILOT DATASET READY FOR STT-2`.

Participant-facing instructions (non-engineers):
[datasets/recording-pack/PARTICIPANT-GUIDE.md](../datasets/recording-pack/PARTICIPANT-GUIDE.md).
