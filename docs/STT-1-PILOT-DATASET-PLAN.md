# STT-1 Pilot Dataset Plan

Date: 2026-09-22
Status: **RECORDING PACK READY — WAITING FOR HUMAN RECORDINGS**

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

Until real recordings exist, the validator verdict is
`PILOT DATASET NOT READY` with a `WAITING FOR HUMAN RECORDINGS` deficit.
The current state of this repository contains **zero human recordings**;
no speaker/hour/quality claim is made. Quality benchmarking starts at STT-2
(faster-whisper `base` + `small` only) once the verdict flips to
`PILOT DATASET READY FOR STT-2`.
