# STT-1 Pilot Recording Pack

Text-only materials for recording the STT-1 pilot dataset (4–6 speakers,
1–2 hours total, all 15 domains). **This pack contains no audio and no
personal data** — only prompts, templates, and instructions. It is safe to
commit.

Full plan: [docs/STT-1-PILOT-DATASET-PLAN.md](../../docs/STT-1-PILOT-DATASET-PLAN.md)

## Contents

| File | Purpose |
|---|---|
| `pilot-prompts.csv` | 180 read/conversational prompts across all 15 domains |
| `speaker-session-template.csv` | Per-session metadata template (speaker pseudonyms only) |
| `transcripts-template.csv` | Schema example for the transcripts CSV fed to `prepare_manifest.py` |

## Prompt bank design

- **15 domains × 12 prompts** (`pp_001`–`pp_180`): general, navigation,
  cash_flow, transaction, receivable, payable, intercompany, reporting,
  project, counterparty, currency_fx, amount, date_time, management,
  assistant_query.
- **Registers vary deliberately** (`style` column): formal commands
  ("Tampilkan piutang Medita bulan ini."), casual speech ("Tampilin piutang
  Medita bulan ini.", "Coba lihat invoice yang overdue."), short commands,
  questions, hesitations/self-corrections ("Eh, maksud saya USD, bukan IDR.").
  Do **not** normalize speakers to one register — natural variation is the
  point.
- **Critical confusion cases are intentionally included** (see
  `evaluation/fixtures/terms.yaml` → `critical_confusions`): Medita vs media
  contexts, Morves phonetics, Creatifyl vs kreatif ("Kreativitas tim
  Creatifyl bagus..."), piutang ↔ utang, debit ↔ kredit, ribu/juta/miliar/
  triliun, IDR ↔ USD, rupiah ↔ dolar, September ↔ Oktober corrections.
- **Numbers stay spoken-form**: "dua ratus lima puluh juta rupiah",
  "satu koma lima miliar", "dua koma lima persen", "nol koma nol lima",
  dates as words ("dua belas September dua ribu dua puluh enam").

## Session assignment (recommended)

~120 prompts per speaker ≈ 15–20 minutes of speech each; 5 speakers ≈ 1.2–1.6 h.

- **Shared core** `pp_001–pp_060`: every speaker reads these → enables
  cross-speaker comparison.
- **Rotation blocks** `pp_061–pp_180` (4 blocks of 30): each speaker gets a
  different block so all 180 prompts are covered collectively.

Recording conditions (mix them, don't standardize to one):

- `recording_profile`: `laptop_quiet` | `phone_quiet` | `office_normal` | `headset_mic`
- `noise_profile`: `quiet` | `ambient` | `mild_noise`
- Normal speaking pace; do not artificially slow down or over-articulate.
  Do not heavily degrade audio either — no added effects, no room echoes on
  purpose.

## Privacy rules (hard gate)

1. **Raw and processed audio are NEVER committed.** `datasets/raw/**` and
   `datasets/processed/**` are gitignored and stay local. No exceptions,
   no "just one sample".
2. **Private manifests and transcript-bearing reports are NEVER committed**
   (`datasets/manifests/*.private.*`, `evaluation/private-manifest.jsonl`,
   `evaluation/reports/*` are gitignored).
3. **No real speaker names anywhere in the repo.** Use `spk_001`-style
   pseudonyms (enforced by the manifest schema). The session template's
   `notes` column must never contain a name.
4. **No customer or production audio.** Recordings must be read/ad-libbed
   from these prompts by consenting adults only.
5. **Explicit consent** recorded per speaker *before* recording
   (`consent_confirmed` column in the session template). Keep the signed
   consent records outside the repository.
6. Synthetic audio is fine for pipeline plumbing but **never counts** toward
   speaker count, hours, or any quality claim.

## Workflow

```bash
# 1. Record: speaker reads assigned prompts; save per-utterance WAV files
#    under datasets/raw/spk_00X/ (any format WAV/FLAC/MP3/OGG works; 16 kHz+).

# 2. Normalize to canonical 16 kHz mono PCM16:
python scripts/normalize_audio.py --input datasets/raw/spk_001 --output datasets/processed/spk_001

# 3. Transcribe manually into a CSV modeled on transcripts-template.csv
#    (audio path relative to datasets/, e.g. processed/spk_001/000001.wav).

# 4. Build speaker-independent split manifests:
python scripts/prepare_manifest.py --csv datasets/transcripts/spk_001-005.csv

# 5. Validate + pilot readiness verdict (one command):
python scripts/validate_dataset.py
```

Step 5 prints the verdict `PILOT DATASET READY FOR STT-2` only when all
floors are met (≥4 speakers, ≥1 h total, all 15 domains, zero leakage, real
speech in test). Until then it reports `PILOT DATASET NOT READY` with the
exact deficits — including `WAITING FOR HUMAN RECORDINGS` when no real audio
has been prepared yet.
