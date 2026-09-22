# Morves Local STT

Locally deployable Indonesian speech-to-text optimized for Morves Finance
terminology. One purpose:

```
AUDIO → LOCAL ASR → TRANSCRIPT
```

This repository is **completely separate** from `morves-finance-core`.

## What it IS

- A local ASR stack for Indonesian finance-domain speech (Whisper weights via
  faster-whisper/CTranslate2 int8 today; fine-tuned domain model planned)
- A dataset pipeline with a strict privacy model and speaker-independent splits
- A domain-aware evaluation harness (WER/CER + entity/finance/amount/date/
  currency accuracies + critical-confusion tracking, plus semantic
  amount/date/currency diagnostics that compare numeric meaning — STRICT
  metrics always stay authoritative and are reported alongside)
- A minimal local HTTP transcription service

## What it is NOT

- No Morves PostgreSQL connection, no financial queries, no business logic
- No transaction posting, no user authentication, no AI Assistant, no tool calling
- Integration happens later, in morves-finance-core, through a
  `SpeechToTextProvider` adapter (future provider id: `morves-local`)

## Architecture

```
Browser → Morves Backend (SpeechToTextProvider) → this service → transcript
```

Details: [docs/INTEGRATION-CONTRACT.md](docs/INTEGRATION-CONTRACT.md).

## Quick Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -e ".[dev]"
```

Python 3.10+ (developed on 3.10.6; all dependencies support it).

## Hardware Requirements

- CPU inference (int8): 4-core modern CPU, 8 GB RAM — works today
- GPU inference (optional): ≥4 GB VRAM + cuDNN 9/cuBLAS DLLs on Windows (see
  [docs/HARDWARE-BASELINE.md](docs/HARDWARE-BASELINE.md))
- Disk: ~2 GB for code + one model

## Dataset Privacy (read before adding any audio)

Raw/processed audio is **never committed**. Speakers are pseudonymous
(`spk_001`). Test sets contain real human speech only. Full policy:
[datasets/README.md](datasets/README.md) and
[docs/DATASET-GUIDE.md](docs/DATASET-GUIDE.md).

## Preparing Data (STT-1 pilot)

Recording pack and session design: [datasets/recording-pack/README.md](datasets/recording-pack/README.md)
and [docs/STT-1-PILOT-DATASET-PLAN.md](docs/STT-1-PILOT-DATASET-PLAN.md).
Participant instructions (non-engineers):
[datasets/recording-pack/PARTICIPANT-GUIDE.md](datasets/recording-pack/PARTICIPANT-GUIDE.md).

```bash
# per speaker batch: transcription-assisted mapping, dry-run then --apply
python scripts/ingest_speaker_batch.py --speaker spk_002 --input-dir <recordings>

# already-named recovery takes (e.g. missing p017):
python scripts/ingest_speaker_batch.py --speaker spk_001 \
    --input-dir datasets/raw/spk_001/recovery --direct --apply

# manual path (normalize + records.csv) still works:
python scripts/normalize_audio.py datasets/raw --out-dir datasets/processed
python scripts/prepare_manifest.py --csv datasets/transcripts/records.csv

python scripts/validate_dataset.py --extra-manifest evaluation/private-manifest.jsonl
```

`validate_dataset.py` prints `PILOT DATASET READY FOR STT-2` only when all
pilot floors are met (≥4 speakers, ≥1 h total, all 15 domains, anchor terms
spoken by enough speakers, zero speaker leakage, real speech in test);
otherwise `PILOT DATASET NOT READY` with the exact deficits. Per-speaker
private quality reports: `python scripts/data_quality.py --extra-manifest <manifest>`.

## Running Tests

```bash
pytest                # unit + service tests; no model download required
pytest -m integration # slower multi-component tests
```

Marked `model` tests require a downloaded ASR model and are opt-in.

## Baseline Benchmark

```bash
# quality benchmark (requires real private speech manifest):
python scripts/benchmark_baseline.py --config configs/baseline.yaml

# pipeline smoke test (synthetic audio; latency only, NOT quality):
python scripts/benchmark_baseline.py --smoke
```

Reports land in `evaluation/reports/` (gitignored — they may contain
transcripts of private audio).

## Local Service

```bash
.venv/Scripts/python -m service.main
# GET  /health  /ready  /v1/model
# POST /v1/transcriptions  (multipart: file=<wav>, language=id)
```

## Static Quality

```bash
ruff format . && ruff check .
mypy
```

## Training Roadmap

| Stage | Scope |
|---|---|
| STT-0 | bootstrap, hardware audit, dataset contract, eval harness, integration contract (this repo state) |
| STT-1 | dataset pipeline hardening on real recordings (recording pack ready; status: WAITING FOR HUMAN RECORDINGS) |
| STT-2 | pretrained baseline benchmark on pilot dataset |
| STT-3 | domain dataset collection (speakers/devices/noise) |
| STT-4 | fine-tuning (LoRA on whisper-small; see docs/BASELINE-RESEARCH.md) |
| STT-5 | evaluation of fine-tuned model on the same harness |
| STT-6 | quantization / runtime optimization |
| STT-7 | production local inference service |
| STT-8 | Morves provider integration (in morves-finance-core) |
| STT-9 | A/B comparison vs 9Router on the same private test set |

## Future Morves Finance Integration

Via `SpeechToTextProvider` with provider id `morves-local`, calling the HTTP
contract in [docs/INTEGRATION-CONTRACT.md](docs/INTEGRATION-CONTRACT.md). Not
implemented in this repo, by design.

## Status

STT-0 foundations. **No accuracy claims have been made** — none are valid until
the STT-2 benchmark runs on real speech.
