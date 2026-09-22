# STT-0 Bootstrap Report

Date: 2026-09-22

## Repository

- **Repository**: `morves-local-stt` at `E:\Reza(Jangan Dihapus!!!\morves-local-stt` (sibling of `morves-finance-core`; no crossover)
  - Relocation note (STT-1A, 2026-09-22): the canonical location is now
    `E:\Reza(Jangan Dihapus!!!)\morves-local-stt`. The path above is the
    historical STT-0 working location, preserved for accuracy — see
    [REPOSITORY-LOCATION.md](REPOSITORY-LOCATION.md).
- **Branch**: `phase/stt-0-bootstrap` (from empty `main`)
- **Starting HEAD**: none (repository had zero commits; `main` unborn)
- **Final HEAD**: `c9a909b` — this closure report commits on top (run `git log -1`)

## Machine

- **CPU**: Intel Core i5-9300H (4C/8T)
- **RAM**: 23.9 GB total
- **GPU**: NVIDIA GeForce GTX 1050, 4 GB VRAM, compute capability 6.1, driver 572.83
- **CUDA**: CT2 4.8.2 reports 1 CUDA device; **cuDNN/cuBLAS DLLs not installed** → GPU model load unverified; STT-0 ran CPU int8

## Verdicts

- **Primary baseline**: faster-whisper (Whisper weights, CTranslate2) — verified live
- **Secondary baseline**: whisper.cpp (quantized GGML, streaming-capable fallback)
- **Training recommendation**: `openai/whisper-small` + PEFT LoRA — local pilot only; realistic production path is remote training + local CT2 inference
- **Inference recommendation**: faster-whisper `small` int8 CPU (config default); `base` verified live; GPU `turbo` int8 as a later upgrade after DLL setup

## Deliverable Checklist

| Item | Status |
|---|---|
| Dataset contract (schema, validation) | PASS |
| Speaker-independent split (zero-overlap enforcement + tests) | PASS |
| Audio normalization (decode → 16 kHz mono PCM16, clipping guard) | PASS |
| Evaluation harness (JSON+MD reports, per-domain/per-speaker) | PASS |
| WER | IMPLEMENTED |
| CER | IMPLEMENTED |
| Entity metric | IMPLEMENTED |
| Finance-term metric | IMPLEMENTED |
| Amount metric | IMPLEMENTED |
| Date metric | IMPLEMENTED |
| Currency metric | IMPLEMENTED |
| Critical-confusion tracking | IMPLEMENTED |
| Baseline adapter (faster-whisper) | PASS |
| Live baseline inference | PASS (smoke; quality benchmark pending real audio) |
| Local API skeleton | PASS |
| Health endpoint | PASS |
| Ready endpoint (503 on model-load failure, health stays 200) | PASS |
| Privacy controls (no audio logging, temp cleanup, gitignore walls) | PASS |
| Tests | 82 passed / 0 failed |
| Static quality (ruff format+lint, mypy) | PASS |

## Live Baseline Inference Record (§40 smoke)

- Model: whisper `base` int8, CPU (Systran CT2 conversion, cached under `models/`)
- Audio: 3 locally generated synthetic tone/noise clips (1.6–2.5 s) — **not** quality material
- Latency: mean 1,682 ms / p50 1,478 ms / p95 2,158 ms (after 1 warmup, machine under background load)
- Real-time factor: 0.81 mean (< 1.0 → faster than real time for `base` int8 CPU)
- Output: empty transcripts (VAD correctly filtered non-speech tones) — infrastructure verified end-to-end
- VRAM: 0 (CPU run); RAM not instrumented (see limitations)
- Report artifact: `evaluation/reports/20260922-211833-baseline.*` (gitignored)

**No quality claims are made from this run.** Quality benchmarking requires the
real private speech manifest (STT-2).

## Known Limitations

1. GPU inference unverified on this machine — Windows cuDNN 9/cuBLAS DLLs missing; CPU int8 is the verified path (documented in HARDWARE-BASELINE.md).
2. No real Indonesian speech has been benchmarked yet — every quality number is pending STT-2.
3. Term-category accuracies are occurrence-based and position-blind (conservative STT-0 proxy); alignment-based positional scoring is future work.
4. Peak VRAM/RAM capture not instrumented (latency and RTF are).
5. ffmpeg absent → M4A/AAC dataset decode unavailable until installed (WAV/FLAC/MP3/OGG work via libsndfile).
6. Python 3.10.6 (below the 3.11+ preference; machine constraint, no global installs) — upgrade before STT-4 (3.10 EOL Oct 2026).
7. No word-timestamp exposure in the service response yet (adapter supports segments; add when a consumer needs it).
8. Windows console cp1252 mangled some tool output during audit; captured facts were re-verified via targeted queries.

## Next Recommended Stage

**STT-1** — collect the pilot dataset (4–6 speakers, 1–2 h, all 15 domains per
DATASET-GUIDE) — immediately followed by **STT-2** running this harness on it.
The engineering foundation (pipeline, metrics, benchmark runner, service) is
already in place for both.
