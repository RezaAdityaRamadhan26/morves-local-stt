# Hardware Baseline — STT-0 Audit

Captured: 2026-09-22, via `scripts/inspect_hardware.py` + manual audit.
Machine: development laptop running Windows 11 Home Single Language (10.0.26200).

## Machine Summary

| Component | Value |
|---|---|
| OS | Windows 11 Home Single Language, 10.0.26200 |
| CPU | Intel Core i5-9300H @ 2.40 GHz (Coffee Lake mobile) |
| Cores / Threads | 4 / 8 |
| RAM (total) | 23.9 GB |
| RAM (free at audit) | 2.8 GB (background load; close apps before benchmarking) |
| Disk (E:) | 932 GB total, 765 GB free |
| GPU | NVIDIA GeForce GTX 1050 (laptop) |
| VRAM | 4096 MB |
| Compute capability | 6.1 (Pascal — **no tensor cores, no bf16**) |
| Driver | 572.83 (supports up to CUDA 12.x) |
| Python | 3.10.6 (system) — project venv built on this |
| ffmpeg | NOT installed (optional; extends dataset decode formats) |
| torch | NOT installed (not required for STT-0 inference; needed for STT-4) |

## Verdict

**Hardware suitability: LIMITED TRAINING — plan around INFERENCE FIRST.**

### Inference suitability: GOOD (with constraints)

- **CPU path (recommended for STT-0)**: faster-whisper `int8` on this 4C/8T CPU is
  a first-class supported configuration. Expect real-time or near-real-time factors
  for `small`; `medium` will be slower but usable for batch evaluation.
- **GPU path (optional upgrade)**: 4 GB VRAM fits whisper `small` fp16 (~2 GB) and
  `medium`/`turbo` int8 (~1.5–3 GB) — *however* CTranslate2 on Windows requires
  cuBLAS (CUDA 12) + cuDNN 9 DLLs that are not currently installed. Until those
  DLLs are provided (Purfview standalone archive or NVIDIA installer + PATH),
  GPU inference will fail to load and the service must run CPU-only.
  Pascal also lacks tensor cores, so fp16 gains are modest.
- `large-v3` (fp16 ~4.5 GB / int8 ~2.9 GB + activations) is at or beyond the
  4 GB ceiling — not recommended on this GPU.

### Training suitability: LIMITED (pilot-scale only)

- 4 GB VRAM rules out full fine-tuning of anything above `tiny`/`base`.
- LoRA/PEFT on `whisper-small` with gradient checkpointing and batch size 1
  is feasible but slow; quality iteration will be painful.
- Pascal cc 6.1: no bf16, no tensor-core fp16 acceleration; use fp16 or fp32.
- **Recommendation**: treat this machine as the *inference and evaluation*
  machine. If fine-tuning (STT-4) needs more throughput, either (a) accept slow
  local LoRA pilots on `base`/`small`, or (b) train remotely (cloud/rented GPU)
  and deploy the converted weights locally via CTranslate2. Decide after STT-2.

### CPU-only fallback: REALISTIC

Yes. CPU int8 inference of `small` (and `base`) is expected to run at or near
real time for short command-style utterances (2–8 s), which matches the Morves
voice-command use case. Actual RTF will be measured in STT-2 — no claims until
measured.

## Recommended Model-Size Ceiling

| Context | Ceiling | Reason |
|---|---|---|
| CPU int8 inference | `small` daily / `medium` batch eval | 4C/8T throughput |
| GPU inference (after DLL setup) | `turbo`/`medium` int8 | 4 GB VRAM |
| Local fine-tuning | `base`/`small` LoRA only | VRAM + no tensor cores |

## Recommended Batch / Precision Starting Points

- Inference service: batch = 1 (single-utterance request model), beam_size 5, VAD on.
- Benchmark: warmup 1 utterance before timing.
- Training (STT-4, if local): micro-batch 1, grad-accum 8, fp16, gradient
  checkpointing on, LoRA r=32 — see `configs/training.yaml`.

## Bottlenecks to Expect

1. CPU contention — only 2.8 GB RAM free at audit time; ASR processes need headroom.
2. Autoregressive decoder latency on long audio (cap requests at 120 s in service).
3. GPU path blocked on missing cuDNN/cuBLAS DLLs (Windows-specific friction).
4. No ffmpeg on PATH — M4A/AAC decode unavailable until installed.

## Python Version Note

Spec prefers 3.11/3.12; the machine's local interpreter is 3.10.6 and `uv` is
not installed. Per the project-local-environment rule (no global installs), the
venv uses 3.10.6, which all current dependencies support. Python 3.10 reaches
end-of-life in October 2026 — **upgrade the interpreter before STT-4** training
work begins.
