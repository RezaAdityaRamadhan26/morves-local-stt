# Baseline Research — Local ASR for Indonesian (STT-0)

Date: 2026-09-22. Sources: official model cards, GitHub repositories, and papers
(URLs inline). Where a figure is an estimate or from a secondary source it is
flagged. Machine context: Windows 11, i5-9300H, 24 GB RAM, GTX 1050 4 GB
(compute capability 6.1) — see [HARDWARE-BASELINE.md](HARDWARE-BASELINE.md).

## Candidates Considered

| Family | Indonesian quality evidence | License | VRAM (inference) | CPU feasible | Quantization | Fine-tune maturity | Word timestamps | Streaming | Export |
|---|---|---|---|---|---|---|---|---|---|
| **Whisper (OpenAI)** | FLEURS `id` WER: small 16.3 / medium 10.2 / large-v2 7.1 (paper Table 13, [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)); large-v3 claims 10–20% relative improvement ([HF card](https://huggingface.co/openai/whisper-large-v3)) | MIT (weights+code, [GitHub](https://github.com/openai/whisper)); HF large-v3 card says Apache-2.0; turbo card MIT — all permissive | tiny~1 / small~2 / medium~5 / large~10 / turbo~6 GB (README) | tiny–small comfortably; medium slow | via runtimes | **Best**: HF Trainer + PEFT/LoRA first-party guides ([blog](https://huggingface.co/blog/fine-tune-whisper)) | segment native; word via DTW | no (30-s window) | CTranslate2, GGML, ONNX, OpenVINO, CoreML |
| **faster-whisper (CT2)** | identical weights → identical quality; up to 4x faster ([repo](https://github.com/SYSTRAN/faster-whisper)) | MIT | large-v2: fp16 4.5 GB / int8 2.9 GB; **turbo: fp16 2.5 GB / int8 1.5 GB** (README + issue #1030) | **yes — int8 CPU is a documented path** | float16/int8/int8_float16/int16 | convert-after-HF-train (`ct2-transformers-converter`) | yes (`word_timestamps=True`) | batched only (not live) | is itself CTranslate2 |
| **whisper.cpp** | identical weights | MIT | similar; q5 quantized large ~2 GB | **yes — primary use case** | q5_0/q5_1/q8_0 | none (inference only) | yes (`-ml 1` + DTW, experimental) | **yes** (`whisper-stream`, 500 ms) | ggml `.bin` |
| **WhisperX** | = Whisper + alignment | BSD-2-Clause ([repo](https://github.com/m-bain/whisperX)); pyannote dep CC-BY-4.0 (gated) | <8 GB for large-v2 | marginal | inherits CT2 | n/a (wrapper) | **best-in-class** (wav2vec2 forced align; default `id` aligner = cahya/wav2vec2-large-xlsr-indonesian) | no | consumes CT2 models |
| **wav2vec2/XLS-R id fine-tunes** | best CV `id` WER 14.29 ([indonesian-nlp](https://huggingface.co/indonesian-nlp/wav2vec2-large-xlsr-indonesian)) — well behind Whisper; CTC output lacks punctuation/casing | **apache-2.0** on id fine-tunes and Meta XLS-R bases (verified on HF metadata; the oft-repeated "XLS-R is CC-BY-NC" does not match current cards) | ~1 GB fp16 (317 M) | yes, fast CTC | ONNX int8 (optimum) | classic HF CTC path, cheap | frame-level native | chunked CTC (unofficial) | ONNX, CT2 |
| **Meta MMS** ([card](https://huggingface.co/facebook/mms-1b-all)) | 1162 langs incl. `ind`; FLEURS-54 avg WER 24.8/18.7 vs Whisper-v2 44.3 ([paper](https://arxiv.org/abs/2305.13516)); edge is low-resource langs, not `id` | **CC-BY-NC 4.0 — no commercial/internal-business use** | ~2 GB fp16 | yes (CTC) | community ONNX | adapter fine-tune | frame-level | unofficial | ONNX only |
| **SeamlessM4T v2** ([card](https://huggingface.co/facebook/seamless-m4t-v2-large)) | `ind` ASR+S2T supported; FLEURS-77 avg WER 18.5 vs Whisper-v2 41.7 (Nature 2025 paper) | **CC-BY-NC 4.0 — disqualified** | ~6 GB fp16 | no | none | thin | none | separate model | none |
| **NVIDIA Parakeet/Canary** ([docs](https://docs.nvidia.com/nemo/speech/3.0.0/starthere/choosing_a_model.html)) | **no Indonesian** (EN or 25 European languages) | CC-BY-4.0 | 0.6 B ≈ 2.5 GB fp16 | partial | NeMo int8 | NeMo (Linux/WSL) | native | v3 variants | NeMo/ONNX |
| **Moonshine** ([repo](https://github.com/usefulsensors/moonshine)) | **no Indonesian** (EN + 7 others) | MIT | tiny/base <0.5 GB | excellent | ONNX | community | no | realtime variants | ONNX/MLX |
| **Vosk** ([models](https://alphacephei.com/vosk/models)) | **no public Indonesian model** (commercial-only per maintainer, issue #923) | Apache 2.0 (most) | ~300 MB RAM | excellent | n/a | Kaldi (CUDA) | native word | **native streaming** | n/a |
| Community id fine-tunes | [Dafisns/whisper-turbo-multilingual-fleurs](https://huggingface.co/Dafisns/whisper-turbo-multilingual-fleurs): self-reported id WER 6.97 (small ~15k-sample train set — validate on domain audio); cahya Indonesian Whisper collection (Mar 2025) | Apache-2.0 | as base model | as base model | as base model | standard HF | — | — | — |

## Comparison Reasoning on THIS Machine

1. **Whisper family wins on evidence**: it is the only family with published
   per-language Indonesian WER across sizes, a permissive license, mature
   fine-tuning tooling, and multiple local runtimes.
2. **faster-whisper is the right runtime**: int8 CPU inference works today with
   zero CUDA setup; the model-size aliases (`turbo` → mobiuslabsgmbh CT2 repo)
   are verified in `faster_whisper/utils.py`. GPU enablement on Windows needs
   cuBLAS/cuDNN DLLs (Purfview archive method) — documented, not yet installed.
3. **License traps eliminated**: MMS and SeamlessM4T are CC-BY-NC (excludes
   internal business use, not just selling) — research reference only.
   Vosk/Moonshine/Parakeet lack Indonesian entirely.
4. **Quality caveat that matters for Morves**: FLEURS/Common Voice numbers are
   clean read speech; the one published mixed-variability Indonesian eval found
   roughly 2x WER degradation on spontaneous speech (Adila et al.,
   [arXiv:2410.08828](https://arxiv.org/abs/2410.08828)). Expect the same on
   conversational finance commands → domain fine-tuning (STT-4) is anticipated,
   and entity terms (Medita, Morves, Creatifyl) will need it most.

## Verdicts

- **PRIMARY BASELINE CANDIDATE**: `faster-whisper` with Whisper weights.
  STT-0 executes on **CPU int8, model `small`** (483 MB download — verified
  reasonable). When DLLs are sorted and quality demands it, upgrade path is
  `turbo` int8 (1.5 GB VRAM) on GPU with no code change (config only).
- **SECONDARY BASELINE CANDIDATE**: `whisper.cpp` (q5 quantized) — zero Python/
  CUDA dependency, native streaming (`whisper-stream`), same weights; the
  fallback if CTranslate2 CPU throughput disappoints in STT-2.
- **TRAINING CANDIDATE**: `openai/whisper-small` (HF transformers) + PEFT LoRA —
  first-party docs, fits constrained GPUs; warm-start option
  Dafisns/whisper-turbo-multilingual-fleurs for `turbo`-class later. Convert
  back to CT2 for serving. On this 4 GB machine, local training is pilot-scale
  only; remote training + local inference is the realistic production path.
- **LIGHTWEIGHT INFERENCE CANDIDATE**: faster-whisper `base`/`tiny` int8 CPU
  for lowest-latency smoke/dev loops; whisper.cpp `small` q5 + VAD +
  `whisper-stream` if live streaming becomes a requirement.

## Fine-Tuning Strategy (STT-4 plan — do NOT execute in STT-0)

- **Base model**: `openai/whisper-small` (spec helpers and language coverage
  with trainable size). Consider `base` if VRAM forces it; `medium` only on a
  rented GPU.
- **Objective**: standard seq2seq ASR cross-entropy on (audio, verbatim spoken-form
  transcript). No text normalization of numbers — the model must learn to emit
  "dua ratus lima puluh juta".
- **PEFT vs full**: LoRA (r=32, alpha=64, dropout 0.05, q/k/v/o projections)
  first; full fine-tune only if LoRA under-delivers on entity terms AND a
  bigger GPU is available.
- **Precision**: fp16 (Pascal lacks bf16). Mixed precision via HF Trainer
  `fp16=True`; watch loss-scale underflow on tiny batches.
- **Batching**: micro-batch 1–2 + gradient accumulation 8 (effective 8–16),
  gradient checkpointing on.
- **Checkpoints**: save every 200 steps, evaluate every 100, keep best-3 by
  validation WER; early stop after 5 evaluations without improvement.
- **Overfitting risks**: small speaker pool is the biggest one — enforce
  minimum 20 speakers before training (see DATASET-GUIDE staged targets);
  monitor train/val WER divergence per speaker.
- **Speaker diversity requirement**: hard gate, not a guideline — no training
  run with <20 speakers.
- **Noise augmentation**: additive noise + RIR only after a clean pilot run
  reproduces; never let augmentation mask recording-quality problems.
- **Validation frequency**: every 100 steps on the fixed speaker-disjoint
  validation split; track entity-term accuracy, not just WER.
- **Conversion**: after training, `ct2-transformers-converter` → serve with the
  existing faster-whisper adapter (no service code changes).

## Staged Data Quantity Targets

Detailed in [DATASET-GUIDE.md](DATASET-GUIDE.md). Summary:

| Stage | Speakers | Hours | Purpose |
|---|---|---|---|
| Pilot (STT-2) | 4–6 | 1–2 | Baseline benchmark of pretrained models on real domain speech |
| MVP domain (STT-3/4) | ≥20 | 10–20 | First LoRA fine-tune with speaker-disjoint splits |
| Production-quality | ≥50 | 40+ | Multi-device, multi-condition coverage for deployment claims |

Quality and speaker diversity beat raw hours; these are floors, not goals to
pad with synthetic audio.
