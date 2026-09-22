# Model Card — <model name / version>

> Template for future fine-tuned Morves STT models (STT-4 onward).
> Fill every section; "N/A" must be justified.

## Base Model
- Architecture / checkpoint fine-tuned from (e.g. openai/whisper-small)
- Runtime format (CTranslate2 int8, ONNX, ...)

## License
- Base model license + this checkpoint's license
- Commercial/internal deployment confirmation

## Training Dataset Summary
- Source (recorded sessions; any public corpora mixed in)
- Total hours train/validation/test
- Speaker count and demographics summary (aggregate only, no identities)
- Recording conditions (devices, environments, noise profiles)

## Languages
- Primary: Indonesian (id); note any mixed EN/ID handling

## Domain
- Morves Finance terminology coverage (domains present, entity terms)

## Hardware
- Trained on (GPU, VRAM, precision)
- Inference requirements measured (CPU int8 / GPU fp16, RAM)

## Training Method
- Objective, PEFT/LoRA config or full fine-tune, optimizer, LR schedule,
  epochs/steps, augmentation applied

## Evaluation Results
- WER / CER (speaker-disjoint test set; state exact test set composition)
- Entity / finance-term / amount / date / currency accuracies
- Critical confusion counts (medita→media etc.)
- Latency p50/p95, RTF

## Known Limitations
- Accents/dialects underrepresented
- Noise conditions where quality degrades
- Number/date/currency edge cases observed failing

## Bias / Coverage Limitations
- Speaker demographic skews, device skews, register coverage gaps

## Intended Use
- Indonesian finance-domain voice commands to Morves Finance via the
  SpeechToTextProvider pipeline

## Prohibited Use
- Any use producing legal/financial decisions without human review
- Speaker identification or voice profiling of users
- Processing audio without user awareness/consent
- Languages/domains outside evaluated coverage

## Privacy Considerations
- Training data provenance and consent basis
- What the model can memorize/leak (transcripts only; no speaker identity use)
- Deployment privacy posture (local-only processing)
