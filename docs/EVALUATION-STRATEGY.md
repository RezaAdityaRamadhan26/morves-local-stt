# Evaluation Strategy

## Principles

1. **No claims without evidence** — no "excellent Indonesian accuracy", no
   "beats 9Router", no "production ready" until measured on the private
   real-speech test set.
2. **WER alone lies in this domain.** A transcript with WER 5% that says
   "media" instead of "Medita", or "miliar" instead of "juta", is a business
   failure. Domain metrics make those errors visible.
3. **Speaker-disjoint test data or the numbers are fiction.**

## Metrics (implemented in `src/morves_stt/metrics.py`)

| Metric | Definition |
|---|---|
| WER | (S+D+I)/N over normalized tokens, Levenshtein alignment with backtrace |
| CER | same at character level (whitespace removed) |
| Entity Term Accuracy | occurrence-based accuracy over entity terms (Medita, Morves, Creatifyl, BCA, Mandiri...) |
| Financial Term Accuracy | over finance vocabulary (piutang, neraca saldo, jatuh tempo...) |
| Amount Token Accuracy | over amount marker tokens (ribu, juta, miliar, belas, puluh, koma...) |
| Date Token Accuracy | over date markers (month/day names, tanggal, bulan, tahun...) |
| Currency Token Accuracy | over currency tokens (rupiah, dolar, IDR, USD) |
| Domain Phrase Accuracy | exact normalized match rate, reported per domain |
| Critical Term Errors | explicit confusion counts (see below) |

Term categories are occurrence-based and position-blind in STT-0
(`correct = min(expected, predicted)` per term) — a conservative proxy until
forced-alignment tooling arrives. Multi-word terms ("neraca saldo") are matched
as sliding windows.

### Critical Term Error Model

Confusion pairs (configurable in `evaluation/fixtures/terms.yaml`) are counted
explicitly and never hidden inside aggregate WER:

- `medita → media/berita/medika` — entity corruption
- `morves → moris/modus` — entity corruption
- `creatifyl → kreatif/creative` — entity corruption
- `piutang ↔ utang/hutang` — financial semantic flip
- `debit ↔ kredit` — accounting direction flip
- `juta ↔ miliar ↔ ribu` — amount magnitude error
- `IDR ↔ USD`, `rupiah ↔ dolar` — currency error
- deletion of any critical term (entity silently missing) is also a critical error

Example: reference "Tampilkan piutang Medita" vs hypothesis "Tampilkan piutang
media" scores WER 25% but Entity Term Accuracy 0% with one critical event.

## Normalization Rules (exact)

Applied before all metric comparisons (`morves_stt.normalize`):

1. Unicode NFKC.
2. Lowercase.
3. Strip punctuation/symbols (anything not letter/digit/whitespace/hyphen);
   hyphens are word boundaries (antar-perusahaan == antarperusahaan).
4. Collapse whitespace.

**Never normalized** (semantic, must remain distinct):
digits vs number words ("12" ≠ "dua belas"), `debit`/`kredit`,
`juta`/`miliar`, `Medita`/`media`, `rupiah`/`dolar`, `Rp250.000.000` vs
`dua ratus lima puluh juta rupiah`. Presentation-only collapse (removing "."
inside digit strings) is allowed.

## Operational Metrics

Recorded per benchmark run: `latency_ms` (mean/p50/p95), `audio_duration`,
`real_time_factor` (processing_time / audio_duration), utterance and skip
counts. RAM/VRAM peak capture is best-effort and currently deferred (tooling
note in STT-0 report limitations).

## Fixtures

- Committed: text-only term lists and sentence bank
  (`evaluation/fixtures/`) — no audio in Git, ever.
- Tests use locally generated synthetic signals (sine/noise).
- Real benchmarks read `evaluation/private-manifest.jsonl` (gitignored) via
  `scripts/benchmark_baseline.py`; without it the runner reports BLOCKED rather
  than inventing numbers. `--smoke` runs the pipeline on synthetic audio for
  latency/infrastructure verification only and labels reports accordingly.

## Future A/B: Morves Local STT vs 9Router STT

Same private test set, same normalization, same metric implementation, run
back-to-back. Compare: WER, CER, entity accuracy, finance-term accuracy,
amount/date/currency accuracy, latency, RTF, and the qualitative
privacy/offline characteristics (local: no audio leaves the machine; cloud
router: does). No superiority claim before those numbers exist.

## Roadmap

STT-2 runs the pretrained baseline benchmark on the pilot dataset; STT-5
re-runs the identical harness on the fine-tuned model. The harness does not
change between stages — that is what makes the comparison honest.
