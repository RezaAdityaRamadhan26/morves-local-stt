# Evaluation Harness

This directory contains evaluation fixtures, benchmark reporting templates, and private benchmark results.

## Layout

```
evaluation/
├── README.md                  <- This document
├── fixtures/
│   ├── terms.yaml             <- Canonical domain entities, financial terms, and confusion pairs
│   └── sentence_bank.jsonl    <- Curated reference text sentences across 15 domains
├── reports/
│   ├── .gitkeep
│   └── <timestamp>-*.json     <- [GITIGNORED] Machine-readable benchmark outputs
│   └── <timestamp>-*.md       <- [GITIGNORED] Human-readable benchmark summaries
└── private-manifest.jsonl     <- [GITIGNORED] Local-only manifest pointing to real private audio
```

## Running the Benchmark

When real private audio is available locally:

1. Create `evaluation/private-manifest.jsonl` matching the manifest schema.
2. Run:
   ```bash
   python scripts/benchmark_baseline.py --config configs/baseline.yaml
   ```
3. A JSON report and a Markdown summary are written to `evaluation/reports/`.

If no private audio is yet available, the benchmark script reports `BLOCKED/SKIPPED` with an exact explanation. It NEVER generates fake benchmark numbers.
