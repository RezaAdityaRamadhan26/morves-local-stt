# Integration Contract — morves-finance-core ↔ morves-local-stt

## Scope Boundary (absolute)

morves-local-stt does exactly one thing:

```
AUDIO → LOCAL ASR → TRANSCRIPT
```

It MUST NOT: connect to Morves PostgreSQL, execute financial queries, contain
Finance business logic, post or modify transactions, authenticate Morves users,
contain AI Assistant logic, or perform finance tool calling. Nothing in this
repository may grow those capabilities.

## Architecture

```
Morves Browser
   │  (audio capture, WAV PCM mono 16 kHz 16-bit)
   ▼
Morves Backend (morves-finance-core)
   │  SpeechToTextProvider adapter  (future provider id: "morves-local")
   ▼
morves-local-stt  (this repo — HTTP service)
   │
   ▼
TRANSCRIPT  →  returned to Morves Backend → Assistant/NLP layer
```

- The browser **never** calls local STT directly in the final architecture.
- Morves Backend owns: authorization, rate limiting, audio policy, provider
  selection, audit metadata.
- morves-local-stt owns: audio decoding, ASR inference, transcript result.
- No financial business logic crosses into local STT.

## Service API (implemented skeleton)

Base: `http://127.0.0.1:8000` (bind configurable in `configs/inference.yaml`)

| Endpoint | Method | Contract |
|---|---|---|
| `/health` | GET | liveness — 200 `{"status":"ok"}` regardless of model state |
| `/ready` | GET | readiness — 200 `{"ready":true,"model":...}` when the model is loaded; **503** `{"ready":false,"reason":...}` when it cannot load (health stays 200 — liveness and readiness are different) |
| `/v1/model` | GET | model metadata (id, adapter, device, compute type) |
| `/v1/transcriptions` | POST | multipart/form-data: `file` (WAV), optional `language` (default `id`) |

Response:

```json
{
  "text": "Tampilkan piutang Medita bulan ini.",
  "language": "id",
  "durationMs": 1320,
  "model": "morves-stt-baseline"
}
```

Errors are structured: `{"error": {"code": "...", "message": "..."}}` with
HTTP 400 (invalid audio / too long), 413 (payload too large), 415
(unsupported format/type).

## Service Security / Privacy Controls

- Never logs raw or base64 audio (hard-coded deny, not a config flag).
- Transcript logging disabled by default (`log_transcripts: false`).
- Bounded upload size (default 25 MB) and audio duration (default 120 s).
- Extension + MIME allowlist; canonical decode to 16 kHz mono PCM regardless.
- Uploads processed via temp files that are deleted in a `finally` block.
- No URL audio fetching, no shell execution, no filesystem path exposure, no
  database access, no audio persistence.

## Future Provider ID

Provider id candidate: **`morves-local`** (stable, lowercase, hyphenated).
The SpeechToTextProvider implementation lives in morves-finance-core and will
be built in a later Finance task against the HTTP contract above. This repo
does not implement, duplicate, or preempt Finance routing logic (no
nine-router replication here). Fallback/routing between `9router` and
`morves-local` is a Finance-side decision.

## Future A/B Evaluation

See [EVALUATION-STRATEGY.md](EVALUATION-STRATEGY.md): same private test set,
same harness, both providers, measured — not assumed — comparison.
