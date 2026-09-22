# Repository Location

## Canonical location (STT-1A, 2026-09-22)

```
E:\Reza(Jangan Dihapus!!!)\morves-local-stt
```

The repository was bootstrapped (STT-0) at
`E:\Reza(Jangan Dihapus!!!\morves-local-stt` — a parent directory missing its
closing parenthesis (typo). STT-1A relocated the complete repository
(`.git`, working tree, untracked local files, `.venv`, `models/` cache,
private manifests) to the canonical path above. A whole-directory rename was
blocked by a live workspace handle on the old folder, so the relocation was
performed as a **verified full copy** (every file, not just tracked files),
followed by post-move verification at the canonical path: same
branch/HEAD/origin, `git fsck` clean, full test suite green, ruff+mypy
clean — zero behavioral change.

## History (not falsified)

- **STT-0 (2026-09-22)**: all six bootstrap commits were authored at the old
  path. `docs/STT-0-BOOTSTRAP-REPORT.md` intentionally still records the old
  location; it is a historical document.
- **STT-1A (2026-09-22)**: relocation performed as a whole-directory
  move/copy. No history rewrite, no rebase, no force push.

## Old parent disposition

`E:\Reza(Jangan Dihapus!!!\morves-local-stt` (old repository folder) is
**retained** because the open editor workspace holds a handle on it (rename
is refused with "used by another process"). It is a transitional duplicate:
after closing any editor/session that has it open, delete the old
`morves-local-stt` folder manually once the canonical copy is confirmed
complete. The old parent directory itself is **not empty** and must not be
deleted either — it also contains the unrelated `morves-finance-core` folder
(a single stub file, not a git repository; the real one lives under the new
parent).

## Notes for future sessions

- Open the workspace at the canonical path only. The old path must never be
  used for new work; if both exist, the canonical one is authoritative
  (verify with `git remote -v` → `github.com/RezaAdityaRamadhan26/morves-local-stt`).
- If a `.venv` breaks after any future move (Windows venvs embed absolute
  paths in script shims), recreate it project-locally:
  `python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"` — never
  global installs, never committed.
