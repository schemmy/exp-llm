# MLE Transition — Entry Point

**Owner**: Chenxin Ma
**Started**: 2026-08-30
**Not job-search-urgent** — this is long-term skill build alongside the Databricks day job.

---

## What this is

12-month plan to reposition from Senior DS (Databricks Serverless) to **ML Platform/Infra + Inference/Training MLE**. Structured as 4 flagship projects, each producing a public GitHub repo + technical blog on chenxin-ma.github.io.

**Not**:
- A job application plan (that's `~/projects/job_hunting/`, currently paused)
- An applied-ML / model-training plan (already halfway there via the Databricks Migration Prediction Model)

---

## File map

| File | Purpose | When to read |
|---|---|---|
| `README.md` (this file) | Entry-point map | First |
| `progress.md` | Weekly status log (52 rows) | Every session — tells you current week |
| `weekXX.md` | Concrete task list for one week | The current week + upcoming |
| `PLAN.md` | 12-month arc, 4 phases, decision points | Once at start; revisit at phase boundaries |
| `resources.md` | GPU providers, papers, courses, OSS targets | On demand |

Weekly files (`week01.md`, `week02.md`, ...) are added as needed — not all 52 exist yet.

---

## Assistant instructions (for future sessions)

When Chenxin references "week N", "the transition", "vLLM benchmark", "Modal GPU", or similar:

1. **Read `progress.md` first** — get current week + status
2. **Read `weekXX.md`** for the current week — this is the working document
3. Only touch `PLAN.md` if he asks about future phases or arc-level questions
4. Only touch `resources.md` if he asks about tools / links / papers
5. When he finishes a week, help update `progress.md` (status column + artifact link)
6. When starting a new week, create `weekXX+1.md` from the template of the previous week

**Assumed context from memory** (`project_mle_transition.md`):
- Direction chosen: ML Platform/Infra + Inference/Training MLE (not applied ML)
- Prior CUDA state exists at `~/projects/fun_cuda_kernals` (steps 01-05 + FA1 done on T4) — P4 resumes that repo
- **Never** frame P1-P4 public repos as "interview prep" (see `feedback_repo_tone.md`)

---

## Current state (as of 2026-09-04)

- **Phase**: P1 — Inference foundations
- **Week**: 9 of 52
- **This week's task**: tensor parallelism (see `week09.md`)
- **Blog #1 target date**: ~2026-12-05 (end of Wk 14)

For live status, always check `progress.md` — this section is a snapshot.

---

## How Chenxin should use this folder

- Open `weekXX.md` at the start of each week — that's your task list
- Update `progress.md` at end of each week (30 sec: ✅ or 🟡, add artifact link)
- Refer to `PLAN.md` at phase transitions (Wk 14, 26, 38) for decisions and re-scoping
- Add papers you actually read to `resources.md` — kill entries that turn out low-yield
