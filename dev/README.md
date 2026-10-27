# dev/ — verification only

**These files are not part of any experiment.** They exist only to check that
the extraction module computes exactly what `docs/MAPPING.md` specifies. No
number derived from them may appear in the paper, and the β* calibrated on
`dev_warmup.jsonl` is a verification value only: the definitive calibration is
run once the datasets are fixed.

| File | Content |
|---|---|
| `dev_questions.jsonl` | 24 hand-written `{id, question}`: 8 well-known facts (`dev-known-*`), 8 little-known facts (`dev-obscure-*`), 8 misconception-style questions (`dev-misc-*`) |
| `dev_warmup.jsonl` | 12 different hand-written questions (`dev-warm-*`), disjoint from the above, to exercise `calibrate` |
| `reference.py` | deliberately naive float64 re-implementation of every quantity (explicit loops, no `energy_llm` code), used by `energy-llm verify` and `tests/test_reference.py` |

See `hpc/RUNBOOK.md` for the verification sequence on the cluster.
