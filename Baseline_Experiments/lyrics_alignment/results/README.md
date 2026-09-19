# Released alignment results

`overall_metrics.csv` contains the corpus-pooled word- and syllable-level
metrics reported in the paper. `summary.txt` presents the same values rounded
for convenient inspection.

PCS is the interval-overlap unit-and-inter-unit-gap variant defined in the task
README and evaluator.

These compact results are tracked in Git. Re-running `evaluate.py` writes the
full per-unit, per-track, language-group, and overall results beneath
`../outputs/evaluation/`; that generated directory is intentionally ignored.
