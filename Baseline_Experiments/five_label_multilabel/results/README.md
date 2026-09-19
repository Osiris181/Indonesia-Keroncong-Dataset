# Reported five-label results

These files contain the paper results for the selected cached-frame comparison:

- `temporal_five_label_musicfm`: 1,024 framewise MusicFM channels;
- `temporal_five_label_musicfm_pitch`: the same channels plus track-relative
  semitone, TorchCREPE periodicity, and binary pitch reliability.

Each metric is averaged across 15 independently trained models: five fixed
performer-grouped fold rotations multiplied by seeds 64, 128, and 256. Label
support counts unique test syllables across the five folds and is therefore
not multiplied by the three seeds.

- `overall_metrics.csv` contains the overall means and sample standard
  deviations.
- `per_label_metrics.csv` contains per-technique means, sample standard
  deviations, validation-selected thresholds, and support.
- `experiment_summary.txt` provides the same metrics in a readable text form.
- `pairwise_<setting>_row_normalized_percent.csv` contains the mean pairwise
  prediction-association percentages across the 15 models.
- `pairwise_<setting>_pooled_raw_counts.csv` contains the corresponding counts
  pooled across all 15 test predictions.

In each pairwise matrix, rows are ground-truth techniques and columns are
predicted techniques. Diagonal cells represent true positives. Off-diagonal
cells count prediction of the column technique only when that technique is
absent from a syllable containing the row technique; correctly predicted
co-labels are not treated as errors. Percentage rows are normalized by the
ground-truth support within each model before averaging. Therefore, because
this is a multi-label task, a row can sum to more than 100 percent.

The files were regenerated from the completed per-model predictions using
`../report_results.py`. By default, that evaluator writes new reports under
`../outputs/evaluation/`, so running it cannot overwrite these tracked paper
results.
