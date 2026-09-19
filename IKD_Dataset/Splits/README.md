# Performer-Grouped Multi-Label Folds

These are the fixed-seed, five-fold cross-validation partitions for the IKD
low-resource ornament-classification experiment. The primary split follows a
complete-fold rotation: every song is used for training three times,
validation once, and testing once.

## Split policy

- The split unit is `performer_group_id`, not an individual syllable or track.
- The 18 performer groups are assigned once to five fixed base folds. Every
  performer group, and therefore every song, stays in the same base fold.
- In run `k`, base fold `k` is the test set and the next base fold cyclically
  is the validation set. The other three complete base folds are training.
- Consequently, every song and performer group has exactly three `train`, one
  `validation`, and one `test` role across the five runs.
- Training, validation, and test are performer-disjoint within every run.
- The five positive targets remain `luk`, `cengkok`, `gregel`, `embat`, and
  `nggandhul`.
- `no_melisma` is an all-zero target. Its count and total syllable count were
  secondary fold-balancing statistics; it is not a sixth model target.

Every train, validation, and test partition contains positive examples of all
five techniques.

The exact rotation is:

| Run | Train base folds | Validation base fold | Test base fold |
|---|---|---|---|
| 1 | 3, 4, 5 | 2 | 1 |
| 2 | 1, 4, 5 | 3 | 2 |
| 3 | 1, 2, 5 | 4 | 3 |
| 4 | 1, 2, 3 | 5 | 4 |
| 5 | 2, 3, 4 | 1 | 5 |

Because performer groups cannot be divided merely to equalize song counts,
the five runs contain 17--19 training songs, 5--7 validation songs, and 5--7
test songs.

## Files

- `five_fold_song_groups.csv` is the simple teacher-review table. It has one
  row per song and records only its fixed base fold plus identifying metadata;
  it does not assign training, validation, or test roles.
- `track_fold_assignments.csv` has one row per track, per outer fold, with its
  fixed `base_fold` and its `train`, `validation`, or `test` role in that
  run. Here, `outer_fold` is the run number and `base_fold` is the song's
  permanent group. This is the corrected primary filter.
- `fold_summary.csv` reports group, track, syllable, label, and genre counts
  for each role in each outer fold.
- `performer_group_summary.csv` aggregates the same counts by performer group.
- `split_manifest.json` records the deterministic seed, balance objective,
  label order, configuration, and complete group assignments.
- `split_validation.json` records structural and label-coverage validation.

## Performer metadata

The parent folder's `performer_group_resolution.csv` preserves the original
source artist value alongside the curated `performer_group_id` used here.
Some YouTube channel, band, or multi-vocalist labels remain source-label proxy
groups because the exact individual singer is not known. Do not claim stronger
singer identification for those groups without later metadata verification.
