# Indonesian Keroncong Dataset

This folder contains the final release of the 30-song Indonesian Keroncong
Dataset (IKD), including track metadata, word- and syllable-level annotations,
vocal-technique events, rhythmic-pattern annotations, performer-group metadata,
and fixed evaluation splits.

## Use these files

- `Indonesian_Keroncong_Dataset.csv` is the 30-track index.
  Its `annotation_file` column points to the full annotation JSON for each
  track. Its `melisma_events_file` and `melisma_event_ids` fields point to the
  final human-reviewed melisma events. Its `rhythmic_patterns_file` field
  points to the section-level rhythmic-pattern annotation for each track.
- `Song_Annotations/` contains one complete JSON file per song. Each file has
  metadata, audio details, words, their nested syllables, melisma labels, Luk
  directions, and the derived event list.
- `Melisma_Events/` contains one compact, final melisma-event JSON file per
  song.
- `Rhythmic_Pattern_Annotations/` contains one section-level rhythmic-pattern
  JSON file per song.
- `Indonesian_Keroncong_Dataset_events.jsonl` combines every
  reviewed event into one JSON Lines file.
- `performer_group_resolution.csv` preserves the source artist value and the
  curated `performer_group_id` used for leakage-safe evaluation.
- `Splits/` contains the fixed-seed five-fold performer-grouped,
  multi-label-balanced train/validation/test partitions.
- `Scripts/download_script.py` reconstructs the original-mix audio from the
  source URLs in the completed metadata.
- `Scripts/demucs_separation.py` applies the same HT-Demucs vocal separation
  used by the released baselines.

The `lyrics` column in the completed CSV intentionally retains its existing
legacy value. Use `annotation_file` as the canonical full-annotation path.

## Audio preparation

The audio recordings are not redistributed in this repository. Install the
preparation dependencies with `uv` and ensure that the `ffmpeg` executable is
available:

```bash
uv pip install yt-dlp pandas tqdm demucs
```

If you encounter an HTTP 403 error, update `yt-dlp` before downloading, especially when reusing an
existing environment:

```bash
uv pip install --upgrade yt-dlp
```
Then run the download script from the repository root:

```bash
python IKD_Dataset/Scripts/download_script.py
```

The downloader reads `Indonesian_Keroncong_Dataset.csv` by
default and creates one original mix per track. Next, create the vocal and
accompaniment stems:

```bash
python IKD_Dataset/Scripts/demucs_separation.py
```

HT-Demucs uses CUDA by default. Pass `--device cpu` when CUDA is unavailable.
Both commands support `--track-ids` for a selected subset. The resulting layout
is:

```text
IKD_Dataset/
└── Audio/
    └── IKD_001/
        ├── original_mix.wav
        ├── vocals.wav
        └── no_vocals.wav
```

`no_vocals.wav` is the residual accompaniment stem produced automatically by
HT-Demucs in two-stem vocal-separation mode. We did not use this residual accompaniment
in our experiments

Pass `IKD_Dataset/Audio/` as `--audio-root` to the baseline scripts. The
directory is excluded by the release `.gitignore`. Availability and content at external
URLs may change, and users are responsible for observing the source platform
terms and applicable copyright conditions.

## Annotation conventions

- Stable `source_region_id` values link words, syllables, and technique events
  across the released annotation files.
- A syllable may have multiple melisma labels. Every positive technique creates
  one event, ordered by syllable time and then by: `luk`, `cengkok`, `gregel`,
  `nggandhul`, `embat`.
- `no_melisma` remains on the syllable annotation but creates no event.
- `luk_direction` is present only on a `luk` event when the expert selected a
  direction. A missing direction is represented by `null`.
- Event IDs are new deterministic IDs such as `IKD_001_E0001`, so they do not
  collide with legacy automatic event IDs.
- Legacy automatic-proposal attributes such as `semitone_range`,
  `type_suggested`, confidence, and beat/pitch measurements are intentionally
  excluded from these final reviewed events.

## Dataset statistics

The final release contains 2,943 lexical words, nine non-lexical segments,
6,772 syllables, 3,559 reviewed vocal-technique events, and 1,590 explicit Luk
directions. Of the syllables, 2,676 contain at least one vocal technique,
4,096 contain no technique, and 736 contain multiple technique labels.

| Technique | Number of annotations |
|---|---:|
| `luk` | 1,658 |
| `cengkok` | 374 |
| `gregel` | 613 |
| `embat` | 563 |
| `nggandhul` | 351 |
| **Total** | **3,559** |

The rhythmic annotations contain 86 section annotations: 35 `engkel`, 31
`double`, and 20 `other` label occurrences.

The CSV `track_duration_sec` values are synchronized to the duration of the
annotated source audio; the original source CSV remains unchanged.

## Performer groups and folds

The completed CSV and per-song metadata contain a curated
`performer_group_id`. Confirmed aliases and corrections are recorded in
`performer_group_resolution.csv`; the original CSV artist value remains in
`artist_source_value` for traceability. There are 18 resulting performer
groups.

The `Splits/` folder provides five fixed performer-grouped base folds with a
cyclic train, validation, and test rotation. Every train, validation, and test
partition includes examples of all five positive techniques. See
`Splits/README.md` for the exact protocol and the limitation for source-label
proxy groups whose exact singer is unknown.

