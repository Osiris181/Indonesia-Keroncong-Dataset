# Lyrics alignment

This folder implements the paper's reference-text lyrics-alignment baseline
for all 30 IKD songs. The task compares the complete original mix with the
HT-Demucs vocal stem at both word and syllable levels.

No model is trained on IKD; all 30 recordings are evaluated directly without
fold rotation or random seeds.

## Task definition

The pipeline receives a complete song and the correct, ordered lexical lyrics.
It predicts where each provided word and syllable occurs:

```text
complete audio + provided lexical lyrics -> word and syllable timestamps
```

No annotated word or syllable timestamp is available to alignment inference.
The timing-free orthographic syllabification is supplied only to map the shared
character alignment into syllable units. No temporal regions are removed from
either condition: the original mix retains its accompaniment, and both
conditions retain annotated non-lexical passages.
However, `[non-lexical]` and its intervals are not provided to the model.
Consequently, a lexical unit placed inside such a passage remains an alignment
error; there is no masked alignment policy.

## Method

WhisperX loads each recording as mono 16-kHz audio and aligns one full-track
segment, from zero to the decoded duration, containing the complete normalized
lexical text. The frozen
`indonesian-nlp/wav2vec2-indonesian-javanese-sundanese` CTC model is loaded
with language code `id` for every recording. Character alignment is enabled,
and missing character times use nearest interpolation, following the values in
`config.json`. No boundary stretching or other post-processing is applied.

One character alignment is shared by both output levels. A predicted word
extends from its first to its last aligned character. A predicted syllable
extends from the first to last character assigned by the provided orthographic
syllabification; a second model pass is not performed.

The reference run used Python 3.12.11, WhisperX 3.8.6, PyTorch 2.8.0, and
Transformers 4.57.6. The model revision observed for that run was
`e5e699fa5aa5bdce999276a90a29589587b58ac9`. WhisperX's loader accepts the
model identifier but does not expose revision pinning here, so `config.json`
records that revision as provenance rather than claiming to enforce it.

## Prepare the alignment data

`prepare_data.py` reads only the final annotations under
`IKD_Dataset/Song_Annotations/` and writes two deliberately separate files:

- `data/alignment_inputs.json` contains ordered word and syllable spellings,
  but no timestamps or non-lexical intervals;
- `data/alignment_ground_truth.json` contains manually reviewed reference
  timestamps and is read only by the evaluator.

The fixed lexical normalization is shared with the transcription task: Unicode
is normalized to NFC and lowercased; apostrophes are removed without adding a
word boundary; other Unicode punctuation becomes spaces; whitespace is
collapsed; and spelling and diacritics are preserved. Complete
`[non-lexical]` entries are omitted from the provided lyrics.

From the repository root, regenerate the tracked inputs with:

```bash
python Baseline_Experiments/lyrics_alignment/prepare_data.py
```

The released data contain 30 tracks, 2,943 lexical words, 6,772 syllables, and
nine non-lexical intervals retained only in evaluation ground truth. The
preparation script validates these totals when reading the default released
annotations.

## Run alignment

Arrange both audio conditions beneath a user-selected root:

```text
<audio-root>/
├── IKD_001/
│   ├── original_mix.wav
│   └── vocals.wav
├── IKD_002/
│   ├── original_mix.wav
│   └── vocals.wav
└── ...
```

Run all 60 track-source jobs:

```bash
python Baseline_Experiments/lyrics_alignment/align.py \
  --audio-root IKD_Dataset/Audio
```

If you want to run a selected job with the same configuration:

```bash
python Baseline_Experiments/lyrics_alignment/align.py \
  --audio-root IKD_Dataset/Audio \
  --track-id IKD_001 \
  --audio-source vocal_stem
```

Predictions are written to `outputs/predictions/original_mix/` and
`outputs/predictions/vocal_stem/`. A completed job is skipped only when its configuration, timing-free input,
and audio-file hashes match. Use `--overwrite` to replace it.

## Evaluate

After all predictions finish, run:

```bash
python Baseline_Experiments/lyrics_alignment/evaluate.py
```

The evaluator writes `metrics.json`, `per_unit_metrics.csv`,
`summary_metrics.csv`, and `summary.txt` beneath `outputs/evaluation/`. It
reports coverage; onset and offset mean, median, and signed errors; onset,
offset, and joint correctness at 50, 100, 200, and 300 ms; interval IoU; and
the following paper metrics:

- **Onset MAE/MedAE:** mean/median absolute onset displacement in seconds.
- **PCO@0.3:** fraction of all reference units whose onset error is at most
  0.3 seconds. An unaligned unit counts as incorrect.
- **PCS (interval-overlap variant):** correctly overlapped unit and
  inter-unit-gap duration divided by the summed evaluated
  reference spans. The reported value is corpus-pooled and duration-weighted,
  not an unweighted mean over tracks.

The original paper results written in `/results` directory are:

| Audio input | Level | Onset MAE (s) | Onset MedAE (s) | PCO@0.3 | PCS |
| --- | --- | ---: | ---: | ---: | ---: |
| Original mix | Word | 60.244 | 50.216 | 0.1329 | 0.1112 |
| Original mix | Syllable | 61.537 | 51.744 | 0.1260 | 0.0963 |
| Vocal stem | Word | 4.571 | 0.040 | 0.7635 | 0.6374 |
| Vocal stem | Syllable | 4.525 | 0.048 | 0.7448 | 0.5624 |

## Annotation-model provenance

The same CTC model previously proposed syllable character timings within
already manually verified word intervals during annotation. Those proposals
were subsequently reviewed and corrected manually. The released boundaries
are therefore human-verified, and this baseline is less constrained because it
aligns one full track at a time. Consequently, its syllable-level result
should not be interpreted as an
evaluation against reference boundaries developed entirely independently of
the evaluated model. This provenance should be disclosed when reporting the
result.
