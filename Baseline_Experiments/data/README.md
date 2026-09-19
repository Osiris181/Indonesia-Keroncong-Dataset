# Shared baseline data

`syllable_dataset.csv` is the common modeling index used by the vocal-
technique baselines. It is generated exclusively from the final files under `../../IKD_Dataset/` to serve an easier data loading; it is not an additional annotation source.

If you want to regenerate it, please rebuild it from the release root with:

```bash
python Baseline_Experiments/data/build_syllable_dataset.py
```

The builder verifies the published corpus statistics: 30 tracks, 6,772
syllables, 2,676 syllables with at least one technique, 4,096 without a
technique, 736 multi-label syllables, and the support of every technique. It
also verifies that every track occurs in three training, one validation, and
one test partition across the five fixed runs.

Audio is deliberately not encoded as a machine-specific absolute path. The
`vocal_audio_relative_path` column stores `TRACK_ID/vocals.wav`; scripts that
need audio receive its parent directory through `--audio-root`.

The current `syllable_dataset.csv` can be used directly without requiring user to
run the reproduction code.