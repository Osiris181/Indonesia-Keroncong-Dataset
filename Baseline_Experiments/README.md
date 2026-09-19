# IKD baseline experiments

This directory contains one self-contained entry point for each benchmark in
the IKD paper. Shared dataset and reproducibility utilities live in `data/`
and `common/`; task-specific models, configurations, generated outputs, and
compact result tables remain inside each task directory.

Current task directories:

- `five_label_multilabel/`: residual Conv1D comparison using cached framewise
  MusicFM layer-7 features with and without three aligned TorchCREPE pitch
  channels.

- `lyrics_transcription/`: zero-shot WhisperX/Faster-Whisper `large-v3`
  transcription of original mixes and vocal stems, with masked and unmasked
  lexical scoring.

- `lyrics_alignment/`: frozen CTC alignment of provided lexical lyrics to
  original mixes and vocal stems, evaluated at word and syllable levels.

Generated audio features, model checkpoints, prediction tables, and training
histories are stored under each task's `outputs/` directory and are intentionally
ignored by Git. Compact metrics reported with the release are stored under the
corresponding `results/` directory.

The classification experiment uses the fixed performer-grouped rotations in
`../IKD_Dataset/Splits/`. Task-specific differences are recorded explicitly
instead of being hidden in a single multi-purpose training script.

