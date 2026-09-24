# Lyrics transcription

This folder implements the paper's zero-shot lyrics-transcription baseline for
all 30 IKD songs. It compares the complete original mix with the HT-Demucs
vocal stem using one fixed WhisperX/Faster-Whisper `large-v3` configuration.

## Method

WhisperX segments the complete audio with Pyannote VAD and decodes each segment
with Faster-Whisper `large-v3`. Both audio conditions use float16 CUDA
inference, batch size 16, a 30-second VAD chunk size, and the decoding values in
`config.json`. There is no initial prompt, hotword list, transcript correction,
or dataset-specific output filter. Indonesian tracks use language code `id`,
Javanese tracks use `jw`, and the mixed-language track uses automatic language
detection.

The ASR text is then aligned to the same audio with
`indonesian-nlp/wav2vec2-indonesian-javanese-sundanese`. This CTC step receives
only the predicted ASR text and produces the word timestamps needed for masked
evaluation. It does not receive reference lyrics, annotated word or syllable
boundaries, or annotated non-lexical intervals.

The reported run used:

- Python 3.12.11, PyTorch 2.8.0, WhisperX 3.8.6,
  Faster-Whisper 1.2.1, and CTranslate2 4.8.1;
- `Systran/faster-whisper-large-v3` revision
  `edaa852ec7e145841d8ffdb056a99866b5f0a478`;
- `indonesian-nlp/wav2vec2-indonesian-javanese-sundanese` revision
  `e5e699fa5aa5bdce999276a90a29589587b58ac9`.

Install WhisperX and the matching runtime versions in a CUDA environment. For
example:

```bash
uv pip install \
  whisperx==3.8.6 faster-whisper==1.2.1 ctranslate2==4.8.1
```

## References and scoring

`prepare_references.py` reads the released files under
`IKD_Dataset/Song_Annotations/`. Entries whose complete word text is
`[non-lexical]` after NFC normalization, trimming, and lowercasing are excluded
from the lexical reference. Their intervals are retained only for evaluation.
Ordinary vocal tokens such as `oh`, `o`, and `la` remain lexical.

The fixed normalization applied to references and hypotheses is:

- normalize Unicode to NFC and lowercase it;
- remove `'`, `’`, `‘`, `ʼ`, and `＇` without adding a word boundary;
- replace every other Unicode punctuation character with a space;
- collapse whitespace and trim its outer edges;
- preserve spelling, diacritics, digits, and non-punctuation symbols.

The tracked reference contains 2,943 lexical words and nine non-lexical
intervals across 30 tracks. Regenerate it from the repository root with:

```bash
python Baseline_Experiments/lyrics_transcription/prepare_references.py
```

WER uses normalized whitespace-separated tokens. CER uses normalized
non-whitespace Unicode characters. The primary values are corpus-level micro
rates, obtained by summing edits over all tracks before dividing by the summed
reference units. Word substitution, deletion, and insertion rates all use the
number of reference words as their denominator. Equal-cost Levenshtein paths
use the fixed priority substitution, deletion, then insertion.

Each prediction is scored under two policies:

- `masked`: discard a timestamped hypothesis word only when its temporal
  midpoint lies in an annotated non-lexical interval;
- `unmasked`: retain the complete full-audio hypothesis.

Untimestamped hypothesis words are retained under both policies. For a
recording with annotated non-lexical intervals, the evaluator requires the
aligned word sequence to reconstruct the raw ASR hypothesis after normalization
before applying the mask. When a recording has no non-lexical interval, the
masked and unmasked hypotheses are identical by definition; raw ASR words that
the CTC aligner could not timestamp are retained and counted as untimestamped.

## Run inference

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

From the repository root, run all 60 track-source jobs:

```bash
python Baseline_Experiments/lyrics_transcription/transcribe.py \
  --audio-root IKD_Dataset/Audio
```

To run one job with the same locked configuration:

```bash
python Baseline_Experiments/lyrics_transcription/transcribe.py \
  --audio-root IKD_Dataset/Audio \
  --track-id IKD_001 \
  --audio-source vocal_stem
```

Predictions are written to `outputs/predictions/original_mix/` and
`outputs/predictions/vocal_stem/`. Completed outputs with the same
configuration fingerprint are skipped. Use `--overwrite` to replace them.
The `outputs/` directory is ignored by Git.

## Evaluate

After all predictions finish, please run:

```bash
python Baseline_Experiments/lyrics_transcription/evaluate.py
```

The output files of `metrics.json`, `per_track_metrics.csv`,
`overall_metrics.csv`, and `summary.txt` will be written in `outputs/evaluation/` directory.
The original paper results that were noted in `results/` are:

| Audio input | Policy | WER | CER | Substitution rate | Deletion rate | Insertion rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original mix | Masked | 0.6391 | 0.4834 | 0.2263 | 0.4030 | 0.0099 |
| Original mix | Unmasked | 0.6425 | 0.4881 | 0.2416 | 0.3911 | 0.0099 |
| Vocal stem | Masked | 0.3761 | 0.1418 | 0.2779 | 0.0601 | 0.0381 |
| Vocal stem | Unmasked | 0.3901 | 0.1535 | 0.2803 | 0.0581 | 0.0516 |
