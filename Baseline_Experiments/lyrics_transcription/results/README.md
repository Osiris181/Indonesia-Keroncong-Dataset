# Published lyrics-transcription results

`overall_metrics.csv` contains the four corpus-level rows reported for the
zero-shot lyrics-transcription baseline: original mix and vocal stem, each with
masked and unmasked scoring. `summary.txt` is a four-decimal rendering of the
same values.

WER, CER, and the word substitution, deletion, and insertion rates are micro
rates over all 30 songs. The three word edit rates use the number of reference
words as their common denominator. CER excludes whitespace from its character
sequence. Predictions and per-track evaluation details are reproducible with
the task scripts and are written under the gitignored `outputs/` directory.
