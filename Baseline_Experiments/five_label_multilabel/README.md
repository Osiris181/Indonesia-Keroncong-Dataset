# Five-label multi-label vocal-technique classification

This directory reproduces the paper comparison that predicts `luk`, `cengkok`,
`gregel`, `embat`, and `nggandhul` for every annotated syllable. A syllable
with none of these techniques is represented by an all-zero target vector.

The two configurations are:

- **MusicFM:** 1,024 cached framewise MusicFM-MSD layer-7 features.
- **MusicFM + pitch:** the same MusicFM features concatenated with three
  framewise TorchCREPE features, producing a 1,027-dimensional input.

Both configurations use the same folds, seeds, Conv1D classifier, loss,
optimizer, early stopping, and validation-threshold procedure. Their only
experimental difference is the inclusion of the three pitch channels.

## Temporal features

The frozen MusicFM encoder processes each HT-Demucs vocal stem at 24 kHz and
produces 1,024-dimensional layer-7 representations on a 25 Hz grid. Tracks are
processed as 30-second non-overlapping cores with five seconds of real context
on each available side. Context frames are discarded, and no artificial audio
padding is added at track boundaries. A non-learned LayerNorm is applied to
each saved MusicFM frame before training.

TorchCREPE processes the same vocal stem at 16 kHz with its `full` model, a
160-sample hop, a 50--1,000 Hz range, and deterministic Viterbi bin-center
decoding. This produces a 100 Hz contour. Every fourth pitch frame is selected
to obtain the exact MusicFM 25 Hz grid without interpolation. The three model
channels are:

1. track-relative F0 in semitones, computed relative to the median reliable F0
   and set to zero on unreliable frames;
2. continuous TorchCREPE periodicity; and
3. a binary reliability indicator defined by periodicity greater than or equal
   to 0.50.

The combined pipeline is:

```text
24 kHz vocal stem -> frozen MusicFM layer 7 at 25 Hz -> 1,024 channels
16 kHz vocal stem -> TorchCREPE at 100 Hz -> exact 25 Hz selection -> 3 channels
                                   |
                         optional concatenation
                                   |
                    residual dilated Conv1D
                                   |
                         five logits per frame
                                   |
              mean logits within each syllable interval
                                   |
                       five sigmoid probabilities
```

## MusicFM requirement

MusicFM is an external repository and must be supplied with `--musicfm-root`.
The expected checkout contains `model/musicfm_25hz.py`,
`data/pretrained_msd.pt`, and `data/msd_stats.json`. The published setup uses:

- MusicFM commit `b83ebedb401bcef639b26b05c0c8bee1dc2dfe71`;
- `pretrained_msd.pt` SHA-256
  `218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded`;
- `msd_stats.json` SHA-256
  `c36c61ab10ca4d2e7fdfefc3fcc15205316bec276a06a47baa3641a62c546f22`.

The bundled `musicfm_conformer_config.json` fixes the conformer architecture
without requiring a network lookup.

## Prepare the inputs

Run all commands from the repository root. First build the common syllable
index if it is not already present:

```bash
python Baseline_Experiments/data/build_syllable_dataset.py
```

Download and separate the audio as described in `IKD_Dataset/README.md`. Then
extract the two frozen framewise feature sources:

```bash
python Baseline_Experiments/five_label_multilabel/extract_musicfm.py \
  --musicfm-root /path/to/musicfm \
  --device cuda

python Baseline_Experiments/five_label_multilabel/extract_pitch.py \
  --device cuda

python Baseline_Experiments/five_label_multilabel/align_pitch.py
```

The scripts resume completed tracks by default. Use `--overwrite` only when an
intentional re-extraction is required. Generated arrays are written to
`outputs/features/` and are ignored by Git.

## Validate and train

List the locked configurations:

```bash
python Baseline_Experiments/five_label_multilabel/train.py --list-experiments
```

Validate the cached arrays, the 1,024-D and 1,027-D inputs, and all five split
rotations without training:

```bash
python Baseline_Experiments/five_label_multilabel/train.py \
  --all-enabled \
  --dry-run
```

Train all 30 models: two configurations, five fold rotations, and three seeds
(64, 128, and 256):

```bash
python Baseline_Experiments/five_label_multilabel/train.py \
  --all-enabled \
  --device cuda
```

A single model can be run with:

```bash
python Baseline_Experiments/five_label_multilabel/train.py \
  --experiment temporal_five_label_musicfm_pitch \
  --run 1 \
  --seed 64 \
  --device cuda
```

Training writes checkpoints, histories, per-syllable predictions, and run
summaries to `outputs/training/<EXPERIMENT>/run_<RUN>/seed_<SEED>/`.
Completed runs are skipped. A partial or intentionally repeated run requires
`--overwrite`.

## Training configuration

Both models use a 1x1 input projection to 256 channels and four residual
Conv1D blocks with 256 channels, kernel size 5, dilations 1, 2, 4, and 8,
ReLU, per-frame LayerNorm, and channel dropout 0.3. The final 1x1 convolution
produces five framewise logits. Logits whose frame centers fall in each
syllable interval are averaged before sigmoid activation.

Training uses whole-track batches of size one, AdamW with learning rate
`1e-3` and weight decay `1e-4`, a maximum of 100 epochs, and gradient clipping
at 5. Weighted binary cross-entropy uses one negative-to-positive weight per
label calculated only from the training partition. Early stopping monitors
validation mAP with patience 12 and minimum improvement `1e-4`. One threshold
per label is selected on validation data by maximizing F1 over 0.05 to 0.95 in
steps of 0.01. Test data is evaluated only after checkpoint and threshold
selection.

## Aggregate results

After training, run:

```bash
python Baseline_Experiments/five_label_multilabel/report_results.py
```

The command prints MusicFM first and MusicFM + pitch second, following the
paper table. Reproduced reports are written to `outputs/evaluation/`; the
tracked `results/` directory preserves the values reported in the paper and
is never overwritten by the default command. The reporter also writes one
row-normalized pairwise prediction-association matrix and one pooled raw-count
matrix for each configuration.

The reported means across 15 models per configuration are:

| Setting | Macro P | Macro R | Macro F1 | mAP |
|---|---:|---:|---:|---:|
| MusicFM | 0.2769 | 0.4811 | 0.3294 | 0.2878 |
| MusicFM + pitch | 0.2724 | 0.5199 | 0.3326 | 0.2901 |

## Numerical reproducibility note

The feature extraction and model-training settings are fully specified and the
training stage is deterministic for a fixed feature set. Fresh HT-Demucs
separation can nevertheless produce slightly different vocal stems because
the released separation configuration uses one randomized shift. Consequently,
a complete reconstruction from the source videos can produce small numerical
differences while following the same methodology. The exact historical paper
values are preserved under `results/`.
