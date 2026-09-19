# Indonesian Keroncong Dataset (IKD)

This directory is the reproducibility release for the Indonesian Keroncong
Dataset (IKD). IKD provides metadata and annotations for 30 keroncong
recordings totaling 2.41 hours. The release contains manually reviewed word-
and syllable-level lyric boundaries, expert syllable-level annotations of
`luk`, `cengkok`, `gregel`, `embat`, and `nggandhul`, and fixed
performer-grouped evaluation splits.

It also contains the implementations and reported results for three baseline
tasks:

- five-label multi-label vocal-technique classification;
- lyrics transcription; and
- lyrics alignment.

## Repository structure

```text
Indonesia-Keroncong-Dataset/
|-- IKD_Dataset/
|   |-- Song_Annotations/
|   |-- Melisma_Events/
|   |-- Splits/
|   |-- Scripts/
|   `-- Audio/                 # generated locally and ignored by Git
|-- Baseline_Experiments/
|   |-- data/
|   |-- five_label_multilabel/
|   |-- lyrics_transcription/
|   `-- lyrics_alignment/
|-- requirements.txt
`-- README.md
```

See [IKD_Dataset/README.md](IKD_Dataset/README.md) for the dataset schema,
annotation conventions, statistics, audio preparation, and fold definitions.
See
[Baseline_Experiments/README.md](Baseline_Experiments/README.md)
for an overview of the benchmark implementations.

## Installation

The reference environment used Python 3.12.11 and `uv` 0.9.7. Create the
virtual environment and install the pinned dependencies with `uv`:

```bash
uv venv --python 3.12.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

FFmpeg must also be available as a system executable. GPU experiments require
an NVIDIA driver and CUDA installation compatible with PyTorch. The reported
experiments used PyTorch 2.8.0 with CUDA 12.8; users on other platforms may
need to install the corresponding PyTorch build separately.

The five-label vocal-technique baseline additionally requires the external
MusicFM repository and MSD checkpoint. Its README records the expected MusicFM
revision and checkpoint hashes.

## Prepare the audio

The audio recordings are not redistributed. From this directory, reconstruct
the original mixes from the source URLs recorded in the metadata:

```bash
python IKD_Dataset/Scripts/download_script.py
```

Then generate the vocal and accompaniment stems with HT-Demucs:

```bash
python IKD_Dataset/Scripts/demucs_separation.py
```

The scripts create the following layout:

```text
IKD_Dataset/Audio/
`-- IKD_001/
    |-- original_mix.wav
    |-- vocals.wav
    `-- no_vocals.wav
```

Both scripts resume completed tracks and support `--track-ids` for processing
a subset. HT-Demucs uses CUDA by default; pass `--device cpu` when necessary.

## Run the baselines

The common syllable index is already included. It can be regenerated from the
released annotations and splits with:

```bash
python Baseline_Experiments/data/build_syllable_dataset.py
```

Task-specific setup, training or inference commands, evaluation procedures,
and reported metrics are documented in:

- [Five-label multi-label classification](Baseline_Experiments/five_label_multilabel/README.md)
- [Lyrics transcription](Baseline_Experiments/lyrics_transcription/README.md)
- [Lyrics alignment](Baseline_Experiments/lyrics_alignment/README.md)

Use `IKD_Dataset/Audio` as `--audio-root` when running a baseline.

## Generated files

Downloaded audio and separated stems under `IKD_Dataset/Audio/` are ignored
by Git. Baseline features, checkpoints, predictions, and detailed evaluations
are written under each task's `outputs/` directory and are also ignored.
Compact tables containing the paper results remain tracked under each task's
`results/` directory.

Availability and content at external media URLs may change. Users are
responsible for complying with the source platform terms and applicable
copyright and licensing conditions.
