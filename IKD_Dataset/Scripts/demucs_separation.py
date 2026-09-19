#!/usr/bin/env python3
"""Create IKD vocal and accompaniment stems with HT-Demucs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
DEFAULT_AUDIO_ROOT = DATASET_DIR / "Audio"
MODEL_NAME = "htdemucs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=DEFAULT_AUDIO_ROOT,
        help="Root containing <track_id>/original_mix.wav.",
    )
    parser.add_argument(
        "--track-ids",
        nargs="+",
        default=None,
        help="Optional track subset, for example: IKD_001 IKD_002.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device used by Demucs (default: cuda).",
    )
    parser.add_argument(
        "--force-separation",
        action="store_true",
        help="Regenerate vocals.wav and no_vocals.wav when both already exist.",
    )
    return parser.parse_args()


def discover_tracks(
    audio_root: Path,
    requested: list[str] | None,
) -> list[tuple[str, Path]]:
    available = {
        track_dir.name: track_dir / "original_mix.wav"
        for track_dir in sorted(audio_root.glob("IKD_*"))
        if track_dir.is_dir() and (track_dir / "original_mix.wav").is_file()
    }
    if not available:
        raise FileNotFoundError(
            f"No <track_id>/original_mix.wav files were found under {audio_root}"
        )

    if requested is None:
        track_ids = sorted(available)
    else:
        track_ids = sorted(set(requested))
        missing = sorted(set(track_ids) - set(available))
        if missing:
            raise FileNotFoundError(
                f"Missing original mixes for: {', '.join(missing)}"
            )

    return [(track_id, available[track_id]) for track_id in track_ids]


def run_separation(
    track_id: str,
    input_path: Path,
    device: str,
    force: bool,
) -> str:
    track_dir = input_path.parent
    vocals_path = track_dir / "vocals.wav"
    no_vocals_path = track_dir / "no_vocals.wav"

    outputs_exist = all(
        path.is_file() and path.stat().st_size > 0
        for path in (vocals_path, no_vocals_path)
    )
    if outputs_exist and not force:
        return "skipped"

    with tempfile.TemporaryDirectory(
        prefix=".demucs_",
        dir=track_dir,
    ) as temporary_directory:
        output_root = Path(temporary_directory)
        command = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            MODEL_NAME,
            "--two-stems=vocals",
            "-d",
            device,
            "-o",
            str(output_root),
            str(input_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            diagnostic = "\n".join(completed.stderr.splitlines()[-20:])
            raise RuntimeError(
                f"Demucs exited with status {completed.returncode}:\n{diagnostic}"
            )

        generated_dir = output_root / MODEL_NAME / input_path.stem
        generated_vocals = generated_dir / "vocals.wav"
        generated_no_vocals = generated_dir / "no_vocals.wav"
        missing = [
            str(path)
            for path in (generated_vocals, generated_no_vocals)
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            raise FileNotFoundError(
                "Demucs did not create the expected output(s): "
                + ", ".join(missing)
            )

        shutil.copy2(generated_vocals, vocals_path)
        shutil.copy2(generated_no_vocals, no_vocals_path)

    return "separated"


def main() -> int:
    args = parse_args()
    audio_root = args.audio_root.resolve()
    try:
        tracks = discover_tracks(audio_root, args.track_ids)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    separated = 0
    skipped = 0
    failures: list[str] = []

    for track_id, input_path in tqdm(
        tracks,
        desc="HT-Demucs vocal separation",
        unit="track",
    ):
        try:
            outcome = run_separation(
                track_id,
                input_path,
                args.device,
                args.force_separation,
            )
        except Exception as error:
            failures.append(track_id)
            tqdm.write(f"FAILED {track_id}: {error}")
            continue

        if outcome == "separated":
            separated += 1
        else:
            skipped += 1

    print(
        f"Completed: {separated} separated, {skipped} already present, "
        f"{len(failures)} failed."
    )
    if failures:
        print(f"Failed track IDs: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
