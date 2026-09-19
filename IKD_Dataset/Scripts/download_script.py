#!/usr/bin/env python3
"""Download the IKD recordings listed in the released metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yt_dlp
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
DEFAULT_METADATA = DATASET_DIR / "Indonesian_Keroncong_Dataset.csv"
DEFAULT_AUDIO_ROOT = DATASET_DIR / "Audio"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Completed IKD metadata CSV.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=DEFAULT_AUDIO_ROOT,
        help="Destination root for the reconstructed audio.",
    )
    parser.add_argument(
        "--track-ids",
        nargs="+",
        default=None,
        help="Optional track subset, for example: IKD_001 IKD_002.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download a track again when original_mix.wav already exists.",
    )
    return parser.parse_args()


def load_metadata(path: Path, requested: list[str] | None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    table = pd.read_csv(path)
    required = {"track_id", "preview_url"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing metadata columns: {', '.join(missing)}")
    if table["track_id"].isna().any() or table["track_id"].duplicated().any():
        raise ValueError("track_id values must be present and unique")

    if requested is not None:
        requested_set = set(requested)
        known = set(table["track_id"].astype(str))
        unknown = sorted(requested_set - known)
        if unknown:
            raise ValueError(f"Unknown track IDs: {', '.join(unknown)}")
        table = table[table["track_id"].astype(str).isin(requested_set)]

    return table


def download_track(
    track_id: str,
    url: str,
    audio_root: Path,
    overwrite: bool,
) -> str:
    track_dir = audio_root / track_id
    output_path = track_dir / "original_mix.wav"

    if output_path.is_file() and output_path.stat().st_size > 0 and not overwrite:
        return "skipped"

    track_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(track_dir / "original_mix.%(ext)s"),
        "noplaylist": True,
        "overwrites": overwrite,
        "continuedl": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        return_code = downloader.download([url])
    if return_code != 0:
        raise RuntimeError(f"yt-dlp returned exit status {return_code}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("the expected original_mix.wav was not created")
    return "downloaded"


def main() -> int:
    args = parse_args()
    metadata = args.metadata.resolve()
    audio_root = args.audio_root.resolve()

    try:
        table = load_metadata(metadata, args.track_ids)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    downloaded = 0
    skipped = 0
    failures: list[str] = []

    for row in tqdm(
        table.itertuples(index=False),
        total=len(table),
        desc="Downloading IKD",
        unit="track",
    ):
        track_id = str(row.track_id)
        url = str(row.preview_url).strip()
        if not url or url.lower() == "nan":
            failures.append(track_id)
            tqdm.write(f"FAILED {track_id}: missing preview_url")
            continue

        try:
            outcome = download_track(track_id, url, audio_root, args.overwrite)
        except Exception as error:
            failures.append(track_id)
            tqdm.write(f"FAILED {track_id}: {error}")
            continue

        if outcome == "downloaded":
            downloaded += 1
        else:
            skipped += 1

    print(
        f"Completed: {downloaded} downloaded, {skipped} already present, "
        f"{len(failures)} failed."
    )
    if failures:
        print(f"Failed track IDs: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
