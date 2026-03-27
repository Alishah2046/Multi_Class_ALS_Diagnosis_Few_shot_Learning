from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)

    # Truncate the frequency at 5 kHz and calculate Mel spectogram
    ms = librosa.feature.melspectrogram(y=y, sr=sr, fmax=5000)

    # Convert spectogram to decibel scale
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close()


def create_augmented_spectogram(audio_file, image_file, augmented_image_file):
    # Open the saved image
    original_image = Image.open(image_file)

    # Flip the image horizontally
    flipped_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Save the flipped image
    flipped_image.save(augmented_image_file)


def _guess_workspace_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "Dataset").exists():
            return parent
    return p.parents[2]


def _resolve_wav_path(row: pd.Series, *, data_root: Path, workspace_root: Path) -> Path:
    """
    Force wav lookup under:
      data_root / <class> / <subject> / <filename.wav>

    We derive <class>/<subject>/<filename> from row['file_path'] (old CSV style):
      Dataset/raw_new_balanced/<class>/<subject>/<filename.wav>

    If file_path is in a different but similar shape, we fall back to using the last 3 parts.
    """
    fp = str(row.get("file_path", "")).strip()
    if not fp:
        raise FileNotFoundError("Row has empty file_path")

    p = Path(fp)

    cls = subj = fname = None

    # Expected old pattern: Dataset/raw_new_balanced/<cls>/<subj>/<file.wav>
    parts = p.parts
    if len(parts) >= 5 and parts[0] == "Dataset":
        cls, subj, fname = parts[2], parts[3], parts[4]
    elif len(parts) >= 3:
        # Fallback: take last 3 parts as <cls>/<subj>/<file.wav>
        cls, subj, fname = parts[-3], parts[-2], parts[-1]

    full_wav = (data_root / cls / subj / fname).resolve()
    if full_wav.is_file():
        return full_wav

    # Last resort: if CSV already has an absolute resolved_path that exists, use it
    rp = row.get("resolved_path", None)
    if rp is not None:
        rp_path = Path(str(rp))
        if rp_path.is_absolute() and rp_path.is_file():
            return rp_path

    # Another fallback: workspace_root + file_path (old behavior)
    full_old = (workspace_root / fp).resolve()
    if full_old.is_file():
        return full_old

    raise FileNotFoundError(
        f"Could not find wav.\n"
        f"  tried new  : {full_wav}\n"
        f"  tried old  : {full_old}\n"
        f"  file_path  : {fp}\n"
        f"  resolved_path: {rp}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate (and augment) mel-spectrogram PNGs for wavs in a feature CSV."
    )
    parser.add_argument(
        "--workspace-root",
        type=str,
        default=None,
        help="Path to ALS_Diagnosis_Meta root. If omitted, attempts to auto-detect.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/filtered/final_metadata_acoustic_features.csv",
        help="Input CSV that includes `file_path`.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/filtered/spectogram_acoustic_features.csv",
        help="Output CSV with added `spectogram_file_path` and `spectogram_type` columns.",
    )
    parser.add_argument(
        "--orig-dirname",
        type=str,
        default="spectogram",
        help="Folder name (next to each wav) to store original spectrograms.",
    )
    parser.add_argument(
        "--aug-dirname",
        type=str,
        default="augmented_spectogram",
        help="Folder name (next to each wav) to store augmented spectrograms.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N rows.")
    parser.add_argument(
        "--no-augment", action="store_true", help="Skip augmented spectrogram generation."
    )
    args = parser.parse_args()

    sota_root = Path(__file__).resolve().parents[2]
    workspace_root = Path(args.workspace_root) if args.workspace_root else _guess_workspace_root()

    # YOUR NEW DATA ROOT (force using this)
    data_root = Path("/mnt/d/PhD/ALS_PROJECT1/ALS_Diagnosis_Meta/Dataset/ALI_new/All")
    if not data_root.is_dir():
        raise SystemExit(f"DATA ROOT not found: {data_root}")

    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = sota_root / input_csv
    if not input_csv.exists():
        raise SystemExit(f"input csv not found: {input_csv}")

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = sota_root / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if "file_path" not in df.columns:
        raise SystemExit(f"`file_path` column missing in: {input_csv}")

    updated_rows = []
    updated_column_names = df.columns.to_list() + ["spectogram_file_path", "spectogram_type"]

    n = 0
    max_n = args.limit if args.limit is not None else len(df)
    for _, row in df.head(max_n).iterrows():
        full_wav = _resolve_wav_path(row, data_root=data_root, workspace_root=workspace_root)

        one_dir_up = full_wav.parent

        orig_dir = one_dir_up / args.orig_dirname
        aug_dir = one_dir_up / args.aug_dirname
        orig_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_augment:
            aug_dir.mkdir(parents=True, exist_ok=True)

        orig_png = (orig_dir / full_wav.name).with_suffix(".png")
        aug_png = (aug_dir / full_wav.name).with_suffix(".png")

        updated_rows.append(row.to_list() + [str(orig_png), "original"])
        if not args.no_augment:
            updated_rows.append(row.to_list() + [str(aug_png), "augmented"])

        if not orig_png.is_file():
            create_spectrogram(str(full_wav), str(orig_png))
        else:
            print("Original spectogram exists.")

        # Crop the original spectogram to remove borders/axes
        image = Image.open(orig_png)
        left_crop, right_crop, top_crop = 20, 20, 45
        width, height = image.size
        cropped_image = image.crop((left_crop, top_crop, width - right_crop, height))
        cropped_image.save(orig_png)

        if not args.no_augment:
            if not aug_png.is_file():
                create_augmented_spectogram(str(full_wav), str(orig_png), str(aug_png))
            else:
                print("Augmented spectogram already exists.")

        print(f"{n}. {full_wav}")
        n += 1

    pd.DataFrame(updated_rows, columns=updated_column_names).to_csv(output_csv, index=False)
    print(f"Wrote: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
