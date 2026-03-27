from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from acoustics import acstc_anlys
from utilities import get_voiced

def run_analysis_single(
    wav_path: str, praat_script: str, textgrid_script: str, get_acoustics: bool = True
) -> pd.DataFrame:
    '''
    A function to analyze 

    Parameters
    ----------
    wav_path : str
        Filepath to -converted.wav audio file (absolute path). MUST be the audio file
    praat_script : str
        Praat script for extracting voiced segments from audio data (absolute path).
    textgrid_script : str
        Praat script for textgrid parsing for timing analysis (absolute path).
        Praat script for analyzing pauses from audio data (absolute path).
    get_acoustics : bool
        Extract acoustic features?

    Returns
    -------
    output : pd.DataFrame
        Single-row dataframe containing all extracted features.

    '''
    
    # Verify inputs
    assert (
        isinstance(praat_script, str) and len(praat_script) > 3
    ), "Please specify location of voice extraction Praat script"

    file = wav_path.replace("\\", "/")
    data_dir = os.path.dirname(file).replace("\\", "/")
    
    fnew_test = os.path.dirname(file) + "/" + "voiced/" + os.path.basename(file).replace(".wav", "_OnlyVoiced.wav")
    if get_acoustics==True:
        if os.path.isfile(fnew_test):
            print("Voiced file already extracted")
            pass
        else:
            get_voiced(file = file, 
                                        data_dir = data_dir,
                                        praat_script = praat_script,
                                        textgrid_script = textgrid_script
                                        )
        # get acoustic features
        output = acstc_anlys(file = file,
                                    data_dir = data_dir,
                                    f0_min = 75,
                                    f0_max = 600
                                    )
    
    output.index = [os.path.basename(file).split('.')[0]]
    
    return output    

def _guess_workspace_root() -> Path:
    """
    Best-effort guess of the ALS_Diagnosis_Meta workspace root.
    Walks upwards until a `Dataset/` directory is found.
    """
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "Dataset").exists():
            return parent
    return p.parents[2]  # Speech-Disorder-Classification-ML root


def _iter_rows(df: pd.DataFrame, start: int, stop: int | None, limit: int | None) -> Iterable[tuple[int, pd.Series]]:
    end = stop if stop is not None else len(df)
    if limit is not None:
        end = min(end, start + limit)
    for i in range(start, end):
        yield i, df.iloc[i]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract acoustic features for wavs listed in a metadata CSV."
    )
    parser.add_argument(
        "--workspace-root",
        type=str,
        default=None,
        help="Path to ALS_Diagnosis_Meta root (where Dataset/ lives). If omitted, attempts to auto-detect.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="Dataset/raw_new_balanced_metadata.csv",
        help="Metadata CSV containing at least `file_path` and `label` columns (path relative to workspace root).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/filtered/final_metadata_acoustic_features.csv",
        help="Output CSV path (relative to Speech-Disorder-Classification-ML repo root unless absolute).",
    )
    parser.add_argument(
        "--errors-csv",
        type=str,
        default="data/filtered/extract_acoustic_features_errors.csv",
        help="Where to write paths that failed (relative to repo root unless absolute).",
    )
    parser.add_argument(
        "--praat-script",
        type=str,
        default=None,
        help="Path to `extract_voiced_segments.praat` (defaults to this repo's copy).",
    )
    parser.add_argument(
        "--textgrid-script",
        type=str,
        default=None,
        help="Path to `textgrid2py.praat` (defaults to this repo's copy).",
    )
    parser.add_argument("--start", type=int, default=0, help="Start row index (0-based).")
    parser.add_argument("--stop", type=int, default=None, help="Stop row index (exclusive).")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N rows.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If output exists, skip rows whose `file_path` is already present.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Write accumulated rows every N processed files.",
    )

    args = parser.parse_args()

    sota_root = Path(__file__).resolve().parents[2]

    workspace_root = Path(args.workspace_root) if args.workspace_root else _guess_workspace_root()
    if not workspace_root.exists():
        raise SystemExit(f"workspace root not found: {workspace_root}")

    metadata_csv = Path(args.metadata_csv)
    if not metadata_csv.is_absolute():
        metadata_csv = workspace_root / metadata_csv
    if not metadata_csv.exists():
        raise SystemExit(f"metadata csv not found: {metadata_csv}")

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = sota_root / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    errors_csv = Path(args.errors_csv)
    if not errors_csv.is_absolute():
        errors_csv = sota_root / errors_csv
    errors_csv.parent.mkdir(parents=True, exist_ok=True)

    praat_script = Path(args.praat_script) if args.praat_script else (Path(__file__).resolve().parent / "extract_voiced_segments.praat")
    textgrid_script = Path(args.textgrid_script) if args.textgrid_script else (Path(__file__).resolve().parent / "textgrid2py.praat")
    praat_script = praat_script.resolve()
    textgrid_script = textgrid_script.resolve()

    df_meta = pd.read_csv(metadata_csv)
    required_cols = {"file_path", "label"}
    missing = required_cols - set(df_meta.columns)
    if missing:
        raise SystemExit(f"metadata csv missing required columns: {sorted(missing)}")

    # Resume: skip already-processed file_path entries
    if args.resume and output_csv.exists():
        try:
            done = pd.read_csv(output_csv, usecols=["file_path"])
            done_set = set(done["file_path"].astype(str).tolist())
            df_meta = df_meta[~df_meta["file_path"].astype(str).isin(done_set)].reset_index(drop=True)
            print(f"Resume enabled: skipping {len(done_set)} already in {output_csv}")
        except Exception as e:
            print(f"Resume enabled but failed to load {output_csv}: {e}. Continuing without skipping.")

    rows_out: list[dict] = []
    errors: list[dict] = []

    # Determine feature columns once (run on first valid file we can open)
    sample_features_cols: list[str] | None = None

    processed = 0
    for idx, row in _iter_rows(df_meta, start=args.start, stop=args.stop, limit=args.limit):
        rel_path = str(row["file_path"])
        wav_abs = (workspace_root / rel_path).resolve()

        print(f"{idx}: {rel_path}")

        if not wav_abs.exists():
            errors.append({"file_path": rel_path, "error": "missing_wav"})
            continue

        try:
            feat_df = run_analysis_single(
                str(wav_abs), str(praat_script), str(textgrid_script), get_acoustics=True
            )
            feat_row = feat_df.iloc[0].to_dict()

            voiced_name = Path(wav_abs).name.replace(".wav", "_OnlyVoiced.wav")
            voiced_abs = wav_abs.parent / "voiced" / voiced_name
            voiced_rel = os.path.relpath(voiced_abs, workspace_root)

            out_row = row.to_dict()
            out_row["resolved_path"] = str(wav_abs)
            out_row["voiced_file_path"] = voiced_rel.replace("\\", "/")
            out_row.update(feat_row)

            if sample_features_cols is None:
                sample_features_cols = list(feat_row.keys())

            rows_out.append(out_row)
            processed += 1

            if processed % max(1, args.flush_every) == 0:
                df_out = pd.DataFrame(rows_out)
                write_header = not output_csv.exists()
                df_out.to_csv(output_csv, index=False, mode="a", header=write_header)
                rows_out.clear()

        except Exception as e:
            errors.append({"file_path": rel_path, "error": repr(e)})

    # flush remaining
    if rows_out:
        df_out = pd.DataFrame(rows_out)
        write_header = not output_csv.exists()
        df_out.to_csv(output_csv, index=False, mode="a", header=write_header)

    if errors:
        pd.DataFrame(errors).to_csv(errors_csv, index=False)
        print(f"Wrote errors: {errors_csv} ({len(errors)})")

    print(f"Wrote features: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

    


