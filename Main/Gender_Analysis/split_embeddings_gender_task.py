from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


META_FILES = {"labels.npy", "subject_ids.npy", "file_list.npy", "label_map.npy"}


@dataclass(frozen=True)
class GroupKey:
    gender: str
    task: str


def _gender_from_subject_id(subject_id: str) -> str:
    # Examples seen in this repo: "Control/F1", "ALSwDysarthria/M12"
    token = subject_id.split("/")[-1].strip()
    if not token:
        return "UNKNOWN"
    first = token[0].upper()
    if first == "F":
        return "FEMALE"
    if first == "M":
        return "MALE"
    return "UNKNOWN"


_TASK_RE = re.compile(r"_([A-Za-z]\d+)\.wav$", re.IGNORECASE)


def _task_from_file_path(file_path: str) -> str:
    # Examples seen: "..._C1.wav", "..._W45.wav", "..._o1.wav"
    m = _TASK_RE.search(file_path)
    if m:
        return m.group(1).upper()
    return Path(file_path).stem


def _load_required_meta(model_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    labels_path = model_dir / "labels.npy"
    subject_ids_path = model_dir / "subject_ids.npy"
    file_list_path = model_dir / "file_list.npy"

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing: {labels_path}")
    if not subject_ids_path.exists():
        raise FileNotFoundError(f"Missing: {subject_ids_path}")
    if not file_list_path.exists():
        raise FileNotFoundError(f"Missing: {file_list_path}")

    labels = np.load(labels_path, allow_pickle=True)
    subject_ids = np.load(subject_ids_path, allow_pickle=True).astype(str)
    file_list = np.load(file_list_path, allow_pickle=True).astype(str)

    if not (len(labels) == len(subject_ids) == len(file_list)):
        raise ValueError(
            f"Length mismatch in {model_dir}:\n"
            f"  labels     : {len(labels)}\n"
            f"  subject_ids: {len(subject_ids)}\n"
            f"  file_list  : {len(file_list)}"
        )

    return labels, subject_ids, file_list, (model_dir / "label_map.npy")


def _embedding_files(model_dir: Path) -> list[Path]:
    files = []
    for p in sorted(model_dir.glob("*.npy")):
        if p.name in META_FILES:
            continue
        files.append(p)
    return files


def split_model_dir(
    model_dir: Path,
    *,
    overwrite: bool,
    dry_run: bool,
    include_unknown_gender: bool,
) -> None:
    labels, subject_ids, file_list, label_map_path = _load_required_meta(model_dir)

    genders = np.array([_gender_from_subject_id(s) for s in subject_ids], dtype=object)
    tasks = np.array([_task_from_file_path(p) for p in file_list], dtype=object)

    groups: dict[GroupKey, np.ndarray] = {}
    for g in sorted(set(genders.tolist())):
        if g == "UNKNOWN" and not include_unknown_gender:
            continue
        g_mask = genders == g
        for t in sorted(set(tasks[g_mask].tolist())):
            idx = np.flatnonzero(g_mask & (tasks == t))
            groups[GroupKey(gender=g, task=t)] = idx

    emb_files = _embedding_files(model_dir)
    if not emb_files:
        raise FileNotFoundError(f"No embedding .npy files found in: {model_dir}")

    print(f"\n== {model_dir} ==")
    print(f"rows: {len(labels)} | genders: {sorted(set(genders.tolist()))} | tasks: {len(set(tasks.tolist()))}")
    print(f"embedding files ({len(emb_files)}): {[p.name for p in emb_files]}")
    print(f"groups (gender x task): {len(groups)}")

    for key, idx in groups.items():
        out_dir = model_dir / key.gender / key.task
        if dry_run:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata once per (gender, task) folder
        meta_files_to_write: list[tuple[Path, np.ndarray]] = [
            (out_dir / "indices.npy", idx.astype(np.int64)),
            (out_dir / "labels.npy", labels[idx]),
            (out_dir / "subject_ids.npy", subject_ids[idx]),
            (out_dir / "file_list.npy", file_list[idx]),
        ]
        for out_path, arr in meta_files_to_write:
            if out_path.exists() and not overwrite:
                continue
            np.save(out_path, arr, allow_pickle=True)

        if label_map_path.exists():
            lm_out = out_dir / "label_map.npy"
            if overwrite or not lm_out.exists():
                np.save(lm_out, np.load(label_map_path, allow_pickle=True), allow_pickle=True)

        meta_json = out_dir / "meta.json"
        if overwrite or not meta_json.exists():
            meta_json.write_text(
                json.dumps(
                    {
                        "model_dir": str(model_dir),
                        "gender": key.gender,
                        "task": key.task,
                        "n": int(len(idx)),
                    },
                    indent=2,
                )
            )

    # Save embeddings per group
    for emb_path in emb_files:
        X = np.load(emb_path, mmap_mode="r", allow_pickle=True)
        if len(X) != len(labels):
            raise ValueError(f"Embedding length mismatch for {emb_path}: got {len(X)} expected {len(labels)}")

        for key, idx in groups.items():
            out_path = model_dir / key.gender / key.task / emb_path.name
            if out_path.exists() and not overwrite:
                continue
            if dry_run:
                continue
            np.save(out_path, np.asarray(X[idx]), allow_pickle=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Split embedding .npy files into folders by GENDER (MALE/FEMALE) and TASK code (e.g., C1, W10) "
            "based on subject_ids.npy + file_list.npy."
        )
    )
    parser.add_argument(
        "--emb-root",
        type=str,
        default="embeddings",
        help="Path to embeddings root (contains Hubert/, data2vec/, wav2vec/, wavlm/).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Model subfolders to process (default: all under --emb-root). Example: --models Hubert wavlm",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split files.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done, but write nothing.")
    parser.add_argument(
        "--include-unknown-gender",
        action="store_true",
        help="Also create UNKNOWN gender folder (if any subject ids can't be parsed).",
    )
    args = parser.parse_args()

    emb_root = Path(args.emb_root).resolve()
    if not emb_root.is_dir():
        raise SystemExit(f"emb root not found: {emb_root}")

    model_dirs = (
        [emb_root / m for m in args.models]
        if args.models
        else [p for p in sorted(emb_root.iterdir()) if p.is_dir()]
    )

    for model_dir in model_dirs:
        if not model_dir.is_dir():
            raise SystemExit(f"model dir not found: {model_dir}")
        split_model_dir(
            model_dir,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            include_unknown_gender=args.include_unknown_gender,
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
