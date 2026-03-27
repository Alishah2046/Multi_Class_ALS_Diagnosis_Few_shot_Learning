from __future__ import annotations

import json
import os
import hashlib
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def _require_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torchvision.transforms as T  # type: ignore
        from torchvision import models  # type: ignore
    except Exception as e:
        msg = str(e)
        if "operator torchvision::nms does not exist" in msg:
            raise SystemExit(
                "Torchvision failed to import because it is incompatible with your installed torch build.\n"
                "This usually happens when torch/torchvision versions (or CUDA builds) don't match.\n\n"
                "What to do:\n"
                "- Use a Python environment where BOTH `torch` and `torchvision` import cleanly.\n"
                "- Avoid Python 3.13 for this repo; Python 3.12 is a safer choice.\n"
                "- Reinstall a matching torch+torchvision pair in the same environment.\n\n"
                f"Import error: {e}"
            )
        if "No module named 'torchvision'" in msg:
            raise SystemExit(
                "This Python environment has `torch` but does not have `torchvision` installed.\n"
                "Select an interpreter that has both installed, or install torchvision into this environment.\n\n"
                f"Import error: {e}"
            )
        raise SystemExit(
            "PyTorch/torchvision could not be imported in this Python interpreter.\n"
            "In VS Code: Ctrl+Shift+P → Python: Select Interpreter → choose an environment that has BOTH torch + torchvision.\n"
            f"Import error: {e}"
        )
    return torch, nn, T, models


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _guess_workspace_root() -> Path:
    # FIXED to your real workspace (where Dataset/ exists)
    p = Path("/mnt/d/PhD/ALS_PROJECT1/ALS_Diagnosis_Meta")
    if (p / "Dataset").is_dir():
        return p
    raise FileNotFoundError(f"Workspace root not found or missing Dataset/: {p}")



def _load_features_csv(repo_root: Path) -> Path:
    path = repo_root / "data" / "filtered" / "final_metadata_acoustic_features.csv"
    if path.exists():
        return path
    raise FileNotFoundError(f"Could not find features CSV at: {path}")



def _safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in s)


def _df_fingerprint(df: pd.DataFrame) -> str:
    """
    Stable fingerprint to ensure resume/checkpoints match the same dataset ordering.
    """
    h = hashlib.sha1()
    for fp in df["file_path"].astype(str).tolist():
        h.update(fp.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def _atomic_torch_save(torch: Any, obj: Any, path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _safe_torch_load(torch: Any, path: Path, *, label: str) -> Any | None:
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine = path.with_name(f"{path.stem}.corrupt_{stamp}{path.suffix}")
        print(f"{label}: failed to load {path} ({type(e).__name__}: {e})", flush=True)
        try:
            os.replace(path, quarantine)
            print(f"{label}: moved unreadable checkpoint to {quarantine}", flush=True)
        except OSError as move_err:
            print(
                f"{label}: could not quarantine unreadable checkpoint {path} ({type(move_err).__name__}: {move_err})",
                flush=True,
            )
        return None


def _wav_abs(workspace_root: Path, rel_path: str) -> Path:
    p = str(rel_path).replace("\\", "/")

    old_prefix = "Dataset/raw_new_balanced/"
    new_prefix = "Dataset/ALI_new/All/"

    if p.startswith(old_prefix):
        p = new_prefix + p[len(old_prefix):]

    return (workspace_root / p).resolve()



def _spectrogram_png_path(
    repo_root: Path, label: str, subject_uid: str, file_stem: str
) -> Path:
    base = repo_root / "data" / "als_diagnosis_meta_spectrograms"
    return base / _safe_name(label) / _safe_name(subject_uid) / f"{_safe_name(file_stem)}.png"


def _wav_to_png(wav_path: Path, png_path: Path, *, sr: int = 16000, n_mels: int = 128, fmax: int = 5000) -> None:
    import librosa  # local import (heavy)
    from PIL import Image

    y, _sr = librosa.load(str(wav_path), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError("empty_audio")

    ms = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    log_ms = librosa.power_to_db(ms, ref=np.max)

    # Normalize to [0, 255] uint8
    log_ms = log_ms - float(np.nanmin(log_ms))
    denom = float(np.nanmax(log_ms)) + 1e-9
    log_ms = log_ms / denom
    img = (log_ms * 255.0).astype(np.uint8)

    pil = Image.fromarray(img, mode="L").resize((224, 224))
    pil = pil.convert("RGB")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(png_path)


class SpectrogramDataset:
    def __init__(self, df: pd.DataFrame, indices: np.ndarray, transform: Any):
        self.df = df.iloc[indices].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        from PIL import Image

        row = self.df.iloc[idx]
        img = Image.open(row["png_path"]).convert("RGB")
        x = self.transform(img)
        y = int(row["label_id"])
        subject = str(row["subject_uid"])
        return x, y, subject


@dataclass(frozen=True)
class FoldMetrics:
    fold: int
    n_subjects: int
    subject_accuracy: float
    subject_f1_macro: float
    subject_auc_ovr_macro: float


@dataclass(frozen=True)
class RunMetrics:
    run: int
    seed: int
    n_subjects_test: int
    subject_accuracy: float
    subject_f1_macro: float
    subject_auc_ovr_macro: float
    utterance_accuracy: float = float("nan")
    utterance_f1_macro: float = float("nan")
    utterance_auc_ovr_macro: float = float("nan")


@dataclass(frozen=True)
class SubjectSplit:
    seed: int
    n_subjects_total: int
    n_subjects_train: int
    n_subjects_test: int
    train_subjects: list[str]
    test_subjects: list[str]


def _subject_wise_split_80_20(
    *,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> SubjectSplit:
    """
    Subject-wise split: 80% TRAIN and 20% TEST (no separate VAL).
    Mirrors the protocol used in your updated main run.
    """
    uniq_subjects = np.unique(groups).astype(str)

    # Majority label per subject (robust to accidental mixed rows)
    subj_y: list[int] = []
    mixed = 0
    for sid in uniq_subjects:
        ys = y[groups == sid]
        vals, cnts = np.unique(ys, return_counts=True)
        subj_y.append(int(vals[int(np.argmax(cnts))]))
        if len(vals) > 1:
            mixed += 1

    subj_y_arr = np.asarray(subj_y, dtype=int)
    subj_train, subj_test, _, _ = train_test_split(
        uniq_subjects,
        subj_y_arr,
        test_size=0.2,
        stratify=subj_y_arr,
        random_state=seed,
    )

    set_train = set(map(str, subj_train.tolist()))
    set_test = set(map(str, subj_test.tolist()))
    if set_train & set_test:
        raise RuntimeError("Subject leakage detected in split (overlapping subject_uids).")
    if mixed:
        print(f"Split note: {mixed} subjects had mixed utterance labels (majority label used).", flush=True)

    return SubjectSplit(
        seed=int(seed),
        n_subjects_total=int(len(uniq_subjects)),
        n_subjects_train=int(len(set_train)),
        n_subjects_test=int(len(set_test)),
        train_subjects=sorted(set_train),
        test_subjects=sorted(set_test),
    )


def _load_subject_split(path: Path) -> SubjectSplit | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Accept either this script's SubjectSplit schema, or the tabular baseline SplitInfo schema.
        if "train_subjects" in raw and "test_subjects" in raw:
            train_subjects = list(map(str, raw["train_subjects"]))
            test_subjects = list(map(str, raw["test_subjects"]))
            return SubjectSplit(
                seed=int(raw.get("seed", 42)),
                n_subjects_total=int(raw.get("n_subjects_total", len(set(train_subjects) | set(test_subjects)))),
                n_subjects_train=int(raw.get("n_subjects_train", len(set(train_subjects)))),
                n_subjects_test=int(raw.get("n_subjects_test", len(set(test_subjects)))),
                train_subjects=sorted(set(train_subjects)),
                test_subjects=sorted(set(test_subjects)),
            )
    except Exception:
        return None
    return None


def _subject_metrics(
    *,
    subject_ids: list[str],
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
) -> tuple[float, float, float]:
    # Aggregate by subject: mean probas
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    subj_true: dict[str, int] = {}

    for s, yt, yp in zip(subject_ids, y_true.tolist(), y_proba):
        if s not in sums:
            sums[s] = np.zeros((n_classes,), dtype=np.float64)
            counts[s] = 0
            subj_true[s] = int(yt)
        sums[s] += yp.astype(np.float64)
        counts[s] += 1

    subjects = sorted(sums.keys())
    y_true_sub = np.array([subj_true[s] for s in subjects], dtype=int)
    y_proba_sub = np.vstack([sums[s] / max(1, counts[s]) for s in subjects])
    y_pred_sub = np.argmax(y_proba_sub, axis=1)

    acc = float(accuracy_score(y_true_sub, y_pred_sub))
    f1m = float(f1_score(y_true_sub, y_pred_sub, average="macro"))

    try:
        auc = float(
            roc_auc_score(
                y_true_sub,
                y_proba_sub,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        auc = float("nan")
    return acc, f1m, auc


def _utterance_metrics(
    *,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
) -> tuple[float, float, float]:
    """
    Utterance-level (micro) metrics on ALL test utterances/images.
    Note: this weights subjects with more utterances more heavily.
    """
    y_pred = np.argmax(y_proba, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    try:
        auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return acc, f1m, auc


def _create_model(models: Any, nn: Any, arch: str, num_classes: int):
    arch = arch.lower().strip()

    if arch == "mobilenet_v2":
        try:
            m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        except Exception:
            m = models.mobilenet_v2(pretrained=True)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    if arch == "resnet50":
        try:
            m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            m = models.resnet50(pretrained=True)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    raise ValueError(f"Unknown arch: {arch}. Use 'mobilenet_v2' or 'resnet50'.")

def _split_train_val_subjects(
    *,
    df_train: pd.DataFrame,
    seed: int,
    val_subject_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split training rows into (train_indices, val_indices) by *subject_uid* so that
    per-epoch monitoring does not touch the test fold (no leakage).
    """
    subj = (
        df_train[["subject_uid", "label_id"]]
        .drop_duplicates(subset=["subject_uid"])
        .reset_index(drop=True)
    )
    if len(subj) < 3:
        # too few subjects to split sensibly; return all as train, empty val
        return df_train.index.to_numpy(), np.array([], dtype=int)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_subject_frac, random_state=seed)
    train_s, val_s = next(splitter.split(subj["subject_uid"], subj["label_id"]))
    train_subjects = set(subj.loc[train_s, "subject_uid"].astype(str).tolist())
    val_subjects = set(subj.loc[val_s, "subject_uid"].astype(str).tolist())

    train_idx = df_train.index[df_train["subject_uid"].astype(str).isin(train_subjects)].to_numpy()
    val_idx = df_train.index[df_train["subject_uid"].astype(str).isin(val_subjects)].to_numpy()
    return train_idx, val_idx


def main() -> int:
    # Click-run configuration (3-class)
    # The original repo's CNN script used MobileNetV2 and ResNet50.
    ARCHS = ["mobilenet_v2", "resnet50"]  # remove one if you want a faster run
    SEED = 42
    N_RUNS = 30  # match main: repeat training with different init seeds (same fixed split)
    EPOCHS = 10
    BATCH_SIZE = 32
    LR = 1e-4
    NUM_WORKERS = 0  # safer for WSL/Windows setups
    RESUME = True  # resumes from checkpoints if the system restarts
    RUN_DIRNAME = "latest"  # change to a custom name to keep multiple runs

    torch, nn, T, models = _require_torch()

    repo_root = _repo_root()
    os.chdir(repo_root)

    workspace_root = _guess_workspace_root()
    features_csv = _load_features_csv(repo_root)

    results_dir = repo_root / "results" / "als_diagnosis_meta_3class_repo_cnn"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run folder (for resumable training)
    run_root = results_dir / RUN_DIRNAME
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "state.json"
    results_json_path = run_root / "cnn_subject_metrics.json"
    results_csv_path = run_root / "cnn_subject_metrics_summary.csv"
    results_csv_pm_path = run_root / "cnn_subject_metrics_summary_pm.csv"
    split_path = run_root / f"subject_split_seed{SEED}.json"

    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("3-class CNN training (repo script)", flush=True)
    print(f"Input CSV: {features_csv}", flush=True)
    print(f"Workspace root: {workspace_root}", flush=True)
    print(f"Results dir: {results_dir}", flush=True)
    print(f"Run dir: {run_root}", flush=True)
    print(f"Protocol: subject split 80/20 (seed={SEED}); repeated runs: {N_RUNS}", flush=True)

    df = pd.read_csv(features_csv)
    required = {"label", "subject_uid", "file_path", "file_name"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in features CSV: {sorted(missing)}")

    # Deduplicate (your feature CSV may have duplicates due to append)
    before = len(df)
    df = df.drop_duplicates(subset=["file_path"], keep="last").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Dedup: removed {removed} duplicate rows by `file_path`.", flush=True)

    labels = sorted(df["label"].astype(str).unique().tolist())
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    df["label_id"] = df["label"].astype(str).map(label_to_id).astype(int)

    # Try to reuse the exact same subject split as the tabular baselines (if available).
    # This keeps the CNN baselines comparable to your fixed 80/20 main split.
    tabular_split_path = repo_root / "results" / "als_diagnosis_meta_3class_repo_baselines" / f"subject_split_seed{SEED}.json"
    split = _load_subject_split(tabular_split_path) if tabular_split_path.exists() else None
    if split is None:
        y_for_split = df["label_id"].to_numpy(dtype=int)
        g_for_split = df["subject_uid"].astype(str).to_numpy()
        split = _subject_wise_split_80_20(y=y_for_split, groups=g_for_split, seed=SEED)
    split_path.write_text(json.dumps(asdict(split), indent=2), encoding="utf-8")
    print(f"Subject split: train={split.n_subjects_train} test={split.n_subjects_test} (seed={split.seed})", flush=True)

    # Prepare (and cache) PNG spectrograms
    png_paths = []
    created = 0
    for _, row in df.iterrows():
        wav = _wav_abs(workspace_root, str(row["file_path"]))
        png = _spectrogram_png_path(
            repo_root, str(row["label"]), str(row["subject_uid"]), Path(str(row["file_name"])).stem
        )
        if not png.exists():
            try:
                _wav_to_png(wav, png)
                created += 1
            except Exception:
                # Skip generating now; the file may be problematic. We'll exclude such rows below.
                pass
        png_paths.append(str(png))

    df["png_path"] = png_paths
    ok_mask = df["png_path"].apply(lambda p: Path(str(p)).exists())
    dropped = int((~ok_mask).sum())
    if dropped:
        df = df[ok_mask].reset_index(drop=True)
        print(f"PNG generation: dropped {dropped} rows (failed to create/read PNG).", flush=True)
    print(f"PNG generation: created {created} new PNG(s).", flush=True)

    print(f"Rows: {len(df)} | Labels: {labels}", flush=True)
    print(f"Unique subjects: {df['subject_uid'].nunique()}", flush=True)

    # Ensure the fixed split subjects still exist after dropping PNG failures.
    subj_now = set(df["subject_uid"].astype(str).unique().tolist())
    missing_train = sorted(set(split.train_subjects) - subj_now)
    missing_test = sorted(set(split.test_subjects) - subj_now)
    if missing_train or missing_test:
        raise SystemExit(
            "Fixed subject split contains subjects that are missing after PNG filtering.\n"
            "Fix PNG generation for those subjects (or regenerate) to keep the evaluation protocol consistent.\n"
            f"missing_train_subjects: {missing_train}\n"
            f"missing_test_subjects : {missing_test}"
        )

    df_hash = _df_fingerprint(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_tf = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_tf = train_tf

    X_dummy = np.zeros((len(df), 1), dtype=np.int8)  # splitter needs X
    y = df["label_id"].to_numpy(dtype=int)
    groups = df["subject_uid"].astype(str).to_numpy()
    train_idx = np.where(df["subject_uid"].astype(str).isin(set(split.train_subjects)))[0]
    test_idx = np.where(df["subject_uid"].astype(str).isin(set(split.test_subjects)))[0]

    # Load previous run state if resuming
    if RESUME and state_path.exists() and results_json_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("protocol") not in (None, "subject_split_80_train_20_test"):
            raise SystemExit(
                "Resume requested but the existing run uses a different evaluation protocol.\n"
                "To avoid mixing results, delete the run dir or change RUN_DIRNAME.\n"
                f"Run dir: {run_root}\n"
                f"found protocol: {state.get('protocol')}"
            )
        if state.get("df_hash") != df_hash:
            raise SystemExit(
                "Resume requested but dataset fingerprint changed. "
                "Delete the run dir or set RUN_DIRNAME to start a fresh run.\n"
                f"Run dir: {run_root}"
            )
        if int(state.get("n_runs", N_RUNS)) != int(N_RUNS):
            raise SystemExit(
                "Resume requested but N_RUNS changed.\n"
                "To avoid mixing results, delete the run dir or change RUN_DIRNAME.\n"
                f"Run dir: {run_root}\n"
                f"existing n_runs: {state.get('n_runs')}, requested: {N_RUNS}"
            )
        created_at = str(state.get("created_at", created_at))
        all_out: dict[str, Any] = json.loads(results_json_path.read_text(encoding="utf-8"))
        print(f"Resuming existing run (created_at={created_at})", flush=True)
    else:
        state = {
            "created_at": created_at,
            "df_hash": df_hash,
            "archs": ARCHS,
            "folds": 1,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "seed": SEED,
            "protocol": "subject_split_80_train_20_test",
            "split_path": str(split_path),
            "n_runs": int(N_RUNS),
            "completed": {},  # arch -> list[int]
        }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        all_out = {
            "task": "3-class",
            "labels": labels,
            "features_csv": str(features_csv),
            "workspace_root": str(workspace_root),
            "n_rows": int(len(df)),
            "n_subjects": int(df["subject_uid"].nunique()),
            "folds": 1,
            "seed": int(SEED),
            "protocol": "subject_split_80_train_20_test",
            "split_path": str(split_path),
            "n_runs": int(N_RUNS),
            "epochs": int(EPOCHS),
            "batch_size": int(BATCH_SIZE),
            "lr": float(LR),
            "archs": ARCHS,
            "created_at": created_at,
            "models": {},
        }
        results_json_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")

    for arch in ARCHS:
        # If resuming, preload already-completed run metrics from the saved JSON
        run_metrics: list[RunMetrics] = []
        existing_arch = all_out.get("models", {}).get(arch)
        if isinstance(existing_arch, dict):
            if "per_run" in existing_arch:
                try:
                    run_metrics = [RunMetrics(**m) for m in existing_arch.get("per_run", [])]
                except Exception:
                    run_metrics = []
            elif "per_fold" in existing_arch:
                # Backward-compat: older versions stored a single fixed split as fold_0.
                try:
                    fold_metrics = [FoldMetrics(**m) for m in existing_arch.get("per_fold", [])]
                    run_metrics = [
                        RunMetrics(
                            run=int(m.fold),
                            seed=int(SEED + int(m.fold)),
                            n_subjects_test=int(m.n_subjects),
                            subject_accuracy=float(m.subject_accuracy),
                            subject_f1_macro=float(m.subject_f1_macro),
                            subject_auc_ovr_macro=float(m.subject_auc_ovr_macro),
                            utterance_accuracy=float("nan"),
                            utterance_f1_macro=float("nan"),
                            utterance_auc_ovr_macro=float("nan"),
                        )
                        for m in fold_metrics
                    ]
                except Exception:
                    run_metrics = []
        print(f"\n### CNN: {arch} ###", flush=True)

        completed_runs = set(state.get("completed", {}).get(arch, [])) | {m.run for m in run_metrics}
        for run in range(int(N_RUNS)):
            if run in completed_runs:
                print(f"run {run}: already completed (skipping)", flush=True)
                continue
            run_seed = int(SEED + run)

            # Make this run reproducible
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)

            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_test = df.iloc[test_idx].reset_index(drop=True)

            # Train/val split inside the training fold (subject-wise) for per-epoch monitoring
            tr_sub_idx, val_sub_idx = _split_train_val_subjects(
                df_train=df_train, seed=SEED, val_subject_frac=0.2
            )
            if len(val_sub_idx) == 0:
                print(
                    f"run {run}: warning: not enough subjects for a val split; "
                    "per-epoch metrics will be skipped.",
                    flush=True,
                )

            # Class weights (sample-level) to reduce imbalance
            counts = np.bincount(df_train["label_id"].to_numpy(), minlength=len(labels)).astype(np.float64)
            weights = (counts.sum() / np.maximum(counts, 1.0))
            weights = weights / weights.sum() * len(labels)
            class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

            # NOTE: SpectrogramDataset is built from the *full* df and uses passed indices.
            # Here we construct indices into the full df.
            full_train_idx = df.iloc[train_idx].index.to_numpy()
            # Map df_train (0..n-1) back to full df indices for the split
            full_tr_sub_idx = full_train_idx[tr_sub_idx]
            full_val_sub_idx = full_train_idx[val_sub_idx] if len(val_sub_idx) else np.array([], dtype=int)

            train_ds = SpectrogramDataset(df, full_tr_sub_idx, transform=train_tf)
            val_ds = SpectrogramDataset(df, full_val_sub_idx, transform=test_tf) if len(full_val_sub_idx) else None
            test_ds = SpectrogramDataset(df, test_idx, transform=test_tf)

            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
            )
            val_loader = (
                torch.utils.data.DataLoader(
                    val_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                )
                if val_ds is not None
                else None
            )
            test_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
            )

            model = _create_model(models, nn, arch=arch, num_classes=len(labels)).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)

            ckpt_dir = run_root / "checkpoints" / arch / f"run_{run}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "last.pt"
            best_path = ckpt_dir / "best.pt"

            start_epoch = 0
            best_val_f1 = float("-inf")
            if RESUME and ckpt_path.exists():
                ckpt = _safe_torch_load(torch, ckpt_path, label=f"run {run}")
                if (
                    isinstance(ckpt, dict)
                    and ckpt.get("arch") == arch
                    and int(ckpt.get("run", -1)) == int(run)
                    and ckpt.get("df_hash") == df_hash
                ):
                    start_epoch = int(ckpt.get("epoch", -1)) + 1
                    model.load_state_dict(ckpt["model_state"])
                    opt.load_state_dict(ckpt["opt_state"])
                    best_val_f1 = float(ckpt.get("best_val_f1", best_val_f1))
                    print(f"run {run}: resumed from epoch {start_epoch}/{EPOCHS}", flush=True)
                elif ckpt is not None:
                    print(
                        f"run {run}: ignoring checkpoint with mismatched metadata; starting from scratch",
                        flush=True,
                    )

            for epoch in range(start_epoch, EPOCHS):
                model.train()
                total_loss = 0.0
                n_seen = 0

                for xb, yb, _sb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    opt.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    opt.step()
                    total_loss += float(loss.item()) * int(xb.shape[0])
                    n_seen += int(xb.shape[0])

                avg_loss = total_loss / max(1, n_seen)
                line = f"run {run} epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f}"

                # Validation metrics for monitoring (no test leakage)
                if val_loader is not None:
                    model.eval()
                    sub_ids: list[str] = []
                    y_true: list[int] = []
                    y_prob: list[np.ndarray] = []
                    with torch.no_grad():
                        for xb, yb, sb in val_loader:
                            xb = xb.to(device)
                            logits = model(xb)
                            prob = torch.softmax(logits, dim=1).cpu().numpy()
                            y_prob.extend(prob)
                            y_true.extend(yb.numpy().tolist())
                            sub_ids.extend([str(s) for s in sb])

                    y_true_arr = np.asarray(y_true, dtype=int)
                    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
                    v_acc, v_f1, v_auc = _subject_metrics(
                        subject_ids=sub_ids,
                        y_true=y_true_arr,
                        y_proba=y_prob_arr,
                        n_classes=len(labels),
                    )
                    line += f" | val_subject_acc={v_acc:.4f} val_macro_f1={v_f1:.4f} val_macro_auc={v_auc:.4f}"
                    if v_f1 > best_val_f1:
                        best_val_f1 = float(v_f1)
                        _atomic_torch_save(
                            torch,
                            {"model_state": model.state_dict(), "best_val_f1": best_val_f1},
                            best_path,
                        )

                print(line, flush=True)

                # Save checkpoint every epoch so you can resume after shutdown
                _atomic_torch_save(
                    torch,
                    {
                        "arch": arch,
                        "run": int(run),
                        "epoch": int(epoch),
                        "model_state": model.state_dict(),
                        "opt_state": opt.state_dict(),
                        "df_hash": df_hash,
                        "best_val_f1": best_val_f1,
                    },
                    ckpt_path,
                )

            # Evaluate ONCE on the held-out test split.
            # If we have a best validation checkpoint, use it; otherwise use final epoch weights.
            if best_path.exists():
                best = _safe_torch_load(torch, best_path, label=f"run {run}")
                if isinstance(best, dict) and "model_state" in best:
                    model.load_state_dict(best["model_state"])
            model.eval()
            sub_ids: list[str] = []
            y_true: list[int] = []
            y_prob: list[np.ndarray] = []
            with torch.no_grad():
                for xb, yb, sb in test_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    prob = torch.softmax(logits, dim=1).cpu().numpy()
                    y_prob.extend(prob)
                    y_true.extend(yb.numpy().tolist())
                    sub_ids.extend([str(s) for s in sb])

            y_true_arr = np.asarray(y_true, dtype=int)
            y_prob_arr = np.asarray(y_prob, dtype=np.float64)
            u_acc, u_f1, u_auc = _utterance_metrics(y_true=y_true_arr, y_proba=y_prob_arr, n_classes=len(labels))
            b_acc, b_f1, b_auc = _subject_metrics(
                subject_ids=sub_ids,
                y_true=y_true_arr,
                y_proba=y_prob_arr,
                n_classes=len(labels),
            )
            print(
                f"run {run} test (utterance): acc={u_acc:.4f} macro_f1={u_f1:.4f} macro_auc={u_auc:.4f}",
                flush=True,
            )
            print(
                f"run {run} test (subject)  : acc={b_acc:.4f} macro_f1={b_f1:.4f} macro_auc={b_auc:.4f}",
                flush=True,
            )
            rm = RunMetrics(
                run=int(run),
                seed=int(run_seed),
                n_subjects_test=int(df_test["subject_uid"].nunique()),
                subject_accuracy=float(b_acc),
                subject_f1_macro=float(b_f1),
                subject_auc_ovr_macro=float(b_auc),
                utterance_accuracy=float(u_acc),
                utterance_f1_macro=float(u_f1),
                utterance_auc_ovr_macro=float(u_auc),
            )
            run_metrics.append(rm)
            print(
                f"run {run} done: utterance_acc={rm.utterance_accuracy:.4f} subject_acc={rm.subject_accuracy:.4f}",
                flush=True,
            )

            # Mark run completed and write intermediate results so shutdown won't lose everything
            state.setdefault("completed", {}).setdefault(arch, [])
            if int(run) not in state["completed"][arch]:
                state["completed"][arch].append(int(run))
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            all_out.setdefault("models", {}).setdefault(arch, {})
            all_out["models"][arch]["arch"] = arch
            all_out["models"][arch]["per_run"] = [asdict(m) for m in run_metrics]
            results_json_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")

        u_accs = np.array([m.utterance_accuracy for m in run_metrics], dtype=np.float64)
        u_f1s = np.array([m.utterance_f1_macro for m in run_metrics], dtype=np.float64)
        u_aucs = np.array([m.utterance_auc_ovr_macro for m in run_metrics], dtype=np.float64)

        accs = np.array([m.subject_accuracy for m in run_metrics], dtype=np.float64)
        f1s = np.array([m.subject_f1_macro for m in run_metrics], dtype=np.float64)
        aucs = np.array([m.subject_auc_ovr_macro for m in run_metrics], dtype=np.float64)

        model_summary = {
            "arch": arch,
            "utterance_accuracy_mean": float(np.nanmean(u_accs)),
            "utterance_accuracy_std": float(np.nanstd(u_accs)),
            "utterance_f1_macro_mean": float(np.nanmean(u_f1s)),
            "utterance_f1_macro_std": float(np.nanstd(u_f1s)),
            "utterance_auc_ovr_macro_mean": float(np.nanmean(u_aucs)),
            "utterance_auc_ovr_macro_std": float(np.nanstd(u_aucs)),
            "subject_accuracy_mean": float(np.nanmean(accs)),
            "subject_accuracy_std": float(np.nanstd(accs)),
            "subject_f1_macro_mean": float(np.nanmean(f1s)),
            "subject_f1_macro_std": float(np.nanstd(f1s)),
            "subject_auc_ovr_macro_mean": float(np.nanmean(aucs)),
            "subject_auc_ovr_macro_std": float(np.nanstd(aucs)),
            "per_run": [asdict(m) for m in run_metrics],
        }
        all_out["models"][arch] = model_summary

        print(
            f"{arch} summary (utterance): acc={model_summary['utterance_accuracy_mean']:.4f}±{model_summary['utterance_accuracy_std']:.4f} "
            f"f1_macro={model_summary['utterance_f1_macro_mean']:.4f}±{model_summary['utterance_f1_macro_std']:.4f} "
            f"auc={model_summary['utterance_auc_ovr_macro_mean']:.4f}±{model_summary['utterance_auc_ovr_macro_std']:.4f}",
            flush=True,
        )
        print(
            f"{arch} summary (subject)  : acc={model_summary['subject_accuracy_mean']:.4f}±{model_summary['subject_accuracy_std']:.4f} "
            f"f1_macro={model_summary['subject_f1_macro_mean']:.4f}±{model_summary['subject_f1_macro_std']:.4f} "
            f"auc={model_summary['subject_auc_ovr_macro_mean']:.4f}±{model_summary['subject_auc_ovr_macro_std']:.4f}",
            flush=True,
        )

    results_json_path.write_text(json.dumps(all_out, indent=2), encoding="utf-8")
    print(f"\nSaved: {results_json_path}", flush=True)

    # Summary CSVs (one row per arch)
    rows_mean = []
    rows_pm = []
    for arch, summ in all_out["models"].items():
        rows_mean.append(
            {
                "model": arch,
                "utterance_acc": summ.get("utterance_accuracy_mean"),
                "utterance_macro_f1": summ.get("utterance_f1_macro_mean"),
                "utterance_macro_auc_ovr": summ.get("utterance_auc_ovr_macro_mean"),
                "subject_acc": summ.get("subject_accuracy_mean"),
                "subject_macro_f1": summ.get("subject_f1_macro_mean"),
                "subject_macro_auc_ovr": summ.get("subject_auc_ovr_macro_mean"),
            }
        )

        def pm(mean_key: str, std_key: str) -> str:
            m = summ.get(mean_key)
            s = summ.get(std_key)
            if m is None or s is None or (isinstance(m, float) and np.isnan(m)) or (isinstance(s, float) and np.isnan(s)):
                return ""
            return f"{float(m):.4f} ± {float(s):.4f}"

        rows_pm.append(
            {
                "model": arch,
                "utterance_acc": pm("utterance_accuracy_mean", "utterance_accuracy_std"),
                "utterance_macro_f1": pm("utterance_f1_macro_mean", "utterance_f1_macro_std"),
                "utterance_macro_auc_ovr": pm("utterance_auc_ovr_macro_mean", "utterance_auc_ovr_macro_std"),
                "subject_acc": pm("subject_accuracy_mean", "subject_accuracy_std"),
                "subject_macro_f1": pm("subject_f1_macro_mean", "subject_f1_macro_std"),
                "subject_macro_auc_ovr": pm("subject_auc_ovr_macro_mean", "subject_auc_ovr_macro_std"),
            }
        )

    pd.DataFrame(
        rows_mean,
        columns=[
            "model",
            "utterance_acc",
            "utterance_macro_f1",
            "utterance_macro_auc_ovr",
            "subject_acc",
            "subject_macro_f1",
            "subject_macro_auc_ovr",
        ],
    ).to_csv(results_csv_path, index=False)
    print(f"Saved: {results_csv_path}", flush=True)

    pd.DataFrame(
        rows_pm,
        columns=[
            "model",
            "utterance_acc",
            "utterance_macro_f1",
            "utterance_macro_auc_ovr",
            "subject_acc",
            "subject_macro_f1",
            "subject_macro_auc_ovr",
        ],
    ).to_csv(results_csv_pm_path, index=False)
    print(f"Saved: {results_csv_pm_path}", flush=True)

    # Also write timestamped snapshots for paper bookkeeping
    snap_json = results_dir / f"cnn_subject_metrics_{created_at}.json"
    snap_csv = results_dir / f"cnn_subject_metrics_summary_{created_at}.csv"
    snap_csv_pm = results_dir / f"cnn_subject_metrics_summary_pm_{created_at}.csv"
    shutil.copyfile(results_json_path, snap_json)
    shutil.copyfile(results_csv_path, snap_csv)
    shutil.copyfile(results_csv_pm_path, snap_csv_pm)
    print(f"Saved snapshot: {snap_json}", flush=True)
    print(f"Saved snapshot: {snap_csv}", flush=True)
    print(f"Saved snapshot: {snap_csv_pm}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
