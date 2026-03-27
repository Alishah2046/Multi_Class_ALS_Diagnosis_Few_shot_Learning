from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class FoldMetrics:
    fold: int
    n_subjects: int
    utterance_accuracy: float
    utterance_f1_macro: float
    utterance_auc_ovr_macro: float
    subject_accuracy: float
    subject_f1_macro: float
    subject_auc_ovr_macro: float


@dataclass(frozen=True)
class SplitInfo:
    seed: int
    n_subjects_total: int
    n_subjects_train: int
    n_subjects_val: int
    n_subjects_test: int
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_features_csv(repo_root: Path) -> Path:
    path = repo_root / "data" / "filtered" / "final_metadata_acoustic_features.csv"
    if path.exists():
        return path
    raise FileNotFoundError(f"Could not find features CSV at: {path}")


def _prepare_xy_groups(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    required = {"label", "subject_uid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "file_path" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["file_path"], keep="last").reset_index(drop=True)
        removed = before - len(df)
        if removed:
            print(f"Dedup: removed {removed} duplicate rows by `file_path`.", flush=True)

    y = df["label"].astype(str).to_numpy()
    groups = df["subject_uid"].astype(str).to_numpy()

    drop_cols = {
        "label",
        "subjectID",
        "subject_uid",
        "file_path",
        "file_name",
        "utterance_type",
        "utterance_id",
        "resolved_path",
        "voiced_file_path",
        "updated_file_path",
        "filtered_file_path",
        "Dataset",
        "Phoneme",
        "Sex",
        "Age",
        "Severity",
    }
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X_df = X_df.select_dtypes(include=[np.number]).copy()
    X_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if X_df.shape[1] == 0:
        raise ValueError("No numeric feature columns found in the CSV.")

    labels = sorted(pd.unique(y).tolist())
    X = X_df.to_numpy(dtype=np.float64, copy=False)
    return X, y, groups, labels


def _subject_wise_split_80_20(
    *,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> SplitInfo:
    """
    Subject-wise split: 80% TRAIN, 20% TEST (no separate VAL).

    This matches the protocol where you previously created an 80/10/10 split and then
    merged VAL+TEST into a single 20% TEST set.
    """
    uniq_subjects = np.unique(groups).astype(str)

    # Majority label per subject (handles the rare case where subject has mixed utterance labels)
    subj_y: list[str] = []
    mixed = 0
    for sid in uniq_subjects:
        ys = y[groups == sid]
        vals, cnts = np.unique(ys, return_counts=True)
        maj = str(vals[int(np.argmax(cnts))])
        subj_y.append(maj)
        if len(vals) > 1:
            mixed += 1

    subj_y_arr = np.asarray(subj_y, dtype=object)

    subj_train, subj_test, _, _ = train_test_split(
        uniq_subjects,
        subj_y_arr,
        test_size=0.2,
        stratify=subj_y_arr,
        random_state=seed,
    )

    set_train = set(map(str, subj_train.tolist()))
    set_test = set(map(str, subj_test.tolist()))

    # Leakage checks (subject overlaps must be empty)
    if set_train & set_test:
        raise RuntimeError("Subject leakage detected in split (overlapping subject_uids).")

    if mixed:
        print(f"Split note: {mixed} subjects had mixed utterance labels (majority label used).", flush=True)

    return SplitInfo(
        seed=int(seed),
        n_subjects_total=int(len(uniq_subjects)),
        n_subjects_train=int(len(set_train)),
        n_subjects_val=0,
        n_subjects_test=int(len(set_test)),
        train_subjects=sorted(set_train),
        val_subjects=[],
        test_subjects=sorted(set_test),
    )


def _scores_for_auc(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            # binary case (not expected for your 3-class run)
            return np.vstack([-scores, scores]).T
        return scores
    raise ValueError("Model does not support predict_proba or decision_function (cannot compute AUC).")


def _aggregate_subject_level(
    *,
    subject_ids: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (y_true_subject, y_score_subject) where y_score is averaged over utterances.
    """
    df = pd.DataFrame({"subject": subject_ids, "y_true": y_true})
    for i, lab in enumerate(labels):
        df[f"score_{lab}"] = y_score[:, i]

    # true label per subject (should be constant)
    y_true_sub = df.groupby("subject")["y_true"].first()
    score_cols = [f"score_{lab}" for lab in labels]
    y_score_sub = df.groupby("subject")[score_cols].mean()

    return y_true_sub.to_numpy(), y_score_sub.to_numpy(dtype=np.float64)


def _metrics_subject_level(
    *,
    y_true_sub: np.ndarray,
    y_score_sub: np.ndarray,
    labels: list[str],
) -> tuple[float, float, float]:
    y_pred_sub = np.asarray([labels[i] for i in np.argmax(y_score_sub, axis=1)])

    acc = float(accuracy_score(y_true_sub, y_pred_sub))
    f1m = float(f1_score(y_true_sub, y_pred_sub, average="macro"))

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    y_true_int = np.asarray([label_to_idx[v] for v in y_true_sub], dtype=int)
    auc = float(roc_auc_score(y_true_int, y_score_sub, multi_class="ovr", average="macro"))
    return acc, f1m, auc


def _metrics_utterance_level(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: list[str],
) -> tuple[float, float, float]:
    """
    Utterance-level (micro) metrics on ALL test utterances.
    Note: this will weight subjects with more utterances more heavily.
    """
    y_pred = np.asarray([labels[i] for i in np.argmax(y_score, axis=1)])
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro"))

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    y_true_int = np.asarray([label_to_idx[str(v)] for v in y_true], dtype=int)
    try:
        auc = float(roc_auc_score(y_true_int, y_score, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return acc, f1m, auc


def _run_model_cv(
    *,
    name: str,
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
    folds: int,
    seed: int,
) -> dict[str, Any]:
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_rows: list[FoldMetrics] = []

    print(f"\n== {name} ==", flush=True)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_test = groups[test_idx]

        pipeline.fit(X_train, y_train)
        y_score = _scores_for_auc(pipeline, X_test)

        # Ensure score columns match `labels` order (roc_auc_score expects consistent ordering)
        clf = pipeline.named_steps.get("clf", pipeline)
        if hasattr(clf, "classes_"):
            classes = [str(c) for c in clf.classes_]
            if set(classes) == set(labels) and classes != labels:
                col_idx = [classes.index(lab) for lab in labels]
                y_score = y_score[:, col_idx]

        utt_acc, utt_f1m, utt_auc = _metrics_utterance_level(y_true=y_test, y_score=y_score, labels=labels)

        y_true_sub, y_score_sub = _aggregate_subject_level(
            subject_ids=g_test, y_true=y_test, y_score=y_score, labels=labels
        )
        try:
            acc, f1m, auc = _metrics_subject_level(
                y_true_sub=y_true_sub, y_score_sub=y_score_sub, labels=labels
            )
        except ValueError as e:
            # If a fold is missing a class at subject-level, AUC cannot be computed.
            acc = float(accuracy_score(y_true_sub, np.asarray([labels[i] for i in np.argmax(y_score_sub, axis=1)])))
            f1m = float(f1_score(y_true_sub, np.asarray([labels[i] for i in np.argmax(y_score_sub, axis=1)]), average="macro"))
            auc = float("nan")
            print(f"fold {fold}: AUC skipped ({e})", flush=True)

        fm = FoldMetrics(
            fold=fold,
            n_subjects=int(len(np.unique(g_test))),
            utterance_accuracy=float(utt_acc),
            utterance_f1_macro=float(utt_f1m),
            utterance_auc_ovr_macro=float(utt_auc),
            subject_accuracy=acc,
            subject_f1_macro=f1m,
            subject_auc_ovr_macro=auc,
        )
        fold_rows.append(fm)
        print(
            f"fold {fold}: subject_acc={acc:.4f} subject_f1_macro={f1m:.4f} subject_auc={auc:.4f} (n_subjects={fm.n_subjects})",
            flush=True,
        )

    u_accs = np.array([r.utterance_accuracy for r in fold_rows], dtype=np.float64)
    u_f1s = np.array([r.utterance_f1_macro for r in fold_rows], dtype=np.float64)
    u_aucs = np.array([r.utterance_auc_ovr_macro for r in fold_rows], dtype=np.float64)
    accs = np.array([r.subject_accuracy for r in fold_rows], dtype=np.float64)
    f1s = np.array([r.subject_f1_macro for r in fold_rows], dtype=np.float64)
    aucs = np.array([r.subject_auc_ovr_macro for r in fold_rows], dtype=np.float64)

    summary = {
        "model": name,
        "folds": folds,
        "seed": seed,
        "utterance_accuracy_mean": float(u_accs.mean()),
        "utterance_accuracy_std": float(u_accs.std()),
        "utterance_f1_macro_mean": float(u_f1s.mean()),
        "utterance_f1_macro_std": float(u_f1s.std()),
        "utterance_auc_ovr_macro_mean": float(u_aucs.mean()),
        "utterance_auc_ovr_macro_std": float(u_aucs.std()),
        "subject_accuracy_mean": float(accs.mean()),
        "subject_accuracy_std": float(accs.std()),
        "subject_f1_macro_mean": float(f1s.mean()),
        "subject_f1_macro_std": float(f1s.std()),
        "subject_auc_ovr_macro_mean": float(aucs.mean()),
        "subject_auc_ovr_macro_std": float(aucs.std()),
        "per_fold": [asdict(r) for r in fold_rows],
    }

    print(
        f"{name} summary (utterance): acc={summary['utterance_accuracy_mean']:.4f}±{summary['utterance_accuracy_std']:.4f} "
        f"f1_macro={summary['utterance_f1_macro_mean']:.4f}±{summary['utterance_f1_macro_std']:.4f} "
        f"auc={summary['utterance_auc_ovr_macro_mean']:.4f}±{summary['utterance_auc_ovr_macro_std']:.4f}",
        flush=True,
    )
    print(
        f"{name} summary: acc={summary['subject_accuracy_mean']:.4f}±{summary['subject_accuracy_std']:.4f} "
        f"f1_macro={summary['subject_f1_macro_mean']:.4f}±{summary['subject_f1_macro_std']:.4f} "
        f"auc={summary['subject_auc_ovr_macro_mean']:.4f}±{summary['subject_auc_ovr_macro_std']:.4f}",
        flush=True,
    )
    return summary


def _run_model_fixed_split(
    *,
    name: str,
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
    split: SplitInfo,
    merge_train_val: bool,
) -> dict[str, Any]:
    set_train = set(split.train_subjects)
    set_val = set(split.val_subjects)
    set_test = set(split.test_subjects)

    train_idx = np.where(np.isin(groups, list(set_train)))[0]
    val_idx = np.where(np.isin(groups, list(set_val)))[0]
    test_idx = np.where(np.isin(groups, list(set_test)))[0]

    if merge_train_val:
        fit_idx = np.concatenate([train_idx, val_idx])
    else:
        fit_idx = train_idx

    X_fit, y_fit = X[fit_idx], y[fit_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    g_test = groups[test_idx]

    print(f"\n== {name} ==", flush=True)
    print(
        f"split: subjects train/test = {split.n_subjects_train}/{split.n_subjects_test} "
        f"(seed={split.seed}, merge_train_val={merge_train_val})",
        flush=True,
    )
    print(f"rows  : fit={len(fit_idx)} test={len(test_idx)}", flush=True)

    pipeline.fit(X_fit, y_fit)
    y_score = _scores_for_auc(pipeline, X_test)

    # Ensure score columns match `labels` order (roc_auc_score expects consistent ordering)
    clf = pipeline.named_steps.get("clf", pipeline)
    if hasattr(clf, "classes_"):
        classes = [str(c) for c in clf.classes_]
        if set(classes) == set(labels) and classes != labels:
            col_idx = [classes.index(lab) for lab in labels]
            y_score = y_score[:, col_idx]

    utt_acc, utt_f1m, utt_auc = _metrics_utterance_level(y_true=y_test, y_score=y_score, labels=labels)

    y_true_sub, y_score_sub = _aggregate_subject_level(
        subject_ids=g_test, y_true=y_test, y_score=y_score, labels=labels
    )
    acc, f1m, auc = _metrics_subject_level(y_true_sub=y_true_sub, y_score_sub=y_score_sub, labels=labels)

    summary = {
        "model": name,
        "seed": int(split.seed),
        "protocol": "subject_split_80_train_20_test",
        "merge_train_val": bool(merge_train_val),
        "n_subjects_total": int(split.n_subjects_total),
        "n_subjects_train": int(split.n_subjects_train),
        "n_subjects_val": int(split.n_subjects_val),
        "n_subjects_test": int(split.n_subjects_test),
        "n_rows_fit": int(len(fit_idx)),
        "n_rows_test": int(len(test_idx)),
        "utterance_accuracy": float(utt_acc),
        "utterance_f1_macro": float(utt_f1m),
        "utterance_auc_ovr_macro": float(utt_auc),
        "subject_accuracy": float(acc),
        "subject_f1_macro": float(f1m),
        "subject_auc_ovr_macro": float(auc),
    }

    print(
        f"{name} test (utterance): acc={summary['utterance_accuracy']:.4f} "
        f"f1_macro={summary['utterance_f1_macro']:.4f} "
        f"auc={summary['utterance_auc_ovr_macro']:.4f}",
        flush=True,
    )
    print(
        f"{name} test (subject)  : acc={summary['subject_accuracy']:.4f} "
        f"f1_macro={summary['subject_f1_macro']:.4f} "
        f"auc={summary['subject_auc_ovr_macro']:.4f}",
        flush=True,
    )
    return summary


def main() -> int:
    # Click-run config (3-class ALSwDysarthria vs ALSwoDysarthria vs Control)
    # Match your updated protocol: 80% train, 20% test (subject-wise).
    SEED = 42
    MERGE_TRAIN_VAL = False

    repo_root = _repo_root()
    os.chdir(repo_root)

    features_csv = _load_features_csv(repo_root)
    results_dir = repo_root / "results" / "als_diagnosis_meta_3class_repo_baselines"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("3-class baseline training (repo script)", flush=True)
    print(f"Input CSV: {features_csv}", flush=True)
    print(f"Results dir: {results_dir}", flush=True)
    print(f"Seed: {SEED} | protocol: 80% train / 20% test (subject-wise)", flush=True)

    df = pd.read_csv(features_csv)
    X, y, groups, labels = _prepare_xy_groups(df)

    print(f"Rows: {len(y)} | Labels: {labels}", flush=True)
    print(f"Unique subjects: {len(np.unique(groups))}", flush=True)
    print(f"Numeric features: {X.shape[1]}", flush=True)

    split = _subject_wise_split_80_20(y=y, groups=groups, seed=SEED)
    split_path = results_dir / f"subject_split_seed{SEED}.json"
    split_path.write_text(json.dumps(asdict(split), indent=2), encoding="utf-8")
    print(f"Saved split: {split_path}", flush=True)

    models: list[tuple[str, Pipeline]] = [
        (
            "logreg",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=4000,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        ),
        (
            "svm_linear",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    # Use SVC(probability=True) so multiclass AUC (OvR) is well-defined
                    # with probabilities that sum to 1 across classes.
                    ("clf", SVC(kernel="linear", class_weight="balanced", probability=True, random_state=SEED)),
                ]
            ),
        ),
        (
            "random_forest",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=500,
                            random_state=SEED,
                            n_jobs=-1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            ),
        ),
    ]

    all_results: dict[str, Any] = {
        "task": "3-class",
        "labels": labels,
        "features_csv": str(features_csv),
        "n_rows": int(len(y)),
        "n_subjects": int(len(np.unique(groups))),
        "n_features_numeric": int(X.shape[1]),
        "seed": int(SEED),
        "protocol": "subject_split_80_train_20_test",
        "merge_train_val": bool(MERGE_TRAIN_VAL),
        "split_path": str(split_path),
        "models": {},
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    for name, pipe in models:
        all_results["models"][name] = _run_model_fixed_split(
            name=name,
            pipeline=pipe,
            X=X,
            y=y,
            groups=groups,
            labels=labels,
            split=split,
            merge_train_val=MERGE_TRAIN_VAL,
        )

    out_path = results_dir / f"baselines_subject_metrics_{all_results['created_at']}.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}", flush=True)

    summary_rows = []
    for model_name, summary in all_results["models"].items():
        summary_rows.append(
            {
                "model": model_name,
                "utterance_acc": summary.get("utterance_accuracy"),
                "utterance_macro_f1": summary.get("utterance_f1_macro"),
                "utterance_macro_auc_ovr": summary.get("utterance_auc_ovr_macro"),
                "subject_acc": summary.get("subject_accuracy"),
                "subject_macro_f1": summary.get("subject_f1_macro"),
                "subject_macro_auc_ovr": summary.get("subject_auc_ovr_macro"),
            }
        )
    summary_csv = results_dir / f"baselines_subject_metrics_summary_{all_results['created_at']}.csv"
    pd.DataFrame(
        summary_rows,
        columns=[
            "model",
            "utterance_acc",
            "utterance_macro_f1",
            "utterance_macro_auc_ovr",
            "subject_acc",
            "subject_macro_f1",
            "subject_macro_auc_ovr",
        ],
    ).to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
