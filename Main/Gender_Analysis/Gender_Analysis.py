# Auto-generated from code/experiment4_gender_fair_wav2vec_stability.ipynb

from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# ===================== Paths (robust to running from /code) =====================

def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / 'Results').is_dir() and (p / 'code').is_dir() and (p / 'embeddings').is_dir():
            return p
    return start

BASE = find_project_root(Path.cwd())
MODEL_DIR = BASE / 'embeddings' / 'wav2vec'

# Output run folder (keeps results separate)
RUN_TAG = datetime.now().strftime('run_%Y%m%d_%H%M%S')
RESULTS_DIR = BASE / 'Results' / 'AL_MODELS' / 'exp4_gender_fair_wav2vec_stability' / RUN_TAG
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Output files
SUMMARY_CSV = RESULTS_DIR / 'experiment_summary_gender_fair_wav2vec_stability.csv'
PER_SEED_CSV = RESULTS_DIR / 'gender_fair_wav2vec_stability_per_seed.csv'
SUBJECT_DETAILS_CSV = RESULTS_DIR / 'gender_fair_wav2vec_stability_subject_details.csv'
META_JSONL = RESULTS_DIR / 'gender_fair_wav2vec_stability_meta.jsonl'

print('BASE       :', BASE)
print('MODEL_DIR  :', MODEL_DIR, '| exists:', MODEL_DIR.is_dir())
print('RESULTS_DIR:', RESULTS_DIR)
print('SUMMARY_CSV:', SUMMARY_CSV)

# ===================== Fixed split + run settings =====================
SPLIT_SEED = 42
INIT_SEEDS = list(range(30))

# No validation
TEST_SIZE = 0.20

# Optional down-sampling to match per-class counts across genders (computed below)
DO_DOWNSAMPLE = False
DOWNSAMPLE_SEED = 42
TARGET_CLASS_COUNTS = None  # computed below

# Optional SMOTE (train only)
DO_SMOTE = True

# ===================== Model + training configs (match Experiment 1) =====================
K_SHOT = 5
Q_QUERY = 15

MODEL_CFG = dict(
    hidden_dim=512,
    feature_dim=256,
    context_dim=64,
    dropout_p1=0.3,
    dropout_p2=0.2,
    use_metric_learner=True,
    distance_metric='cosine',
    distance_scale_init=10.0,
)

TRAIN_CFG = dict(
    num_epochs=120,
    meta_batch_size=16,
    patience=10**9,
    lr=3e-5,
    weight_decay=0.0,
    scheduler_patience=10,
    scheduler_factor=0.5,
    explicit_l2=True,
)

def _load_all() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    X = np.load(MODEL_DIR / 'xwav2vec2_stability.npy')
    y_raw = np.load(MODEL_DIR / 'labels.npy', allow_pickle=True)
    subj = np.load(MODEL_DIR / 'subject_ids.npy', allow_pickle=True).astype(str)
    files = np.load(MODEL_DIR / 'file_list.npy', allow_pickle=True).astype(str)
    label_map = np.load(MODEL_DIR / 'label_map.npy', allow_pickle=True)
    assert len(X) == len(y_raw) == len(subj) == len(files)
    return X, y_raw, subj, files, label_map


def _gender_from_subject_id(subject_id: str) -> str:
    token = str(subject_id).split('/')[-1].strip()
    if not token:
        return 'UNKNOWN'
    first = token[0].upper()
    if first == 'F':
        return 'FEMALE'
    if first == 'M':
        return 'MALE'
    return 'UNKNOWN'


def _downsample_to_targets(
    X: np.ndarray,
    y: np.ndarray,
    subj: np.ndarray,
    files: np.ndarray,
    targets: dict[int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Downsample utterances to match per-class target counts, while keeping every subject (>=1 utt).
    # Assumes each subject has a single label.
    rng = np.random.default_rng(seed)
    keep_idx: list[int] = []

    for cls, n_target in targets.items():
        idx_cls = np.where(y == cls)[0]
        if len(idx_cls) < n_target:
            raise ValueError(f'Not enough samples for class {cls}: have {len(idx_cls)} need {n_target}')

        # ensure every subject in this class keeps at least 1 utterance
        subj_cls = subj[idx_cls]
        uniq_subj = np.unique(subj_cls)
        if n_target < len(uniq_subj):
            raise ValueError(
                f'Target {n_target} is smaller than number of subjects {len(uniq_subj)} for class {cls}.'
            )

        first_picks = []
        remaining = set(idx_cls.tolist())
        for s in uniq_subj:
            idx_s = idx_cls[subj_cls == s]
            pick = int(rng.choice(idx_s))
            first_picks.append(pick)
            remaining.discard(pick)

        need_more = n_target - len(first_picks)
        if need_more > 0:
            rem_list = np.array(sorted(remaining), dtype=int)
            extra = rng.choice(rem_list, size=need_more, replace=False)
            picks = np.concatenate([np.array(first_picks, dtype=int), extra])
        else:
            picks = np.array(first_picks, dtype=int)

        keep_idx.extend(picks.tolist())

    keep_idx = np.array(sorted(keep_idx), dtype=int)
    return X[keep_idx], y[keep_idx], subj[keep_idx], files[keep_idx]


def _subject_stratified_split_80_20(
    subj: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    uniq_subj = np.unique(subj)
    subj_y = np.zeros(len(uniq_subj), dtype=int)
    for i, sid in enumerate(uniq_subj):
        ys = y[subj == sid]
        vals, cnts = np.unique(ys, return_counts=True)
        subj_y[i] = int(vals[np.argmax(cnts)])

    subj_train, subj_test, _, _ = train_test_split(
        uniq_subj,
        subj_y,
        test_size=test_size,
        stratify=subj_y,
        random_state=seed,
    )
    return subj_train.astype(str), subj_test.astype(str)


def maybe_apply_smote(X_train, y_train, do_smote, seed=SPLIT_SEED):
    """
    Apply SMOTE to the training utterances only.

    Notes
    - Primary implementation uses `imblearn` when available.
    - Falls back to a minimal NumPy/Scikit-learn SMOTE implementation when
      `imblearn` is not installed, so this script can run in lightweight envs.
    """

    if not do_smote:
        return X_train, y_train

    # Try imbalanced-learn first (matches the main project notebooks).
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore

        sm = SMOTE(random_state=seed)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        return X_res, y_res
    except Exception:
        pass

    # Fallback: basic SMOTE (interpolation within each minority class).
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(int(seed))
    X = np.asarray(X_train)
    y = np.asarray(y_train)

    classes, counts = np.unique(y, return_counts=True)
    max_count = int(counts.max())

    X_parts = [X]
    y_parts = [y]

    for cls, cnt in zip(classes, counts):
        cnt = int(cnt)
        if cnt >= max_count:
            continue

        X_c = X[y == cls]
        need = max_count - cnt

        if len(X_c) == 0:
            continue
        if len(X_c) == 1:
            # Cannot interpolate with neighbors; duplicate the single sample.
            idx = rng.integers(0, 1, size=need)
            X_new = X_c[idx]
        else:
            k_eff = min(5, len(X_c) - 1)
            nn = NearestNeighbors(n_neighbors=k_eff + 1, metric='euclidean')
            nn.fit(X_c)
            neigh = nn.kneighbors(X_c, return_distance=False)[:, 1:]  # drop self

            base_idx = rng.integers(0, len(X_c), size=need)
            neigh_choice = rng.integers(0, k_eff, size=need)
            neigh_idx = neigh[base_idx, neigh_choice]

            lam = rng.random(need).astype(X.dtype, copy=False).reshape(-1, 1)
            X_new = X_c[base_idx] + lam * (X_c[neigh_idx] - X_c[base_idx])

        y_new = np.full((len(X_new),), cls, dtype=y.dtype)
        X_parts.append(X_new.astype(X.dtype, copy=False))
        y_parts.append(y_new)

    X_res = np.concatenate(X_parts, axis=0)
    y_res = np.concatenate(y_parts, axis=0)
    return X_res, y_res


def normalize(X_train, X_test):
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    return X_train, X_test

# ===================== Load data and compute gender labels =====================
X_ALL, y_raw_all, subj_all, files_all, label_map = _load_all()

LE_GLOBAL = LabelEncoder()
y_all = LE_GLOBAL.fit_transform(y_raw_all)
n_classes = int(len(LE_GLOBAL.classes_))
print('LE_GLOBAL.classes_:', [str(x) for x in LE_GLOBAL.classes_])

gender_all = np.array([_gender_from_subject_id(s) for s in subj_all], dtype=object)
mask_f = gender_all == 'FEMALE'
mask_m = gender_all == 'MALE'
print('Female utterances:', int(mask_f.sum()), 'Male utterances:', int(mask_m.sum()))

# ====================== TASK GENERATOR ======================
class BalancedTaskGenerator:
    def __init__(self, features, labels, n_way, k_shot, q_query):
        self.features = features
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        self.class_indices = {}
        for idx, label in enumerate(labels):
            self.class_indices.setdefault(int(label), []).append(idx)
        self.classes = list(self.class_indices.keys())

    def create_task(self):
        selected_classes = random.sample(self.classes, self.n_way)
        support_set = []
        query_set = []
        for cls in selected_classes:
            samples = random.sample(self.class_indices[cls], self.k_shot + self.q_query)
            support_set.extend([(samples[i], cls) for i in range(self.k_shot)])
            query_set.extend(
                [(samples[i], cls) for i in range(self.k_shot, self.k_shot + self.q_query)]
            )
        support_features = torch.stack([torch.FloatTensor(self.features[idx]) for idx, _ in support_set])
        support_labels = torch.LongTensor([label for _, label in support_set])
        query_features = torch.stack([torch.FloatTensor(self.features[idx]) for idx, _ in query_set])
        query_labels = torch.LongTensor([label for _, label in query_set])
        return support_features, support_labels, query_features, query_labels


# ====================== MODEL DEFINITIONS ======================
class ClassLSTM(nn.Module):
    def __init__(self, feature_dim, context_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=context_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(context_dim, feature_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        h_mean = output.mean(dim=1)
        return self.fc(h_mean)


class HyperMetaLearner(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        feature_dim,
        context_dim,
        num_classes,
        use_metric_learner=False,
        distance_metric="cosine",
        distance_scale_init=10.0,
        dropout_p1=0.3,
        dropout_p2=0.2,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.use_metric_learner = use_metric_learner
        self.distance_metric = distance_metric

        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p2),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        self.class_lstm = ClassLSTM(feature_dim=feature_dim, context_dim=context_dim)

        self.metric_learner = (
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim),
            )
            if use_metric_learner
            else nn.Identity()
        )

        self.distance_scale = nn.Parameter(torch.tensor(float(distance_scale_init)))

    def forward(self, x):
        return self.feature_net(x)

    def compute_prototypes(self, support, support_labels, n_way):
        device_ = next(self.parameters()).device
        support_features = self.feature_net(support)
        prototypes = torch.zeros(n_way, self.feature_dim).to(device_)
        for cls in range(n_way):
            class_mask = support_labels == cls
            class_features = support_features[class_mask]
            if len(class_features) > 0:
                class_features = class_features.unsqueeze(0)
                prototypes[cls] = self.class_lstm(class_features).squeeze(0)
        return prototypes

    def compute_distances(self, prototypes, query_features):
        prototypes = prototypes + self.metric_learner(prototypes)
        query_features = query_features + self.metric_learner(query_features)
        if self.distance_metric == "cosine":
            q = F.normalize(query_features, p=2, dim=-1)
            p = F.normalize(prototypes, p=2, dim=-1)
            return (1 - torch.matmul(q, p.T)) * self.distance_scale
        if self.distance_metric == "euclidean":
            return torch.cdist(query_features, prototypes, p=2) * self.distance_scale
        raise ValueError(f"Unsupported distance metric: {self.distance_metric}")


# ====================== LOSS & METRICS ======================
def prototypical_loss(model, support, support_labels, query, query_labels, n_way):
    device_ = next(model.parameters()).device
    support, query = support.to(device_), query.to(device_)
    support_labels, query_labels = support_labels.to(device_), query_labels.to(device_)

    query_features = model(query)
    prototypes = model.compute_prototypes(support, support_labels, n_way)
    distances = model.compute_distances(prototypes, query_features)

    log_p_y = F.log_softmax(-distances, dim=1)
    loss = F.nll_loss(log_p_y, query_labels)
    _, predictions = torch.max(log_p_y, 1)
    acc = torch.mean((predictions == query_labels).float())

    class_probs = F.softmax(-distances, dim=1).detach().cpu().numpy()
    return loss, acc, predictions, query_labels, class_probs


def calculate_auc(all_probs, all_labels, n_classes):
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning

    if n_classes == 2:
        if all_probs.ndim == 2 and all_probs.shape[1] == 2:
            all_probs = all_probs[:, 1]
        elif all_probs.ndim == 2 and all_probs.shape[1] == 1:
            all_probs = all_probs.ravel()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            auc = roc_auc_score(all_labels, all_probs)
        return auc, [auc]
    binarized_labels = label_binarize(all_labels, classes=range(n_classes))
    auc_scores = []
    for i in range(n_classes):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            try:
                auc = roc_auc_score(binarized_labels[:, i], all_probs[:, i])
            except ValueError:
                # Undefined when a class is absent in the evaluated subset.
                auc = float("nan")
        auc_scores.append(float(auc))
    macro_auc = float("nan") if any(np.isnan(auc_scores)) else float(np.mean(auc_scores))
    return macro_auc, auc_scores


def train_model(
    model,
    train_task_gen,
    n_classes,
    num_epochs=120,
    meta_batch_size=16,
    lr=3e-5,
    weight_decay=0.0,
    scheduler_patience=10,
    scheduler_factor=0.5,
    explicit_l2=True,
):
    device_ = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=scheduler_patience, factor=scheduler_factor
    )
    best_monitor = -1e9
    best_state_dict = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        for _ in range(meta_batch_size):
            support, support_labels, query, query_labels = train_task_gen.create_task()
            optimizer.zero_grad()
            loss, acc, _, _, _ = prototypical_loss(
                model, support, support_labels, query, query_labels, n_way=n_classes
            )
            if explicit_l2 and weight_decay != 0.0:
                l2_reg = 0.0
                for p in model.parameters():
                    l2_reg = l2_reg + torch.norm(p)
                loss = loss + weight_decay * l2_reg
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            epoch_acc += float(acc.item())

        avg_train_acc = epoch_acc / meta_batch_size
        scheduler.step(avg_train_acc)

        if avg_train_acc > best_monitor:
            best_monitor = avg_train_acc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best_state_dict


def test_model(model, test_task_gen, n_classes, meta_batch_size=16):
    device_ = next(model.parameters()).device
    model.eval()
    test_acc = 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for _ in range(meta_batch_size * 2):
            support, support_labels, query, query_labels = test_task_gen.create_task()
            support, query = support.to(device_), query.to(device_)
            support_labels, query_labels = support_labels.to(device_), query_labels.to(device_)
            loss, acc, preds, true_labels, probs = prototypical_loss(
                model, support, support_labels, query, query_labels, n_way=n_classes
            )
            test_acc += float(acc.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
            all_probs.extend(probs)
    test_acc = test_acc / (meta_batch_size * 2)
    return test_acc, np.asarray(all_preds), np.asarray(all_labels), np.asarray(all_probs)


def compute_class_prototypes_from_train(model, X_tr, y_tr, n_classes, batch_size=512):
    device_ = next(model.parameters()).device
    model.eval()
    sums = torch.zeros(n_classes, model.feature_dim, device=device_)
    counts = torch.zeros(n_classes, device=device_, dtype=torch.long)
    with torch.no_grad():
        for i in range(0, len(X_tr), batch_size):
            xb = torch.FloatTensor(X_tr[i : i + batch_size]).to(device_)
            yb = torch.LongTensor(y_tr[i : i + batch_size]).to(device_)
            feats = model(xb)
            for cls in range(n_classes):
                m = yb == cls
                if torch.any(m):
                    sums[cls] += feats[m].sum(dim=0)
                    counts[cls] += int(m.sum().item())
    prototypes = sums / counts.clamp(min=1).unsqueeze(1)
    return prototypes


def predict_probs_from_prototypes(model, prototypes, X):
    device_ = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xb = torch.FloatTensor(X).to(device_)
        q = model(xb)
        d = model.compute_distances(prototypes, q)
        probs = F.softmax(-d, dim=1).detach().cpu().numpy()
    return probs


def subject_metrics_from_all_test_utterances(subj, y_true, probs, n_classes, return_details=False):
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    true_by_subj: dict[str, int] = {}
    for sid, yt, pr in zip(subj, y_true, probs):
        sid = str(sid)
        if sid not in sums:
            sums[sid] = np.zeros((n_classes,), dtype=np.float64)
            counts[sid] = 0
            true_by_subj[sid] = int(yt)
        sums[sid] += np.asarray(pr, dtype=np.float64)
        counts[sid] += 1

    subjects = sorted(sums.keys())
    y_true_sub = np.array([true_by_subj[s] for s in subjects], dtype=int)
    y_prob_sub = np.vstack([sums[s] / max(1, counts[s]) for s in subjects])
    y_pred_sub = np.argmax(y_prob_sub, axis=1)

    subj_acc = float(np.mean(y_pred_sub == y_true_sub))
    subj_f1 = float(f1_score(y_true_sub, y_pred_sub, average="macro"))
    try:
        subj_auc = float(roc_auc_score(y_true_sub, y_prob_sub, multi_class="ovr", average="macro"))
    except ValueError:
        subj_auc = float("nan")

    if return_details:
        return subj_acc, subj_f1, subj_auc, {
            "subjects": subjects,
            "n_utts": [int(counts[s]) for s in subjects],
            "y_true_sub": y_true_sub,
            "y_pred_sub": y_pred_sub,
            "y_prob_sub": y_prob_sub,
        }
    return subj_acc, subj_f1, subj_auc

def _subject_majority_labels(subj: np.ndarray, y: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in np.unique(subj.astype(str)):
        ys = y[subj.astype(str) == s]
        vals, cnts = np.unique(ys, return_counts=True)
        out[str(s)] = int(vals[int(np.argmax(cnts))])
    return out
def _counts_by_class(subjects: list[str], subj_to_y: dict[str, int]) -> dict[int, int]:
    c: dict[int, int] = {}
    for s in subjects:
        y = int(subj_to_y[str(s)])
        c[y] = c.get(y, 0) + 1
    return c
def _sample_subjects_stratified(pool: list[str], subj_to_y: dict[str, int], target_counts: dict[int, int], seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    by_cls: dict[int, list[str]] = {}
    for s in pool:
        cls = int(subj_to_y[str(s)])
        by_cls.setdefault(cls, []).append(str(s))
    out: list[str] = []
    for cls, n in target_counts.items():
        cand = by_cls.get(int(cls), [])
        if len(cand) < int(n):
            raise ValueError(f'Not enough subjects in class {cls}: have {len(cand)} need {n}')
        picks = rng.choice(np.array(cand, dtype=object), size=int(n), replace=False)
        out.extend([str(x) for x in picks.tolist()])
    return out
print('=== Experiment 4B: Fair gender evaluation on the same fixed test split ===')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)
# Subject-disjoint split on the full dataset
subj_train, subj_test = _subject_stratified_split_80_20(subj=subj_all, y=y_all, seed=SPLIT_SEED, test_size=TEST_SIZE)
set_train, set_test = set(subj_train.tolist()), set(subj_test.tolist())
train_idx = np.where(np.isin(subj_all, list(set_train)))[0]
test_idx = np.where(np.isin(subj_all, list(set_test)))[0]
X_train_raw, y_train_raw = X_ALL[train_idx], y_all[train_idx]
X_test_raw, y_test = X_ALL[test_idx], y_all[test_idx]
subj_test_arr = subj_all[test_idx].astype(str)
gender_test_arr = gender_all[test_idx].astype(str)
test_subjects = sorted(np.unique(subj_test_arr).tolist())
subj_to_y = _subject_majority_labels(subj_all.astype(str), y_all)
subj_to_gender = {str(s): _gender_from_subject_id(str(s)) for s in np.unique(subj_all.astype(str))}
female_test_subjects = [s for s in test_subjects if subj_to_gender.get(str(s), 'UNKNOWN') == 'FEMALE']
male_test_subjects = [s for s in test_subjects if subj_to_gender.get(str(s), 'UNKNOWN') == 'MALE']
print('test subjects total:', len(test_subjects), 'female:', len(female_test_subjects), 'male:', len(male_test_subjects))
print('female class counts (subjects):', _counts_by_class(female_test_subjects, subj_to_y))
print('male   class counts (subjects):', _counts_by_class(male_test_subjects, subj_to_y))
# Matching settings for optional fair comparison
DO_MATCH_MALE_TO_FEMALE = True
MATCH_ITERS = 200
per_seed_rows: list[dict] = []
details_rows: list[dict] = []
for seed in tqdm(INIT_SEEDS, desc='init seeds'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # SMOTE on training utterances only
    X_tr, y_tr = maybe_apply_smote(X_train_raw, y_train_raw, DO_SMOTE, seed=SPLIT_SEED)
    # Normalize using training statistics
    X_tr, X_te = normalize(X_tr, X_test_raw.copy())
    train_task_gen = BalancedTaskGenerator(X_tr, y_tr, n_way=n_classes, k_shot=K_SHOT, q_query=Q_QUERY)
    model = HyperMetaLearner(
        input_dim=int(X_tr.shape[1]),
        hidden_dim=MODEL_CFG['hidden_dim'],
        feature_dim=MODEL_CFG['feature_dim'],
        context_dim=MODEL_CFG['context_dim'],
        num_classes=n_classes,
        use_metric_learner=MODEL_CFG['use_metric_learner'],
        distance_metric=MODEL_CFG['distance_metric'],
        distance_scale_init=MODEL_CFG['distance_scale_init'],
        dropout_p1=MODEL_CFG['dropout_p1'],
        dropout_p2=MODEL_CFG['dropout_p2'],
    ).to(device)
    best_state = train_model(
        model,
        train_task_gen,
        n_classes,
        num_epochs=TRAIN_CFG['num_epochs'],
        meta_batch_size=TRAIN_CFG['meta_batch_size'],
        lr=TRAIN_CFG['lr'],
        weight_decay=TRAIN_CFG['weight_decay'],
        scheduler_patience=TRAIN_CFG['scheduler_patience'],
        scheduler_factor=TRAIN_CFG['scheduler_factor'],
        explicit_l2=TRAIN_CFG['explicit_l2'],
    )
    model.load_state_dict(best_state)
    # Prototype-based probabilities for ALL test utterances
    prototypes = compute_class_prototypes_from_train(model, X_tr, y_tr, n_classes)
    test_probs_full = predict_probs_from_prototypes(model, prototypes, X_te)
    for gender in ['FEMALE', 'MALE']:
        idxg = np.where(gender_test_arr == gender)[0]
        if len(idxg) == 0:
            continue
        y_true_g = y_test[idxg]
        probs_g = test_probs_full[idxg]
        y_pred_g = probs_g.argmax(axis=1)
        utt_acc = float((y_pred_g == y_true_g).mean())
        utt_f1 = float(f1_score(y_true_g, y_pred_g, average='macro', labels=list(range(n_classes)), zero_division=0))
        utt_auc, _ = calculate_auc(probs_g, y_true_g, n_classes)
        subj_ids_g = subj_test_arr[idxg]
        subj_acc, subj_f1, subj_auc, det = subject_metrics_from_all_test_utterances(
            subj_ids_g, y_true_g, probs_g, n_classes, return_details=True
        )
        per_seed_rows.append({
            'mode': 'full',
            'gender': gender,
            'do_smote': bool(DO_SMOTE),
            'split_seed': int(SPLIT_SEED),
            'init_seed': int(seed),
            'n_test_subjects': int(len(np.unique(subj_ids_g))),
            'utt_acc': float(utt_acc),
            'utt_macro_f1': float(utt_f1),
            'utt_macro_auc': float(utt_auc),
            'subject_acc': float(subj_acc),
            'subject_f1': float(subj_f1),
            'subject_auc': float(subj_auc),
        })
        for sid, n_utts, yt, yp, p in zip(det['subjects'], det['n_utts'], det['y_true_sub'], det['y_pred_sub'], det['y_prob_sub']):
            details_rows.append({
                'mode': 'full',
                'gender': gender,
                'do_smote': bool(DO_SMOTE),
                'split_seed': int(SPLIT_SEED),
                'init_seed': int(seed),
                'subject_id': str(sid),
                'n_utts': int(n_utts),
                'y_true': int(yt),
                'y_pred': int(yp),
                'p0': float(p[0]),
                'p1': float(p[1]),
                'p2': float(p[2]),
            })
    # Optional matched male analysis with utterance and subject metrics
    if DO_MATCH_MALE_TO_FEMALE and len(female_test_subjects) > 0 and len(male_test_subjects) > 0:
        target_counts = _counts_by_class(female_test_subjects, subj_to_y)
        male_metrics = []
        for r in range(int(MATCH_ITERS)):
            sampled = _sample_subjects_stratified(
                male_test_subjects, subj_to_y, target_counts,
                seed=int(SPLIT_SEED) * 100000 + int(seed) * 1000 + r
            )
            idxm = np.where(np.isin(subj_test_arr, np.array(sampled, dtype=object)))[0]
            y_true_m = y_test[idxm]
            probs_m = test_probs_full[idxm]
            y_pred_m = probs_m.argmax(axis=1)
            utt_acc_m = float((y_pred_m == y_true_m).mean())
            utt_f1_m = float(f1_score(y_true_m, y_pred_m, average='macro', labels=list(range(n_classes)), zero_division=0))
            utt_auc_m, _ = calculate_auc(probs_m, y_true_m, n_classes)
            subj_ids_m = subj_test_arr[idxm]
            subj_acc_m, subj_f1_m, subj_auc_m = subject_metrics_from_all_test_utterances(
                subj_ids_m, y_true_m, probs_m, n_classes, return_details=False
            )
            male_metrics.append((utt_acc_m, utt_f1_m, float(utt_auc_m), float(subj_acc_m), float(subj_f1_m), float(subj_auc_m)))
        arr = np.array(male_metrics, dtype=float)
        utt_auc_mean = float(np.nanmean(arr[:, 2])) if np.isfinite(arr[:, 2]).any() else float('nan')
        subj_auc_mean = float(np.nanmean(arr[:, 5])) if np.isfinite(arr[:, 5]).any() else float('nan')
        per_seed_rows.append({
            'mode': 'male_matched',
            'gender': 'MALE',
            'do_smote': bool(DO_SMOTE),
            'split_seed': int(SPLIT_SEED),
            'init_seed': int(seed),
            'n_test_subjects': int(len(female_test_subjects)),
            'utt_acc': float(np.mean(arr[:, 0])),
            'utt_macro_f1': float(np.mean(arr[:, 1])),
            'utt_macro_auc': utt_auc_mean,
            'subject_acc': float(np.mean(arr[:, 3])),
            'subject_f1': float(np.mean(arr[:, 4])),
            'subject_auc': subj_auc_mean,
            'matched_iters': int(MATCH_ITERS),
        })
per_seed_df = pd.DataFrame(per_seed_rows)
details_df = pd.DataFrame(details_rows)
per_seed_df.to_csv(PER_SEED_CSV, index=False)
details_df.to_csv(SUBJECT_DETAILS_CSV, index=False)
print('Saved:', PER_SEED_CSV)
print('Saved:', SUBJECT_DETAILS_CSV)
# Summary
summary_rows = []
for mode in sorted(per_seed_df['mode'].unique().tolist()):
    for gender in sorted(per_seed_df[per_seed_df['mode']==mode]['gender'].unique().tolist()):
        sub = per_seed_df[(per_seed_df['mode']==mode) & (per_seed_df['gender']==gender)]
        summary_rows.append({
            'mode': mode,
            'gender': gender,
            'do_smote': bool(DO_SMOTE),
            'split_seed': int(SPLIT_SEED),
            'n_init_seeds': int(len(sub)),
            'utt_acc_mean': float(sub['utt_acc'].mean()),
            'utt_acc_std': float(sub['utt_acc'].std()),
            'utt_f1_mean': float(sub['utt_macro_f1'].mean()),
            'utt_f1_std': float(sub['utt_macro_f1'].std()),
            'utt_auc_mean': float(sub['utt_macro_auc'].mean()),
            'utt_auc_std': float(sub['utt_macro_auc'].std()),
            'subject_acc_mean': float(sub['subject_acc'].mean()),
            'subject_acc_std': float(sub['subject_acc'].std()),
            'subject_f1_mean': float(sub['subject_f1'].mean()),
            'subject_f1_std': float(sub['subject_f1'].std()),
            'subject_auc_mean': float(sub['subject_auc'].mean()),
            'subject_auc_std': float(sub['subject_auc'].std()),
            'n_test_subjects': int(sub['n_test_subjects'].iloc[0]) if len(sub) else 0,
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)
print('Saved:', SUMMARY_CSV)
print('\n=== FINAL SUMMARY (mean±std over init seeds) ===')
print(summary_df)
meta = {
    'created_at': datetime.now().strftime('%Y%m%d_%H%M%S'),
    'split_seed': int(SPLIT_SEED),
    'test_size': float(TEST_SIZE),
    'init_seeds': INIT_SEEDS,
    'do_smote': bool(DO_SMOTE),
    'k_shot': int(K_SHOT),
    'q_query': int(Q_QUERY),
    'model_cfg': MODEL_CFG,
    'train_cfg': TRAIN_CFG,
    'n_subjects_train': int(len(np.unique(subj_train))),
    'n_subjects_test': int(len(np.unique(subj_test))),
    'n_test_subjects_female': int(len(female_test_subjects)),
    'n_test_subjects_male': int(len(male_test_subjects)),
    'do_match_male_to_female': bool(DO_MATCH_MALE_TO_FEMALE),
    'match_iters': int(MATCH_ITERS),
}
with META_JSONL.open('a', encoding='utf-8') as f:
    f.write(json.dumps(meta) + '\n')
