"""Head-only MOMENT fine-tuning for IMWUT with dual-head feature export.

This module trains frozen-encoder MOMENT classification heads for multiple
labeling schemes and exports features for both head architectures.

How to run:

# Full profile
`
python trustME/source/finetune_imwut_heads.py \
  --config trustME/configs/finetune_imwut_heads.full.yaml
`

# Fast stratified subset profile
`
python trustME/source/finetune_imwut_heads.py \
  --config trustME/configs/finetune_imwut_heads.quick_subset.yaml
`

# Override one thing from CLI (example)
`
python trustME/source/finetune_imwut_heads.py \
  --config trustME/configs/finetune_imwut_heads.quick_subset.yaml \
  --subset-fraction 0.4 \
  --epochs 5
`

"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import random
import shutil
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


LOGGER = logging.getLogger(__name__)
PIPELINE_VERSION = "1.0.0"

# Transformers<->Torch compatibility warning (harmless for current execution).
# Keep this narrowly scoped so unrelated FutureWarnings remain visible.
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.utils\._pytree\._register_pytree_node` is deprecated\..*",
    category=FutureWarning,
    module=r"transformers\.utils\.generic",
)


# Canonical labeling definitions from:
# /home/ppg/eyetracking/CognitiveLoadFromEyeTracking/scripts2/ML/labelling.py
DISTINCT_LABELS = (
    "central_position",
    "questionnaire",
    "2_back",
    "3_back",
    "stroop_easy",
    "stroop_difficult",
    "rest",
    "memory_easy",
    "memory_difficult",
    "images_questions_easy",
    "images_questions_difficult",
    "listen_music",
    "difference_images_easy",
    "difference_images_difficult",
    "pursuit_easy",
    "pursuit_difficult",
    "passive_viewing",
)
NO_LOAD = ("central_position", "questionnaire", "rest", "listen_music", "passive_viewing")
REST_LABELS = ("central_position", "rest", "listen_music", "passive_viewing")
EASY_TASKS = tuple([x for x in DISTINCT_LABELS if ("easy" in x or "2_back" in x)])
DIFFICULT_TASKS = tuple([x for x in DISTINCT_LABELS if ("difficult" in x or "3_back" in x)])
ATTENTION_TASKS = ("2_back", "3_back", "stroop_easy", "stroop_difficult")
MEMORY_TASKS = ("memory_easy", "memory_difficult", "images_questions_easy", "images_questions_difficult")
VISUAL_TASKS = ("difference_images_easy", "difference_images_difficult", "pursuit_easy", "pursuit_difficult")


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for fine-tuning and feature export."""

    model_name: str = "AutonLab/MOMENT-1-large"
    batch_size: int = 64
    epochs: int = 10
    patience: int = 3
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    subject_train_frac: float = 0.70
    subject_val_frac: float = 0.15
    subject_test_frac: float = 0.15
    schemes: tuple[str, ...] = ("binary", "edr", "avm")
    head_types: tuple[str, ...] = ("linear", "mlp")
    mlp_hidden_dim: int = 256
    mlp_dropout: float = 0.1
    selection_metric: str = "val_balanced_accuracy"
    drop_central_and_questionnaire: bool = True
    optional_drop_labels: tuple[str, ...] = ("central_position", "questionnaire")
    subset_fraction: float | None = None
    subset_min_per_class: int = 1
    subset_seed: int = 42
    clear_cuda_cache_between_heads: bool = True
    save_model: bool = True
    save_metrics: bool = True
    save_base_embeddings: bool = True
    save_head_features: bool = True


@dataclass(frozen=True)
class SplitIndices:
    """Index arrays for train/validation/test splits."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class SchemeArtifacts:
    """Paths to artifacts exported for one scheme/head pair."""

    scheme: str
    head_type: str
    model_path: Path | None
    head_features_path: Path | None
    base_embeddings_path: Path | None
    metrics_path: Path | None


class MLPClassificationHead(nn.Module):
    """Small MLP classification head over pooled patch features."""

    def __init__(self, d_model: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor | None = None) -> torch.Tensor:
        _ = input_mask
        pooled = torch.mean(x, dim=1)
        return self.net(pooled)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    # `force=True` ensures INFO logs are visible even if another logger was initialized earlier.
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", force=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _maybe_clear_cuda_cache() -> None:
    """Release cached CUDA memory if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return model state dict cloned onto CPU."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        conf[t, p] += 1
    return conf


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, Any]:
    """Compute accuracy, balanced accuracy, macro F1, and row-normalized confusion matrix."""
    conf = _confusion_matrix(y_true=y_true, y_pred=y_pred, n_classes=n_classes)
    support = conf.sum(axis=1)
    pred_count = conf.sum(axis=0)
    tp = np.diag(conf).astype(np.float64)

    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=np.float64), where=support > 0)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp, dtype=np.float64), where=pred_count > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(precision + recall) > 0,
    )

    valid_classes = support > 0
    balanced_accuracy = float(recall[valid_classes].mean()) if np.any(valid_classes) else 0.0
    macro_f1 = float(f1[valid_classes].mean()) if np.any(valid_classes) else 0.0
    accuracy = float(np.mean(y_true == y_pred)) if y_true.size > 0 else 0.0

    row_sum = conf.sum(axis=1, keepdims=True)
    conf_norm = np.divide(conf, row_sum, out=np.zeros_like(conf, dtype=np.float64), where=row_sum > 0)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "per_class_recall": recall.tolist(),
        "confusion_matrix": conf.tolist(),
        "confusion_matrix_normalized": conf_norm.tolist(),
    }


def apply_label_scheme(
    df: pd.DataFrame,
    scheme: str,
    drop_central_and_questionnaire: bool = True,
    optional_drop_labels: tuple[str, ...] = ("central_position", "questionnaire"),
) -> pd.DataFrame:
    """Apply canonical IMWUT label scheme and return filtered dataframe with `scheme_label`.

    Parameters
    ----------
    df:
        Input dataframe that must include `Label`.
    scheme:
        One of `binary`, `edr`, `avm`.
    drop_central_and_questionnaire:
        If True (default), drop rows with labels `central_position` and `questionnaire`
        before scheme mapping. If False, retain them and let scheme rules decide how they
        are mapped/kept.
    optional_drop_labels:
        Labels to drop when `drop_central_and_questionnaire=True`.
    """
    supported = {"binary", "edr", "avm"}
    if scheme not in supported:
        raise ValueError(f"Unsupported scheme={scheme}. Supported: {sorted(supported)}")

    work = df.dropna(subset=["Label"]).copy()
    if drop_central_and_questionnaire:
        work = work.loc[~work["Label"].isin(list(optional_drop_labels))].copy()

    if scheme == "binary":
        labels = work["Label"].astype(str)
        labels = np.where(labels.isin(REST_LABELS), "rest", labels)
        labels = np.where(pd.Series(labels).isin(NO_LOAD), "rest", "load")
        work["scheme_label"] = labels

    elif scheme == "edr":
        labels = work["Label"].astype(str)
        labels = np.where(labels.isin(REST_LABELS), "rest", labels)
        labels = np.where(pd.Series(labels).isin(EASY_TASKS), "low_load", labels)
        labels = np.where(pd.Series(labels).isin(DIFFICULT_TASKS), "high_load", labels)
        work["scheme_label"] = labels

    else:  # avm
        avm_map = {
            "2_back": "attention_task",
            "3_back": "attention_task",
            "stroop_easy": "attention_task",
            "stroop_difficult": "attention_task",
            "memory_easy": "memory_task",
            "memory_difficult": "memory_task",
            "images_questions_easy": "memory_task",
            "images_questions_difficult": "memory_task",
            "difference_images_easy": "visual_task",
            "difference_images_difficult": "visual_task",
            "pursuit_easy": "visual_task",
            "pursuit_difficult": "visual_task",
        }
        if not drop_central_and_questionnaire:
            avm_map = {
                **avm_map,
                "central_position": "central_position",
                "questionnaire": "questionnaire",
            }
        work["scheme_label"] = work["Label"].map(avm_map)
        allowed_labels = ["attention_task", "memory_task", "visual_task"]
        if not drop_central_and_questionnaire:
            allowed_labels.extend(["central_position", "questionnaire"])
        work = work.loc[work["scheme_label"].isin(allowed_labels)]

    return work.reset_index(drop=True)


def _allocate_subject_counts(
    n_subjects: int, train_frac: float, val_frac: float, test_frac: float
) -> tuple[int, int, int]:
    if n_subjects < 3:
        raise ValueError("Need at least 3 subjects for train/val/test holdout split.")
    frac_sum = train_frac + val_frac + test_frac
    if not np.isclose(frac_sum, 1.0, atol=1e-6):
        raise ValueError(f"Split fractions must sum to 1.0, got {frac_sum}")

    n_train = max(1, int(round(n_subjects * train_frac)))
    n_val = max(1, int(round(n_subjects * val_frac)))
    n_test = n_subjects - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_train = max(1, n_subjects - 2)
            n_val = 1
        n_test = n_subjects - n_train - n_val
    while n_train + n_val + n_test < n_subjects:
        n_train += 1
    while n_train + n_val + n_test > n_subjects:
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1
    return n_train, n_val, n_test


def split_subject_holdout(
    df: pd.DataFrame,
    subject_col: str = "Subject",
    label_col: str = "scheme_label",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    max_attempts: int = 500,
) -> SplitIndices:
    """Create deterministic subject-level holdout split with class-coverage checks."""
    subjects = np.asarray(sorted(df[subject_col].astype(str).unique()))
    n_train, n_val, n_test = _allocate_subject_counts(
        n_subjects=len(subjects), train_frac=train_frac, val_frac=val_frac, test_frac=test_frac
    )
    full_label_set = set(df[label_col].astype(str).unique())
    required_val_classes = min(2, len(full_label_set))

    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        perm = rng.permutation(subjects)
        train_subjects = set(perm[:n_train])
        val_subjects = set(perm[n_train : n_train + n_val])
        test_subjects = set(perm[n_train + n_val : n_train + n_val + n_test])

        train_idx = np.flatnonzero(df[subject_col].astype(str).isin(train_subjects).to_numpy())
        val_idx = np.flatnonzero(df[subject_col].astype(str).isin(val_subjects).to_numpy())
        test_idx = np.flatnonzero(df[subject_col].astype(str).isin(test_subjects).to_numpy())

        train_labels = set(df.iloc[train_idx][label_col].astype(str).unique())
        val_labels = set(df.iloc[val_idx][label_col].astype(str).unique())

        if train_labels != full_label_set:
            continue
        if len(val_labels) < required_val_classes:
            continue
        if test_idx.size <= 0:
            continue

        return SplitIndices(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    raise ValueError(
        "Failed to find subject holdout split with required class coverage. "
        f"subjects={len(subjects)} classes={sorted(full_label_set)}"
    )


def build_moment_model(
    head_type: str,
    num_classes: int,
    model_name: str,
    device: str,
    num_channels: int,
    mlp_hidden_dim: int,
    mlp_dropout: float,
) -> Any:
    """Build MOMENT classification model with frozen encoder and selected head type."""
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "classification",
            "n_channels": num_channels,
            "num_class": num_classes,
            "freeze_encoder": True,
            "freeze_embedder": True,
            "freeze_head": False,
            "reduction": "mean",
            "enable_gradient_checkpointing": False,
        },
    )
    model.init()
    if head_type == "mlp":
        d_model = int(getattr(model.config, "d_model"))
        model.head = MLPClassificationHead(
            d_model=d_model,
            hidden_dim=mlp_hidden_dim,
            num_classes=num_classes,
            dropout=mlp_dropout,
        )
    elif head_type != "linear":
        raise ValueError(f"Unsupported head_type={head_type}. Supported: ['linear', 'mlp']")

    model.to(device)
    return model


def _make_dataloader(
    x: np.ndarray,
    input_mask: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x[indices]).float(),
        torch.from_numpy(input_mask[indices]).long(),
        torch.from_numpy(labels[indices]).long(),
        torch.from_numpy(indices).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


@torch.no_grad()
def _run_head_inference(model: Any, loader: DataLoader, device: str) -> dict[str, np.ndarray]:
    model.eval()
    idx_out: list[np.ndarray] = []
    logits_out: list[np.ndarray] = []
    probs_out: list[np.ndarray] = []
    pred_out: list[np.ndarray] = []
    y_out: list[np.ndarray] = []

    for x_batch, mask_batch, y_batch, idx_batch in loader:
        output = model(x_enc=x_batch.to(device), input_mask=mask_batch.to(device), reduction="mean")
        logits = output.logits.detach().cpu().numpy()
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        pred = np.argmax(probs, axis=1).astype(np.int64)

        idx_out.append(idx_batch.cpu().numpy())
        logits_out.append(logits.astype(np.float32))
        probs_out.append(probs.astype(np.float32))
        pred_out.append(pred)
        y_out.append(y_batch.cpu().numpy().astype(np.int64))

    idx = np.concatenate(idx_out, axis=0)
    order = np.argsort(idx)
    return {
        "idx": idx[order],
        "logits": np.concatenate(logits_out, axis=0)[order],
        "probs": np.concatenate(probs_out, axis=0)[order],
        "pred": np.concatenate(pred_out, axis=0)[order],
        "y_true": np.concatenate(y_out, axis=0)[order],
    }


@torch.no_grad()
def _run_embed_inference(model: Any, loader: DataLoader, device: str) -> dict[str, np.ndarray]:
    model.eval()
    idx_out: list[np.ndarray] = []
    emb_out: list[np.ndarray] = []
    for x_batch, mask_batch, _, idx_batch in loader:
        output = model.embed(x_enc=x_batch.to(device), input_mask=mask_batch.to(device), reduction="mean")
        emb = output.embeddings.detach().cpu().numpy().astype(np.float32)
        idx_out.append(idx_batch.cpu().numpy())
        emb_out.append(emb)
    idx = np.concatenate(idx_out, axis=0)
    order = np.argsort(idx)
    return {
        "idx": idx[order],
        "embeddings": np.concatenate(emb_out, axis=0)[order],
    }


def train_one_head(
    *,
    train_config: TrainConfig,
    head_type: str,
    model_name: str,
    num_classes: int,
    num_channels: int,
    x: np.ndarray,
    input_mask: np.ndarray,
    labels: np.ndarray,
    split_indices: SplitIndices,
) -> dict[str, Any]:
    """Train one head type and return trained state, metrics, and outputs."""
    _seed_everything(train_config.seed)
    device = _resolve_device(train_config.device)
    model = build_moment_model(
        head_type=head_type,
        num_classes=num_classes,
        model_name=model_name,
        device=device,
        num_channels=num_channels,
        mlp_hidden_dim=train_config.mlp_hidden_dim,
        mlp_dropout=train_config.mlp_dropout,
    )

    train_loader = _make_dataloader(
        x=x,
        input_mask=input_mask,
        labels=labels,
        indices=split_indices.train_idx,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    val_loader = _make_dataloader(
        x=x,
        input_mask=input_mask,
        labels=labels,
        indices=split_indices.val_idx,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )
    test_loader = _make_dataloader(
        x=x,
        input_mask=input_mask,
        labels=labels,
        indices=split_indices.test_idx,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )
    full_loader = _make_dataloader(
        x=x,
        input_mask=input_mask,
        labels=labels,
        indices=np.arange(x.shape[0], dtype=np.int64),
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    class_count = np.bincount(labels[split_indices.train_idx], minlength=num_classes).astype(np.float32)
    if np.any(class_count == 0):
        raise ValueError("Training split is missing at least one class, cannot build weighted loss.")
    class_weight = class_count.sum() / (num_classes * class_count)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weight).to(device))

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    # Initialize any lazy modules before taking the initial best-state snapshot.
    first_batch = next(iter(train_loader))
    with torch.no_grad():
        _ = model(
            x_enc=first_batch[0].to(device),
            input_mask=first_batch[1].to(device),
            reduction="mean",
        )

    best_val = -1.0
    best_epoch = 0
    best_state = _state_dict_to_cpu(model)
    bad_epochs = 0

    for epoch in range(1, train_config.epochs + 1):
        model.train()
        loss_values: list[float] = []
        for x_batch, mask_batch, y_batch, _ in train_loader:
            optimizer.zero_grad()
            output = model(x_enc=x_batch.to(device), input_mask=mask_batch.to(device), reduction="mean")
            loss = criterion(output.logits, y_batch.to(device))
            loss.backward()
            optimizer.step()
            loss_values.append(float(loss.detach().cpu().item()))

        val_out = _run_head_inference(model=model, loader=val_loader, device=device)
        val_metrics = _classification_metrics(
            y_true=val_out["y_true"], y_pred=val_out["pred"], n_classes=num_classes
        )
        val_score = float(val_metrics["balanced_accuracy"])
        train_loss = float(np.mean(loss_values)) if loss_values else float("nan")
        LOGGER.info(
            "head=%s epoch=%d train_loss=%.6f val_balanced_accuracy=%.6f",
            head_type,
            epoch,
            train_loss,
            val_score,
        )

        if val_score > best_val + 1e-12:
            best_val = val_score
            best_epoch = epoch
            best_state = _state_dict_to_cpu(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= train_config.patience:
                LOGGER.info(
                    "head=%s early_stop epoch=%d best_epoch=%d best_val=%.6f",
                    head_type,
                    epoch,
                    best_epoch,
                    best_val,
                )
                break

    model.load_state_dict(best_state)

    train_out = _run_head_inference(model=model, loader=train_loader, device=device)
    val_out = _run_head_inference(model=model, loader=val_loader, device=device)
    test_out = _run_head_inference(model=model, loader=test_loader, device=device)
    full_head_out = _run_head_inference(model=model, loader=full_loader, device=device)
    full_embed_out = _run_embed_inference(model=model, loader=full_loader, device=device)

    metrics = {
        "best_epoch": int(best_epoch),
        "train": _classification_metrics(train_out["y_true"], train_out["pred"], num_classes),
        "val": _classification_metrics(val_out["y_true"], val_out["pred"], num_classes),
        "test": _classification_metrics(test_out["y_true"], test_out["pred"], num_classes),
    }
    metrics["val_balanced_accuracy"] = float(metrics["val"]["balanced_accuracy"])
    metrics["test_balanced_accuracy"] = float(metrics["test"]["balanced_accuracy"])

    model.to("cpu")
    _maybe_clear_cuda_cache()

    return {
        "state_dict": best_state,
        "metrics": metrics,
        "head_outputs": full_head_out,
        "base_embeddings": full_embed_out,
    }


def export_head_artifacts(
    *,
    scheme_dir: Path,
    scheme: str,
    head_type: str,
    model_state_dict: dict[str, Any],
    head_outputs: dict[str, np.ndarray],
    base_embeddings: dict[str, np.ndarray],
    metrics: dict[str, Any],
    segment_ids: np.ndarray,
    label_int: np.ndarray,
    label_str: np.ndarray,
    split: np.ndarray,
) -> SchemeArtifacts:
    """Export model checkpoint, head features, base embeddings, and metrics for one head."""
    head_dir = scheme_dir / head_type
    head_dir.mkdir(parents=True, exist_ok=True)

    model_path = head_dir / "model.pt"
    torch.save(model_state_dict, model_path)

    head_features_path = head_dir / "head_features.npz"
    np.savez_compressed(
        head_features_path,
        logits=head_outputs["logits"].astype(np.float32),
        probs=head_outputs["probs"].astype(np.float32),
        pred_label_int=head_outputs["pred"].astype(np.int64),
        segment_id=segment_ids.astype(str),
        label_int=label_int.astype(np.int64),
        label_str=label_str.astype(str),
        split=split.astype(str),
    )

    base_embeddings_path = head_dir / "base_embeddings.npz"
    np.savez_compressed(
        base_embeddings_path,
        embeddings=base_embeddings["embeddings"].astype(np.float32),
        segment_id=segment_ids.astype(str),
        label_int=label_int.astype(np.int64),
        label_str=label_str.astype(str),
        split=split.astype(str),
    )

    metrics_path = head_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    return SchemeArtifacts(
        scheme=scheme,
        head_type=head_type,
        model_path=model_path,
        head_features_path=head_features_path,
        base_embeddings_path=base_embeddings_path,
        metrics_path=metrics_path,
    )


def choose_best_head(results: dict[str, dict[str, Any]]) -> str:
    """Select best head by validation balanced accuracy (tie-break by head name)."""
    ranked = sorted(
        results.items(),
        key=lambda item: (
            float(item[1]["metrics"].get("val_balanced_accuracy", -1.0)),
            item[0],
        ),
        reverse=True,
    )
    if not ranked:
        raise ValueError("No head results available for selection.")
    return ranked[0][0]


def _load_cached_inputs(input_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load aligned cached inputs and kept metadata."""
    inputs_path = input_dir / "moment_inputs.npz"
    metadata_path = input_dir / "segments_metadata.parquet"
    if not inputs_path.exists():
        raise FileNotFoundError(f"Missing inputs file: {inputs_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    payload = np.load(inputs_path, allow_pickle=False)
    x = payload["x"].astype(np.float32)
    input_mask = payload["input_mask"].astype(np.int64)
    segment_ids = payload["segment_id"].astype(str)

    metadata = pd.read_parquet(metadata_path)
    kept = metadata.loc[metadata["kept"]].copy()
    kept = kept.set_index("segment_id", drop=False)

    missing = sorted(set(segment_ids) - set(kept.index.astype(str)))
    if missing:
        raise ValueError(f"Missing kept metadata rows for segment IDs: {missing[:5]} ...")
    kept_aligned = kept.loc[segment_ids].reset_index(drop=True)
    if x.shape[0] != kept_aligned.shape[0]:
        raise ValueError("Input and metadata row counts differ after alignment.")
    return x, input_mask, kept_aligned


def _parse_csv_list(raw: str) -> tuple[str, ...]:
    values = tuple([part.strip() for part in raw.split(",") if part.strip()])
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def _copy_best_artifacts(scheme_dir: Path, selected: SchemeArtifacts) -> None:
    shutil.copy2(selected.model_path, scheme_dir / "best_model.pt")
    shutil.copy2(selected.head_features_path, scheme_dir / "best_head_features.npz")
    shutil.copy2(selected.base_embeddings_path, scheme_dir / "best_base_embeddings.npz")


def _stratified_subset_indices(
    labels: np.ndarray,
    fraction: float | None,
    seed: int,
    min_per_class: int = 1,
) -> np.ndarray:
    """Return stratified subset indices; keep all rows if fraction is None or >=1."""
    if fraction is None or fraction >= 1.0:
        return np.arange(labels.shape[0], dtype=np.int64)
    if fraction <= 0.0:
        raise ValueError(f"subset_fraction must be in (0, 1], got {fraction}")
    if min_per_class < 1:
        raise ValueError(f"subset_min_per_class must be >=1, got {min_per_class}")

    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    labels = labels.astype(str)
    for label in sorted(np.unique(labels).tolist()):
        class_idx = np.flatnonzero(labels == label)
        target_n = int(round(class_idx.size * fraction))
        target_n = max(min_per_class, target_n)
        target_n = min(target_n, class_idx.size)
        sampled = rng.choice(class_idx, size=target_n, replace=False)
        keep.append(np.sort(sampled))
    out = np.sort(np.concatenate(keep, axis=0).astype(np.int64))
    return out


def _config_value(defaults: dict[str, Any], key: str, fallback: Any) -> Any:
    return defaults.get(key, fallback)


def _csv_default(defaults: dict[str, Any], key: str, fallback: str) -> str:
    value = defaults.get(key, fallback)
    if isinstance(value, (list, tuple)):
        return ",".join([str(x) for x in value])
    return str(value)


def _path_default(defaults: dict[str, Any], key: str, fallback: Path) -> Path:
    value = defaults.get(key, fallback)
    return value if isinstance(value, Path) else Path(str(value))


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config file for this pipeline."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for config files. Install it in `moment4ET` (e.g. `pip install pyyaml`)."
        ) from exc
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping, got {type(payload)}")
    return payload


def _run_rebuild_if_requested(args: argparse.Namespace) -> None:
    if not args.rebuild_inputs:
        return
    from segment_imwut import SegmentConfig, build_tobii_segment_inputs

    seg_csv = args.seg_csv or (args.raw_root / "all_segments.csv")
    LOGGER.info("Rebuilding cached inputs into %s", args.input_dir)
    build_tobii_segment_inputs(
        raw_root=args.raw_root,
        seg_csv=seg_csv,
        out_dir=args.input_dir,
        config=SegmentConfig(),
    )


def run_pipeline(args: argparse.Namespace) -> Path:
    """Run dual-head fine-tuning and export pipeline."""
    _seed_everything(args.seed)
    LOGGER.info(
        "Starting head fine-tuning pipeline \n" \
        "                               input_dir=%s\n" \
        "                               out_dir=%s \n" \
        "                               schemes=%s \n" \
        "                               heads=%s \n" \
        "                               device=%s",
        args.input_dir,
        args.out_dir,
        args.schemes,
        args.head_types,
        args.device,
    )
    _run_rebuild_if_requested(args)

    x, input_mask, metadata = _load_cached_inputs(args.input_dir)
    LOGGER.info(
        "Loaded cached inputs | \n" \
        "                           n_segments=%d\n" \
        "                           x_shape=%s\n" \
        "                           mask_shape=%s",
        x.shape[0],
        tuple(x.shape),
        tuple(input_mask.shape),
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_config = TrainConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        subject_train_frac=args.subject_train_frac,
        subject_val_frac=args.subject_val_frac,
        subject_test_frac=args.subject_test_frac,
        schemes=args.schemes,
        head_types=args.head_types,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.mlp_dropout,
        drop_central_and_questionnaire=args.drop_central_and_questionnaire,
        subset_fraction=args.subset_fraction,
        subset_min_per_class=args.subset_min_per_class,
        subset_seed=args.subset_seed,
        clear_cuda_cache_between_heads=args.clear_cuda_cache_between_heads,
    )

    run_manifest: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "config_path": str(args.config),
        "input_dir": str(args.input_dir),
        "out_dir": str(out_dir),
        "config": asdict(train_config),
        "schemes": {},
    }

    for scheme_idx, scheme in enumerate(train_config.schemes):
        LOGGER.info("")
        LOGGER.info("*"*50)
        LOGGER.info("Processing scheme=%s", scheme)
        scheme_df = metadata.copy()
        scheme_df["row_idx"] = np.arange(scheme_df.shape[0], dtype=np.int64)
        scheme_df = apply_label_scheme(
            scheme_df,
            scheme=scheme,
            drop_central_and_questionnaire=train_config.drop_central_and_questionnaire,
            optional_drop_labels=train_config.optional_drop_labels,
        )
        if scheme_df.empty:
            raise ValueError(f"No rows remained after applying scheme={scheme}")

        subset_idx = _stratified_subset_indices(
            labels=scheme_df["scheme_label"].to_numpy(dtype=str),
            fraction=train_config.subset_fraction,
            seed=train_config.subset_seed + scheme_idx,
            min_per_class=train_config.subset_min_per_class,
        )
        if subset_idx.size < scheme_df.shape[0]:
            original_n = int(scheme_df.shape[0])
            scheme_df = scheme_df.iloc[subset_idx].reset_index(drop=True)
            LOGGER.info(
                "Scheme=%s stratified subset active | kept=%d/%d (%.2f%%) fraction=%.4f",
                scheme,
                scheme_df.shape[0],
                original_n,
                100.0 * scheme_df.shape[0] / max(original_n, 1),
                float(train_config.subset_fraction or 1.0),
            )

        scheme_row_idx = scheme_df["row_idx"].to_numpy(dtype=np.int64)
        x_scheme = x[scheme_row_idx]
        mask_scheme = input_mask[scheme_row_idx]
        segment_ids = scheme_df["segment_id"].astype(str).to_numpy()
        label_str = scheme_df["scheme_label"].astype(str).to_numpy()

        classes = sorted(np.unique(label_str).tolist())
        LOGGER.info(
            "Scheme=%s prepared | n_samples=%d n_classes=%d classes=%s",
            scheme,
            scheme_df.shape[0],
            len(classes),
            classes,
        )
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
        label_int = np.asarray([class_to_idx[name] for name in label_str], dtype=np.int64)

        split_indices = split_subject_holdout(
            scheme_df,
            subject_col="Subject",
            label_col="scheme_label",
            train_frac=train_config.subject_train_frac,
            val_frac=train_config.subject_val_frac,
            test_frac=train_config.subject_test_frac,
            seed=train_config.seed,
        )

        split = np.full(shape=(scheme_df.shape[0],), fill_value="unused", dtype=object)
        split[split_indices.train_idx] = "train"
        split[split_indices.val_idx] = "val"
        split[split_indices.test_idx] = "test"

        scheme_dir = out_dir / scheme
        scheme_dir.mkdir(parents=True, exist_ok=True)

        head_results: dict[str, dict[str, Any]] = {}
        head_artifacts: dict[str, SchemeArtifacts] = {}
        head_summary: dict[str, Any] = {}

        for head_type in train_config.head_types:
            LOGGER.info("Training scheme=%s head=%s", scheme, head_type)
            result = train_one_head(
                train_config=train_config,
                head_type=head_type,
                model_name=train_config.model_name,
                num_classes=len(classes),
                num_channels=x_scheme.shape[1],
                x=x_scheme,
                input_mask=mask_scheme,
                labels=label_int,
                split_indices=split_indices,
            )

            artifacts = export_head_artifacts(
                scheme_dir=scheme_dir,
                scheme=scheme,
                head_type=head_type,
                model_state_dict=result["state_dict"],
                head_outputs=result["head_outputs"],
                base_embeddings=result["base_embeddings"],
                metrics=result["metrics"],
                segment_ids=segment_ids,
                label_int=label_int,
                label_str=label_str,
                split=split,
            )
            head_artifacts[head_type] = artifacts
            head_summary[head_type] = {
                "val_balanced_accuracy": float(result["metrics"]["val_balanced_accuracy"]),
                "test_balanced_accuracy": float(result["metrics"]["test_balanced_accuracy"]),
                "metrics_path": str(artifacts.metrics_path),
            }
            head_results[head_type] = {"metrics": result["metrics"]}
            del result
            if train_config.clear_cuda_cache_between_heads:
                _maybe_clear_cuda_cache()
        best_head = choose_best_head(head_results)
        selected = head_artifacts[best_head]
        _copy_best_artifacts(scheme_dir=scheme_dir, selected=selected)

        best_head_json = {
            "scheme": scheme,
            "selection_metric": train_config.selection_metric,
            "best_head": best_head,
            "classes": classes,
            "n_samples": int(scheme_df.shape[0]),
            "head_summary": head_summary,
            "artifacts": {
                "best_model": str(scheme_dir / "best_model.pt"),
                "best_head_features": str(scheme_dir / "best_head_features.npz"),
                "best_base_embeddings": str(scheme_dir / "best_base_embeddings.npz"),
            },
        }
        best_head_path = scheme_dir / "best_head.json"
        best_head_path.write_text(json.dumps(best_head_json, indent=2, sort_keys=True))

        run_manifest["schemes"][scheme] = {
            "best_head": best_head,
            "classes": classes,
            "n_samples": int(scheme_df.shape[0]),
            "head_types": list(train_config.head_types),
            "head_summary": head_summary,
            "best_head_json": str(best_head_path),
            "best_head_json_sha256": _sha256_file(best_head_path),
        }
        LOGGER.info(
            "Scheme=%s complete | best_head=%s val_bal_acc=%.6f",
            scheme,
            best_head,
            float(head_results[best_head]["metrics"]["val_balanced_accuracy"]),
        )

    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True))
    LOGGER.info("Saved run manifest: %s", manifest_path)
    return out_dir


def build_arg_parser(
    config_defaults: dict[str, Any] | None = None,
    default_config_path: Path = Path("trustME/configs/finetune_imwut_heads.full.yaml"),
) -> argparse.ArgumentParser:
    """Build CLI parser for head-only fine-tuning pipeline."""
    defaults = config_defaults or {}
    parser = argparse.ArgumentParser(description="IMWUT head-only MOMENT fine-tuning and dual-head export.")
    parser.add_argument("--config", type=Path, default=default_config_path)
    parser.add_argument("--input-dir", type=Path, default=_path_default(defaults, "input_dir", Path("trustME/data/processed/imwut_tobii")))
    parser.add_argument("--raw-root", type=Path, default=_path_default(defaults, "raw_root", Path("trustME/data/raw/imwut")))
    parser.add_argument("--seg-csv", type=Path, default=_path_default(defaults, "seg_csv", Path("")) if defaults.get("seg_csv") else None)
    parser.add_argument("--rebuild-inputs", action="store_true", default=bool(_config_value(defaults, "rebuild_inputs", False)))
    parser.add_argument("--schemes", type=str, default=_csv_default(defaults, "schemes", "binary,edr,avm"))
    parser.add_argument("--head-types", type=str, default=_csv_default(defaults, "head_types", "linear,mlp"))
    parser.add_argument("--model-name", type=str, default=str(_config_value(defaults, "model_name", "AutonLab/MOMENT-1-large")))
    parser.add_argument("--batch-size", type=int, default=int(_config_value(defaults, "batch_size", 64)))
    parser.add_argument("--epochs", type=int, default=int(_config_value(defaults, "epochs", 10)))
    parser.add_argument("--patience", type=int, default=int(_config_value(defaults, "patience", 3)))
    parser.add_argument("--lr", type=float, default=float(_config_value(defaults, "lr", 1e-4)))
    parser.add_argument("--weight-decay", type=float, default=float(_config_value(defaults, "weight_decay", 1e-4)))
    parser.add_argument("--seed", type=int, default=int(_config_value(defaults, "seed", 42)))
    parser.add_argument("--device", type=str, default=str(_config_value(defaults, "device", "auto")))
    parser.add_argument("--subject-train-frac", type=float, default=float(_config_value(defaults, "subject_train_frac", 0.70)))
    parser.add_argument("--subject-val-frac", type=float, default=float(_config_value(defaults, "subject_val_frac", 0.15)))
    parser.add_argument("--subject-test-frac", type=float, default=float(_config_value(defaults, "subject_test_frac", 0.15)))
    parser.add_argument("--mlp-hidden-dim", type=int, default=int(_config_value(defaults, "mlp_hidden_dim", 256)))
    parser.add_argument("--mlp-dropout", type=float, default=float(_config_value(defaults, "mlp_dropout", 0.1)))
    parser.add_argument("--subset-fraction", type=float, default=_config_value(defaults, "subset_fraction", None))
    parser.add_argument("--subset-min-per-class", type=int, default=int(_config_value(defaults, "subset_min_per_class", 1)))
    parser.add_argument("--subset-seed", type=int, default=int(_config_value(defaults, "subset_seed", 42)))
    parser.add_argument("--num-workers", type=int, default=int(_config_value(defaults, "num_workers", 0)))
    parser.add_argument(
        "--clear-cuda-cache-between-heads",
        dest="clear_cuda_cache_between_heads",
        action="store_true",
        default=bool(_config_value(defaults, "clear_cuda_cache_between_heads", True)),
        help="Call `torch.cuda.empty_cache()` after each head training stage (default).",
    )
    parser.add_argument(
        "--no-clear-cuda-cache-between-heads",
        dest="clear_cuda_cache_between_heads",
        action="store_false",
        help="Disable CUDA cache clearing between heads.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_path_default(defaults, "out_dir", Path("trustME/data/processed/imwut_tobii_finetuned")),
    )
    parser.add_argument(
        "--drop-central-and-questionnaire",
        dest="drop_central_and_questionnaire",
        action="store_true",
        default=bool(_config_value(defaults, "drop_central_and_questionnaire", True)),
        help="Drop `central_position` and `questionnaire` before label scheme mapping (default).",
    )
    parser.add_argument(
        "--retain-central-and-questionnaire",
        dest="drop_central_and_questionnaire",
        action="store_false",
        help="Retain `central_position` and `questionnaire` instead of dropping them.",
    )
    parser.add_argument("--verbose", action="store_true", default=bool(_config_value(defaults, "verbose", False)))
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=Path("trustME/configs/finetune_imwut_heads.full.yaml"))
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_defaults = _load_yaml_config(pre_args.config)

    parser = build_arg_parser(config_defaults=config_defaults, default_config_path=pre_args.config)
    args = parser.parse_args(argv)
    _setup_logging(verbose=args.verbose)
    args.schemes = _parse_csv_list(args.schemes)
    args.head_types = _parse_csv_list(args.head_types)
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
