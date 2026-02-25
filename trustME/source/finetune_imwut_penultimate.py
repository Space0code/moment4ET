"""Fine-tune MOMENT on IMWUT labels with configurable penultimate-layer scopes.

This script fine-tunes one classification model per label scheme (`binary`,
`edr`, `avm`) and exports scheme-specific artifacts.

Examples
--------
# Quick subset run from YAML profile
python trustME/source/finetune_imwut_penultimate.py \
  --config trustME/configs/finetune_imwut_penultimate.quick_subset.yaml

# Full run with CLI overrides
python trustME/source/finetune_imwut_penultimate.py \
  --config trustME/configs/finetune_imwut_penultimate.full.yaml \
  --encoder-tune-scope last_n_layernorm \
  --unfreeze-last-n-blocks 1 \
  --epochs 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import warnings
from contextlib import nullcontext
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
SUPPORTED_SCHEMES = ("binary", "edr", "avm")
SUPPORTED_HEAD_TYPES = ("linear", "mlp")
SUPPORTED_ENCODER_TUNE_SCOPES = ("last_n_layernorm", "last_n_mlp", "last_n_blocks")
SUPPORTED_WEIGHTS_FORMATS = ("trainable_only", "full")
FORCED_DROP_LABELS = ("questionnaire", "central_position")

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


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for penultimate fine-tuning and artifact export."""

    model_name: str = "AutonLab/MOMENT-1-large"
    batch_size: int = 64
    epochs: int = 10
    patience: int = 3
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 2
    non_blocking: bool = True
    use_amp: bool = True
    amp_dtype: str = "bf16"
    enable_tf32: bool = True
    schemes: tuple[str, ...] = ("binary", "edr", "avm")
    subject_train_frac: float = 0.70
    subject_val_frac: float = 0.15
    subject_test_frac: float = 0.15
    subset_fraction: float | None = None
    subset_min_per_class: int = 1
    subset_seed: int = 42
    head_type: str = "linear"
    mlp_hidden_dim: int = 256
    mlp_dropout: float = 0.1
    encoder_tune_scope: str = "last_n_blocks"
    unfreeze_last_n_blocks: int = 1
    weights_format: str = "trainable_only"
    save_model_weights: bool = False
    save_embeddings: bool = True
    save_metrics: bool = True
    save_predictions: bool = False


@dataclass(frozen=True)
class SplitIndices:
    """Index arrays for train/validation/test splits."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class MLPClassificationHead(nn.Module):
    """MLP classification head over pooled patch features."""

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


def _resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    """Resolve AMP dtype string to torch dtype."""
    amp_dtype_lower = amp_dtype.strip().lower()
    if amp_dtype_lower == "bf16":
        return torch.bfloat16
    if amp_dtype_lower == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported amp_dtype={amp_dtype}. Supported: ['bf16', 'fp16']")


def _configure_tf32(enable_tf32: bool, device: str) -> None:
    """Configure TF32 backend flags when CUDA is used."""
    if not device.startswith("cuda"):
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
    torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
    LOGGER.info("TF32 enabled=%s", bool(enable_tf32))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, Any]:
    """Compute accuracy, balanced accuracy, macro F1, and row-normalized confusion matrix."""
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        conf[t, p] += 1
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


def apply_label_scheme(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Map raw IMWUT labels into one scheme after forced label drops.

    Processing order is fixed:
    1) Drop rows with NaN `Label`.
    2) Drop `questionnaire` and `central_position`.
    3) Apply scheme mapping.
    """
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(f"Unsupported scheme={scheme}. Supported: {list(SUPPORTED_SCHEMES)}")

    work = df.dropna(subset=["Label"]).copy()
    work = work.loc[~work["Label"].isin(list(FORCED_DROP_LABELS))].copy()

    if scheme == "binary":
        labels = work["Label"].astype(str)
        labels = np.where(labels.isin(REST_LABELS), "rest", labels)
        labels = np.where(pd.Series(labels).isin(NO_LOAD), "rest", "load")
        work["scheme_label"] = labels
        return work.reset_index(drop=True)

    if scheme == "edr":
        labels = work["Label"].astype(str)
        labels = np.where(labels.isin(REST_LABELS), "rest", labels)
        labels = np.where(pd.Series(labels).isin(EASY_TASKS), "low_load", labels)
        labels = np.where(pd.Series(labels).isin(DIFFICULT_TASKS), "high_load", labels)
        work["scheme_label"] = labels
        return work.reset_index(drop=True)

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
    work["scheme_label"] = work["Label"].map(avm_map)
    work = work.loc[work["scheme_label"].isin(["attention_task", "memory_task", "visual_task"])].copy()
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
    """Create deterministic subject-level split with class-coverage checks."""
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
    labels = labels.astype(str)
    keep: list[np.ndarray] = []
    for label in sorted(np.unique(labels).tolist()):
        class_idx = np.flatnonzero(labels == label)
        target_n = int(round(class_idx.size * fraction))
        target_n = max(min_per_class, target_n)
        target_n = min(target_n, class_idx.size)
        sampled = rng.choice(class_idx, size=target_n, replace=False)
        keep.append(np.sort(sampled))
    return np.sort(np.concatenate(keep, axis=0).astype(np.int64))


def _validate_unfreeze_request(encoder: nn.Module, unfreeze_last_n_blocks: int) -> list[int]:
    if not hasattr(encoder, "block"):
        raise ValueError("Expected encoder to expose `encoder.block` (T5 stack).")
    blocks = getattr(encoder, "block")
    n_blocks = len(blocks)
    if unfreeze_last_n_blocks < 1:
        raise ValueError(f"unfreeze_last_n_blocks must be >= 1, got {unfreeze_last_n_blocks}")
    if unfreeze_last_n_blocks > n_blocks:
        raise ValueError(
            f"unfreeze_last_n_blocks={unfreeze_last_n_blocks} exceeds encoder blocks={n_blocks}"
        )
    return list(range(n_blocks - unfreeze_last_n_blocks, n_blocks))


def _apply_unfreeze_scope(
    model: nn.Module, encoder_tune_scope: str, unfreeze_last_n_blocks: int
) -> tuple[list[int], list[str]]:
    """Freeze all parameters, then unfreeze head and selected encoder parameters."""
    if encoder_tune_scope not in SUPPORTED_ENCODER_TUNE_SCOPES:
        raise ValueError(
            f"Unsupported encoder_tune_scope={encoder_tune_scope}. "
            f"Supported: {list(SUPPORTED_ENCODER_TUNE_SCOPES)}"
        )

    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    selected_block_idx = _validate_unfreeze_request(model.encoder, unfreeze_last_n_blocks)
    for block_idx in selected_block_idx:
        block = model.encoder.block[block_idx]
        if encoder_tune_scope == "last_n_blocks":
            for param in block.parameters():
                param.requires_grad = True
        elif encoder_tune_scope == "last_n_mlp":
            for param in block.layer[1].DenseReluDense.parameters():
                param.requires_grad = True
        else:  # last_n_layernorm
            for param in block.layer[0].layer_norm.parameters():
                param.requires_grad = True
            for param in block.layer[1].layer_norm.parameters():
                param.requires_grad = True

    trainable_names = sorted([name for name, param in model.named_parameters() if param.requires_grad])
    if not trainable_names:
        raise RuntimeError("No trainable parameters after unfreeze policy was applied.")
    return selected_block_idx, trainable_names


def build_moment_model(
    *,
    head_type: str,
    num_classes: int,
    model_name: str,
    device: str,
    num_channels: int,
    mlp_hidden_dim: int,
    mlp_dropout: float,
    encoder_tune_scope: str,
    unfreeze_last_n_blocks: int,
) -> tuple[Any, list[str], list[int]]:
    """Build MOMENT classification model and configure trainable parameters."""
    if head_type not in SUPPORTED_HEAD_TYPES:
        raise ValueError(f"Unsupported head_type={head_type}. Supported: {list(SUPPORTED_HEAD_TYPES)}")
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

    selected_block_idx, trainable_names = _apply_unfreeze_scope(
        model=model,
        encoder_tune_scope=encoder_tune_scope,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
    )
    model.to(device)
    return model, trainable_names, selected_block_idx


def _make_dataloader(
    *,
    x: np.ndarray,
    input_mask: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x[indices]).float(),
        torch.from_numpy(input_mask[indices]).long(),
        torch.from_numpy(labels[indices]).long(),
        torch.from_numpy(indices).long(),
    )
    dataloader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **dataloader_kwargs)


@torch.no_grad()
def _run_head_inference(
    model: Any,
    loader: DataLoader,
    device: str,
    non_blocking: bool,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    model.eval()
    idx_out: list[np.ndarray] = []
    logits_out: list[np.ndarray] = []
    probs_out: list[np.ndarray] = []
    pred_out: list[np.ndarray] = []
    y_out: list[np.ndarray] = []

    for x_batch, mask_batch, y_batch, idx_batch in loader:
        with (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp and device.startswith("cuda"))
            if device.startswith("cuda")
            else nullcontext()
        ):
            output = model(
                x_enc=x_batch.to(device, non_blocking=non_blocking),
                input_mask=mask_batch.to(device, non_blocking=non_blocking),
                reduction="mean",
            )
        logits_tensor = output.logits.detach().float()
        logits = logits_tensor.cpu().numpy().astype(np.float32, copy=False)
        probs = torch.softmax(logits_tensor, dim=1).cpu().numpy().astype(np.float32, copy=False)
        pred = np.argmax(probs, axis=1).astype(np.int64)

        idx_out.append(idx_batch.cpu().numpy())
        logits_out.append(logits)
        probs_out.append(probs)
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
def _run_embed_inference(
    model: Any,
    loader: DataLoader,
    device: str,
    non_blocking: bool,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    model.eval()
    idx_out: list[np.ndarray] = []
    emb_out: list[np.ndarray] = []
    for x_batch, mask_batch, _, idx_batch in loader:
        with (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp and device.startswith("cuda"))
            if device.startswith("cuda")
            else nullcontext()
        ):
            output = model.embed(
                x_enc=x_batch.to(device, non_blocking=non_blocking),
                input_mask=mask_batch.to(device, non_blocking=non_blocking),
                reduction="mean",
            )
        emb = output.embeddings.detach().float().cpu().numpy().astype(np.float32, copy=False)
        idx_out.append(idx_batch.cpu().numpy())
        emb_out.append(emb)
    idx = np.concatenate(idx_out, axis=0)
    order = np.argsort(idx)
    return {"idx": idx[order], "embeddings": np.concatenate(emb_out, axis=0)[order]}


def _state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _filter_state_dict(state_dict: dict[str, torch.Tensor], keys: list[str]) -> dict[str, torch.Tensor]:
    return {key: state_dict[key] for key in keys if key in state_dict}


def load_trainable_only_checkpoint(model: nn.Module, checkpoint_path: Path | str) -> list[str]:
    """Load trainable-only checkpoint payload into an initialized model."""
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("Expected trainable-only checkpoint payload with `state_dict` key.")
    trainable_state = payload["state_dict"]
    if not isinstance(trainable_state, dict):
        raise ValueError("Checkpoint payload `state_dict` must be a mapping.")
    model_state = model.state_dict()
    missing_model_keys = sorted([key for key in trainable_state.keys() if key not in model_state])
    if missing_model_keys:
        raise ValueError(f"Checkpoint keys are not present in model: {missing_model_keys[:5]}")
    merged_state = {**model_state, **trainable_state}
    model.load_state_dict(merged_state, strict=False)
    return sorted(trainable_state.keys())


def train_one_scheme(
    *,
    config: TrainConfig,
    x: np.ndarray,
    input_mask: np.ndarray,
    labels: np.ndarray,
    split_indices: SplitIndices,
) -> dict[str, Any]:
    """Train one scheme model and return metrics, outputs, and weights."""
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    amp_dtype = _resolve_amp_dtype(config.amp_dtype)
    amp_enabled = bool(config.use_amp and device.startswith("cuda"))
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    effective_pin_memory = bool(config.pin_memory and device.startswith("cuda"))
    LOGGER.info(
        "Runtime options | device=%s amp=%s amp_dtype=%s tf32=%s pin_memory=%s persistent_workers=%s prefetch_factor=%s non_blocking=%s",
        device,
        amp_enabled,
        config.amp_dtype,
        config.enable_tf32,
        effective_pin_memory,
        config.persistent_workers,
        config.prefetch_factor,
        config.non_blocking,
    )
    model, trainable_names, selected_blocks = build_moment_model(
        head_type=config.head_type,
        num_classes=int(np.max(labels)) + 1,
        model_name=config.model_name,
        device=device,
        num_channels=x.shape[1],
        mlp_hidden_dim=config.mlp_hidden_dim,
        mlp_dropout=config.mlp_dropout,
        encoder_tune_scope=config.encoder_tune_scope,
        unfreeze_last_n_blocks=config.unfreeze_last_n_blocks,
    )

    train_loader = _make_dataloader(
        x=x,
        input_mask=input_mask,
        labels=labels,
        indices=split_indices.train_idx,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    val_loader = _make_dataloader(
        x=x,
        input_mask=input_mask,
        labels=labels,
        indices=split_indices.val_idx,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    test_loader = _make_dataloader(
        x=x,
        input_mask=input_mask,
        labels=labels,
        indices=split_indices.test_idx,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    full_loader = _make_dataloader(
        x=x,
        input_mask=input_mask,
        labels=labels,
        indices=np.arange(x.shape[0], dtype=np.int64),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )

    class_count = np.bincount(labels[split_indices.train_idx], minlength=int(np.max(labels)) + 1).astype(np.float32)
    if np.any(class_count == 0):
        raise ValueError("Training split is missing at least one class, cannot build weighted loss.")
    class_weight = class_count.sum() / (class_count.shape[0] * class_count)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weight).to(device))

    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)

    first_batch = next(iter(train_loader))
    with torch.no_grad():
        warmup_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
            if device.startswith("cuda")
            else nullcontext()
        )
        with warmup_ctx:
            _ = model(
                x_enc=first_batch[0].to(device, non_blocking=config.non_blocking),
                input_mask=first_batch[1].to(device, non_blocking=config.non_blocking),
                reduction="mean",
            )

    best_val = -1.0
    best_epoch = 0
    best_state = _state_dict_to_cpu(model)
    bad_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_values: list[float] = []
        for x_batch, mask_batch, y_batch, _ in train_loader:
            optimizer.zero_grad()
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)
                if device.startswith("cuda")
                else nullcontext()
            )
            with autocast_ctx:
                output = model(
                    x_enc=x_batch.to(device, non_blocking=config.non_blocking),
                    input_mask=mask_batch.to(device, non_blocking=config.non_blocking),
                    reduction="mean",
                )
                loss = criterion(output.logits, y_batch.to(device, non_blocking=config.non_blocking))

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            loss_values.append(float(loss.detach().cpu().item()))

        val_out = _run_head_inference(
            model=model,
            loader=val_loader,
            device=device,
            non_blocking=config.non_blocking,
            use_amp=amp_enabled,
            amp_dtype=amp_dtype,
        )
        val_metrics = _classification_metrics(
            y_true=val_out["y_true"],
            y_pred=val_out["pred"],
            n_classes=int(np.max(labels)) + 1,
        )
        val_score = float(val_metrics["balanced_accuracy"])
        train_loss = float(np.mean(loss_values)) if loss_values else float("nan")
        LOGGER.info(
            "epoch=%d train_loss=%.6f val_balanced_accuracy=%.6f",
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
            if bad_epochs >= config.patience:
                LOGGER.info("early_stop epoch=%d best_epoch=%d best_val=%.6f", epoch, best_epoch, best_val)
                break

    model.load_state_dict(best_state)

    train_out = _run_head_inference(
        model=model,
        loader=train_loader,
        device=device,
        non_blocking=config.non_blocking,
        use_amp=amp_enabled,
        amp_dtype=amp_dtype,
    )
    val_out = _run_head_inference(
        model=model,
        loader=val_loader,
        device=device,
        non_blocking=config.non_blocking,
        use_amp=amp_enabled,
        amp_dtype=amp_dtype,
    )
    test_out = _run_head_inference(
        model=model,
        loader=test_loader,
        device=device,
        non_blocking=config.non_blocking,
        use_amp=amp_enabled,
        amp_dtype=amp_dtype,
    )
    full_head_out = _run_head_inference(
        model=model,
        loader=full_loader,
        device=device,
        non_blocking=config.non_blocking,
        use_amp=amp_enabled,
        amp_dtype=amp_dtype,
    )
    full_embed_out = _run_embed_inference(
        model=model,
        loader=full_loader,
        device=device,
        non_blocking=config.non_blocking,
        use_amp=amp_enabled,
        amp_dtype=amp_dtype,
    )

    metrics = {
        "best_epoch": int(best_epoch),
        "train": _classification_metrics(train_out["y_true"], train_out["pred"], int(np.max(labels)) + 1),
        "val": _classification_metrics(val_out["y_true"], val_out["pred"], int(np.max(labels)) + 1),
        "test": _classification_metrics(test_out["y_true"], test_out["pred"], int(np.max(labels)) + 1),
    }
    metrics["val_balanced_accuracy"] = float(metrics["val"]["balanced_accuracy"])
    metrics["test_balanced_accuracy"] = float(metrics["test"]["balanced_accuracy"])

    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "metrics": metrics,
        "head_outputs": full_head_out,
        "base_embeddings": full_embed_out,
        "best_state_dict": best_state,
        "trainable_param_names": trainable_names,
        "selected_encoder_blocks": selected_blocks,
    }


def _save_scheme_artifacts(
    *,
    scheme_dir: Path,
    scheme: str,
    result: dict[str, Any],
    config: TrainConfig,
    classes: list[str],
    segment_ids: np.ndarray,
    label_int: np.ndarray,
    label_str: np.ndarray,
    split: np.ndarray,
) -> dict[str, str]:
    scheme_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, str] = {}

    if config.save_embeddings:
        emb_path = scheme_dir / "embeddings.npz"
        np.savez_compressed(
            emb_path,
            embeddings=result["base_embeddings"]["embeddings"].astype(np.float32),
            segment_id=segment_ids.astype(str),
            label_int=label_int.astype(np.int64),
            label_str=label_str.astype(str),
            split=split.astype(str),
        )
        artifact_paths["embeddings"] = str(emb_path)

    if config.save_predictions:
        pred_path = scheme_dir / "predictions.npz"
        np.savez_compressed(
            pred_path,
            logits=result["head_outputs"]["logits"].astype(np.float32),
            probs=result["head_outputs"]["probs"].astype(np.float32),
            pred_label_int=result["head_outputs"]["pred"].astype(np.int64),
            segment_id=segment_ids.astype(str),
            label_int=label_int.astype(np.int64),
            label_str=label_str.astype(str),
            split=split.astype(str),
        )
        artifact_paths["predictions"] = str(pred_path)

    if config.save_metrics:
        metrics_path = scheme_dir / "metrics.json"
        metrics_path.write_text(json.dumps(result["metrics"], indent=2, sort_keys=True))
        artifact_paths["metrics"] = str(metrics_path)

    if config.save_model_weights:
        model_path = scheme_dir / "model.pt"
        if config.weights_format == "full":
            torch.save(result["best_state_dict"], model_path)
        else:
            trainable_state = _filter_state_dict(result["best_state_dict"], result["trainable_param_names"])
            payload = {
                "weights_format": "trainable_only",
                "scheme": scheme,
                "head_type": config.head_type,
                "encoder_tune_scope": config.encoder_tune_scope,
                "unfreeze_last_n_blocks": config.unfreeze_last_n_blocks,
                "classes": classes,
                "state_dict": trainable_state,
                "trainable_parameter_names": result["trainable_param_names"],
            }
            torch.save(payload, model_path)
        artifact_paths["model"] = str(model_path)

    return artifact_paths


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
    kept = metadata.loc[metadata["kept"]].copy() if "kept" in metadata.columns else metadata.copy()
    kept = kept.set_index("segment_id", drop=False)

    missing = sorted(set(segment_ids) - set(kept.index.astype(str)))
    if missing:
        raise ValueError(f"Missing kept metadata rows for segment IDs: {missing[:5]} ...")
    kept_aligned = kept.loc[segment_ids].reset_index(drop=True)
    if x.shape[0] != kept_aligned.shape[0]:
        raise ValueError("Input and metadata row counts differ after alignment.")
    return x, input_mask, kept_aligned


def _log_loaded_dataset_sanity(metadata: pd.DataFrame) -> None:
    """Log high-level sanity statistics for the aligned canonical dataset."""
    n_segments = int(metadata.shape[0])
    n_subjects = int(metadata["Subject"].astype(str).nunique()) if "Subject" in metadata.columns else 0

    if "Label" not in metadata.columns:
        LOGGER.warning(
            "Loaded canonical dataset: n_segments=%d, n_subjects=%d, missing `Label` column.",
            n_segments,
            n_subjects,
        )
        return

    label_counts = metadata["Label"].dropna().astype(str).value_counts(sort=False).sort_index()
    n_source_labels = int(label_counts.shape[0])

    LOGGER.info(
        "Loaded canonical dataset: n_segments=%d, n_subjects=%d, n_source_labels=%d",
        n_segments,
        n_subjects,
        n_source_labels,
    )
    LOGGER.info("Source label counts:")
    for label, count in label_counts.items():
        LOGGER.info("  %s: %d", label, int(count))


def _log_scheme_label_counts_once(
    scheme_df: pd.DataFrame,
    scheme: str,
    source_total_count: int,
) -> None:
    """Log one concise summary of scheme-label counts after subset selection."""
    n_samples = int(scheme_df.shape[0])
    n_subjects = int(scheme_df["Subject"].astype(str).nunique()) if "Subject" in scheme_df.columns else 0
    scheme_counts = (
        scheme_df["scheme_label"].dropna().astype(str).value_counts(sort=False).sort_index()
        if "scheme_label" in scheme_df.columns
        else pd.Series(dtype=np.int64)
    )
    LOGGER.info(
        "Scheme dataset (%s): n_samples=%d (source_total=%d), n_subjects=%d, n_scheme_labels=%d",
        scheme,
        n_samples,
        int(source_total_count),
        n_subjects,
        int(scheme_counts.shape[0]),
    )
    LOGGER.info("Scheme label counts (%s):", scheme)
    for label, count in scheme_counts.items():
        LOGGER.info("  %s: %d ", label, int(count))


def _parse_csv_list(raw: str) -> tuple[str, ...]:
    values = tuple([part.strip() for part in raw.split(",") if part.strip()])
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


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


def run_pipeline(args: argparse.Namespace) -> Path:
    """Run per-scheme fine-tuning and artifact export pipeline."""
    _seed_everything(args.seed)
    x, input_mask, metadata = _load_cached_inputs(args.input_dir)
    _log_loaded_dataset_sanity(metadata)
    canonical_source_total = int(metadata.shape[0])

    config = TrainConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        non_blocking=args.non_blocking,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        enable_tf32=args.enable_tf32,
        schemes=args.schemes,
        subject_train_frac=args.subject_train_frac,
        subject_val_frac=args.subject_val_frac,
        subject_test_frac=args.subject_test_frac,
        subset_fraction=args.subset_fraction,
        subset_min_per_class=args.subset_min_per_class,
        subset_seed=args.subset_seed,
        head_type=args.head_type,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.mlp_dropout,
        encoder_tune_scope=args.encoder_tune_scope,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        weights_format=args.weights_format,
        save_model_weights=args.save_model_weights,
        save_embeddings=args.save_embeddings,
        save_metrics=args.save_metrics,
        save_predictions=args.save_predictions,
    )
    runtime_device = _resolve_device(config.device)
    _configure_tf32(enable_tf32=config.enable_tf32, device=runtime_device)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_manifest: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "config_path": str(args.config),
        "input_dir": str(args.input_dir),
        "out_dir": str(out_dir),
        "config": asdict(config),
        "schemes": {},
    }

    for scheme_idx, scheme in enumerate(config.schemes):
        LOGGER.info("Processing scheme=%s", scheme)
        scheme_df = metadata.copy()
        scheme_df["row_idx"] = np.arange(scheme_df.shape[0], dtype=np.int64)
        scheme_df = apply_label_scheme(scheme_df, scheme=scheme)
        if scheme_df.empty:
            raise ValueError(f"No rows remained after applying scheme={scheme}")

        subset_idx = _stratified_subset_indices(
            labels=scheme_df["scheme_label"].to_numpy(dtype=str),
            fraction=config.subset_fraction,
            seed=config.subset_seed + scheme_idx,
            min_per_class=config.subset_min_per_class,
        )
        if subset_idx.size < scheme_df.shape[0]:
            scheme_df = scheme_df.iloc[subset_idx].reset_index(drop=True)
        _log_scheme_label_counts_once(
            scheme_df=scheme_df,
            scheme=scheme,
            source_total_count=canonical_source_total,
        )

        scheme_row_idx = scheme_df["row_idx"].to_numpy(dtype=np.int64)
        x_scheme = x[scheme_row_idx]
        mask_scheme = input_mask[scheme_row_idx]
        segment_ids = scheme_df["segment_id"].astype(str).to_numpy()
        label_str = scheme_df["scheme_label"].astype(str).to_numpy()
        classes = sorted(np.unique(label_str).tolist())
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
        label_int = np.asarray([class_to_idx[name] for name in label_str], dtype=np.int64)

        split_indices = split_subject_holdout(
            scheme_df,
            subject_col="Subject",
            label_col="scheme_label",
            train_frac=config.subject_train_frac,
            val_frac=config.subject_val_frac,
            test_frac=config.subject_test_frac,
            seed=config.seed,
        )

        split = np.full(shape=(scheme_df.shape[0],), fill_value="unused", dtype=object)
        split[split_indices.train_idx] = "train"
        split[split_indices.val_idx] = "val"
        split[split_indices.test_idx] = "test"

        result = train_one_scheme(
            config=config,
            x=x_scheme,
            input_mask=mask_scheme,
            labels=label_int,
            split_indices=split_indices,
        )

        scheme_dir = out_dir / scheme
        artifacts = _save_scheme_artifacts(
            scheme_dir=scheme_dir,
            scheme=scheme,
            result=result,
            config=config,
            classes=classes,
            segment_ids=segment_ids,
            label_int=label_int,
            label_str=label_str,
            split=split,
        )
        artifact_hashes = {
            key: {"path": value, "sha256": _sha256_file(Path(value))}
            for key, value in artifacts.items()
            if Path(value).exists()
        }

        run_manifest["schemes"][scheme] = {
            "classes": classes,
            "n_samples": int(scheme_df.shape[0]),
            "encoder_tune_scope": config.encoder_tune_scope,
            "unfreeze_last_n_blocks": config.unfreeze_last_n_blocks,
            "head_type": config.head_type,
            "selected_encoder_blocks": result["selected_encoder_blocks"],
            "trainable_parameter_count": int(len(result["trainable_param_names"])),
            "metrics_summary": {
                "val_balanced_accuracy": float(result["metrics"]["val_balanced_accuracy"]),
                "test_balanced_accuracy": float(result["metrics"]["test_balanced_accuracy"]),
            },
            "artifacts": artifact_hashes,
        }

    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True))
    LOGGER.info("Saved run manifest: %s", manifest_path)
    return out_dir


def build_arg_parser(
    config_defaults: dict[str, Any] | None = None,
    default_config_path: Path = Path("trustME/configs/finetune_imwut_penultimate.full.yaml"),
) -> argparse.ArgumentParser:
    """Build CLI parser for penultimate-layer fine-tuning pipeline."""
    defaults = config_defaults or {}
    parser = argparse.ArgumentParser(description="IMWUT MOMENT penultimate-layer fine-tuning pipeline.")
    parser.add_argument("--config", type=Path, default=default_config_path)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_path_default(defaults, "input_dir", Path("trustME/data/processed/imwut_tobii")),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_path_default(defaults, "out_dir", Path("trustME/data/processed/imwut_tobii_finetuned_penultimate")),
    )
    parser.add_argument("--schemes", type=str, default=_csv_default(defaults, "schemes", "binary,edr,avm"))
    parser.add_argument("--model-name", type=str, default=str(_config_value(defaults, "model_name", "AutonLab/MOMENT-1-large")))
    parser.add_argument("--batch-size", type=int, default=int(_config_value(defaults, "batch_size", 64)))
    parser.add_argument("--epochs", type=int, default=int(_config_value(defaults, "epochs", 10)))
    parser.add_argument("--patience", type=int, default=int(_config_value(defaults, "patience", 3)))
    parser.add_argument("--lr", type=float, default=float(_config_value(defaults, "lr", 1e-4)))
    parser.add_argument("--weight-decay", type=float, default=float(_config_value(defaults, "weight_decay", 1e-4)))
    parser.add_argument("--seed", type=int, default=int(_config_value(defaults, "seed", 42)))
    parser.add_argument("--device", type=str, default=str(_config_value(defaults, "device", "auto")))
    parser.add_argument("--num-workers", type=int, default=int(_config_value(defaults, "num_workers", 0)))
    parser.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        default=bool(_config_value(defaults, "pin_memory", True)),
    )
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument(
        "--persistent-workers",
        dest="persistent_workers",
        action="store_true",
        default=bool(_config_value(defaults, "persistent_workers", True)),
    )
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--prefetch-factor", type=int, default=_config_value(defaults, "prefetch_factor", 2))
    parser.add_argument(
        "--non-blocking",
        dest="non_blocking",
        action="store_true",
        default=bool(_config_value(defaults, "non_blocking", True)),
    )
    parser.add_argument("--no-non-blocking", dest="non_blocking", action="store_false")
    parser.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_true",
        default=bool(_config_value(defaults, "use_amp", True)),
    )
    parser.add_argument("--no-use-amp", dest="use_amp", action="store_false")
    parser.add_argument("--amp-dtype", type=str, choices=["bf16", "fp16"], default=str(_config_value(defaults, "amp_dtype", "bf16")))
    parser.add_argument(
        "--enable-tf32",
        dest="enable_tf32",
        action="store_true",
        default=bool(_config_value(defaults, "enable_tf32", True)),
    )
    parser.add_argument("--disable-tf32", dest="enable_tf32", action="store_false")
    parser.add_argument("--subject-train-frac", type=float, default=float(_config_value(defaults, "subject_train_frac", 0.70)))
    parser.add_argument("--subject-val-frac", type=float, default=float(_config_value(defaults, "subject_val_frac", 0.15)))
    parser.add_argument("--subject-test-frac", type=float, default=float(_config_value(defaults, "subject_test_frac", 0.15)))
    parser.add_argument("--subset-fraction", type=float, default=_config_value(defaults, "subset_fraction", None))
    parser.add_argument("--subset-min-per-class", type=int, default=int(_config_value(defaults, "subset_min_per_class", 1)))
    parser.add_argument("--subset-seed", type=int, default=int(_config_value(defaults, "subset_seed", 42)))
    parser.add_argument("--head-type", type=str, choices=list(SUPPORTED_HEAD_TYPES), default=str(_config_value(defaults, "head_type", "linear")))
    parser.add_argument("--mlp-hidden-dim", type=int, default=int(_config_value(defaults, "mlp_hidden_dim", 256)))
    parser.add_argument("--mlp-dropout", type=float, default=float(_config_value(defaults, "mlp_dropout", 0.1)))
    parser.add_argument(
        "--encoder-tune-scope",
        type=str,
        choices=list(SUPPORTED_ENCODER_TUNE_SCOPES),
        default=str(_config_value(defaults, "encoder_tune_scope", "last_n_blocks")),
    )
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=int(_config_value(defaults, "unfreeze_last_n_blocks", 1)))
    parser.add_argument(
        "--weights-format",
        type=str,
        choices=list(SUPPORTED_WEIGHTS_FORMATS),
        default=str(_config_value(defaults, "weights_format", "trainable_only")),
    )
    parser.add_argument(
        "--save-model-weights",
        dest="save_model_weights",
        action="store_true",
        default=bool(_config_value(defaults, "save_model_weights", False)),
    )
    parser.add_argument("--no-save-model-weights", dest="save_model_weights", action="store_false")
    parser.add_argument(
        "--save-embeddings",
        dest="save_embeddings",
        action="store_true",
        default=bool(_config_value(defaults, "save_embeddings", True)),
    )
    parser.add_argument("--no-save-embeddings", dest="save_embeddings", action="store_false")
    parser.add_argument(
        "--save-metrics",
        dest="save_metrics",
        action="store_true",
        default=bool(_config_value(defaults, "save_metrics", True)),
    )
    parser.add_argument("--no-save-metrics", dest="save_metrics", action="store_false")
    parser.add_argument(
        "--save-predictions",
        dest="save_predictions",
        action="store_true",
        default=bool(_config_value(defaults, "save_predictions", False)),
    )
    parser.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    parser.add_argument("--verbose", action="store_true", default=bool(_config_value(defaults, "verbose", False)))
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=Path("trustME/configs/finetune_imwut_penultimate.full.yaml"))
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_defaults = _load_yaml_config(pre_args.config)

    parser = build_arg_parser(config_defaults=config_defaults, default_config_path=pre_args.config)
    args = parser.parse_args(argv)
    _setup_logging(verbose=args.verbose)

    args.schemes = _parse_csv_list(args.schemes)
    unsupported_schemes = sorted(set(args.schemes) - set(SUPPORTED_SCHEMES))
    if unsupported_schemes:
        raise ValueError(f"Unsupported schemes requested: {unsupported_schemes}")
    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
