"""Trust-me MOMENT embedding pipeline.

By default, consumes a preprocessed `.npz` cache produced by
`preprocess_trustme_cache.py` and computes one embedding per kept window.
Optionally, it can rebuild MOMENT inputs from `cleaned_windowed` parquet files.

Examples
--------
Default cache-first path:
python trustME_trustme/source/segment_trustme.py \
  --input-npz trustME_trustme/data/cleaned_windowed_preprocessed/trustme_preprocessed_moment_inputs.npz \
  --out-dir trustME_trustme/data/processed/trustme_tobii_0shot \
  --batch-size 64 --device auto --model-name AutonLab/MOMENT-1-large

Legacy rebuild path:
python trustME_trustme/source/segment_trustme.py \
  --rebuild-inputs \
  --cleaned-root trustME_trustme/data/cleaned_windowed \
  --out-dir trustME_trustme/data/processed/trustme_tobii_0shot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


LOGGER = logging.getLogger(__name__)
PIPELINE_VERSION = "1.2.0"
VALIDITY_COLUMNS = (
    "ValidityLeft",
    "ValidityRight",
    "PupilValidityLeft",
    "PupilValidityRight",
)
EXTRA_LABEL_COLUMNS = (
    "sleep_feedback",
    "prompt_id",
    "prompt_time",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
)


@dataclass(frozen=True)
class SegmentConfig:
    """Configuration for Trust-me segment preprocessing and MOMENT input creation."""

    seq_len: int = 512
    channels: tuple[str, ...] = (
        "GazePointX",
        "GazePointY",
        "PupilSizeLeft",
        "PupilSizeRight",
        "AverageDistance",
    )
    max_invalid_frames: int = 60
    min_valid_fraction: float = 0.3
    # Locked from user decisions (CLUES CLI defaults used previously).
    min_valid_frames: int = 32
    # Keep external normalization off by default; MOMENT RevIN standardizes per sample.
    normalize: str = "none"
    label_column: str = "sleep_feedback"
    require_label: bool = False


@dataclass
class SegmentProcessResult:
    """Container for one segment's processed payload and QC metadata."""

    segment_id: str
    used_len: int
    pad_len: int
    kept: bool
    drop_reason: str
    orig_len: int
    valid_fraction: float
    max_invalid_run: int
    x: np.ndarray | None = None
    input_mask: np.ndarray | None = None


@dataclass
class BuildStats:
    """Summary statistics for Trust-me input building."""

    subjects_processed: int
    parquet_files_processed: int
    total_candidates: int
    total_kept: int
    total_dropped: int
    drop_reason_counts: dict[str, int]


def _setup_logging(verbose: bool, log_file: Path | None = None) -> None:
    """Configure console logging and optional file logging."""
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def _format_duration_hhmmss(total_seconds: float) -> str:
    """Format elapsed time in HH:MM:SS."""
    whole_seconds = int(round(total_seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _sha256_file(path: Path) -> str:
    """Return SHA256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_hash(config: SegmentConfig) -> str:
    """Return a stable hash for the segmentation config."""
    config_json = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()


def _compute_run_lengths(mask: np.ndarray) -> np.ndarray:
    """Compute run-length per position for consecutive True values."""
    run_lengths = np.zeros(mask.shape[0], dtype=np.int32)
    run_start = -1
    for idx, is_true in enumerate(mask):
        if is_true and run_start < 0:
            run_start = idx
        elif not is_true and run_start >= 0:
            run_lengths[run_start:idx] = idx - run_start
            run_start = -1
    if run_start >= 0:
        run_lengths[run_start:] = mask.shape[0] - run_start
    return run_lengths


def _build_invalid_mask(segment_df: pd.DataFrame, channels: tuple[str, ...]) -> np.ndarray:
    """Return per-row invalid mask using validity flags and sentinel values."""
    signals = segment_df.loc[:, list(channels)]
    sentinel_invalid = (signals == -1.0).any(axis=1).to_numpy()
    nan_invalid = signals.isna().any(axis=1).to_numpy()

    available_validity_cols = [col for col in VALIDITY_COLUMNS if col in segment_df.columns]
    if available_validity_cols:
        validity_frame = segment_df.loc[:, available_validity_cols]
        validity_invalid = (~validity_frame.eq(1)).any(axis=1).to_numpy()
    else:
        validity_invalid = np.zeros(len(segment_df), dtype=bool)

    return sentinel_invalid | nan_invalid | validity_invalid


def _process_segment(segment_df: pd.DataFrame, segment_id: str, config: SegmentConfig) -> SegmentProcessResult:
    """Transform one segment into padded MOMENT inputs and QC metadata."""
    orig_len = len(segment_df)
    if orig_len <= 0:
        return SegmentProcessResult(
            segment_id=segment_id,
            used_len=0,
            pad_len=config.seq_len,
            kept=False,
            drop_reason="empty_segment",
            orig_len=0,
            valid_fraction=0.0,
            max_invalid_run=0,
        )

    signals = segment_df.loc[:, list(config.channels)].astype(float).copy()
    invalid_mask = _build_invalid_mask(segment_df, config.channels)
    valid_count = int((~invalid_mask).sum())
    valid_fraction = valid_count / orig_len
    run_lengths = _compute_run_lengths(invalid_mask)
    max_invalid_run = int(run_lengths.max()) if run_lengths.size else 0

    if valid_fraction < config.min_valid_fraction:
        return SegmentProcessResult(
            segment_id=segment_id,
            used_len=0,
            pad_len=config.seq_len,
            kept=False,
            drop_reason="low_valid_fraction",
            orig_len=orig_len,
            valid_fraction=valid_fraction,
            max_invalid_run=max_invalid_run,
        )
    if valid_count < config.min_valid_frames:
        return SegmentProcessResult(
            segment_id=segment_id,
            used_len=0,
            pad_len=config.seq_len,
            kept=False,
            drop_reason="too_few_valid_frames",
            orig_len=orig_len,
            valid_fraction=valid_fraction,
            max_invalid_run=max_invalid_run,
        )

    signals.loc[invalid_mask, :] = np.nan
    # Pandas interpolate(limit=K) can error when K+1 > segment length.
    interp_limit = min(max(config.max_invalid_frames - 1, 0), max(orig_len - 1, 0))
    signals = signals.interpolate(method="linear", limit=interp_limit)
    signals = signals.ffill().bfill()

    if signals.isna().any(axis=None):
        return SegmentProcessResult(
            segment_id=segment_id,
            used_len=0,
            pad_len=config.seq_len,
            kept=False,
            drop_reason="nan_after_interpolation",
            orig_len=orig_len,
            valid_fraction=valid_fraction,
            max_invalid_run=max_invalid_run,
        )

    win = signals.to_numpy(dtype=np.float32).T
    if config.normalize == "per_segment_zscore":
        mean = win.mean(axis=1, keepdims=True)
        std = win.std(axis=1, keepdims=True) + 1e-8
        win = (win - mean) / std
    elif config.normalize != "none":
        raise ValueError(f"Unsupported normalize mode: {config.normalize}")

    used_len = min(orig_len, config.seq_len)
    x = np.zeros((len(config.channels), config.seq_len), dtype=np.float32)
    x[:, :used_len] = win[:, :used_len]

    input_mask = np.zeros(config.seq_len, dtype=np.uint8)
    input_mask[:used_len] = 1
    pad_len = config.seq_len - used_len

    return SegmentProcessResult(
        segment_id=segment_id,
        used_len=used_len,
        pad_len=pad_len,
        kept=True,
        drop_reason="",
        orig_len=orig_len,
        valid_fraction=valid_fraction,
        max_invalid_run=max_invalid_run,
        x=x,
        input_mask=input_mask,
    )


def _write_manifest(manifest: dict[str, Any], out_dir: Path) -> Path:
    """Write manifest JSON to output directory."""
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return manifest_path


def _parse_cache_manifest(cache_payload: np.lib.npyio.NpzFile) -> dict[str, Any] | None:
    """Parse optional JSON manifest embedded in cache npz payload."""
    if "manifest_json" not in cache_payload.files:
        return None
    try:
        raw_value = cache_payload["manifest_json"]
        if isinstance(raw_value, np.ndarray):
            manifest_text = str(raw_value.item()) if raw_value.ndim == 0 else "".join(raw_value.astype(str).tolist())
        else:
            manifest_text = str(raw_value)
        parsed = json.loads(manifest_text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        LOGGER.warning("Unable to parse `manifest_json` from cache payload; continuing without it.")
        return None


def _materialize_metadata_from_cache(cache_payload: np.lib.npyio.NpzFile, out_dir: Path) -> Path | None:
    """Write embedded metadata parquet from cache payload into out_dir, if present."""
    if "metadata_parquet" not in cache_payload.files:
        return None
    metadata_bytes = np.asarray(cache_payload["metadata_parquet"], dtype=np.uint8).tobytes()
    metadata_path = out_dir / "segments_metadata.parquet"
    metadata_path.write_bytes(metadata_bytes)
    return metadata_path


def _prepare_inputs_from_cache(input_npz: Path, out_dir: Path) -> tuple[Path, Path | None, dict[str, Any] | None]:
    """Validate cache payload and materialize optional metadata sidecar."""
    if not input_npz.exists():
        raise FileNotFoundError(
            f"Missing input npz cache: {input_npz}. "
            "Generate it with `preprocess_trustme_cache.py` or pass `--rebuild-inputs`."
        )
    if not input_npz.is_file():
        raise FileNotFoundError(f"Input npz path is not a file: {input_npz}")

    with np.load(input_npz, allow_pickle=False) as cache_payload:
        required = {"x", "input_mask", "segment_id"}
        missing = sorted(required - set(cache_payload.files))
        if missing:
            raise ValueError(f"Input npz missing required arrays {missing}: {input_npz}")
        metadata_path = _materialize_metadata_from_cache(cache_payload=cache_payload, out_dir=out_dir)
        cache_manifest = _parse_cache_manifest(cache_payload=cache_payload)
    return input_npz, metadata_path, cache_manifest


def _ensure_parquet_support() -> None:
    """Ensure parquet IO backend is available."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        try:
            import fastparquet  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Parquet support is required but neither `pyarrow` nor `fastparquet` is installed."
            ) from exc


def _get_parquet_columns(path: Path) -> list[str]:
    """Return column names for a parquet file without loading full data."""
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        try:
            import fastparquet

            return list(fastparquet.ParquetFile(path).columns)
        except Exception as exc:
            raise RuntimeError(f"Failed to read parquet schema for {path}") from exc


def _get_parquet_num_rows(path: Path) -> int:
    """Return row count for parquet file without loading full table when possible."""
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        try:
            import fastparquet

            return int(fastparquet.ParquetFile(path).count())
        except Exception:
            return int(len(pd.read_parquet(path)))


def _parse_subject_filter(raw_value: str | None) -> set[str] | None:
    """Parse comma-separated subject filter argument."""
    if not raw_value:
        return None
    subjects = {part.strip() for part in raw_value.split(",") if part.strip()}
    return subjects or None


def _discover_subjects(cleaned_root: Path, subject_filter: set[str] | None) -> list[str]:
    """Discover subject folders under cleaned root, optionally filtered."""
    discovered = sorted([p.name for p in cleaned_root.iterdir() if p.is_dir()])
    if subject_filter is None:
        return discovered
    return [subject for subject in discovered if subject in subject_filter]


def _window_id_to_str(window_id: Any) -> str:
    """Convert window id to stable compact string."""
    if isinstance(window_id, (int, np.integer)):
        return str(int(window_id))
    if isinstance(window_id, (float, np.floating)) and float(window_id).is_integer():
        return str(int(window_id))
    return str(window_id)


def _window_scalar(window_df: pd.DataFrame, column: str, segment_id: str) -> Any:
    """Return single scalar value for a column and validate within-window consistency."""
    if column not in window_df.columns:
        return np.nan

    non_null = window_df[column].dropna().unique()
    if len(non_null) > 1:
        raise ValueError(
            f"Conflicting non-null values in column '{column}' for segment={segment_id}: {non_null[:5]}"
        )
    if len(non_null) == 0:
        return np.nan
    return non_null[0]


def build_trustme_window_inputs(
    cleaned_root: Path,
    out_dir: Path,
    config: SegmentConfig,
    subject_filter: set[str] | None = None,
) -> tuple[Path, BuildStats]:
    """Build cleaned fixed-length MOMENT inputs from Trust-me cleaned-windowed parquet files."""
    _ensure_parquet_support()
    LOGGER.info("Building Trust-me segmented MOMENT inputs")
    LOGGER.info("cleaned_root=%s", cleaned_root)
    LOGGER.info("out_dir=%s", out_dir)
    LOGGER.info("segment_config=%s", asdict(config))
    if subject_filter:
        LOGGER.info("subject_filter=%s", sorted(subject_filter))

    if not cleaned_root.exists():
        raise FileNotFoundError(f"Missing cleaned root: {cleaned_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, Any]] = []
    x_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    kept_segment_ids: list[str] = []

    subjects = _discover_subjects(cleaned_root=cleaned_root, subject_filter=subject_filter)
    if not subjects:
        raise ValueError("No subjects found in cleaned root after applying subject filter.")

    required_core_columns = set(config.channels) | set(VALIDITY_COLUMNS) | {"window_id", "TimeStamp"}
    optional_columns = set(EXTRA_LABEL_COLUMNS)

    total_candidates = 0
    total_kept = 0
    total_dropped = 0
    drop_reason_counts: dict[str, int] = {}
    parquet_files_processed = 0

    for subject in subjects:
        tobii_dir = cleaned_root / subject / "tobii"
        if not tobii_dir.exists():
            raise FileNotFoundError(f"Missing Tobii folder for subject={subject}: {tobii_dir}")

        parquet_files = sorted(tobii_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {tobii_dir}")

        LOGGER.info("Subject %s: processing %d parquet files", subject, len(parquet_files))

        subject_kept = 0
        subject_dropped = 0
        subject_drop_reasons: dict[str, int] = {}

        for parquet_path in parquet_files:
            parquet_files_processed += 1
            source_filename = parquet_path.name.removesuffix(".parquet")

            available_columns = set(_get_parquet_columns(parquet_path))
            missing_required = sorted(required_core_columns - available_columns)
            if missing_required:
                raise ValueError(f"Parquet missing required columns {missing_required}: {parquet_path}")

            read_columns = sorted(list(required_core_columns | (optional_columns & available_columns)))
            df = pd.read_parquet(parquet_path, columns=read_columns)

            if df.empty:
                LOGGER.info("Skipping empty parquet file: %s", parquet_path)
                continue

            grouped = df.groupby("window_id", sort=False)
            LOGGER.info(
                "Loaded %s rows=%d windows=%d",
                parquet_path.name,
                len(df),
                grouped.ngroups,
            )

            for window_id, window_df in grouped:
                if pd.isna(window_id):
                    raise ValueError(f"Found NaN window_id in {parquet_path}")

                window_id_str = _window_id_to_str(window_id)
                segment_id = f"{subject}_{window_id_str}"
                source_row_idx = total_candidates
                total_candidates += 1

                label_values = {col: _window_scalar(window_df=window_df, column=col, segment_id=segment_id) for col in EXTRA_LABEL_COLUMNS}
                label_value = label_values.get(config.label_column, np.nan)

                if config.require_label and pd.isna(label_value):
                    total_dropped += 1
                    subject_dropped += 1
                    drop_reason = "missing_label"
                    drop_reason_counts[drop_reason] = drop_reason_counts.get(drop_reason, 0) + 1
                    subject_drop_reasons[drop_reason] = subject_drop_reasons.get(drop_reason, 0) + 1
                    continue

                ts_series = window_df["TimeStamp"].dropna() if "TimeStamp" in window_df.columns else pd.Series(dtype=float)
                if ts_series.empty:
                    start_t = np.nan
                    end_t = np.nan
                else:
                    start_t = float(ts_series.iloc[0])
                    end_t = float(ts_series.iloc[-1])

                result = _process_segment(segment_df=window_df, segment_id=segment_id, config=config)

                if result.kept:
                    if result.x is None or result.input_mask is None:
                        raise RuntimeError(f"Missing processed payload for kept segment: {segment_id}")

                    metadata_row = {
                        "segment_id": segment_id,
                        "source_row_idx": int(source_row_idx),
                        "Subject": subject,
                        "Path": f"{subject}/tobii",
                        "Filename": source_filename,
                        "window_id": window_id,
                        "Start_i": 0,
                        "End_i": max(result.orig_len - 1, 0),
                        "Start_t": start_t,
                        "End_t": end_t,
                        "Label": label_value,
                        "orig_len": int(result.orig_len),
                        "valid_fraction": float(result.valid_fraction),
                        "max_invalid_run": int(result.max_invalid_run),
                        "kept": True,
                        "drop_reason": "",
                        "pad_len": int(result.pad_len),
                    }
                    metadata_row.update(label_values)

                    metadata_rows.append(metadata_row)
                    x_list.append(result.x)
                    mask_list.append(result.input_mask)
                    kept_segment_ids.append(segment_id)
                    total_kept += 1
                    subject_kept += 1
                else:
                    total_dropped += 1
                    subject_dropped += 1
                    drop_reason = result.drop_reason or "unknown_drop_reason"
                    drop_reason_counts[drop_reason] = drop_reason_counts.get(drop_reason, 0) + 1
                    subject_drop_reasons[drop_reason] = subject_drop_reasons.get(drop_reason, 0) + 1

        LOGGER.info("Subject %s done: kept=%d dropped=%d", subject, subject_kept, subject_dropped)
        if subject_drop_reasons:
            LOGGER.info("Subject %s drop reasons: %s", subject, subject_drop_reasons)

    metadata = pd.DataFrame(metadata_rows).sort_values("source_row_idx").reset_index(drop=True)
    metadata_path = out_dir / "segments_metadata.parquet"
    metadata.to_parquet(metadata_path, index=False)

    if x_list:
        x = np.stack(x_list, axis=0).astype(np.float32)
        input_mask = np.stack(mask_list, axis=0).astype(np.uint8)
        segment_ids_arr = np.asarray(kept_segment_ids, dtype=str)
    else:
        x = np.zeros((0, len(config.channels), config.seq_len), dtype=np.float32)
        input_mask = np.zeros((0, config.seq_len), dtype=np.uint8)
        segment_ids_arr = np.asarray([], dtype=str)

    inputs_path = out_dir / "moment_inputs.npz"
    np.savez_compressed(inputs_path, x=x, input_mask=input_mask, segment_id=segment_ids_arr)

    LOGGER.info("Saved metadata rows=%d (kept only)", len(metadata))
    LOGGER.info(
        "Saved %s with x.shape=%s input_mask.shape=%s",
        inputs_path,
        tuple(x.shape),
        tuple(input_mask.shape),
    )
    if drop_reason_counts:
        LOGGER.info("Global drop reasons: %s", drop_reason_counts)

    stats = BuildStats(
        subjects_processed=len(subjects),
        parquet_files_processed=parquet_files_processed,
        total_candidates=total_candidates,
        total_kept=total_kept,
        total_dropped=total_dropped,
        drop_reason_counts=drop_reason_counts,
    )
    return out_dir, stats


def compute_moment_embeddings(
    input_npz: Path,
    out_npz: Path,
    model_name: str,
    batch_size: int,
    device: str,
    reduction: str = "mean",
) -> Path:
    """Compute one MOMENT embedding per cached segment input."""
    from momentfm import MOMENTPipeline

    LOGGER.info("Computing MOMENT embeddings from %s", input_npz)
    with np.load(input_npz, allow_pickle=False) as payload:
        x = payload["x"].astype(np.float32, copy=False)
        input_mask = payload["input_mask"].astype(np.int64, copy=False)
        segment_ids = payload["segment_id"].astype(str, copy=False)

    if x.shape[0] != input_mask.shape[0] or x.shape[0] != segment_ids.shape[0]:
        raise ValueError("Input arrays are misaligned in moment_inputs.npz")

    if x.shape[0] == 0:
        LOGGER.warning("No input segments found; writing empty embeddings to %s", out_npz)
        np.savez_compressed(out_npz, embeddings=np.zeros((0, 0), dtype=np.float32), segment_id=segment_ids)
        return out_npz

    if device == "auto":
        runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        runtime_device = device

    LOGGER.info(
        "Embedding settings: model=%s device=%s batch_size=%d reduction=%s samples=%d",
        model_name,
        runtime_device,
        batch_size,
        reduction,
        x.shape[0],
    )

    model = MOMENTPipeline.from_pretrained(model_name, model_kwargs={"task_name": "embedding"})
    model.init()
    model.to(runtime_device)
    model.eval()

    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(input_mask).long())
    pin_memory = runtime_device == "cuda"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    LOGGER.info("Embedding will run in %d batches", len(dataloader))

    all_embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_idx, (x_batch, mask_batch) in enumerate(dataloader, start=1):
            output = model(
                x_enc=x_batch.to(runtime_device, non_blocking=pin_memory),
                input_mask=mask_batch.to(runtime_device, non_blocking=pin_memory),
                reduction=reduction,
            )
            if output.embeddings is None:
                raise RuntimeError(
                    "MOMENT returned `embeddings=None`. Expected task_name='embedding'. "
                    f"Output fields: embeddings={type(output.embeddings)}, "
                    f"reconstruction={type(output.reconstruction)}, logits={type(output.logits)}."
                )
            emb = output.embeddings.detach().cpu().numpy()
            all_embeddings.append(emb)
            if batch_idx == 1 or batch_idx == len(dataloader) or batch_idx % 25 == 0:
                LOGGER.info("Processed embedding batch %d/%d", batch_idx, len(dataloader))

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    np.savez_compressed(out_npz, embeddings=embeddings, segment_id=segment_ids)
    LOGGER.info("Saved %s with embeddings.shape=%s", out_npz, tuple(embeddings.shape))
    return out_npz


def run_pipeline(
    out_dir: Path,
    config: SegmentConfig,
    input_npz: Path | None = Path(
        "trustME_trustme/data/cleaned_windowed_preprocessed/trustme_preprocessed_moment_inputs.npz"
    ),
    cleaned_root: Path | None = None,
    model_name: str = "AutonLab/MOMENT-1-large",
    batch_size: int = 64,
    device: str = "auto",
    reduction: str = "mean",
    subject_filter: set[str] | None = None,
    skip_embedding: bool = False,
    rebuild_inputs: bool = False,
) -> Path:
    """Run Trust-me embedding pipeline from cache (default) or rebuilt inputs."""
    _ensure_parquet_support()
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Running Trust-me Tobii pipeline")
    LOGGER.info("pipeline_out_dir=%s", out_dir)
    LOGGER.info("input_mode=%s", "rebuild" if rebuild_inputs else "cache")

    stats: BuildStats | None = None
    cache_manifest: dict[str, Any] | None = None
    metadata_path: Path | None = None
    if rebuild_inputs:
        if cleaned_root is None:
            raise ValueError("--cleaned-root must be provided when --rebuild-inputs is enabled.")
        LOGGER.info("Rebuilding MOMENT inputs from cleaned_root=%s", cleaned_root)
        _, stats = build_trustme_window_inputs(
            cleaned_root=cleaned_root,
            out_dir=out_dir,
            config=config,
            subject_filter=subject_filter,
        )
        inputs_path = out_dir / "moment_inputs.npz"
        metadata_path = out_dir / "segments_metadata.parquet"
    else:
        if input_npz is None:
            raise ValueError("--input-npz must be provided unless --rebuild-inputs is enabled.")
        LOGGER.info("Using preprocessed cache input: %s", input_npz)
        inputs_path, metadata_path, cache_manifest = _prepare_inputs_from_cache(
            input_npz=input_npz,
            out_dir=out_dir,
        )
        if metadata_path is not None:
            LOGGER.info("Materialized metadata from cache to %s", metadata_path)
        else:
            LOGGER.warning("No embedded metadata found in cache payload.")

    embeddings_path = out_dir / "moment_embeddings.npz"

    if not skip_embedding:
        compute_moment_embeddings(
            input_npz=inputs_path,
            out_npz=embeddings_path,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            reduction=reduction,
        )
    else:
        LOGGER.info("Skipping embedding stage (--skip-embedding)")

    metadata_rows = 0
    if metadata_path is not None and metadata_path.exists():
        metadata_rows = _get_parquet_num_rows(metadata_path)

    manifest: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "input_dataset": "trustME_trustme/data/cleaned_windowed" if rebuild_inputs else "preprocessed_cache_npz",
        "cleaned_root": str(cleaned_root) if cleaned_root is not None else None,
        "out_dir": str(out_dir),
        "model": {
            "name": model_name,
            "task_name": "embedding",
            "reduction": reduction,
            "batch_size": batch_size,
            "device": device,
            "computed_embeddings": not skip_embedding,
        },
        "artifacts": {
            "moment_inputs": {
                "path": str(inputs_path),
                "sha256": _sha256_file(inputs_path),
            },
        },
    }
    if metadata_path is not None and metadata_path.exists():
        manifest["artifacts"]["segments_metadata"] = {
            "path": str(metadata_path),
            "sha256": _sha256_file(metadata_path),
        }

    if rebuild_inputs:
        if stats is None:
            raise RuntimeError("Missing build stats for rebuild mode.")
        manifest["config"] = asdict(config)
        manifest["config_hash"] = _config_hash(config)
        manifest["counts"] = {
            "segments_total_candidates": int(stats.total_candidates),
            "segments_kept": int(stats.total_kept),
            "segments_dropped": int(stats.total_dropped),
            "drop_reason_counts": stats.drop_reason_counts,
            "subjects_processed": int(stats.subjects_processed),
            "parquet_files_processed": int(stats.parquet_files_processed),
            "metadata_rows": int(metadata_rows),
        }
    else:
        with np.load(inputs_path, allow_pickle=False) as payload:
            segments_kept = int(payload["x"].shape[0])
        cache_counts: dict[str, Any] = {}
        def _cache_int(name: str, default: int) -> int:
            value = cache_counts.get(name, default)
            return default if value is None else int(value)

        if cache_manifest is not None and isinstance(cache_manifest.get("counts"), dict):
            cache_counts = cache_manifest["counts"]
            manifest["cache_manifest"] = cache_manifest
        manifest["config"] = (
            cache_manifest.get("config")
            if cache_manifest is not None and isinstance(cache_manifest.get("config"), dict)
            else asdict(config)
        )
        manifest["config_hash"] = (
            str(cache_manifest.get("config_hash"))
            if cache_manifest is not None and cache_manifest.get("config_hash") is not None
            else _config_hash(config)
        )
        manifest["counts"] = {
            "segments_total_candidates": _cache_int("segments_total_candidates", segments_kept),
            "segments_kept": _cache_int("segments_kept", segments_kept),
            "segments_dropped": _cache_int("segments_dropped", 0),
            "drop_reason_counts": cache_counts.get("drop_reason_counts", {}),
            "subjects_processed": _cache_int("subjects_processed", 0),
            "parquet_files_processed": _cache_int("parquet_files_processed", 0),
            "metadata_rows": _cache_int("metadata_rows", metadata_rows),
        }

    if embeddings_path.exists():
        manifest["artifacts"]["moment_embeddings"] = {
            "path": str(embeddings_path),
            "sha256": _sha256_file(embeddings_path),
        }

    written_manifest_path = _write_manifest(manifest=manifest, out_dir=out_dir)
    LOGGER.info("Saved manifest: %s", written_manifest_path)
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for Trust-me embedding pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Trust-me MOMENT embedding pipeline. "
            "Uses preprocessed cache npz by default; pass --rebuild-inputs to rebuild from cleaned parquet."
        )
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path("trustME_trustme/data/cleaned_windowed_preprocessed/trustme_preprocessed_moment_inputs.npz"),
        help=(
            "Preprocessed input cache (.npz) from preprocess_trustme_cache.py. "
            "Used by default unless --rebuild-inputs is set."
        ),
    )
    parser.add_argument(
        "--rebuild-inputs",
        action="store_true",
        help="Rebuild MOMENT inputs from --cleaned-root parquet data instead of using --input-npz.",
    )
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=Path("trustME_trustme/data/cleaned_windowed"),
        help="Root directory containing per-subject cleaned_windowed Tobii parquet files (used with --rebuild-inputs).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("trustME_trustme/data/processed/trustme_tobii_0shot"),
        help="Output folder for metadata, inputs, embeddings, and manifest.",
    )
    parser.add_argument("--model-name", type=str, default="AutonLab/MOMENT-1-large")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--reduction", type=str, default="mean")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subset, e.g. s_004_pk,s_005_ak")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file. Default: <out-dir>/segment_trustme.log",
    )

    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument(
        "--channels",
        type=str,
        default="GazePointX,GazePointY,PupilSizeLeft,PupilSizeRight,AverageDistance",
        help="Comma-separated signal channel list.",
    )
    parser.add_argument("--max-invalid-frames", type=int, default=60)
    parser.add_argument("--min-valid-fraction", type=float, default=0.3)
    parser.add_argument("--min-valid-frames", type=int, default=32)
    parser.add_argument(
        "--normalize",
        type=str,
        default="none",
        choices=["per_segment_zscore", "none"],
        help=(
            "External preprocessing normalization. "
            "NOTE: keep 'none' unless you intentionally want extra scaling; "
            "MOMENT RevIN already does per-sample standardization."
        ),
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="sleep_feedback",
        choices=list(EXTRA_LABEL_COLUMNS),
        help="Column to copy into canonical metadata `Label`.",
    )
    parser.add_argument(
        "--require-label",
        action="store_true",
        help="Keep only windows where --label-column is non-null.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Trust-me embedding pipeline."""
    parser = build_arg_parser()
    parsed = parser.parse_args(argv)
    log_file = parsed.log_file if parsed.log_file is not None else parsed.out_dir / "segment_trustme.log"
    _setup_logging(verbose=parsed.verbose, log_file=log_file)

    channels = tuple([channel.strip() for channel in parsed.channels.split(",") if channel.strip()])
    if not channels:
        raise ValueError("At least one channel must be provided")

    config = SegmentConfig(
        seq_len=parsed.seq_len,
        channels=channels,
        max_invalid_frames=parsed.max_invalid_frames,
        min_valid_fraction=parsed.min_valid_fraction,
        min_valid_frames=parsed.min_valid_frames,
        normalize=parsed.normalize,
        label_column=parsed.label_column,
        require_label=parsed.require_label,
    )
    subject_filter = _parse_subject_filter(parsed.subjects)

    LOGGER.info("CLI parsed. verbose=%s", parsed.verbose)
    LOGGER.info("Log file: %s", log_file)
    LOGGER.info("input_mode=%s", "rebuild" if parsed.rebuild_inputs else "cache")
    if parsed.rebuild_inputs:
        LOGGER.info("cleaned_root=%s", parsed.cleaned_root)
    else:
        LOGGER.info("input_npz=%s", parsed.input_npz)

    started_at = time.perf_counter()
    exit_code = 0
    try:
        run_pipeline(
            out_dir=parsed.out_dir,
            config=config,
            input_npz=parsed.input_npz,
            cleaned_root=parsed.cleaned_root,
            model_name=parsed.model_name,
            batch_size=parsed.batch_size,
            device=parsed.device,
            reduction=parsed.reduction,
            subject_filter=subject_filter,
            skip_embedding=parsed.skip_embedding,
            rebuild_inputs=parsed.rebuild_inputs,
        )
        LOGGER.info("Pipeline completed: %s", parsed.out_dir)
    except Exception:
        exit_code = 1
        LOGGER.exception("Pipeline failed")
        raise
    finally:
        elapsed_seconds = time.perf_counter() - started_at
        LOGGER.info(
            "Total runtime: %s (%d seconds)",
            _format_duration_hhmmss(elapsed_seconds),
            int(round(elapsed_seconds)),
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
