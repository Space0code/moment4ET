"""Trust-me cleaned-windowed Tobii preprocessing and MOMENT embedding pipeline.

Builds fixed-length MOMENT inputs from `trustME_trustme/data/cleaned_windowed`
Parquet files (grouped by `window_id`) and computes one embedding per kept window.

Example
-------
python trustME_trustme/source/segment_trustme.py \
  --cleaned-root trustME_trustme/data/cleaned_windowed \
  --out-dir trustME_trustme/data/processed/trustme_tobii_0shot \
  --batch-size 64 --device auto --model-name AutonLab/MOMENT-1-large
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


LOGGER = logging.getLogger(__name__)
PIPELINE_VERSION = "1.0.0"
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
    )
    max_invalid_frames: int = 60
    min_valid_fraction: float = 0.3
    # Locked from user decisions (CLUES CLI defaults used previously).
    min_valid_frames: int = 32
    # Keep external normalization off by default; MOMENT RevIN standardizes per sample.
    normalize: str = "none"
    label_column: str = "sleep_feedback"


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


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


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
    payload = np.load(input_npz, allow_pickle=False)
    x = payload["x"].astype(np.float32)
    input_mask = payload["input_mask"].astype(np.int64)
    segment_ids = payload["segment_id"].astype(str)

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    LOGGER.info("Embedding will run in %d batches", len(dataloader))

    all_embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, (x_batch, mask_batch) in enumerate(dataloader, start=1):
            output = model(
                x_enc=x_batch.to(runtime_device),
                input_mask=mask_batch.to(runtime_device),
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
    cleaned_root: Path,
    out_dir: Path,
    config: SegmentConfig,
    model_name: str = "AutonLab/MOMENT-1-large",
    batch_size: int = 64,
    device: str = "auto",
    reduction: str = "mean",
    subject_filter: set[str] | None = None,
    skip_embedding: bool = False,
) -> Path:
    """Run full Trust-me cleaned-windowed preprocessing + embedding pipeline."""
    _ensure_parquet_support()
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Running Trust-me cleaned-windowed Tobii pipeline")
    LOGGER.info("pipeline_out_dir=%s", out_dir)

    _, stats = build_trustme_window_inputs(
        cleaned_root=cleaned_root,
        out_dir=out_dir,
        config=config,
        subject_filter=subject_filter,
    )

    inputs_path = out_dir / "moment_inputs.npz"
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

    metadata_path = out_dir / "segments_metadata.parquet"
    metadata = pd.read_parquet(metadata_path)

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "input_dataset": "trustME_trustme/data/cleaned_windowed",
        "cleaned_root": str(cleaned_root),
        "out_dir": str(out_dir),
        "config": asdict(config),
        "config_hash": _config_hash(config),
        "model": {
            "name": model_name,
            "task_name": "embedding",
            "reduction": reduction,
            "batch_size": batch_size,
            "device": device,
            "computed_embeddings": not skip_embedding,
        },
        "counts": {
            "segments_total_candidates": int(stats.total_candidates),
            "segments_kept": int(stats.total_kept),
            "segments_dropped": int(stats.total_dropped),
            "drop_reason_counts": stats.drop_reason_counts,
            "subjects_processed": int(stats.subjects_processed),
            "parquet_files_processed": int(stats.parquet_files_processed),
            "metadata_rows": int(len(metadata)),
        },
        "artifacts": {
            "segments_metadata": {
                "path": str(metadata_path),
                "sha256": _sha256_file(metadata_path),
            },
            "moment_inputs": {
                "path": str(inputs_path),
                "sha256": _sha256_file(inputs_path),
            },
        },
    }

    if embeddings_path.exists():
        manifest["artifacts"]["moment_embeddings"] = {
            "path": str(embeddings_path),
            "sha256": _sha256_file(embeddings_path),
        }

    manifest_path = _write_manifest(manifest=manifest, out_dir=out_dir)
    LOGGER.info("Saved manifest: %s", manifest_path)
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for Trust-me cleaned-windowed pipeline."""
    parser = argparse.ArgumentParser(
        description="Trust-me cleaned-windowed Tobii segmentation and MOMENT embedding pipeline."
    )
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=Path("trustME_trustme/data/cleaned_windowed"),
        help="Root directory containing per-subject cleaned_windowed Tobii parquet files.",
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

    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument(
        "--channels",
        type=str,
        default="GazePointX,GazePointY,PupilSizeLeft,PupilSizeRight",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Trust-me cleaned-windowed pipeline."""
    parser = build_arg_parser()
    parsed = parser.parse_args(argv)
    _setup_logging(verbose=parsed.verbose)

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
    )
    subject_filter = _parse_subject_filter(parsed.subjects)

    LOGGER.info("CLI parsed. verbose=%s", parsed.verbose)
    run_pipeline(
        cleaned_root=parsed.cleaned_root,
        out_dir=parsed.out_dir,
        config=config,
        model_name=parsed.model_name,
        batch_size=parsed.batch_size,
        device=parsed.device,
        reduction=parsed.reduction,
        subject_filter=subject_filter,
        skip_embedding=parsed.skip_embedding,
    )
    LOGGER.info("Pipeline completed: %s", parsed.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
