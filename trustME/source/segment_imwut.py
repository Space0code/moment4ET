"""IMWUT Tobii segmentation and MOMENT embedding pipeline.

This module builds fixed-length MOMENT inputs from IMWUT Tobii segments and
computes one embedding per segment.
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


@dataclass(frozen=True)
class SegmentConfig:
    """Configuration for segment preprocessing and MOMENT input creation."""

    seq_len: int = 512
    channels: tuple[str, ...] = (
        "GazePointX",
        "GazePointY",
        "PupilSizeLeft",
        "PupilSizeRight",
    )
    max_invalid_frames: int = 60
    min_valid_fraction: float = 0.3
    min_valid_frames: int = 64
    # NOTE: Keep external normalization off by default because MOMENT RevIN
    # already applies per-sample (segment-wise), mask-aware standardization.
    normalize: str = "none"
    modality: str = "tobii"
    excluded_labels: tuple[str, ...] = ("questionnaire", "central_position")


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


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _sha256_file(path: Path) -> str:
    """Return SHA256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_hash(config: SegmentConfig) -> str:
    """Return a stable hash for the segmentation config."""
    config_json = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
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

    validity_cols = [
        "ValidityLeft",
        "ValidityRight",
        "PupilValidityLeft",
        "PupilValidityRight",
    ]
    available_validity_cols = [col for col in validity_cols if col in segment_df.columns]

    if available_validity_cols:
        validity_frame = segment_df.loc[:, available_validity_cols]
        validity_invalid = (~validity_frame.eq(1)).any(axis=1).to_numpy()
    else:
        validity_invalid = np.zeros(len(segment_df), dtype=bool)

    return sentinel_invalid | nan_invalid | validity_invalid


def _process_segment(
    segment_df: pd.DataFrame,
    segment_id: str,
    config: SegmentConfig,
) -> SegmentProcessResult:
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
    interp_limit = max(config.max_invalid_frames - 1, 0)
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
        # NOTE: This external scaling is opt-in only; RevIN already standardizes
        # each sample inside MOMENT.
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


def _prepare_segment_index(
    seg_csv: Path,
    modality: str,
    excluded_labels: tuple[str, ...] = (),
    subject_filter: set[str] | None = None,
) -> pd.DataFrame:
    """Load and filter segment index from CSV with deterministic ordering."""
    seg = pd.read_csv(seg_csv).reset_index(names="source_row_idx")
    seg = seg.loc[seg["Modality"] == modality].copy()
    if excluded_labels:
        seg = seg.loc[~seg["Label"].isin(excluded_labels)].copy()
    if subject_filter is not None:
        seg = seg.loc[seg["Subject"].isin(subject_filter)].copy()
    seg = seg.sort_values(["source_row_idx", "Subject", "Start_i", "End_i"]).reset_index(drop=True)
    seg["segment_id"] = seg.apply(
        lambda row: f"{row['Subject']}_{int(row['source_row_idx'])}",
        axis=1,
    )
    return seg


def _write_manifest(manifest: dict[str, Any], out_dir: Path) -> Path:
    """Write manifest JSON to output directory."""
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
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


def build_tobii_segment_inputs(
    raw_root: Path,
    seg_csv: Path,
    out_dir: Path,
    config: SegmentConfig,
    subject_filter: set[str] | None = None,
) -> Path:
    """Build cleaned fixed-length MOMENT inputs from IMWUT Tobii segment rows.

    Parameters
    ----------
    raw_root:
        Root folder containing IMWUT raw data subjects.
    seg_csv:
        Segmentation CSV path (all segments).
    out_dir:
        Output directory for metadata and `moment_inputs.npz`.
    config:
        Segment preprocessing configuration.
    subject_filter:
        Optional subset of subjects to process.

    Returns
    -------
    Path
        Path to output directory containing cached inputs.
    """
    _ensure_parquet_support()
    LOGGER.info("Building segmented MOMENT inputs")
    LOGGER.info("raw_root=%s", raw_root)
    LOGGER.info("seg_csv=%s", seg_csv)
    LOGGER.info("out_dir=%s", out_dir)
    LOGGER.info("segment_config=%s", asdict(config))
    if subject_filter:
        LOGGER.info("subject_filter=%s", sorted(subject_filter))
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_all = pd.read_csv(seg_csv)
    seg_all_modality = seg_all.loc[seg_all["Modality"] == config.modality].copy()
    label_excluded_count = 0
    if config.excluded_labels:
        label_excluded_count = int(seg_all_modality["Label"].isin(config.excluded_labels).sum())

    seg = _prepare_segment_index(
        seg_csv=seg_csv,
        modality=config.modality,
        excluded_labels=config.excluded_labels,
        subject_filter=subject_filter,
    )
    LOGGER.info(
        "Filtered %d segments for modality=%s across %d subjects",
        len(seg),
        config.modality,
        seg["Subject"].nunique(),
    )
    if config.excluded_labels:
        LOGGER.info(
            "Excluded labels %s removed %d segments before QC",
            list(config.excluded_labels),
            label_excluded_count,
        )

    metadata_rows: list[dict[str, Any]] = []
    x_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    kept_segment_ids: list[str] = []

    for subject, subject_seg in seg.groupby("Subject", sort=True):
        subject_seg = subject_seg.sort_values("source_row_idx").reset_index(drop=True)
        subject_path = raw_root / subject / "tobii" / f"{subject}.csv"
        if not subject_path.exists():
            raise FileNotFoundError(f"Raw Tobii file not found: {subject_path}")

        df_subject = pd.read_csv(subject_path)
        n_rows = len(df_subject)
        LOGGER.info("Loaded subject=%s rows=%d segments=%d", subject, n_rows, len(subject_seg))
        subject_kept = 0
        subject_drop_reasons: dict[str, int] = {}

        for _, row in subject_seg.iterrows():
            start_i = int(row["Start_i"])
            end_i = int(row["End_i"])
            if start_i < 0 or end_i >= n_rows or start_i > end_i:
                raise ValueError(
                    f"Invalid segment bounds for {row['segment_id']}: start={start_i}, end={end_i}, rows={n_rows}"
                )
            segment_df = df_subject.iloc[start_i : end_i + 1]
            result = _process_segment(segment_df=segment_df, segment_id=row["segment_id"], config=config)

            metadata_rows.append(
                {
                    "segment_id": row["segment_id"],
                    "source_row_idx": int(row["source_row_idx"]),
                    "Subject": row["Subject"],
                    "Path": row["Path"],
                    "Filename": row["Filename"],
                    "Start_i": start_i,
                    "End_i": end_i,
                    "Start_t": float(row["Start_t"]),
                    "End_t": float(row["End_t"]),
                    "Label": row["Label"],
                    "orig_len": result.orig_len,
                    "valid_fraction": result.valid_fraction,
                    "max_invalid_run": result.max_invalid_run,
                    "kept": result.kept,
                    "drop_reason": result.drop_reason,
                    "pad_len": result.pad_len,
                }
            )

            if result.kept:
                if result.x is None or result.input_mask is None:
                    raise RuntimeError(f"Missing processed payload for kept segment: {result.segment_id}")
                x_list.append(result.x)
                mask_list.append(result.input_mask)
                kept_segment_ids.append(result.segment_id)
                subject_kept += 1
            else:
                subject_drop_reasons[result.drop_reason] = subject_drop_reasons.get(result.drop_reason, 0) + 1

        subject_dropped = len(subject_seg) - subject_kept
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
    np.savez_compressed(
        inputs_path,
        x=x,
        input_mask=input_mask,
        segment_id=segment_ids_arr,
    )

    LOGGER.info(
        "Saved metadata rows=%d kept=%d dropped=%d",
        len(metadata),
        int(metadata["kept"].sum()),
        int((~metadata["kept"]).sum()),
    )
    drop_counts = metadata.loc[~metadata["kept"], "drop_reason"].value_counts().to_dict()
    if drop_counts:
        LOGGER.info("Global drop reasons: %s", drop_counts)
    LOGGER.info("Saved %s", metadata_path)
    LOGGER.info(
        "Saved %s with x.shape=%s input_mask.shape=%s",
        inputs_path,
        tuple(x.shape),
        tuple(input_mask.shape),
    )
    return out_dir


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
        np.savez_compressed(
            out_npz,
            embeddings=np.zeros((0, 0), dtype=np.float32),
            segment_id=segment_ids,
        )
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
    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={"task_name": "embedding"},
    )
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
                    "MOMENT returned `embeddings=None`. "
                    "Expected task_name='embedding'. "
                    f"Output fields: embeddings={type(output.embeddings)}, "
                    f"reconstruction={type(output.reconstruction)}, logits={type(output.logits)}."
                )
            emb = output.embeddings.detach().cpu().numpy()
            all_embeddings.append(emb)
            if batch_idx == 1 or batch_idx == len(dataloader) or batch_idx % 25 == 0:
                LOGGER.info("Processed embedding batch %d/%d", batch_idx, len(dataloader))

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    np.savez_compressed(
        out_npz,
        embeddings=embeddings,
        segment_id=segment_ids,
    )
    LOGGER.info("Saved %s with embeddings.shape=%s", out_npz, tuple(embeddings.shape))
    return out_npz


def run_pipeline(
    raw_root: Path,
    seg_csv: Path,
    out_dir: Path,
    config: SegmentConfig,
    model_name: str = "AutonLab/MOMENT-1-large",
    batch_size: int = 64,
    device: str = "auto",
    reduction: str = "mean",
    subject_filter: set[str] | None = None,
    skip_embedding: bool = False,
) -> Path:
    """Run full IMWUT Tobii preprocessing + embedding pipeline."""
    _ensure_parquet_support()
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Running IMWUT Tobii pipeline")
    LOGGER.info("pipeline_out_dir=%s", out_dir)
    build_tobii_segment_inputs(
        raw_root=raw_root,
        seg_csv=seg_csv,
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
    n_kept = int(metadata["kept"].sum())
    n_dropped = int((~metadata["kept"]).sum())

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "seg_csv": str(seg_csv),
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
            "segments_total_filtered": int(len(metadata)),
            "segments_kept": n_kept,
            "segments_dropped": n_dropped,
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

    manifest_path = _write_manifest(manifest, out_dir=out_dir)
    LOGGER.info("Saved manifest: %s", manifest_path)
    return out_dir


def _parse_subject_filter(raw_value: str | None) -> set[str] | None:
    """Parse comma-separated subject filter argument."""
    if not raw_value:
        return None
    subjects = {part.strip() for part in raw_value.split(",") if part.strip()}
    return subjects or None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for the IMWUT pipeline."""
    parser = argparse.ArgumentParser(description="IMWUT Tobii segmentation and MOMENT embedding pipeline.")
    parser.add_argument("--raw-root", type=Path, default=Path("trustME/data/raw/imwut"))
    parser.add_argument("--seg-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("trustME/data/processed/imwut_tobii"))
    parser.add_argument("--model-name", type=str, default="AutonLab/MOMENT-1-large")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--reduction", type=str, default="mean")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subset, e.g. s_001,s_002")
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
    parser.add_argument("--modality", type=str, default="tobii")
    parser.add_argument(
        "--excluded-labels",
        type=str,
        default="questionnaire,central_position",
        help="Comma-separated labels to exclude from the pipeline entirely.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for IMWUT Tobii pipeline."""
    parser = build_arg_parser()
    parsed = parser.parse_args(argv)
    _setup_logging(verbose=parsed.verbose)

    seg_csv = parsed.seg_csv or (parsed.raw_root / "all_segments.csv")
    channels = tuple([channel.strip() for channel in parsed.channels.split(",") if channel.strip()])
    excluded_labels = tuple([label.strip() for label in parsed.excluded_labels.split(",") if label.strip()])
    if not channels:
        raise ValueError("At least one channel must be provided")

    config = SegmentConfig(
        seq_len=parsed.seq_len,
        channels=channels,
        max_invalid_frames=parsed.max_invalid_frames,
        min_valid_fraction=parsed.min_valid_fraction,
        min_valid_frames=parsed.min_valid_frames,
        normalize=parsed.normalize,
        modality=parsed.modality,
        excluded_labels=excluded_labels,
    )
    subject_filter = _parse_subject_filter(parsed.subjects)
    LOGGER.info("CLI parsed. verbose=%s", parsed.verbose)

    run_pipeline(
        raw_root=parsed.raw_root,
        seg_csv=seg_csv,
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
