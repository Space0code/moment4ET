"""Build a single-file cache of preprocessed Trust-me cleaned-windowed segments.

This script runs only the data cleaning/segmentation stage used by
`segment_trustme.py` and stores kept windows in one compressed `.npz` file.
No embeddings are computed.

Example
-------
python trustME_trustme/source/preprocess_trustme_cache.py \
  --cleaned-root trustME_trustme/data/cleaned_windowed \
  --out-root trustME_trustme/data/cleaned_windowed_preprocessed \
  --cache-name trustme_preprocessed_moment_inputs.npz \
  --channels GazePointX,GazePointY,PupilSizeLeft,PupilSizeRight,AverageDistance
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from trustME_trustme.source.segment_trustme import (
        EXTRA_LABEL_COLUMNS,
        PIPELINE_VERSION,
        BuildStats,
        SegmentConfig,
        _config_hash,
        _format_duration_hhmmss,
        _parse_subject_filter,
        _setup_logging,
        build_trustme_window_inputs,
    )
except ImportError:
    from segment_trustme import (  # type: ignore
        EXTRA_LABEL_COLUMNS,
        PIPELINE_VERSION,
        BuildStats,
        SegmentConfig,
        _config_hash,
        _format_duration_hhmmss,
        _parse_subject_filter,
        _setup_logging,
        build_trustme_window_inputs,
    )


LOGGER = logging.getLogger(__name__)
CACHE_FORMAT_VERSION = "1.0.0"


def _build_cache_manifest(
    cleaned_root: Path,
    cache_path: Path,
    config: SegmentConfig,
    stats: BuildStats,
    x_shape: tuple[int, int, int],
    input_mask_shape: tuple[int, int],
    metadata_rows: int,
) -> dict[str, Any]:
    """Create reproducibility metadata for the cache artifact."""
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "pipeline_source_version": PIPELINE_VERSION,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "cleaned_root": str(cleaned_root),
        "cache_path": str(cache_path),
        "config": asdict(config),
        "config_hash": _config_hash(config),
        "counts": {
            "segments_total_candidates": int(stats.total_candidates),
            "segments_kept": int(stats.total_kept),
            "segments_dropped": int(stats.total_dropped),
            "drop_reason_counts": stats.drop_reason_counts,
            "subjects_processed": int(stats.subjects_processed),
            "parquet_files_processed": int(stats.parquet_files_processed),
            "metadata_rows": int(metadata_rows),
        },
        "shapes": {
            "x": list(x_shape),
            "input_mask": list(input_mask_shape),
        },
    }


def build_preprocessed_cache(
    cleaned_root: Path,
    out_root: Path,
    cache_name: str,
    config: SegmentConfig,
    subject_filter: set[str] | None,
    overwrite: bool,
) -> Path:
    """Run Trust-me preprocessing only and save a single compressed cache file."""
    out_root.mkdir(parents=True, exist_ok=True)
    cache_path = out_root / cache_name

    if cache_path.exists() and not overwrite:
        raise FileExistsError(
            f"Cache file already exists: {cache_path}. Pass --overwrite to replace it."
        )

    with tempfile.TemporaryDirectory(prefix="tmp_preprocess_", dir=out_root) as tmpdir:
        tmp_out_dir = Path(tmpdir)
        _, stats = build_trustme_window_inputs(
            cleaned_root=cleaned_root,
            out_dir=tmp_out_dir,
            config=config,
            subject_filter=subject_filter,
        )

        with np.load(tmp_out_dir / "moment_inputs.npz", allow_pickle=False) as inputs_payload:
            x = inputs_payload["x"].astype(np.float32)
            input_mask = inputs_payload["input_mask"].astype(np.uint8)
            segment_id = inputs_payload["segment_id"].astype(str)

        metadata_path = tmp_out_dir / "segments_metadata.parquet"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Expected metadata artifact not found: {metadata_path}")
        metadata_parquet_bytes = metadata_path.read_bytes()
        metadata_rows = int(stats.total_kept)

        manifest = _build_cache_manifest(
            cleaned_root=cleaned_root,
            cache_path=cache_path,
            config=config,
            stats=stats,
            x_shape=tuple(x.shape),
            input_mask_shape=tuple(input_mask.shape),
            metadata_rows=metadata_rows,
        )
        manifest_json = json.dumps(manifest, sort_keys=True, default=str)

    np.savez_compressed(
        cache_path,
        x=x,
        input_mask=input_mask,
        segment_id=segment_id,
        metadata_parquet=np.frombuffer(metadata_parquet_bytes, dtype=np.uint8),
        manifest_json=np.asarray(manifest_json),
    )

    LOGGER.info(
        "Saved preprocessed cache: %s (x.shape=%s, input_mask.shape=%s, metadata_rows=%d)",
        cache_path,
        tuple(x.shape),
        tuple(input_mask.shape),
        metadata_rows,
    )
    return cache_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for preprocessing-only Trust-me cache generation."""
    parser = argparse.ArgumentParser(
        description="Build a single-file preprocessed Trust-me cache without computing embeddings."
    )
    parser.add_argument(
        "--cleaned-root",
        type=Path,
        default=Path("trustME_trustme/data/cleaned_windowed"),
        help="Root directory containing per-subject cleaned_windowed Tobii parquet files.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("trustME_trustme/data/cleaned_windowed_preprocessed"),
        help="Output directory where the single cache file is saved.",
    )
    parser.add_argument(
        "--cache-name",
        type=str,
        default="trustme_preprocessed_moment_inputs.npz",
        help="Output cache filename (.npz).",
    )
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subset, e.g. s_004_pk,s_005_ak")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite cache file if it exists.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file. Default: <out-root>/preprocess_trustme_cache.log",
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
        help="External preprocessing normalization option from segment_trustme.py.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="sleep_feedback",
        choices=list(EXTRA_LABEL_COLUMNS),
        help="Column copied into canonical metadata `Label` in reused preprocessing.",
    )
    parser.add_argument(
        "--require-label",
        action="store_true",
        help="Keep only windows where --label-column is non-null.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for preprocessing-only cache generation."""
    parser = build_arg_parser()
    parsed = parser.parse_args(argv)
    parsed.out_root.mkdir(parents=True, exist_ok=True)

    log_file = (
        parsed.log_file
        if parsed.log_file is not None
        else parsed.out_root / "preprocess_trustme_cache.log"
    )
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
    LOGGER.info("out_root=%s", parsed.out_root)
    LOGGER.info("cache_name=%s", parsed.cache_name)

    started_at = time.perf_counter()
    exit_code = 0
    try:
        cache_path = build_preprocessed_cache(
            cleaned_root=parsed.cleaned_root,
            out_root=parsed.out_root,
            cache_name=parsed.cache_name,
            config=config,
            subject_filter=subject_filter,
            overwrite=parsed.overwrite,
        )
        LOGGER.info("Preprocessing cache completed: %s", cache_path)
    except Exception:
        exit_code = 1
        LOGGER.exception("Preprocessing cache failed")
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
