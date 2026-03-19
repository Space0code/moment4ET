"""Build a labeled-only subset of Trust-me cleaned-windowed Tobii data.

Creates a new cleaned dataset root by filtering existing Parquet files under
`--cleaned-root` and keeping only labeled data according to `--label-column`.

Filtering modes
---------------
- `window` (default): keep whole windows where at least one row has a non-null label.
- `row`: keep only rows where the label is non-null.

Example
-------
python trustME_trustme/source/build_labeled_cleaned_subset.py \
  --cleaned-root trustME_trustme/data/cleaned_windowed \
  --out-root trustME_trustme/data/cleaned_windowed_labeled \
  --label-column sleep_feedback \
  --filter-mode window \
  --subjects s_004_pk,s_005_ak
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
SCRIPT_VERSION = "1.0.0"


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for building a labeled cleaned-windowed subset."""

    label_column: str = "sleep_feedback"
    filter_mode: str = "window"  # one of {"window", "row"}
    overwrite: bool = False
    dry_run: bool = False


@dataclass
class FileResult:
    """Per-file filtering summary."""

    filename: str
    status: str
    raw_rows: int
    raw_windows: int
    rows_kept: int
    windows_kept: int
    note: str


@dataclass
class BuildStats:
    """Global summary for subset building."""

    subjects_processed: int
    parquet_files_processed: int
    files_written: int
    files_skipped: int
    rows_input_total: int
    rows_output_total: int
    windows_input_total: int
    windows_output_total: int


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


def _format_duration_hhmmss(total_seconds: float) -> str:
    """Format elapsed time in HH:MM:SS."""
    whole_seconds = int(round(total_seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _validate_inputs(cleaned_root: Path, out_root: Path, config: FilterConfig) -> None:
    """Validate paths and options before processing."""
    if not cleaned_root.exists():
        raise FileNotFoundError(f"Missing cleaned root: {cleaned_root}")
    if cleaned_root.resolve() == out_root.resolve():
        raise ValueError("--out-root must be different from --cleaned-root.")
    if config.filter_mode not in {"window", "row"}:
        raise ValueError(f"Unsupported filter mode: {config.filter_mode}")
    if out_root.exists() and any(out_root.iterdir()) and not config.overwrite and not config.dry_run:
        raise FileExistsError(
            f"Output root already exists and is not empty: {out_root}. "
            "Pass --overwrite to allow writing."
        )


def _collect_window_mode_keep_ids(
    label_view: pd.DataFrame,
    label_column: str,
    source_name: str,
) -> tuple[np.ndarray, int]:
    """Return window IDs to keep and number of labeled windows in window mode."""
    if label_view["window_id"].isna().any():
        raise ValueError(f"Found NaN window_id in {source_name}")

    labeled_rows = label_view.dropna(subset=[label_column])
    if labeled_rows.empty:
        return np.array([], dtype=label_view["window_id"].dtype), 0

    non_null_nunique = labeled_rows.groupby("window_id")[label_column].nunique(dropna=True)
    conflicting = non_null_nunique[non_null_nunique > 1]
    if not conflicting.empty:
        sample_ids = conflicting.index[:5].tolist()
        raise ValueError(
            f"Conflicting non-null labels within window(s) in {source_name}. "
            f"Example window_id values: {sample_ids}"
        )

    keep_window_ids = labeled_rows["window_id"].drop_duplicates().to_numpy()
    return keep_window_ids, int(len(keep_window_ids))


def _write_subject_report(
    subject: str,
    out_subject_dir: Path,
    file_results: list[FileResult],
    stats: dict[str, int],
    config: FilterConfig,
) -> None:
    """Write markdown report for one subject."""
    report_path = out_subject_dir / "processing_report.md"
    lines: list[str] = []
    lines.append(f"# Labeled Cleaned Subset Report: `{subject}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- filter_mode: **{config.filter_mode}**")
    lines.append(f"- label_column: **{config.label_column}**")
    lines.append(f"- parquet_files_processed: **{stats['parquet_files_processed']}**")
    lines.append(f"- files_written: **{stats['files_written']}**")
    lines.append(f"- files_skipped: **{stats['files_skipped']}**")
    lines.append(f"- rows_input_total: **{stats['rows_input_total']}**")
    lines.append(f"- rows_output_total: **{stats['rows_output_total']}**")
    lines.append(f"- windows_input_total: **{stats['windows_input_total']}**")
    lines.append(f"- windows_output_total: **{stats['windows_output_total']}**")
    lines.append("")
    lines.append("## Per-File Details")
    lines.append("")
    lines.append("| filename | status | raw_rows | raw_windows | rows_kept | windows_kept | note |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")

    for result in file_results:
        lines.append(
            f"| {result.filename} | {result.status} | {result.raw_rows} | {result.raw_windows} | "
            f"{result.rows_kept} | {result.windows_kept} | {result.note} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_labeled_subset(
    cleaned_root: Path,
    out_root: Path,
    config: FilterConfig,
    subject_filter: set[str] | None = None,
) -> BuildStats:
    """Build labeled-only cleaned-windowed subset from existing parquet files."""
    _ensure_parquet_support()
    _validate_inputs(cleaned_root=cleaned_root, out_root=out_root, config=config)

    subjects = _discover_subjects(cleaned_root=cleaned_root, subject_filter=subject_filter)
    if not subjects:
        raise ValueError("No subjects found in cleaned root after applying subject filter.")

    if not config.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    parquet_files_processed = 0
    files_written = 0
    files_skipped = 0
    rows_input_total = 0
    rows_output_total = 0
    windows_input_total = 0
    windows_output_total = 0

    for subject in subjects:
        in_tobii_dir = cleaned_root / subject / "tobii"
        if not in_tobii_dir.exists():
            raise FileNotFoundError(f"Missing Tobii folder for subject={subject}: {in_tobii_dir}")

        parquet_files = sorted(in_tobii_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {in_tobii_dir}")

        LOGGER.info("Subject %s: processing %d parquet files", subject, len(parquet_files))
        subject_file_results: list[FileResult] = []

        subject_rows_input = 0
        subject_rows_output = 0
        subject_windows_input = 0
        subject_windows_output = 0
        subject_files_written = 0
        subject_files_skipped = 0

        out_subject_dir = out_root / subject
        out_tobii_dir = out_subject_dir / "tobii"
        if not config.dry_run:
            out_tobii_dir.mkdir(parents=True, exist_ok=True)

        for parquet_path in parquet_files:
            parquet_files_processed += 1
            source_name = parquet_path.name

            label_view = pd.read_parquet(parquet_path, columns=["window_id", config.label_column])
            raw_rows = int(len(label_view))
            raw_windows = int(label_view["window_id"].nunique(dropna=True)) if raw_rows > 0 else 0

            subject_rows_input += raw_rows
            subject_windows_input += raw_windows
            rows_input_total += raw_rows
            windows_input_total += raw_windows

            if config.filter_mode == "window":
                keep_window_ids, windows_kept = _collect_window_mode_keep_ids(
                    label_view=label_view,
                    label_column=config.label_column,
                    source_name=source_name,
                )
                keep_mask = label_view["window_id"].isin(keep_window_ids)
            else:
                keep_mask = label_view[config.label_column].notna()
                windows_kept = int(label_view.loc[keep_mask, "window_id"].nunique(dropna=True))

            rows_kept = int(keep_mask.sum())
            subject_rows_output += rows_kept
            subject_windows_output += windows_kept
            rows_output_total += rows_kept
            windows_output_total += windows_kept

            if rows_kept == 0:
                subject_files_skipped += 1
                files_skipped += 1
                subject_file_results.append(
                    FileResult(
                        filename=source_name,
                        status="skipped",
                        raw_rows=raw_rows,
                        raw_windows=raw_windows,
                        rows_kept=0,
                        windows_kept=0,
                        note=f"No rows kept for label '{config.label_column}'.",
                    )
                )
                continue

            if not config.dry_run:
                full_df = pd.read_parquet(parquet_path)
                if config.filter_mode == "window":
                    full_mask = full_df["window_id"].isin(keep_window_ids)
                else:
                    full_mask = full_df[config.label_column].notna()

                kept_df = full_df.loc[full_mask].copy()
                out_path = out_tobii_dir / source_name
                kept_df.to_parquet(out_path, index=False)

                note = f"Wrote {out_path.name}"
            else:
                note = "Dry run: write skipped."

            subject_files_written += 1
            files_written += 1
            subject_file_results.append(
                FileResult(
                    filename=source_name,
                    status="written" if not config.dry_run else "would_write",
                    raw_rows=raw_rows,
                    raw_windows=raw_windows,
                    rows_kept=rows_kept,
                    windows_kept=windows_kept,
                    note=note,
                )
            )

        subject_stats = {
            "parquet_files_processed": len(parquet_files),
            "files_written": subject_files_written,
            "files_skipped": subject_files_skipped,
            "rows_input_total": subject_rows_input,
            "rows_output_total": subject_rows_output,
            "windows_input_total": subject_windows_input,
            "windows_output_total": subject_windows_output,
        }
        LOGGER.info(
            "Subject %s done: files_written=%d files_skipped=%d rows_in=%d rows_out=%d windows_in=%d windows_out=%d",
            subject,
            subject_files_written,
            subject_files_skipped,
            subject_rows_input,
            subject_rows_output,
            subject_windows_input,
            subject_windows_output,
        )

        if not config.dry_run:
            _write_subject_report(
                subject=subject,
                out_subject_dir=out_subject_dir,
                file_results=subject_file_results,
                stats=subject_stats,
                config=config,
            )

    return BuildStats(
        subjects_processed=len(subjects),
        parquet_files_processed=parquet_files_processed,
        files_written=files_written,
        files_skipped=files_skipped,
        rows_input_total=rows_input_total,
        rows_output_total=rows_output_total,
        windows_input_total=windows_input_total,
        windows_output_total=windows_output_total,
    )


def _write_manifest(
    out_root: Path,
    cleaned_root: Path,
    config: FilterConfig,
    subject_filter: set[str] | None,
    stats: BuildStats,
) -> Path:
    """Write run manifest JSON."""
    manifest = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "cleaned_root": str(cleaned_root),
        "out_root": str(out_root),
        "config": asdict(config),
        "subjects_filter": sorted(subject_filter) if subject_filter else None,
        "counts": {
            "subjects_processed": int(stats.subjects_processed),
            "parquet_files_processed": int(stats.parquet_files_processed),
            "files_written": int(stats.files_written),
            "files_skipped": int(stats.files_skipped),
            "rows_input_total": int(stats.rows_input_total),
            "rows_output_total": int(stats.rows_output_total),
            "windows_input_total": int(stats.windows_input_total),
            "windows_output_total": int(stats.windows_output_total),
        },
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description="Create a labeled-only subset of Trust-me cleaned-windowed parquet data."
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
        default=Path("trustME_trustme/data/cleaned_windowed_labeled"),
        help="Output root where filtered per-subject parquet files are written.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="sleep_feedback",
        help="Label column used to decide what to keep.",
    )
    parser.add_argument(
        "--filter-mode",
        type=str,
        default="window",
        choices=["window", "row"],
        help="`window`: keep whole labeled windows. `row`: keep only rows with non-null label.",
    )
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subset, e.g. s_004_pk,s_005_ak")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty --out-root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and log stats without writing output parquet files.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file. Default: <out-root>/build_labeled_cleaned_subset.log",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_arg_parser()
    parsed = parser.parse_args(argv)
    config = FilterConfig(
        label_column=parsed.label_column,
        filter_mode=parsed.filter_mode,
        overwrite=parsed.overwrite,
        dry_run=parsed.dry_run,
    )
    subject_filter = _parse_subject_filter(parsed.subjects)
    log_file = parsed.log_file if parsed.log_file is not None else parsed.out_root / "build_labeled_cleaned_subset.log"
    _setup_logging(verbose=parsed.verbose, log_file=log_file)

    LOGGER.info("CLI parsed. config=%s", asdict(config))
    if subject_filter:
        LOGGER.info("subject_filter=%s", sorted(subject_filter))

    started_at = time.perf_counter()
    stats = build_labeled_subset(
        cleaned_root=parsed.cleaned_root,
        out_root=parsed.out_root,
        config=config,
        subject_filter=subject_filter,
    )

    if not config.dry_run:
        manifest_path = _write_manifest(
            out_root=parsed.out_root,
            cleaned_root=parsed.cleaned_root,
            config=config,
            subject_filter=subject_filter,
            stats=stats,
        )
        LOGGER.info("Saved manifest: %s", manifest_path)

    elapsed_seconds = time.perf_counter() - started_at
    LOGGER.info(
        "Completed. subjects=%d files_processed=%d files_written=%d files_skipped=%d",
        stats.subjects_processed,
        stats.parquet_files_processed,
        stats.files_written,
        stats.files_skipped,
    )
    LOGGER.info(
        "Rows: input=%d output=%d | Windows: input=%d output=%d",
        stats.rows_input_total,
        stats.rows_output_total,
        stats.windows_input_total,
        stats.windows_output_total,
    )
    LOGGER.info(
        "Total runtime: %s (%d seconds)",
        _format_duration_hhmmss(elapsed_seconds),
        int(round(elapsed_seconds)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
