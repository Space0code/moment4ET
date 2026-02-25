"""Tests for IMWUT Tobii segmentation and embedding pipeline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "trustME" / "source" / "segment_imwut.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("segment_imwut", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load segment_imwut module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_subject_csv(path: Path, n_rows: int, invalid_idx: set[int]) -> None:
    timestamp = np.arange(n_rows) / 60.0
    base = np.linspace(0.0, 1.0, num=n_rows, dtype=np.float64)

    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "Event": [""] * n_rows,
            "GazePointXLeft": base + 0.1,
            "GazePointYLeft": base + 0.2,
            "ValidityLeft": np.ones(n_rows, dtype=int),
            "GazePointXRight": base + 0.3,
            "GazePointYRight": base + 0.4,
            "ValidityRight": np.ones(n_rows, dtype=int),
            "GazePointX": base + 0.5,
            "GazePointY": base + 0.6,
            "PupilSizeLeft": base + 1.0,
            "PupilValidityLeft": np.ones(n_rows, dtype=int),
            "PupilSizeRight": base + 1.2,
            "PupilValidityRight": np.ones(n_rows, dtype=int),
            "label": ["task"] * n_rows,
        }
    )
    for idx in invalid_idx:
        df.loc[idx, ["GazePointX", "GazePointY", "PupilSizeLeft", "PupilSizeRight"]] = -1.0
        df.loc[idx, ["ValidityLeft", "ValidityRight", "PupilValidityLeft", "PupilValidityRight"]] = 0
    df.to_csv(path, index=False)


def _prepare_synthetic_dataset(tmp_path: Path) -> tuple[Path, Path]:
    raw_root = tmp_path / "raw" / "imwut"
    (raw_root / "s_001" / "tobii").mkdir(parents=True)
    (raw_root / "s_002" / "tobii").mkdir(parents=True)

    _make_subject_csv(raw_root / "s_001" / "tobii" / "s_001.csv", n_rows=40, invalid_idx={20})
    _make_subject_csv(raw_root / "s_002" / "tobii" / "s_002.csv", n_rows=35, invalid_idx=set(range(24, 31)))

    seg = pd.DataFrame(
        [
            {
                "Path": "s_001/tobii",
                "Filename": "s_001.csv",
                "Subject": "s_001",
                "Modality": "tobii",
                "Win len": 3.0,
                "Overlap Percent": 25.0,
                "Assumed fs": np.nan,
                "Start_i": 0,
                "End_i": 9,
                "Start_t": 0.0,
                "End_t": 0.15,
                "Label": "task",
            },
            {
                "Path": "s_001/tobii",
                "Filename": "s_001.csv",
                "Subject": "s_001",
                "Modality": "tobii",
                "Win len": 3.0,
                "Overlap Percent": 25.0,
                "Assumed fs": np.nan,
                "Start_i": 10,
                "End_i": 19,
                "Start_t": 0.16,
                "End_t": 0.31,
                "Label": "task",
            },
            {
                "Path": "s_002/tobii",
                "Filename": "s_002.csv",
                "Subject": "s_002",
                "Modality": "tobii",
                "Win len": 3.0,
                "Overlap Percent": 25.0,
                "Assumed fs": np.nan,
                "Start_i": 22,
                "End_i": 31,
                "Start_t": 0.36,
                "End_t": 0.51,
                "Label": "task",
            },
            {
                "Path": "s_001/empatica",
                "Filename": "s_001.csv",
                "Subject": "s_001",
                "Modality": "empatica",
                "Win len": 3.0,
                "Overlap Percent": 25.0,
                "Assumed fs": np.nan,
                "Start_i": 0,
                "End_i": 9,
                "Start_t": 0.0,
                "End_t": 0.15,
                "Label": "irrelevant",
            },
        ]
    )
    seg_csv = raw_root / "all_segments.csv"
    seg.to_csv(seg_csv, index=False)
    return raw_root, seg_csv


def _load_outputs(out_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    metadata = pd.read_parquet(out_dir / "segments_metadata.parquet")
    payload = np.load(out_dir / "moment_inputs.npz", allow_pickle=False)
    return metadata, {"x": payload["x"], "input_mask": payload["input_mask"], "segment_id": payload["segment_id"]}


def test_build_inputs_filters_modality_and_shapes(tmp_path: Path) -> None:
    module = _load_module()
    raw_root, seg_csv = _prepare_synthetic_dataset(tmp_path)
    out_dir = tmp_path / "processed"

    config = module.SegmentConfig(
        seq_len=16,
        max_invalid_frames=3,
        min_valid_fraction=0.5,
        min_valid_frames=5,
    )
    module.build_tobii_segment_inputs(raw_root=raw_root, seg_csv=seg_csv, out_dir=out_dir, config=config)

    metadata, payload = _load_outputs(out_dir)

    assert set(metadata["Subject"].unique()) == {"s_001", "s_002"}
    assert (metadata["Path"] == "s_001/empatica").sum() == 0
    assert len(metadata) == 3
    assert payload["x"].shape == (2, 4, 16)
    assert payload["input_mask"].shape == (2, 16)
    assert payload["segment_id"].shape == (2,)
    assert payload["x"].dtype == np.float32
    assert payload["input_mask"].dtype == np.uint8


def test_mask_sum_matches_capped_segment_length(tmp_path: Path) -> None:
    module = _load_module()
    raw_root, seg_csv = _prepare_synthetic_dataset(tmp_path)
    out_dir = tmp_path / "processed"
    config = module.SegmentConfig(
        seq_len=8,
        max_invalid_frames=3,
        min_valid_fraction=0.5,
        min_valid_frames=5,
    )
    module.build_tobii_segment_inputs(raw_root=raw_root, seg_csv=seg_csv, out_dir=out_dir, config=config)

    metadata, payload = _load_outputs(out_dir)
    kept = metadata.loc[metadata["kept"]].reset_index(drop=True)
    mask_sum = payload["input_mask"].sum(axis=1)
    expected = np.minimum(kept["orig_len"].to_numpy(dtype=np.int64), config.seq_len)
    np.testing.assert_array_equal(mask_sum, expected)


def test_build_is_deterministic_and_id_aligned(tmp_path: Path) -> None:
    module = _load_module()
    raw_root, seg_csv = _prepare_synthetic_dataset(tmp_path)
    out_a = tmp_path / "processed_a"
    out_b = tmp_path / "processed_b"

    config = module.SegmentConfig(
        seq_len=16,
        max_invalid_frames=3,
        min_valid_fraction=0.5,
        min_valid_frames=5,
    )
    module.build_tobii_segment_inputs(raw_root=raw_root, seg_csv=seg_csv, out_dir=out_a, config=config)
    module.build_tobii_segment_inputs(raw_root=raw_root, seg_csv=seg_csv, out_dir=out_b, config=config)

    metadata_a, payload_a = _load_outputs(out_a)
    metadata_b, payload_b = _load_outputs(out_b)

    pd.testing.assert_frame_equal(metadata_a, metadata_b)
    np.testing.assert_array_equal(payload_a["x"], payload_b["x"])
    np.testing.assert_array_equal(payload_a["input_mask"], payload_b["input_mask"])
    np.testing.assert_array_equal(payload_a["segment_id"], payload_b["segment_id"])

    kept_ids = metadata_a.loc[metadata_a["kept"], "segment_id"].to_numpy()
    np.testing.assert_array_equal(kept_ids, payload_a["segment_id"])


def test_invalid_segment_bounds_raises(tmp_path: Path) -> None:
    module = _load_module()
    raw_root, seg_csv = _prepare_synthetic_dataset(tmp_path)
    seg = pd.read_csv(seg_csv)
    seg.loc[0, "End_i"] = 9999
    seg.to_csv(seg_csv, index=False)

    config = module.SegmentConfig(seq_len=16)
    with pytest.raises(ValueError, match="Invalid segment bounds"):
        module.build_tobii_segment_inputs(
            raw_root=raw_root,
            seg_csv=seg_csv,
            out_dir=tmp_path / "processed",
            config=config,
        )


def test_compute_embeddings_preserves_segment_alignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    raw_root, seg_csv = _prepare_synthetic_dataset(tmp_path)
    out_dir = tmp_path / "processed"
    config = module.SegmentConfig(
        seq_len=16,
        max_invalid_frames=3,
        min_valid_fraction=0.5,
        min_valid_frames=5,
    )
    module.build_tobii_segment_inputs(raw_root=raw_root, seg_csv=seg_csv, out_dir=out_dir, config=config)

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_name: str, model_kwargs: dict[str, str]) -> "FakePipeline":
            assert model_kwargs["task_name"] == "embedding"
            return cls()

        def to(self, device: str) -> "FakePipeline":
            _ = device
            return self

        def eval(self) -> "FakePipeline":
            return self

        def __call__(self, x_enc: torch.Tensor, input_mask: torch.Tensor, reduction: str = "mean") -> SimpleNamespace:
            _ = reduction
            emb = x_enc.mean(dim=(1, 2)).unsqueeze(1).repeat(1, 4)
            _ = input_mask
            return SimpleNamespace(embeddings=emb)

    fake_module = ModuleType("momentfm")
    fake_module.MOMENTPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "momentfm", fake_module)

    out_npz = out_dir / "moment_embeddings.npz"
    module.compute_moment_embeddings(
        input_npz=out_dir / "moment_inputs.npz",
        out_npz=out_npz,
        model_name="fake/model",
        batch_size=2,
        device="cpu",
    )

    emb_payload = np.load(out_npz, allow_pickle=False)
    in_payload = np.load(out_dir / "moment_inputs.npz", allow_pickle=False)
    np.testing.assert_array_equal(emb_payload["segment_id"], in_payload["segment_id"])
    assert emb_payload["embeddings"].shape[0] == in_payload["x"].shape[0]
    assert emb_payload["embeddings"].shape[1] == 4
