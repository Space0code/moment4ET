"""Tests for IMWUT head-only fine-tuning and dual-head export pipeline."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "trustME" / "source" / "finetune_imwut_heads.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("finetune_imwut_heads", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load finetune_imwut_heads module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_fake_moment(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLinearHead(torch.nn.Module):
        def __init__(self, d_model: int, n_classes: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(d_model, n_classes)

        def forward(self, x: torch.Tensor, input_mask: torch.Tensor | None = None) -> torch.Tensor:
            _ = input_mask
            pooled = torch.mean(x, dim=1)
            return self.linear(pooled)

    class FakePipeline(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, model_name: str, model_kwargs: dict[str, object]) -> "FakePipeline":
            _ = model_name
            return cls(model_kwargs=model_kwargs)

        def __init__(self, model_kwargs: dict[str, object]) -> None:
            super().__init__()
            self._model_kwargs = model_kwargs
            self.config = SimpleNamespace(d_model=8)
            self.patch_embedding = torch.nn.Linear(1, 1)
            self.encoder = torch.nn.Linear(1, 1)
            self.backbone = torch.nn.LazyLinear(8)
            self.head = FakeLinearHead(d_model=8, n_classes=int(model_kwargs["num_class"]))

            if bool(model_kwargs.get("freeze_embedder", True)):
                for param in self.patch_embedding.parameters():
                    param.requires_grad = False
            if bool(model_kwargs.get("freeze_encoder", True)):
                for param in self.encoder.parameters():
                    param.requires_grad = False
                for param in self.backbone.parameters():
                    param.requires_grad = False

        def init(self) -> "FakePipeline":
            self.head = FakeLinearHead(d_model=8, n_classes=int(self._model_kwargs["num_class"]))
            return self

        def forward(
            self, *, x_enc: torch.Tensor, input_mask: torch.Tensor | None = None, reduction: str = "mean"
        ) -> SimpleNamespace:
            _ = input_mask
            _ = reduction
            base = torch.tanh(self.backbone(x_enc.reshape(x_enc.shape[0], -1)))
            enc = base.unsqueeze(1).repeat(1, 2, 1)
            logits = self.head(enc)
            return SimpleNamespace(logits=logits, embeddings=enc)

        def embed(
            self, *, x_enc: torch.Tensor, input_mask: torch.Tensor | None = None, reduction: str = "mean"
        ) -> SimpleNamespace:
            _ = input_mask
            _ = reduction
            base = torch.tanh(self.backbone(x_enc.reshape(x_enc.shape[0], -1)))
            return SimpleNamespace(embeddings=base)

    fake_module = ModuleType("momentfm")
    fake_module.MOMENTPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "momentfm", fake_module)


def _write_synthetic_processed_dataset(processed_dir: Path) -> None:
    rng = np.random.default_rng(0)
    labels_per_subject = ("2_back", "memory_easy", "pursuit_difficult", "rest")
    rows: list[dict[str, object]] = []
    x_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    seg_ids: list[str] = []

    row_idx = 0
    for subject_num in range(1, 13):
        subject = f"s_{subject_num:03d}"
        for label in labels_per_subject:
            seg_id = f"{subject}_{row_idx}"
            rows.append(
                {
                    "segment_id": seg_id,
                    "source_row_idx": row_idx,
                    "Subject": subject,
                    "Path": f"{subject}/tobii",
                    "Filename": f"{subject}.csv",
                    "Start_i": 0,
                    "End_i": 15,
                    "Start_t": float(row_idx),
                    "End_t": float(row_idx + 1),
                    "Label": label,
                    "orig_len": 16,
                    "valid_fraction": 1.0,
                    "max_invalid_run": 0,
                    "kept": True,
                    "drop_reason": "",
                    "pad_len": 0,
                }
            )
            x_rows.append(rng.normal(loc=0.0, scale=1.0, size=(4, 16)).astype(np.float32))
            mask_rows.append(np.ones((16,), dtype=np.uint8))
            seg_ids.append(seg_id)
            row_idx += 1

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed_dir / "moment_inputs.npz",
        x=np.stack(x_rows, axis=0).astype(np.float32),
        input_mask=np.stack(mask_rows, axis=0).astype(np.uint8),
        segment_id=np.asarray(seg_ids, dtype=str),
    )
    pd.DataFrame(rows).to_parquet(processed_dir / "segments_metadata.parquet", index=False)


def test_apply_label_scheme_matches_canonical_semantics() -> None:
    module = _load_module()
    df = pd.DataFrame(
        {
            "segment_id": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "Subject": ["s_001"] * 8,
            "Label": [
                "2_back",
                "memory_easy",
                "pursuit_difficult",
                "rest",
                "listen_music",
                "passive_viewing",
                "questionnaire",
                "central_position",
            ],
        }
    )

    binary = module.apply_label_scheme(df, scheme="binary")
    assert set(binary["scheme_label"].unique()) == {"load", "rest"}
    assert "questionnaire" not in set(binary["Label"])
    assert "central_position" not in set(binary["Label"])

    edr = module.apply_label_scheme(df, scheme="edr")
    assert set(edr["scheme_label"].unique()) == {"high_load", "low_load", "rest"}
    assert "questionnaire" not in set(edr["Label"])
    assert "central_position" not in set(edr["Label"])

    avm = module.apply_label_scheme(df, scheme="avm")
    assert set(avm["scheme_label"].unique()) == {"attention_task", "memory_task", "visual_task"}
    assert "rest" not in set(avm["Label"])

    binary_keep = module.apply_label_scheme(
        df,
        scheme="binary",
        drop_central_and_questionnaire=False,
    )
    assert "questionnaire" in set(binary_keep["Label"])
    assert "central_position" in set(binary_keep["Label"])

    avm_keep = module.apply_label_scheme(
        df,
        scheme="avm",
        drop_central_and_questionnaire=False,
    )
    assert "questionnaire" in set(avm_keep["scheme_label"])
    assert "central_position" in set(avm_keep["scheme_label"])


def test_split_subject_holdout_is_deterministic_and_disjoint() -> None:
    module = _load_module()
    rows = []
    for subject_num in range(1, 11):
        subject = f"s_{subject_num:03d}"
        rows.append({"Subject": subject, "scheme_label": "class_a"})
        rows.append({"Subject": subject, "scheme_label": "class_b"})
    df = pd.DataFrame(rows)

    split_a = module.split_subject_holdout(df=df, seed=123)
    split_b = module.split_subject_holdout(df=df, seed=123)
    np.testing.assert_array_equal(split_a.train_idx, split_b.train_idx)
    np.testing.assert_array_equal(split_a.val_idx, split_b.val_idx)
    np.testing.assert_array_equal(split_a.test_idx, split_b.test_idx)

    train_subjects = set(df.iloc[split_a.train_idx]["Subject"].tolist())
    val_subjects = set(df.iloc[split_a.val_idx]["Subject"].tolist())
    test_subjects = set(df.iloc[split_a.test_idx]["Subject"].tolist())
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)


def test_stratified_subset_keeps_each_class() -> None:
    module = _load_module()
    labels = np.asarray(["a"] * 50 + ["b"] * 30 + ["c"] * 20, dtype=object)
    idx = module._stratified_subset_indices(labels=labels, fraction=0.2, seed=13, min_per_class=2)
    chosen = labels[idx]
    counts = {label: int((chosen == label).sum()) for label in np.unique(chosen)}
    assert counts["a"] >= 2
    assert counts["b"] >= 2
    assert counts["c"] >= 2
    assert len(idx) < len(labels)


def test_build_moment_model_freezes_encoder_and_keeps_head_trainable(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    _install_fake_moment(monkeypatch)

    linear_model = module.build_moment_model(
        head_type="linear",
        num_classes=3,
        model_name="fake/model",
        device="cpu",
        num_channels=4,
        mlp_hidden_dim=16,
        mlp_dropout=0.1,
    )
    assert all(not p.requires_grad for p in linear_model.patch_embedding.parameters())
    assert all(not p.requires_grad for p in linear_model.encoder.parameters())
    assert any(p.requires_grad for p in linear_model.head.parameters())

    mlp_model = module.build_moment_model(
        head_type="mlp",
        num_classes=3,
        model_name="fake/model",
        device="cpu",
        num_channels=4,
        mlp_hidden_dim=16,
        mlp_dropout=0.1,
    )
    assert any(p.requires_grad for p in mlp_model.head.parameters())
    assert mlp_model.head.__class__.__name__ == "MLPClassificationHead"


def test_choose_best_head_uses_validation_balanced_accuracy() -> None:
    module = _load_module()
    results = {
        "linear": {"metrics": {"val_balanced_accuracy": 0.75}},
        "mlp": {"metrics": {"val_balanced_accuracy": 0.80}},
    }
    assert module.choose_best_head(results) == "mlp"


def test_end_to_end_smoke_dual_head_export_and_best_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    _install_fake_moment(monkeypatch)

    input_dir = tmp_path / "processed"
    out_dir = tmp_path / "out"
    _write_synthetic_processed_dataset(input_dir)

    rc = module.main(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--patience",
            "1",
            "--batch-size",
            "8",
            "--num-workers",
            "0",
            "--seed",
            "7",
            "--schemes",
            "binary,edr,avm",
            "--head-types",
            "linear,mlp",
            "--mlp-hidden-dim",
            "16",
        ]
    )
    assert rc == 0

    manifest_path = out_dir / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["schemes"].keys()) == {"binary", "edr", "avm"}

    for scheme in ("binary", "edr", "avm"):
        scheme_dir = out_dir / scheme
        assert (scheme_dir / "linear" / "model.pt").exists()
        assert (scheme_dir / "linear" / "head_features.npz").exists()
        assert (scheme_dir / "linear" / "base_embeddings.npz").exists()
        assert (scheme_dir / "linear" / "metrics.json").exists()
        assert (scheme_dir / "mlp" / "model.pt").exists()
        assert (scheme_dir / "mlp" / "head_features.npz").exists()
        assert (scheme_dir / "mlp" / "base_embeddings.npz").exists()
        assert (scheme_dir / "mlp" / "metrics.json").exists()
        assert (scheme_dir / "best_model.pt").exists()
        assert (scheme_dir / "best_head_features.npz").exists()
        assert (scheme_dir / "best_base_embeddings.npz").exists()
        assert (scheme_dir / "best_head.json").exists()

        best_payload = json.loads((scheme_dir / "best_head.json").read_text())
        assert best_payload["best_head"] in {"linear", "mlp"}

        linear_head = np.load(scheme_dir / "linear" / "head_features.npz", allow_pickle=False)
        mlp_head = np.load(scheme_dir / "mlp" / "head_features.npz", allow_pickle=False)
        assert linear_head["segment_id"].shape == mlp_head["segment_id"].shape
        assert linear_head["label_int"].shape == mlp_head["label_int"].shape
        assert set(np.unique(linear_head["split"].astype(str))) <= {"train", "val", "test"}


def test_end_to_end_respects_save_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    _install_fake_moment(monkeypatch)

    input_dir = tmp_path / "processed"
    out_dir = tmp_path / "out_no_save"
    _write_synthetic_processed_dataset(input_dir)

    rc = module.main(
        [
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--patience",
            "1",
            "--batch-size",
            "8",
            "--num-workers",
            "0",
            "--seed",
            "11",
            "--schemes",
            "binary",
            "--head-types",
            "linear,mlp",
            "--mlp-hidden-dim",
            "16",
            "--no-save-model",
            "--no-save-metrics",
            "--no-save-base-embeddings",
            "--no-save-head-features",
        ]
    )
    assert rc == 0

    scheme_dir = out_dir / "binary"
    assert not (scheme_dir / "linear" / "model.pt").exists()
    assert not (scheme_dir / "linear" / "head_features.npz").exists()
    assert not (scheme_dir / "linear" / "base_embeddings.npz").exists()
    assert not (scheme_dir / "linear" / "metrics.json").exists()
    assert not (scheme_dir / "mlp" / "model.pt").exists()
    assert not (scheme_dir / "mlp" / "head_features.npz").exists()
    assert not (scheme_dir / "mlp" / "base_embeddings.npz").exists()
    assert not (scheme_dir / "mlp" / "metrics.json").exists()
    assert not (scheme_dir / "best_model.pt").exists()
    assert not (scheme_dir / "best_head_features.npz").exists()
    assert not (scheme_dir / "best_base_embeddings.npz").exists()
    assert (scheme_dir / "best_head.json").exists()

    payload = json.loads((scheme_dir / "best_head.json").read_text())
    assert payload["artifacts"]["best_model"] is None
    assert payload["artifacts"]["best_head_features"] is None
    assert payload["artifacts"]["best_base_embeddings"] is None
