"""Tests for IMWUT penultimate-layer fine-tuning pipeline."""

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
MODULE_PATH = REPO_ROOT / "trustME" / "source" / "finetune_imwut_penultimate.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("finetune_imwut_penultimate", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load finetune_imwut_penultimate module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_fake_moment(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSelfAttention(torch.nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.q = torch.nn.Linear(d_model, d_model, bias=False)
            self.k = torch.nn.Linear(d_model, d_model, bias=False)
            self.v = torch.nn.Linear(d_model, d_model, bias=False)
            self.o = torch.nn.Linear(d_model, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.o(torch.tanh(self.q(x) + self.k(x) + self.v(x)))

    class FakeDenseReluDense(torch.nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.wi = torch.nn.Linear(d_model, d_model, bias=False)
            self.wo = torch.nn.Linear(d_model, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.wo(torch.relu(self.wi(x)))

    class FakeAttentionLayer(torch.nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.SelfAttention = FakeSelfAttention(d_model)
            self.layer_norm = torch.nn.LayerNorm(d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer_norm(x + self.SelfAttention(x))

    class FakeFFLayer(torch.nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.DenseReluDense = FakeDenseReluDense(d_model)
            self.layer_norm = torch.nn.LayerNorm(d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer_norm(x + self.DenseReluDense(x))

    class FakeBlock(torch.nn.Module):
        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.layer = torch.nn.ModuleList(
                [
                    FakeAttentionLayer(d_model),
                    FakeFFLayer(d_model),
                ]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.layer[0](x)
            x = self.layer[1](x)
            return x

    class FakeEncoder(torch.nn.Module):
        def __init__(self, d_model: int, n_blocks: int) -> None:
            super().__init__()
            self.block = torch.nn.ModuleList([FakeBlock(d_model) for _ in range(n_blocks)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.block:
                x = block(x)
            return x

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
            self.encoder = FakeEncoder(d_model=8, n_blocks=3)
            self.backbone = torch.nn.LazyLinear(8)
            self.head = FakeLinearHead(d_model=8, n_classes=int(model_kwargs["num_class"]))

            if bool(model_kwargs.get("freeze_embedder", True)):
                for param in self.patch_embedding.parameters():
                    param.requires_grad = False
            if bool(model_kwargs.get("freeze_encoder", True)):
                for block in self.encoder.block:
                    for param in block.parameters():
                        param.requires_grad = False
            if bool(model_kwargs.get("freeze_head", False)):
                for param in self.head.parameters():
                    param.requires_grad = False

        def init(self) -> "FakePipeline":
            self.head = FakeLinearHead(d_model=8, n_classes=int(self._model_kwargs["num_class"]))
            return self

        def _encode(self, x_enc: torch.Tensor) -> torch.Tensor:
            batch_size = x_enc.shape[0]
            base = torch.tanh(self.backbone(x_enc.reshape(batch_size, -1)))
            enc = base.unsqueeze(1).repeat(1, 2, 1)
            for block in self.encoder.block:
                enc = block(enc)
            return enc

        def forward(
            self,
            *,
            x_enc: torch.Tensor,
            input_mask: torch.Tensor | None = None,
            reduction: str = "mean",
        ) -> SimpleNamespace:
            _ = input_mask
            _ = reduction
            enc = self._encode(x_enc)
            logits = self.head(enc)
            return SimpleNamespace(logits=logits, embeddings=enc)

        def embed(
            self,
            *,
            x_enc: torch.Tensor,
            input_mask: torch.Tensor | None = None,
            reduction: str = "mean",
        ) -> SimpleNamespace:
            _ = input_mask
            _ = reduction
            enc = self._encode(x_enc)
            emb = enc.mean(dim=1)
            return SimpleNamespace(embeddings=emb)

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
    for subject_num in range(1, 11):
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


def _build_args(input_dir: Path, out_dir: Path, schemes: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        config=Path("dummy.yaml"),
        input_dir=input_dir,
        out_dir=out_dir,
        schemes=schemes,
        model_name="fake/model",
        batch_size=8,
        epochs=2,
        patience=1,
        lr=1e-3,
        weight_decay=0.0,
        seed=7,
        device="cpu",
        num_workers=0,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        non_blocking=True,
        use_amp=True,
        amp_dtype="bf16",
        enable_tf32=True,
        subject_train_frac=0.70,
        subject_val_frac=0.15,
        subject_test_frac=0.15,
        subset_fraction=1.0,
        subset_min_per_class=1,
        subset_seed=11,
        head_type="linear",
        mlp_hidden_dim=16,
        mlp_dropout=0.1,
        encoder_tune_scope="last_n_blocks",
        unfreeze_last_n_blocks=1,
        weights_format="trainable_only",
        save_model_weights=False,
        save_embeddings=True,
        save_metrics=True,
        save_predictions=False,
    )


def test_scope_selection_trainable_params(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    _install_fake_moment(monkeypatch)

    expected_last_block_prefix = "encoder.block.2."

    for scope in ("last_n_layernorm", "last_n_mlp", "last_n_blocks"):
        model, trainable_names, selected_blocks = module.build_moment_model(
            head_type="linear",
            num_classes=3,
            model_name="fake/model",
            device="cpu",
            num_channels=4,
            mlp_hidden_dim=16,
            mlp_dropout=0.1,
            encoder_tune_scope=scope,
            unfreeze_last_n_blocks=1,
        )
        assert selected_blocks == [2]
        assert any(name.startswith("head.") for name in trainable_names)

        non_head = [name for name in trainable_names if not name.startswith("head.")]
        assert non_head, "Expected encoder params to be trainable in penultimate setup."
        assert all(name.startswith(expected_last_block_prefix) for name in non_head)
        assert all(not name.startswith("encoder.block.0.") for name in non_head)
        assert all(not name.startswith("encoder.block.1.") for name in non_head)

        if scope == "last_n_layernorm":
            assert all(".layer_norm." in name for name in non_head)
            assert all(".DenseReluDense." not in name for name in non_head)
            assert all(".SelfAttention." not in name for name in non_head)
        elif scope == "last_n_mlp":
            assert all(".DenseReluDense." in name for name in non_head)
            assert all(".SelfAttention." not in name for name in non_head)
            assert all(".layer_norm." not in name for name in non_head)
        else:
            assert any(".DenseReluDense." in name for name in non_head)
            assert any(".SelfAttention." in name for name in non_head)
            assert any(".layer_norm." in name for name in non_head)

        assert isinstance(model, torch.nn.Module)


def test_apply_label_scheme_drops_questionnaire_and_central_first() -> None:
    module = _load_module()
    df = pd.DataFrame(
        {
            "segment_id": ["a", "b", "c", "d", "e", "f"],
            "Subject": ["s_001"] * 6,
            "Label": ["questionnaire", "central_position", "2_back", "memory_easy", "pursuit_easy", "rest"],
        }
    )

    binary = module.apply_label_scheme(df, scheme="binary")
    edr = module.apply_label_scheme(df, scheme="edr")
    avm = module.apply_label_scheme(df, scheme="avm")

    for out_df in (binary, edr, avm):
        assert "questionnaire" not in set(out_df["Label"])
        assert "central_position" not in set(out_df["Label"])

    assert set(binary["scheme_label"].unique()) == {"load", "rest"}
    assert set(edr["scheme_label"].unique()) == {"low_load", "rest"}
    assert set(avm["scheme_label"].unique()) == {"attention_task", "memory_task", "visual_task"}


def test_stratified_subset_keeps_min_per_class() -> None:
    module = _load_module()
    labels = np.asarray(["a"] * 50 + ["b"] * 30 + ["c"] * 20, dtype=object)
    idx = module._stratified_subset_indices(labels=labels, fraction=0.2, seed=13, min_per_class=2)
    chosen = labels[idx]
    counts = {label: int((chosen == label).sum()) for label in np.unique(chosen)}
    assert counts["a"] >= 2
    assert counts["b"] >= 2
    assert counts["c"] >= 2
    assert len(idx) < len(labels)


def test_split_subject_holdout_is_disjoint() -> None:
    module = _load_module()
    rows = []
    for subject_num in range(1, 11):
        subject = f"s_{subject_num:03d}"
        rows.append({"Subject": subject, "scheme_label": "class_a"})
        rows.append({"Subject": subject, "scheme_label": "class_b"})
    df = pd.DataFrame(rows)

    split = module.split_subject_holdout(df=df, seed=123)
    train_subjects = set(df.iloc[split.train_idx]["Subject"].tolist())
    val_subjects = set(df.iloc[split.val_idx]["Subject"].tolist())
    test_subjects = set(df.iloc[split.test_idx]["Subject"].tolist())
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)


def test_default_artifacts_only_embeddings_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    _install_fake_moment(monkeypatch)

    input_dir = tmp_path / "processed"
    out_dir = tmp_path / "out"
    _write_synthetic_processed_dataset(input_dir)
    args = _build_args(input_dir=input_dir, out_dir=out_dir, schemes=("binary", "edr", "avm"))

    module.run_pipeline(args)

    manifest_path = out_dir / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest["schemes"].keys()) == {"binary", "edr", "avm"}

    for scheme in ("binary", "edr", "avm"):
        scheme_dir = out_dir / scheme
        assert (scheme_dir / "embeddings.npz").exists()
        assert (scheme_dir / "metrics.json").exists()
        assert not (scheme_dir / "predictions.npz").exists()
        assert not (scheme_dir / "model.pt").exists()


def test_trainable_only_checkpoint_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    _install_fake_moment(monkeypatch)

    input_dir = tmp_path / "processed"
    out_dir = tmp_path / "out"
    _write_synthetic_processed_dataset(input_dir)
    args = _build_args(input_dir=input_dir, out_dir=out_dir, schemes=("binary",))
    args.save_model_weights = True
    args.save_predictions = True
    args.weights_format = "trainable_only"
    args.encoder_tune_scope = "last_n_mlp"

    module.run_pipeline(args)
    scheme_dir = out_dir / "binary"
    model_path = scheme_dir / "model.pt"
    assert model_path.exists()
    assert (scheme_dir / "predictions.npz").exists()

    fresh_model, _, _ = module.build_moment_model(
        head_type=args.head_type,
        num_classes=2,
        model_name=args.model_name,
        device="cpu",
        num_channels=4,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.mlp_dropout,
        encoder_tune_scope=args.encoder_tune_scope,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
    )
    loaded_keys = module.load_trainable_only_checkpoint(fresh_model, model_path)
    assert loaded_keys


def test_inference_casts_bfloat16_outputs_to_float32() -> None:
    module = _load_module()

    class Bf16OutputModel(torch.nn.Module):
        def forward(
            self,
            *,
            x_enc: torch.Tensor,
            input_mask: torch.Tensor | None = None,
            reduction: str = "mean",
        ) -> SimpleNamespace:
            _ = input_mask
            _ = reduction
            batch = x_enc.shape[0]
            logits = torch.zeros((batch, 3), dtype=torch.bfloat16, device=x_enc.device)
            logits[:, 0] = 1
            return SimpleNamespace(logits=logits)

        def embed(
            self,
            *,
            x_enc: torch.Tensor,
            input_mask: torch.Tensor | None = None,
            reduction: str = "mean",
        ) -> SimpleNamespace:
            _ = input_mask
            _ = reduction
            batch = x_enc.shape[0]
            emb = torch.ones((batch, 8), dtype=torch.bfloat16, device=x_enc.device)
            return SimpleNamespace(embeddings=emb)

    x = np.random.randn(5, 4, 16).astype(np.float32)
    m = np.ones((5, 16), dtype=np.int64)
    y = np.asarray([0, 1, 2, 1, 0], dtype=np.int64)
    idx = np.arange(5, dtype=np.int64)

    loader = module._make_dataloader(
        x=x,
        input_mask=m,
        labels=y,
        indices=idx,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )

    model = Bf16OutputModel()
    head_out = module._run_head_inference(
        model=model,
        loader=loader,
        device="cpu",
        non_blocking=False,
        use_amp=False,
        amp_dtype=torch.bfloat16,
    )
    emb_out = module._run_embed_inference(
        model=model,
        loader=loader,
        device="cpu",
        non_blocking=False,
        use_amp=False,
        amp_dtype=torch.bfloat16,
    )

    assert head_out["logits"].dtype == np.float32
    assert head_out["probs"].dtype == np.float32
    assert emb_out["embeddings"].dtype == np.float32
