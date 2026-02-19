"""Regression tests for train.py parameter forwarding."""

from __future__ import annotations

import sys
import importlib.util
from types import ModuleType, SimpleNamespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PY = ROOT / "train.py"
_UNSET = object()


class _DummyModel:
    def __init__(self) -> None:
        self.train_kwargs: dict | None = None

    def to(self, dtype=None):
        return self

    def train_model(self, **kwargs):
        self.train_kwargs = kwargs


def _load_train_module():
    fake_torch = ModuleType("torch")
    fake_torch.float32 = "float32"

    fake_gliner = ModuleType("gliner")
    fake_gliner.GLiNER = object

    fake_gliner_utils = ModuleType("gliner.utils")
    fake_gliner_utils.load_config_as_namespace = lambda _path: None
    fake_gliner_utils.namespace_to_dict = lambda _ns: {}

    spec = importlib.util.spec_from_file_location("train_under_test", TRAIN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    with patch.dict(
        sys.modules,
        {"torch": fake_torch, "gliner": fake_gliner, "gliner.utils": fake_gliner_utils},
    ):
        spec.loader.exec_module(module)

    return module


def _training_namespace(
    eval_batch_size: int | None = None,
    bf16: bool | object = _UNSET,
    label_smoothing: float | object = _UNSET,
) -> SimpleNamespace:
    training_kwargs = {
        "num_steps": 100,
        "scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "train_batch_size": 4,
        "lr_encoder": 1e-5,
        "lr_others": 3e-5,
        "weight_decay_encoder": 0.01,
        "weight_decay_other": 0.01,
        "max_grad_norm": 1.0,
        "loss_alpha": -1,
        "loss_gamma": 0,
        "loss_reduction": "sum",
        "negatives": 1.0,
        "masking": "none",
        "eval_every": 50,
        "save_total_limit": 3,
        "eval_batch_size": eval_batch_size,
    }
    if bf16 is not _UNSET:
        training_kwargs["bf16"] = bf16
    if label_smoothing is not _UNSET:
        training_kwargs["label_smoothing"] = label_smoothing
    return SimpleNamespace(**training_kwargs)


def test_main_uses_config_root_dir_and_eval_batch_size_for_train_model_output(tmp_path):
    train_module = _load_train_module()
    model = _DummyModel()
    expected_output_dir = tmp_path / "artifacts"

    cfg = SimpleNamespace(
        model=SimpleNamespace(model_name="dummy-model"),
        data=SimpleNamespace(
            root_dir=str(expected_output_dir),
            train_data="ignored-train.json",
            val_data_dir="none",
        ),
        training=_training_namespace(eval_batch_size=2, bf16=True, label_smoothing=0.25),
    )

    with (
        patch.object(train_module, "load_config_as_namespace", return_value=cfg),
        patch.object(train_module, "namespace_to_dict", return_value={}),
        patch.object(train_module, "load_json_data", return_value=[]),
        patch.object(train_module, "build_model", return_value=model),
    ):
        train_module.main("ignored-config.yaml")

    assert model.train_kwargs is not None
    assert model.train_kwargs["output_dir"] == str(expected_output_dir)
    assert model.train_kwargs["per_device_train_batch_size"] == 4
    assert model.train_kwargs["per_device_eval_batch_size"] == 2
    assert model.train_kwargs["bf16"] is True
    assert model.train_kwargs["label_smoothing"] == 0.25
    assert expected_output_dir.exists()


def test_main_falls_back_to_train_batch_size_when_eval_batch_size_unset(tmp_path):
    train_module = _load_train_module()
    model = _DummyModel()

    cfg = SimpleNamespace(
        model=SimpleNamespace(model_name="dummy-model"),
        data=SimpleNamespace(
            root_dir=str(tmp_path / "artifacts-fallback"),
            train_data="ignored-train.json",
            val_data_dir="none",
        ),
        training=_training_namespace(eval_batch_size=None),
    )

    with (
        patch.object(train_module, "load_config_as_namespace", return_value=cfg),
        patch.object(train_module, "namespace_to_dict", return_value={}),
        patch.object(train_module, "load_json_data", return_value=[]),
        patch.object(train_module, "build_model", return_value=model),
    ):
        train_module.main("ignored-config.yaml")

    assert model.train_kwargs is not None
    assert model.train_kwargs["per_device_train_batch_size"] == 4
    assert model.train_kwargs["per_device_eval_batch_size"] == 4
    assert model.train_kwargs["bf16"] is False
    assert model.train_kwargs["label_smoothing"] == 0.0
