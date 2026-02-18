from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def test_create_training_args_forwards_to_training_arguments(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    BaseGLiNER = importlib.import_module("gliner.model").BaseGLiNER

    captured: dict[str, object] = {}

    class DummyTrainingArguments:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            for key, value in kwargs.items():
                setattr(self, key, value)

    monkeypatch.setattr("gliner.model.TrainingArguments", DummyTrainingArguments)

    args = BaseGLiNER.create_training_args(
        output_dir=tmp_path / "run-out",
        learning_rate=2e-5,
        weight_decay=0.02,
        others_lr=6e-5,
        others_weight_decay=0.08,
        lr_scheduler_type="cosine",
        warmup_ratio=0.25,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=3,
        max_grad_norm=1.7,
        max_steps=77,
        save_steps=11,
        save_total_limit=4,
        logging_steps=6,
        use_cpu=True,
        bf16=True,
        dataloader_num_workers=8,
        report_to="tensorboard",
        label_smoothing=0.13,
        gradient_accumulation_steps=4,
    )

    assert isinstance(args, DummyTrainingArguments)
    assert captured["output_dir"] == tmp_path / "run-out"
    assert captured["learning_rate"] == 2e-5
    assert captured["others_lr"] == 6e-5
    assert captured["others_weight_decay"] == 0.08
    assert captured["per_device_eval_batch_size"] == 3
    assert captured["label_smoothing"] == 0.13
    assert captured["gradient_accumulation_steps"] == 4
    assert captured["report_to"] == "tensorboard"
    assert captured["bf16"] is True
    assert captured["use_cpu"] is True
    assert captured["dataloader_num_workers"] == 8


def test_fake_tensor_cpu_path_reflects_bf16_training_arg(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
    BaseGLiNER = importlib.import_module("gliner.model").BaseGLiNER

    class DummyTrainingArguments:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    monkeypatch.setattr("gliner.model.TrainingArguments", DummyTrainingArguments)

    args = BaseGLiNER.create_training_args(
        output_dir=tmp_path / "fake-run",
        use_cpu=True,
        bf16=True,
        report_to="none",
    )

    with fake_tensor_mod.FakeTensorMode() as fake_mode:
        fake_input = fake_mode.from_tensor(torch.randn(2, 4))
        target_dtype = torch.bfloat16 if args.bf16 else fake_input.dtype
        fake_output = fake_input.to(dtype=target_dtype)

    assert fake_input.device.type == "cpu"
    assert fake_output.device.type == "cpu"
    assert fake_output.dtype == torch.bfloat16


@pytest.mark.xfail(
    strict=True,
    reason="PR14 follow-up: train.py still hardcodes output_dir/eval batch size/bf16",
)
def test_train_main_forwards_yaml_training_values(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    train_entrypoint = importlib.import_module("train")

    captured: dict[str, object] = {}

    class StubModel:
        def to(self, *, dtype):
            captured["model_dtype"] = dtype
            return self

        def train_model(self, **kwargs):
            captured["train_kwargs"] = kwargs

    cfg = _ns(
        data=_ns(
            root_dir=str(tmp_path / "artifacts"),
            train_data=str(tmp_path / "train.json"),
            val_data_dir=str(tmp_path / "val.json"),
        ),
        model=_ns(
            model_name="microsoft/deberta-v3-small",
            span_mode="markerV0",
            max_len=384,
        ),
        training=_ns(
            prev_path=None,
            num_steps=12,
            scheduler_type="cosine",
            warmup_ratio=0.2,
            train_batch_size=4,
            eval_batch_size=2,
            lr_encoder=1e-5,
            lr_others=3e-5,
            weight_decay_encoder=0.01,
            weight_decay_other=0.02,
            max_grad_norm=1.1,
            loss_alpha=0.7,
            loss_gamma=1.5,
            loss_prob_margin=0.05,
            loss_reduction="sum",
            negatives=1.0,
            masking="none",
            eval_every=3,
            save_total_limit=2,
            bf16=False,
        ),
    )

    monkeypatch.setattr(train_entrypoint, "load_config_as_namespace", lambda _path: cfg)
    monkeypatch.setattr(train_entrypoint, "namespace_to_dict", lambda ns: vars(ns).copy())
    monkeypatch.setattr(train_entrypoint, "load_json_data", lambda _path: [{"text": "x", "ner": []}])
    monkeypatch.setattr(train_entrypoint, "build_model", lambda *_args, **_kwargs: StubModel())

    train_entrypoint.main("dummy.yaml")

    kwargs = captured["train_kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["output_dir"] == str(tmp_path / "artifacts")
    assert kwargs["per_device_eval_batch_size"] == cfg.training.eval_batch_size
    assert kwargs["bf16"] is cfg.training.bf16
