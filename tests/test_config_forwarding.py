"""Tests for configuration field forwarding in the GLiNER training pipeline.

These tests verify that configuration fields are correctly propagated through:

    config.yaml -> train.py -> model.train_model() -> create_training_args() -> TrainingArguments

Tests are organised into two categories:

**Validation tests** (MUST pass): verify that existing, correctly-wired
functionality works.  A failure here means the test itself is wrong or
a regression was introduced.

**Bug-detection tests** (expected to FAIL against current code): each test
targets a specific gap identified in the Standard HuggingFace Configuration
Report.  When the underlying code is fixed, the corresponding test will
start passing.
"""

import json
import inspect
import importlib
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import yaml
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_create_training_args():
    """Import and return the create_training_args classmethod."""
    from gliner.model import BaseGLiNER
    return BaseGLiNER.create_training_args


def _get_signature():
    """Return inspect.Signature of create_training_args."""
    return inspect.signature(_get_create_training_args())


def _build_args(**overrides):
    """Call create_training_args with sensible CPU-safe defaults."""
    fn = _get_create_training_args()
    with tempfile.TemporaryDirectory() as tmp:
        defaults = {"output_dir": tmp, "use_cpu": True, "report_to": "none"}
        defaults.update(overrides)
        return fn(**defaults)


def _write_config(tmp_path, training_overrides=None):
    """Write a minimal config.yaml and dummy training data; return config path."""
    cfg = {
        "model": {
            "model_name": "microsoft/deberta-v3-small",
            "name": "test",
            "max_width": 12,
            "hidden_size": 768,
            "dropout": 0.3,
            "fine_tune": True,
            "subtoken_pooling": "first",
            "fuse_layers": False,
            "span_mode": "markerV0",
            "max_types": 25,
            "max_len": 384,
            "max_neg_type_ratio": 1,
            "decoder_mode": "span",
        },
        "data": {
            "root_dir": str(tmp_path / "logs"),
            "train_data": str(tmp_path / "train.json"),
            "val_data_dir": "none",
        },
        "training": {
            "prev_path": None,
            "num_steps": 100,
            "train_batch_size": 2,
            "eval_every": 50,
            "warmup_ratio": 0.1,
            "scheduler_type": "cosine",
            "loss_alpha": 0.75,
            "loss_gamma": 2,
            "loss_prob_margin": 0.0,
            "label_smoothing": 0.1,
            "loss_reduction": "sum",
            "negatives": 1.0,
            "masking": "none",
            "lr_encoder": 1e-5,
            "lr_others": 3e-5,
            "weight_decay_encoder": 0.1,
            "weight_decay_other": 0.01,
            "max_grad_norm": 10.0,
            "save_total_limit": 3,
            "size_sup": -1,
            "shuffle_types": True,
            "random_drop": True,
            "freeze_components": None,
        },
    }
    if training_overrides:
        cfg["training"].update(training_overrides)

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    train_data = [{"tokenized_text": ["test"], "ner": [], "text": "test"}]
    with open(tmp_path / "train.json", "w") as f:
        json.dump(train_data, f)

    return str(config_path)


def _run_train_main(tmp_path, training_overrides=None):
    """Patch GLiNER and run train.main(); return the train_model call_args."""
    config_path = _write_config(tmp_path, training_overrides)

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.train_model.return_value = MagicMock()

    # Ensure the gliner module can be imported by train.py even in
    # constrained test environments.  We only need the GLiNER name.
    import gliner as _gliner_mod
    if not hasattr(_gliner_mod, "GLiNER"):
        _gliner_mod.GLiNER = MagicMock()

    import train as train_mod
    train_mod = importlib.reload(train_mod)

    with (
        patch.object(train_mod, "GLiNER") as mock_gliner_cls,
        patch.object(train_mod, "load_json_data") as mock_load,
    ):
        mock_gliner_cls.from_config.return_value = mock_model
        mock_load.return_value = [{"text": "t", "ner": []}]
        train_mod.main(config_path)

    mock_model.train_model.assert_called_once()
    return mock_model.train_model.call_args


# =========================================================================== #
#  PART A  --  VALIDATION TESTS  (must all pass)
# =========================================================================== #


class TestCreateTrainingArgsSmoke:
    """Sanity checks: create_training_args returns a valid TrainingArguments
    with correctly-forwarded named parameters.
    """

    def test_returns_training_arguments_instance(self):
        from gliner.training.trainer import TrainingArguments
        args = _build_args()
        assert isinstance(args, TrainingArguments)

    def test_gliner_custom_fields_forwarded(self):
        """GLiNER-specific fields (focal loss, masking, negatives) are set."""
        args = _build_args(
            focal_loss_alpha=0.75,
            focal_loss_gamma=2.0,
            focal_loss_prob_margin=0.05,
            negatives=0.5,
            masking="none",
            loss_reduction="mean",
        )
        assert args.focal_loss_alpha == 0.75
        assert args.focal_loss_gamma == 2.0
        assert args.focal_loss_prob_margin == 0.05
        assert args.negatives == 0.5
        assert args.masking == "none"
        assert args.loss_reduction == "mean"

    def test_standard_hf_named_fields_forwarded(self):
        """All named params in create_training_args reach TrainingArguments."""
        args = _build_args(
            learning_rate=2e-5,
            weight_decay=0.05,
            others_lr=5e-5,
            others_weight_decay=0.02,
            warmup_ratio=0.06,
            lr_scheduler_type="cosine",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            max_grad_norm=5.0,
            max_steps=5000,
            save_steps=500,
            save_total_limit=5,
            logging_steps=25,
            dataloader_num_workers=4,
        )
        assert args.learning_rate == 2e-5
        assert args.weight_decay == 0.05
        assert args.others_lr == 5e-5
        assert args.others_weight_decay == 0.02
        assert args.warmup_ratio == 0.06
        assert args.lr_scheduler_type == "cosine"
        assert args.per_device_train_batch_size == 16
        assert args.per_device_eval_batch_size == 32
        assert args.max_grad_norm == 5.0
        assert args.max_steps == 5000
        assert args.save_steps == 500
        assert args.save_total_limit == 5
        assert args.logging_steps == 25
        assert args.dataloader_num_workers == 4

    def test_kwargs_passthrough_to_training_arguments(self):
        """Extra kwargs reach TrainingArguments via **kwargs passthrough."""
        args = _build_args(seed=42)
        assert args.seed == 42

    def test_report_to_default_is_none(self):
        """Default report_to should be 'none' (no reporting)."""
        sig = _get_signature()
        assert sig.parameters["report_to"].default == "none"

    def test_existing_named_params_present(self):
        """Every named param documented in the current signature exists."""
        sig = _get_signature()
        expected = {
            "output_dir", "learning_rate", "weight_decay",
            "others_lr", "others_weight_decay",
            "focal_loss_alpha", "focal_loss_gamma", "focal_loss_prob_margin",
            "loss_reduction", "negatives", "masking",
            "lr_scheduler_type", "warmup_ratio",
            "per_device_train_batch_size", "per_device_eval_batch_size",
            "max_grad_norm", "max_steps", "save_steps", "save_total_limit",
            "logging_steps", "use_cpu", "bf16",
            "dataloader_num_workers", "report_to",
        }
        actual = set(sig.parameters.keys()) - {"cls", "kwargs"}
        assert expected.issubset(actual), (
            f"Missing named params: {expected - actual}"
        )


class TestTrainScriptForwardingValidation:
    """Validation: fields that train.py DOES forward correctly."""

    @pytest.fixture()
    def kwargs(self, tmp_path):
        _, kw = _run_train_main(tmp_path)
        return kw

    def test_max_steps_forwarded(self, kwargs):
        assert kwargs["max_steps"] == 100

    def test_learning_rate_forwarded(self, kwargs):
        assert kwargs["learning_rate"] == float(1e-5)

    def test_others_lr_forwarded(self, kwargs):
        assert kwargs["others_lr"] == float(3e-5)

    def test_weight_decay_forwarded(self, kwargs):
        assert kwargs["weight_decay"] == float(0.1)

    def test_others_weight_decay_forwarded(self, kwargs):
        assert kwargs["others_weight_decay"] == float(0.01)

    def test_warmup_ratio_forwarded(self, kwargs):
        assert kwargs["warmup_ratio"] == 0.1

    def test_scheduler_type_forwarded(self, kwargs):
        assert kwargs["lr_scheduler_type"] == "cosine"

    def test_focal_loss_alpha_forwarded(self, kwargs):
        assert kwargs["focal_loss_alpha"] == float(0.75)

    def test_focal_loss_gamma_forwarded(self, kwargs):
        assert kwargs["focal_loss_gamma"] == float(2)

    def test_loss_reduction_forwarded(self, kwargs):
        assert kwargs["loss_reduction"] == "sum"

    def test_negatives_forwarded(self, kwargs):
        assert kwargs["negatives"] == float(1.0)

    def test_masking_forwarded(self, kwargs):
        assert kwargs["masking"] == "none"

    def test_save_total_limit_forwarded(self, kwargs):
        assert kwargs["save_total_limit"] == 3

    def test_save_steps_forwarded(self, kwargs):
        assert kwargs["save_steps"] == 50

    def test_output_dir_forwarded(self, kwargs):
        # train.py uses cfg.data.root_dir (relative to tmp_path)
        assert "output_dir" in kwargs
        assert kwargs["output_dir"].endswith("/logs")


class TestTrainingArgumentsCapability:
    """Validation: TrainingArguments supports eval_steps via **kwargs passthrough."""

    def test_eval_steps_settable_via_kwargs(self):
        """TrainingArguments itself supports eval_steps (the plumbing works)."""
        args = _build_args(save_steps=1000, eval_steps=200)
        assert args.save_steps == 1000
        assert args.eval_steps == 200


# =========================================================================== #
#  PART B  --  BUG DETECTION TESTS  (expected to fail against current code)
#
#  Each test documents a specific gap from the Configuration Report.
#  When the bug is fixed, the test will start passing.
# =========================================================================== #


class TestCreateTrainingArgsSignatureGaps:
    """Report: critical HF TrainingArguments fields must be named parameters
    in create_training_args, not hidden behind **kwargs.

    Named parameters provide discoverability, IDE auto-complete, and
    documentation.  Fields only reachable through **kwargs are invisible.
    """

    @pytest.fixture(autouse=True)
    def _load_signature(self):
        self.sig = _get_signature()

    # -- Precision --------------------------------------------------------- #

    def test_fp16_is_named_parameter(self):
        """fp16 must be a named parameter alongside bf16.

        Report: create_training_args exposes bf16 but not fp16.
        """
        assert "fp16" in self.sig.parameters, (
            "fp16 is missing from create_training_args named parameters; "
            "users cannot discover or configure fp16 without reading source"
        )

    # -- Loss -------------------------------------------------------------- #

    def test_label_smoothing_is_named_parameter(self):
        """label_smoothing is on TrainingArguments but hidden from the API.

        Report: TrainingArguments has label_smoothing (trainer.py:82) but
        create_training_args does not expose it as a named parameter.
        """
        assert "label_smoothing" in self.sig.parameters, (
            "label_smoothing is not a named parameter in create_training_args; "
            "it is defined on TrainingArguments but not exposed in the factory"
        )

    # -- Evaluation -------------------------------------------------------- #

    def test_eval_steps_is_named_parameter(self):
        """eval_steps should be independently configurable from save_steps.

        Report: eval_every conflation means eval and save intervals
        cannot be set independently.
        """
        assert "eval_steps" in self.sig.parameters, (
            "eval_steps is not a named parameter; evaluation and checkpoint "
            "saving frequency cannot be set independently"
        )

    # -- Dataloader -------------------------------------------------------- #

    def test_dataloader_pin_memory_is_named_parameter(self):
        """dataloader_pin_memory is used by Trainer.get_train_dataloader (line 322)."""
        assert "dataloader_pin_memory" in self.sig.parameters

    def test_dataloader_persistent_workers_is_named_parameter(self):
        """dataloader_persistent_workers is read by Trainer.get_train_dataloader (line 323)."""
        assert "dataloader_persistent_workers" in self.sig.parameters

    def test_dataloader_prefetch_factor_is_named_parameter(self):
        """dataloader_prefetch_factor is read by Trainer.get_train_dataloader (line 330)."""
        assert "dataloader_prefetch_factor" in self.sig.parameters

    # -- Hub integration --------------------------------------------------- #

    def test_push_to_hub_is_named_parameter(self):
        """push_to_hub is a standard HF TrainingArguments field for Hub uploads."""
        assert "push_to_hub" in self.sig.parameters

    def test_hub_model_id_is_named_parameter(self):
        """hub_model_id controls the Hub repo destination."""
        assert "hub_model_id" in self.sig.parameters

    # -- Serialization ----------------------------------------------------- #

    @pytest.mark.xfail(
        strict=True,
        reason="save_safetensors was removed from HF TrainingArguments in transformers v5",
    )
    def test_save_safetensors_is_named_parameter(self):
        """save_safetensors was removed from HF TrainingArguments in transformers v5.
        This parameter is no longer relevant."""
        assert "save_safetensors" in self.sig.parameters, (
            "save_safetensors is not a named parameter in create_training_args; "
            "removed in transformers v5"
        )

    def test_remove_unused_columns_is_named_parameter(self):
        """remove_unused_columns must be configurable (GLiNER needs False).

        Report: HF Trainer defaults remove_unused_columns=True which drops
        columns not in the model forward() signature.  GLiNER's custom
        batch dictionaries require False.
        """
        assert "remove_unused_columns" in self.sig.parameters, (
            "remove_unused_columns is not a named parameter; GLiNER needs "
            "this set to False to preserve custom batch columns"
        )

    # -- Type annotation --------------------------------------------------- #

    def test_report_to_annotation_allows_list(self):
        """report_to should accept List[str], not just str.

        Report: HF TrainingArguments accepts List[str] for report_to
        (e.g. ['wandb', 'tensorboard']).  Current signature types it as str.
        """
        param = self.sig.parameters["report_to"]
        assert param.annotation is not str, (
            f"report_to is annotated as {param.annotation!r}; it should "
            "accept Union[str, List[str]] to match HF TrainingArguments"
        )


class TestTrainScriptForwardingGaps:
    """Report: train.py silently drops config fields that should reach
    model.train_model() and ultimately TrainingArguments.

    We patch GLiNER, run train.main(), and inspect the kwargs to detect
    missing or incorrect forwarding.
    """

    @pytest.fixture()
    def kwargs(self, tmp_path):
        """Default config kwargs captured from train.main()."""
        _, kw = _run_train_main(tmp_path)
        return kw

    # -- label_smoothing --------------------------------------------------- #

    def test_label_smoothing_forwarded(self, kwargs):
        """label_smoothing is in config.yaml but must reach train_model().

        Report: label_smoothing is defined in TrainingArguments (trainer.py:82)
        and present in config.yaml (value 0.1), but train.py never includes
        it in the train_model() call.
        """
        assert "label_smoothing" in kwargs, (
            "train.py does not forward 'label_smoothing' to model.train_model(); "
            "the config value 0.1 is silently discarded"
        )

    def test_label_smoothing_value_matches_config(self, kwargs):
        """The forwarded label_smoothing value must match the config."""
        assert kwargs.get("label_smoothing") == 0.1, (
            f"Expected label_smoothing=0.1 from config, "
            f"got {kwargs.get('label_smoothing')}"
        )

    # -- bf16 -------------------------------------------------------------- #

    def test_bf16_not_hardcoded(self, kwargs):
        """bf16 should come from config, not be hardcoded to True.

        Report: train.py line 91 hardcodes bf16=True regardless of what the
        config contains.  Users cannot disable bf16 or switch to fp16 via
        configuration alone.
        """
        # The test config does NOT set bf16, so it should NOT be forced True.
        assert kwargs.get("bf16") is not True, (
            "bf16 is hardcoded to True in train.py; it should be read from "
            "the training config section"
        )

    # -- fp16 -------------------------------------------------------------- #

    @pytest.mark.xfail(
        strict=True,
        reason="train.py (legacy) does not forward fp16; training_cli.py does",
    )
    def test_fp16_forwarded_when_in_config(self, tmp_path):
        """If fp16 is set in the config, it should reach train_model()."""
        _, kwargs = _run_train_main(tmp_path, {"fp16": True})
        assert "fp16" in kwargs, (
            "train.py does not forward 'fp16' to model.train_model() "
            "even when it is present in the config"
        )

    # -- eval_every conflation --------------------------------------------- #

    @pytest.mark.xfail(
        strict=True,
        reason="train.py (legacy) uses eval_every for both; training_cli.py has separate logging_steps",
    )
    def test_save_steps_and_logging_steps_are_independent(self, kwargs):
        """save_steps and logging_steps should not both come from eval_every.

        Report: a single 'eval_every' config field drives both save_steps
        and logging_steps, making it impossible to log more frequently
        than you save checkpoints.
        """
        save = kwargs.get("save_steps")
        log = kwargs.get("logging_steps")
        assert save != log, (
            f"save_steps ({save}) and logging_steps ({log}) are both set "
            f"from the same 'eval_every' config field; they should be "
            f"independently configurable"
        )

    @pytest.mark.xfail(
        strict=True,
        reason="train.py (legacy) does not forward eval_steps; training_cli.py does",
    )
    def test_eval_steps_kwarg_forwarded(self, kwargs):
        """An eval_steps kwarg should be passed to train_model().

        Report: train.py never forwards eval_steps; HF Trainer evaluation
        scheduling relies on this parameter.
        """
        assert "eval_steps" in kwargs, (
            "train.py never forwards 'eval_steps'; HF Trainer evaluation "
            "scheduling relies on this parameter"
        )

    # -- per_device_eval_batch_size ---------------------------------------- #

    def test_eval_batch_size_independent_of_train(self, tmp_path):
        """per_device_eval_batch_size should use eval-specific config.

        Report: train.py line 70 uses cfg.training.train_batch_size for
        per_device_eval_batch_size, ignoring any eval-specific setting.
        """
        _, kwargs = _run_train_main(tmp_path, {
            "train_batch_size": 4,
            "eval_batch_size": 8,
        })
        eval_bs = kwargs.get("per_device_eval_batch_size")
        train_bs = kwargs.get("per_device_train_batch_size")
        assert eval_bs != train_bs, (
            f"per_device_eval_batch_size ({eval_bs}) equals "
            f"per_device_train_batch_size ({train_bs}); eval batch size "
            "should come from a separate config field"
        )

    # -- logging_steps ----------------------------------------------------- #

    def test_logging_steps_reads_own_config_field(self, tmp_path):
        """logging_steps should have its own config field, not reuse eval_every.

        Report: logging_steps and save_steps are both driven by eval_every.
        """
        _, kwargs = _run_train_main(tmp_path, {
            "eval_every": 500,
            "logging_steps": 10,
        })
        assert kwargs.get("logging_steps") == 10, (
            f"logging_steps should be 10 (from its own config field), "
            f"got {kwargs.get('logging_steps')} (likely from eval_every)"
        )
