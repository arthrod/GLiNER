"""Tests for training parameter validation and configuration wiring.

These tests verify that training parameters are correctly defined, forwarded,
and defaulted throughout the GLiNER training pipeline. They catch issues
identified in the training parameters and loss configuration report.

Tests marked with ``@pytest.mark.xfail(strict=True)`` document known gaps.
When a gap is fixed the test will unexpectedly pass (XPASS) and the marker
should be removed.

Test categories:
    - TrainingArguments field definitions and defaults
    - create_training_args explicit parameter coverage
    - train_model parameter forwarding
    - Config field consumption (dead fields)
    - Label smoothing name collision risk
"""

import inspect
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest
import transformers

from gliner.training.trainer import TrainingArguments, Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_create_training_args_explicit_params():
    """Return the set of explicit parameter names in create_training_args.

    Imports BaseGLiNER and inspects the classmethod signature, excluding
    'cls', 'self', and '**kwargs'.
    """
    from gliner.model import BaseGLiNER

    sig = inspect.signature(BaseGLiNER.create_training_args)
    return {
        name
        for name, param in sig.parameters.items()
        if name not in ("cls", "self")
        and param.kind
        not in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
    }


def _get_training_args_field_names():
    """Return all field names defined on the custom TrainingArguments dataclass."""
    return {f.name for f in dataclass_fields(TrainingArguments)}


def _make_training_args(output_dir, **overrides):
    """Create a TrainingArguments with a caller-provided output_dir."""
    defaults = {
        "output_dir": str(output_dir),
        "report_to": "none",
        "use_cpu": True,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def _create_training_args_via_classmethod(output_dir, **overrides):
    """Create TrainingArguments through BaseGLiNER.create_training_args."""
    from gliner.model import BaseGLiNER

    defaults = {
        "output_dir": str(output_dir),
        "report_to": "none",
    }
    defaults.update(overrides)
    return BaseGLiNER.create_training_args(**defaults)


def _get_train_model_signature():
    """Return the signature of BaseGLiNER.train_model."""
    from gliner.model import BaseGLiNER

    return inspect.signature(BaseGLiNER.train_model)


def _get_train_model_source():
    """Return the source code of BaseGLiNER.train_model."""
    from gliner.model import BaseGLiNER

    return inspect.getsource(BaseGLiNER.train_model)


# ===========================================================================
# 1. Smoke tests  --  these MUST pass
# ===========================================================================


class TestSmoke:
    """Basic smoke tests for training module imports and instantiation."""

    def test_training_args_import(self):
        from gliner.training import TrainingArguments as TA

        assert TA is not None

    def test_trainer_import(self):
        from gliner.training import Trainer as T

        assert T is not None

    def test_training_args_instantiation(self, tmp_path):
        args = _make_training_args(output_dir=tmp_path)
        assert args.output_dir == str(tmp_path)

    def test_create_training_args_callable(self):
        from gliner.model import BaseGLiNER

        assert callable(BaseGLiNER.create_training_args)

    def test_create_training_args_returns_training_args(self, tmp_path):
        args = _create_training_args_via_classmethod(output_dir=tmp_path)
        assert isinstance(args, TrainingArguments)


# ===========================================================================
# 2. TrainingArguments defaults  --  these MUST pass
# ===========================================================================


class TestTrainingArgumentsDefaults:
    """Verify TrainingArguments field defaults match expected values."""

    def test_masking_default_is_global(self, tmp_path):
        """TrainingArguments.masking should default to 'global'."""
        args = _make_training_args(output_dir=tmp_path)
        assert args.masking == "global"

    def test_label_smoothing_field_exists(self):
        field_names = _get_training_args_field_names()
        assert "label_smoothing" in field_names

    def test_loss_reduction_default_is_sum(self, tmp_path):
        args = _make_training_args(output_dir=tmp_path)
        assert args.loss_reduction == "sum"

    def test_focal_loss_defaults(self, tmp_path):
        args = _make_training_args(output_dir=tmp_path)
        assert args.focal_loss_alpha == -1
        assert args.focal_loss_gamma == 0
        assert args.focal_loss_prob_margin == 0


# ===========================================================================
# 3. TrainingArguments custom fields completeness  --  these MUST pass
# ===========================================================================


class TestTrainingArgumentsFieldCompleteness:
    """Verify all GLiNER-specific fields are properly declared."""

    REQUIRED_GLINER_FIELDS = {
        "focal_loss_alpha",
        "focal_loss_gamma",
        "focal_loss_prob_margin",
        "label_smoothing",
        "loss_reduction",
        "negatives",
        "masking",
        "others_lr",
        "others_weight_decay",
    }

    def test_all_gliner_fields_declared(self):
        field_names = _get_training_args_field_names()
        missing = self.REQUIRED_GLINER_FIELDS - field_names
        assert not missing, f"Missing GLiNER-specific fields: {missing}"

    def test_gliner_fields_have_defaults(self, tmp_path):
        """All GLiNER-specific fields should have defaults (instantiation with only output_dir)."""
        try:
            _make_training_args(output_dir=tmp_path)
        except TypeError as e:
            pytest.fail(f"TrainingArguments cannot be instantiated with defaults only: {e}")


# ===========================================================================
# 4. Compute loss wiring  --  this MUST pass
# ===========================================================================


class TestComputeLossWiring:
    """Verify Trainer.compute_loss correctly forwards all loss parameters."""

    def test_compute_loss_passes_all_loss_params(self):
        source = inspect.getsource(Trainer.compute_loss)
        required_forwards = [
            ("alpha", "self.args.focal_loss_alpha"),
            ("gamma", "self.args.focal_loss_gamma"),
            ("prob_margin", "self.args.focal_loss_prob_margin"),
            ("label_smoothing", "self.args.label_smoothing"),
            ("reduction", "self.args.loss_reduction"),
            ("negatives", "self.args.negatives"),
            ("masking", "self.args.masking"),
        ]
        missing = []
        for param_name, attr_path in required_forwards:
            if attr_path not in source:
                missing.append(f"{param_name} (via {attr_path})")
        assert not missing, f"compute_loss does not forward: {', '.join(missing)}"


# ===========================================================================
# 5. Trainer dataloader wiring  --  these MUST pass
# ===========================================================================


class TestTrainerDataloaderWiring:
    """Verify the Trainer correctly uses dataloader configuration from args."""

    def test_get_train_dataloader_uses_pin_memory(self):
        source = inspect.getsource(Trainer.get_train_dataloader)
        assert "dataloader_pin_memory" in source

    def test_get_train_dataloader_uses_persistent_workers(self):
        source = inspect.getsource(Trainer.get_train_dataloader)
        assert "dataloader_persistent_workers" in source

    def test_get_train_dataloader_uses_prefetch_factor(self):
        source = inspect.getsource(Trainer.get_train_dataloader)
        assert "dataloader_prefetch_factor" in source


# ===========================================================================
# 6. kwargs pass-through integrity  --  these MUST pass
# ===========================================================================


class TestKwargsPassThrough:
    """Verify kwargs passed to create_training_args reach TrainingArguments."""

    def test_label_smoothing_via_kwargs_reaches_training_args(self, tmp_path):
        args = _create_training_args_via_classmethod(
            output_dir=tmp_path, label_smoothing=0.1
        )
        assert args.label_smoothing == 0.1

    def test_fp16_via_kwargs_reaches_training_args(self, tmp_path):
        args = _create_training_args_via_classmethod(output_dir=tmp_path, fp16=True)
        assert args.fp16 is True

    def test_seed_via_kwargs_reaches_training_args(self, tmp_path):
        args = _create_training_args_via_classmethod(output_dir=tmp_path, seed=42)
        assert args.seed == 42


# ===========================================================================
# 7. HF default for remove_unused_columns  --  documents the risk
# ===========================================================================


class TestHFDefaults:
    """Document HF default values that create risk for GLiNER."""

    def test_hf_remove_unused_columns_defaults_to_true(self, tmp_path):
        """HF Trainer defaults remove_unused_columns to True.

        This is dangerous for GLiNER which uses custom batch dictionaries.
        """
        hf_args = transformers.TrainingArguments(output_dir=str(tmp_path), report_to="none")
        assert hf_args.remove_unused_columns is True

    def test_hf_label_smoothing_factor_defaults_to_zero(self, tmp_path):
        """HF label_smoothing_factor defaults to 0 (no double smoothing by default)."""
        args = _make_training_args(output_dir=tmp_path, label_smoothing=0.1)
        hf_ls = getattr(args, "label_smoothing_factor", 0)
        assert hf_ls == 0


# ===========================================================================
#
#   BUG-CATCHING TESTS  --  xfail(strict=True)
#
#   These tests assert the CORRECT behavior. They currently fail because the
#   code has known issues. When an issue is fixed, the test will XPASS and
#   the marker should be removed.
#
# ===========================================================================


# ---------------------------------------------------------------------------
# 8. create_training_args explicit parameter coverage gaps
# ---------------------------------------------------------------------------


class TestCreateTrainingArgsSignature:
    """Verify create_training_args has explicit parameters for critical fields.

    Fields relying on **kwargs are fragile: not discoverable, not documented
    in the signature, and can silently break if TrainingArguments changes.
    """

    @pytest.fixture(scope="class")
    def explicit_params(self):
        """
        Return the set of parameter names explicitly declared by BaseGLiNER.create_training_args.
        
        Returns:
            set[str]: Parameter names that are explicitly declared on the classmethod, excluding `self`, `cls`, `*args`, and `**kwargs`.
        """
        return _get_create_training_args_explicit_params()

    def test_label_smoothing_is_explicit(self, explicit_params):
        assert "label_smoothing" in explicit_params, (
            "label_smoothing is not an explicit parameter in create_training_args; "
            "it relies on **kwargs pass-through which is fragile"
        )

    def test_fp16_is_explicit(self, explicit_params):
        assert "fp16" in explicit_params, (
            "fp16 is not explicit in create_training_args; only bf16 is"
        )

    def test_seed_is_explicit(self, explicit_params):
        assert "seed" in explicit_params

    def test_gradient_checkpointing_is_explicit(self, explicit_params):
        assert "gradient_checkpointing" in explicit_params

    def test_run_name_is_explicit(self, explicit_params):
        assert "run_name" in explicit_params

    def test_push_to_hub_is_explicit(self, explicit_params):
        """
        Asserts that BaseGLiNER.create_training_args exposes `push_to_hub` as an explicit parameter.
        
        Parameters:
            explicit_params (set[str]): Names of parameters explicitly declared on `create_training_args` (excluding `cls`, `self`, `*args`, and `**kwargs`).
        """
        assert "push_to_hub" in explicit_params

    def test_hub_model_id_is_explicit(self, explicit_params):
        """
        Asserts that "hub_model_id" is declared as an explicit parameter of BaseGLiNER.create_training_args.
        
        Parameters:
            explicit_params (set[str]): Set of parameter names explicitly declared on the classmethod signature (excluding `self`, `cls`, `*args`, and `**kwargs`).
        """
        assert "hub_model_id" in explicit_params

    @pytest.mark.xfail(
        strict=True,
        reason="evaluation_strategy/eval_strategy not explicit; evaluation never runs",
    )
    def test_evaluation_strategy_is_explicit(self, explicit_params):
        has_eval = (
            "evaluation_strategy" in explicit_params
            or "eval_strategy" in explicit_params
        )
        assert has_eval, (
            "Neither evaluation_strategy nor eval_strategy is explicit; "
            "evaluation never runs during training"
        )

    def test_eval_steps_is_explicit(self, explicit_params):
        """
        Asserts that "eval_steps" is listed among the explicit parameters returned by create_training_args.
        
        Parameters:
            explicit_params (set[str]): Set of explicit parameter names extracted from BaseGLiNER.create_training_args.
        """
        assert "eval_steps" in explicit_params


# ---------------------------------------------------------------------------
# 9. Masking default mismatch
# ---------------------------------------------------------------------------


class TestMaskingDefaultMismatch:
    """Detect default mismatch between create_training_args and TrainingArguments."""

    def test_create_training_args_masking_matches_training_args_default(self):
        """
        Ensure the default 'masking' value in BaseGLiNER.create_training_args matches the default on TrainingArguments.
        
        Compares the 'masking' parameter default from BaseGLiNER.create_training_args's signature with the TrainingArguments dataclass field default and fails the test if they differ.
        """
        # Actually inspect the real method
        from gliner.model import BaseGLiNER

        real_sig = inspect.signature(BaseGLiNER.create_training_args)
        cta_default = real_sig.parameters["masking"].default

        ta_fields = {f.name: f for f in dataclass_fields(TrainingArguments)}
        ta_default = ta_fields["masking"].default

        assert cta_default == ta_default, (
            f"Masking default mismatch: create_training_args='{cta_default}' "
            f"vs TrainingArguments='{ta_default}'"
        )


# ---------------------------------------------------------------------------
# 10. remove_unused_columns not set to False
# ---------------------------------------------------------------------------


class TestRemoveUnusedColumns:
    """GLiNER uses custom batch dicts that require remove_unused_columns=False."""

    def test_create_training_args_sets_remove_unused_columns_false(self, tmp_path):
        """
        Verify that create_training_args sets remove_unused_columns to False.
        
        Asserts that a TrainingArguments instance produced via BaseGLiNER.create_training_args has
        remove_unused_columns == False so GLiNER's custom batch dictionary keys are preserved.
        """
        args = _create_training_args_via_classmethod(output_dir=tmp_path)
        assert args.remove_unused_columns is False, (
            f"remove_unused_columns is {args.remove_unused_columns}. "
            f"GLiNER needs False to preserve custom batch dictionary keys."
        )

    def test_create_training_args_defaults_remove_unused_columns_false(self, tmp_path):
        """
        Verify create_training_args preserves GLiNER batch keys by defaulting remove_unused_columns to False.
        
        Checks that the classmethod-produced TrainingArguments sets remove_unused_columns to False (HF's default is True, which can cause GLiNER's custom batch dictionary keys to be dropped).
        """
        args = _create_training_args_via_classmethod(output_dir=tmp_path)
        assert args.remove_unused_columns is False, (
            "create_training_args should default remove_unused_columns to False. "
            "HF defaults to True which can cause silent data loss."
        )


# ---------------------------------------------------------------------------
# 11. Evaluation strategy not configured
# ---------------------------------------------------------------------------


class TestEvaluationConfiguration:
    """Verify that evaluation is properly configured during training."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "create_training_args never sets evaluation_strategy='steps'; "
            "evaluation never runs during training"
        ),
    )
    def test_create_training_args_enables_evaluation(self, tmp_path):
        args = _create_training_args_via_classmethod(output_dir=tmp_path, save_steps=500)
        eval_strategy = getattr(args, "eval_strategy", None) or getattr(
            args, "evaluation_strategy", None
        )
        assert eval_strategy == "steps", (
            f"evaluation_strategy is '{eval_strategy}', not 'steps'. "
            f"Evaluation never runs."
        )

    @pytest.mark.xfail(
        strict=True,
        reason="eval_steps is never set by create_training_args",
    )
    def test_create_training_args_forwards_eval_steps(self, tmp_path):
        args = _create_training_args_via_classmethod(output_dir=tmp_path, save_steps=500)
        eval_steps = getattr(args, "eval_steps", None)
        assert eval_steps is not None and eval_steps > 0, (
            f"eval_steps is {eval_steps}; evaluation won't run at a meaningful frequency"
        )


# ---------------------------------------------------------------------------
# 12. Dead config fields (size_sup, shuffle_types, random_drop)
# ---------------------------------------------------------------------------


class TestDeadConfigFields:
    """Config fields that exist but are never consumed by any code."""

    @pytest.fixture(scope="class")
    def gliner_source_dir(self):
        return Path(__file__).parent.parent / "gliner"

    def _field_consumed_in_source(self, field_name, source_dir):
        for py_file in source_dir.rglob("*.py"):
            try:
                content = py_file.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            if field_name in content:
                return True
        return False

    @pytest.mark.xfail(
        strict=True,
        reason="size_sup is in config.yaml but never consumed by gliner/",
    )
    def test_size_sup_is_consumed(self, gliner_source_dir):
        assert self._field_consumed_in_source("size_sup", gliner_source_dir)

    @pytest.mark.xfail(
        strict=True,
        reason="shuffle_types is in config.yaml but never consumed by gliner/",
    )
    def test_shuffle_types_is_consumed(self, gliner_source_dir):
        assert self._field_consumed_in_source("shuffle_types", gliner_source_dir)

    @pytest.mark.xfail(
        strict=True,
        reason="random_drop is in config.yaml but never consumed by gliner/",
    )
    def test_random_drop_is_consumed(self, gliner_source_dir):
        assert self._field_consumed_in_source("random_drop", gliner_source_dir)


# ---------------------------------------------------------------------------
# 13. label_smoothing name collision
# ---------------------------------------------------------------------------


class TestLabelSmoothingCollision:
    """GLiNER's label_smoothing and HF's label_smoothing_factor can collide."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "TrainingArguments inherits label_smoothing_factor from HF AND "
            "adds label_smoothing; both can be active simultaneously"
        ),
    )
    def test_no_dual_label_smoothing_fields(self, tmp_path):
        """TrainingArguments should not have BOTH label_smoothing AND
        label_smoothing_factor, since they represent two different smoothing
        mechanisms that can accidentally combine.
        """
        args = _make_training_args(output_dir=tmp_path)
        has_gliner_ls = hasattr(args, "label_smoothing")
        has_hf_ls = hasattr(args, "label_smoothing_factor")
        assert not (has_gliner_ls and has_hf_ls), (
            "TrainingArguments has BOTH 'label_smoothing' (GLiNER, applied "
            "in model forward) AND 'label_smoothing_factor' (HF, applied in "
            "Trainer.compute_loss). Double smoothing risk."
        )


# ---------------------------------------------------------------------------
# 14. train_model does not support resume_from_checkpoint
# ---------------------------------------------------------------------------


class TestTrainModelResumeSupport:
    """train_model should support checkpoint resumption."""

    def test_train_model_accepts_resume_from_checkpoint(self):
        sig = _get_train_model_signature()
        param_names = set(sig.parameters.keys())
        assert "resume_from_checkpoint" in param_names

    def test_train_model_forwards_resume_to_trainer(self):
        source = _get_train_model_source()
        assert "resume_from_checkpoint" in source


# ---------------------------------------------------------------------------
# 15. create_training_args should expose all custom TrainingArguments fields
# ---------------------------------------------------------------------------


class TestCreateTrainingArgsCoversCustomFields:
    """Every custom field on TrainingArguments should be explicit in
    create_training_args, not buried in **kwargs.
    """

    CUSTOM_FIELDS = {
        "others_lr",
        "others_weight_decay",
        "focal_loss_alpha",
        "focal_loss_gamma",
        "focal_loss_prob_margin",
        "label_smoothing",
        "loss_reduction",
        "negatives",
        "masking",
    }

    def test_all_custom_fields_are_explicit_params(self):
        explicit = _get_create_training_args_explicit_params()
        missing = self.CUSTOM_FIELDS - explicit
        assert not missing, (
            f"These custom TrainingArguments fields are not explicit parameters "
            f"in create_training_args: {missing}. They rely on **kwargs."
        )


# ---------------------------------------------------------------------------
# 16. Config YAML dead field detection (aggregate)
# ---------------------------------------------------------------------------


class TestConfigYamlDeadFields:
    """Verify all training fields in config.yaml are consumed by code."""

    @pytest.fixture(scope="class")
    def config_yaml_path(self):
        return Path(__file__).parent.parent / "configs" / "config.yaml"

    @pytest.fixture(scope="class")
    def config_training_fields(self, config_yaml_path):
        if not config_yaml_path.exists():
            pytest.skip("configs/config.yaml not found")
        yaml = pytest.importorskip("yaml")
        with config_yaml_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        training_config = config.get("training", {})
        if not isinstance(training_config, dict):
            return []

        return [field_name for field_name in training_config if isinstance(field_name, str)]

    @pytest.mark.xfail(
        strict=True,
        reason="size_sup, shuffle_types, random_drop are dead fields in config.yaml",
    )
    def test_all_config_training_fields_have_consumers(self, config_training_fields):
        gliner_dir = Path(__file__).parent.parent / "gliner"
        dead_fields = []
        for field_name in config_training_fields:
            found = False
            for py_file in gliner_dir.rglob("*.py"):
                try:
                    content = py_file.read_text()
                except (OSError, UnicodeDecodeError):
                    continue
                if field_name in content:
                    found = True
                    break
            if not found:
                dead_fields.append(field_name)
        assert not dead_fields, (
            f"Dead config.yaml training fields: {dead_fields}"
        )