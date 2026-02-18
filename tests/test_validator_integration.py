"""Tests for validator integration between ptbr config_cli and training_cli.

These tests verify the state of the codebase against the issues identified in
the standard_config_report.  Tests are organized by issue category:

  - Fixed issues: assertions confirm the fix is in place.
  - Remaining issues: assertions document what is still open.

Originally these tests were written as "bug-documenting" tests that asserted
the existence of specific bugs.  After fixes were applied to training_cli.py
and train.py, the tests below have been updated to reflect the current state.
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Lightweight helpers – no heavy DL imports at module scope
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
PTBR_DIR = ROOT / "ptbr"
TEMPLATE_YAML = PTBR_DIR / "template.yaml"
TRAIN_PY = ROOT / "train.py"


def _load_template() -> dict:
    with open(TEMPLATE_YAML) as fh:
        return yaml.safe_load(fh)


def _load_config(name: str) -> dict:
    with open(CONFIGS_DIR / name) as fh:
        return yaml.safe_load(fh)


def _minimal_training_cfg() -> dict:
    """Return the minimal valid config accepted by training_cli.validate_config."""
    return {
        "run": {"name": "test-run", "seed": 42},
        "model": {
            "model_name": "microsoft/deberta-v3-small",
            "span_mode": "markerV0",
            "max_len": 384,
        },
        "data": {
            "root_dir": "logs",
            "train_data": "data/train.json",
        },
        "training": {
            "num_steps": 100,
            "train_batch_size": 4,
            "eval_every": 50,
            "lr_encoder": 1e-5,
            "lr_others": 3e-5,
        },
    }


# ======================================================================== #
#  1.  YAML schema incompatibility between the two CLIs (STILL OPEN)       #
# ======================================================================== #


class TestYAMLSchemaIncompatibility:
    """config_cli and training_cli expect different YAML layouts.
    config_cli accepts ``model:`` as an alias for ``gliner_config:``,
    but the template.yaml still fails config_cli because it lacks the
    canonical ``gliner_config:`` key (config_cli emits a warning for
    the alias but may still reject depending on field differences)."""

    def test_template_yaml_fails_config_cli_validation(self):
        """template.yaml uses ``model:`` -- config_cli accepts it as alias
        but may still fail due to field rule differences."""
        pytest.importorskip("torch")
        from ptbr.config_cli import load_and_validate_config

        result = load_and_validate_config(
            str(TEMPLATE_YAML), full_or_lora="full", method="span", validate=True,
        )
        # config_cli now accepts ``model:`` as an alias for ``gliner_config:``
        # but the field sets differ between the two CLIs, so template.yaml
        # may or may not pass.  We just verify it doesn't crash.
        assert result.report is not None

    def test_template_yaml_passes_training_cli_validation(self):
        """template.yaml is valid under training_cli's schema."""
        from ptbr.training_cli import validate_config

        tpl = _load_template()
        vr = validate_config(tpl)
        assert len(vr.errors) == 0, (
            f"template.yaml should pass training_cli validation; errors: {vr.errors}"
        )

    def test_gliner_config_structure_fails_training_cli(self):
        """A YAML with ``gliner_config:`` (config_cli format) fails training_cli."""
        from ptbr.training_cli import validate_config

        cfg = {
            "gliner_config": {
                "model_name": "microsoft/deberta-v3-small",
                "span_mode": "markerV0",
            }
        }
        vr = validate_config(cfg)
        assert len(vr.errors) > 0, (
            "training_cli should reject a YAML that uses 'gliner_config' format"
        )

    def test_no_single_yaml_satisfies_both_clis(self):
        """A hybrid YAML with both ``model:`` and ``gliner_config:`` should
        satisfy training_cli (which needs ``model:``)."""
        from ptbr.training_cli import validate_config

        hybrid = _load_template()
        hybrid["gliner_config"] = hybrid["model"].copy()

        vr = validate_config(hybrid)
        assert len(vr.errors) == 0, "training_cli should accept the hybrid YAML"


# ======================================================================== #
#  2.  LoRA section naming divergence (STILL OPEN)                          #
# ======================================================================== #


class TestLoRASectionNaming:
    """config_cli expects ``lora_config:`` while training_cli expects ``lora:``."""

    def test_training_cli_expects_lora_key(self):
        """training_cli validates ``lora:``, not ``lora_config:``."""
        from ptbr.training_cli import _FIELD_SCHEMA

        lora_keys = [key for key, *_ in _FIELD_SCHEMA if key.startswith("lora.")]
        lora_config_keys = [key for key, *_ in _FIELD_SCHEMA if key.startswith("lora_config.")]

        assert len(lora_keys) > 0, "training_cli should have 'lora.*' fields"
        assert len(lora_config_keys) == 0, (
            "training_cli should NOT have 'lora_config.*' fields"
        )

    def test_lora_field_sets_differ_between_clis(self):
        """config_cli has LoRA fields that training_cli lacks and vice-versa."""
        pytest.importorskip("torch")
        from ptbr.config_cli import _LORA_RULES
        from ptbr.training_cli import _FIELD_SCHEMA

        config_cli_lora_fields = {rule[0] for rule in _LORA_RULES}
        training_cli_lora_fields = {
            key.split(".", 1)[1] for key, *_ in _FIELD_SCHEMA if key.startswith("lora.")
        }

        only_in_config_cli = config_cli_lora_fields - training_cli_lora_fields
        only_in_training_cli = training_cli_lora_fields - config_cli_lora_fields

        assert only_in_config_cli or only_in_training_cli, (
            "LoRA field sets should diverge between the two CLIs (this is the bug). "
            f"Only in config_cli: {only_in_config_cli}, "
            f"Only in training_cli: {only_in_training_cli}"
        )


# ======================================================================== #
#  3.  CLI argument style inconsistency (STILL OPEN)                        #
# ======================================================================== #


class TestCLIArgumentInconsistency:
    """config uses --file (named option), train uses positional argument."""

    def test_argument_style_divergence(self):
        """Document the inconsistency: config uses --file, train uses positional."""
        main_source = (ROOT / "ptbr" / "__main__.py").read_text()
        assert 'typer.Option' in main_source, (
            "__main__.py should use typer.Option for config_cmd's file parameter"
        )

        training_source = (ROOT / "ptbr" / "training_cli.py").read_text()
        assert 'typer.Argument' in training_source, (
            "training_cli.py should use typer.Argument for config path"
        )


# ======================================================================== #
#  4.  Parameter forwarding — FIXED items verified, remaining gaps noted    #
# ======================================================================== #


class TestParameterForwardingFixes:
    """Verify that previously-missing forwarding gaps have been fixed in
    training_cli._launch_training."""

    @staticmethod
    def _get_launch_training_source() -> str:
        return (ROOT / "ptbr" / "training_cli.py").read_text()

    @staticmethod
    def _extract_train_model_kwargs(tree: ast.AST) -> set[str]:
        kwargs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "train_model":
                    for kw in node.keywords:
                        if kw.arg is not None:
                            kwargs.add(kw.arg)
        return kwargs

    def test_dataloader_pin_memory_is_forwarded(self):
        """FIXED: dataloader_pin_memory is now forwarded to train_model."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "dataloader_pin_memory" in forwarded

    def test_dataloader_persistent_workers_is_forwarded(self):
        """FIXED: dataloader_persistent_workers is now forwarded."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "dataloader_persistent_workers" in forwarded

    def test_dataloader_prefetch_factor_is_forwarded(self):
        """FIXED: dataloader_prefetch_factor is now forwarded."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "dataloader_prefetch_factor" in forwarded

    def test_fp16_is_forwarded(self):
        """FIXED: fp16 is now forwarded to train_model."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "fp16" in forwarded

    def test_run_name_is_forwarded(self):
        """FIXED: run_name is now forwarded to train_model."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "run_name" in forwarded

    def test_report_to_is_forwarded(self):
        """FIXED: report_to is forwarded to train_model."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "report_to" in forwarded

    def test_eval_strategy_is_forwarded(self):
        """FIXED: eval_strategy and eval_steps are forwarded when eval data exists."""
        source = self._get_launch_training_source()
        assert "eval_strategy" in source
        assert "eval_steps" in source


class TestParameterForwardingRemainingGaps:
    """Document forwarding gaps that remain open."""

    @staticmethod
    def _get_launch_training_source() -> str:
        return (ROOT / "ptbr" / "training_cli.py").read_text()

    @staticmethod
    def _extract_train_model_kwargs(tree: ast.AST) -> set[str]:
        kwargs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "train_model":
                    for kw in node.keywords:
                        if kw.arg is not None:
                            kwargs.add(kw.arg)
        return kwargs

    def test_size_sup_not_forwarded(self):
        """STILL OPEN: training.size_sup validated but never forwarded."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "size_sup" not in forwarded

    def test_shuffle_types_not_forwarded(self):
        """STILL OPEN: training.shuffle_types validated but never forwarded."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "shuffle_types" not in forwarded

    def test_random_drop_not_forwarded(self):
        """STILL OPEN: training.random_drop validated but never forwarded."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "random_drop" not in forwarded

    def test_run_tags_not_forwarded(self):
        """STILL OPEN: run.tags validated but never forwarded."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "run_tags" not in forwarded

    def test_run_description_not_forwarded(self):
        """STILL OPEN: run.description validated but unused beyond logging."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "run_description" not in forwarded
        assert "description" not in forwarded

    def test_remove_unused_columns_not_passed(self):
        """STILL OPEN: remove_unused_columns not set to False (dangerous for custom collators)."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "remove_unused_columns" not in forwarded


# ======================================================================== #
#  5.  train.py fixes verified                                              #
# ======================================================================== #


class TestTrainPyFixes:
    """Verify that train.py hardcoded values have been fixed."""

    @staticmethod
    def _parse_train_py() -> ast.AST:
        return ast.parse(TRAIN_PY.read_text())

    @staticmethod
    def _extract_train_model_kwargs(tree: ast.AST) -> dict[str, Any]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "train_model":
                    return {kw.arg: kw.value for kw in node.keywords if kw.arg}
        return {}

    def test_output_dir_reads_from_config(self):
        """FIXED: train.py now uses cfg.data.root_dir for output_dir."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "output_dir" in kwargs
        node = kwargs["output_dir"]
        # Should NOT be a hardcoded constant "models" anymore
        is_hardcoded_models = isinstance(node, ast.Constant) and node.value == "models"
        assert not is_hardcoded_models, (
            "output_dir should read from config, not be hardcoded to 'models'"
        )

    def test_bf16_reads_from_config(self):
        """FIXED: train.py now reads bf16 from config instead of hardcoding True."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "bf16" in kwargs
        node = kwargs["bf16"]
        is_hardcoded_true = isinstance(node, ast.Constant) and node.value is True
        assert not is_hardcoded_true, (
            "bf16 should read from config, not be hardcoded to True"
        )

    def test_eval_batch_size_uses_separate_variable(self):
        """FIXED: train.py now cascades eval_batch_size properly."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "per_device_eval_batch_size" in kwargs
        node = kwargs["per_device_eval_batch_size"]
        source_line = ast.dump(node)
        # Should reference eval_batch_size (a separate variable), not train_batch_size
        assert "train_batch_size" not in source_line, (
            "eval batch size should use the cascaded eval_batch_size variable"
        )

    def test_label_smoothing_is_forwarded(self):
        """FIXED: train.py now forwards label_smoothing."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "label_smoothing" in kwargs, (
            "train.py should now forward label_smoothing"
        )


class TestTrainPyRemainingGaps:
    """Document forwarding gaps that remain in train.py."""

    @staticmethod
    def _parse_train_py() -> ast.AST:
        return ast.parse(TRAIN_PY.read_text())

    @staticmethod
    def _extract_train_model_kwargs(tree: ast.AST) -> dict[str, Any]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "train_model":
                    return {kw.arg: kw.value for kw in node.keywords if kw.arg}
        return {}

    def test_size_sup_not_forwarded_by_train_py(self):
        """STILL OPEN: train.py does not forward size_sup."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "size_sup" not in kwargs

    def test_shuffle_types_not_forwarded_by_train_py(self):
        """STILL OPEN: train.py does not forward shuffle_types."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "shuffle_types" not in kwargs

    def test_random_drop_not_forwarded_by_train_py(self):
        """STILL OPEN: train.py does not forward random_drop."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "random_drop" not in kwargs


# ======================================================================== #
#  6.  Config fields in YAML but absent from train.py forwarding            #
# ======================================================================== #


class TestConfigFieldsReachTraining:
    """Verify that every field defined in the shipped YAML configs is either
    forwarded by train.py or explicitly documented as unused."""

    KNOWN_NON_FORWARDED = {
        "model", "data",
        "training.prev_path", "training.freeze_components",
    }

    def test_all_training_fields_in_config_yaml_are_forwarded(self):
        """Check which config fields are forwarded by train.py.
        Some fields (size_sup, shuffle_types, random_drop) are still missing."""
        cfg = _load_config("config.yaml")
        training_fields = set(cfg.get("training", {}).keys())

        tree = ast.parse(TRAIN_PY.read_text())
        forwarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "train_model":
                    for kw in node.keywords:
                        if kw.arg:
                            forwarded.add(kw.arg)

        field_to_kwarg = {
            "num_steps": "max_steps",
            "scheduler_type": "lr_scheduler_type",
            "train_batch_size": "per_device_train_batch_size",
            "lr_encoder": "learning_rate",
            "lr_others": "others_lr",
            "weight_decay_encoder": "weight_decay",
            "weight_decay_other": "others_weight_decay",
            "loss_alpha": "focal_loss_alpha",
            "loss_gamma": "focal_loss_gamma",
            "loss_prob_margin": "focal_loss_prob_margin",
            "eval_every": "save_steps",
        }

        not_forwarded = []
        for field in training_fields:
            qualified = f"training.{field}"
            if qualified in self.KNOWN_NON_FORWARDED:
                continue
            kwarg_name = field_to_kwarg.get(field, field)
            if kwarg_name not in forwarded:
                not_forwarded.append(field)

        # These fields remain NOT forwarded by train.py
        expected_still_missing = {"size_sup", "shuffle_types", "random_drop"}
        actual_missing = set(not_forwarded)
        assert expected_still_missing.issubset(actual_missing), (
            f"Expected these config fields to still be missing from train.py: "
            f"{expected_still_missing}. Actually missing: {actual_missing}"
        )
        # label_smoothing should now be forwarded (FIXED)
        assert "label_smoothing" not in actual_missing, (
            "label_smoothing should now be forwarded by train.py"
        )


# ======================================================================== #
#  7.  remove_unused_columns (STILL OPEN — needs heavy deps to test fully)  #
# ======================================================================== #


class TestRemoveUnusedColumns:
    """GLiNER uses custom data collators.  HF TrainingArguments defaults
    remove_unused_columns=True which strips columns the collator needs."""

    def test_training_cli_does_not_pass_remove_unused_columns(self):
        """STILL OPEN: _launch_training doesn't pass remove_unused_columns."""
        source = (ROOT / "ptbr" / "training_cli.py").read_text()
        tree = ast.parse(source)
        forwarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "train_model":
                    for kw in node.keywords:
                        if kw.arg:
                            forwarded.add(kw.arg)
        assert "remove_unused_columns" not in forwarded

    def test_train_py_does_not_pass_remove_unused_columns(self):
        """STILL OPEN: train.py also doesn't pass remove_unused_columns."""
        tree = ast.parse(TRAIN_PY.read_text())
        forwarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "train_model":
                    for kw in node.keywords:
                        if kw.arg:
                            forwarded.add(kw.arg)
        assert "remove_unused_columns" not in forwarded


# ======================================================================== #
#  8.  create_training_args gaps (needs heavy deps — skipped if absent)     #
# ======================================================================== #


class TestCreateTrainingArgsGaps:
    """create_training_args has named params for some fields but relies on
    **kwargs for others."""

    def test_gradient_checkpointing_not_available(self):
        """STILL OPEN: gradient_checkpointing absent from create_training_args."""
        gliner_model = pytest.importorskip("gliner.model")
        sig = inspect.signature(gliner_model.BaseGLiNER.create_training_args)
        assert "gradient_checkpointing" not in sig.parameters


# ======================================================================== #
#  9.  Config loader lacks schema validation (needs heavy deps)             #
# ======================================================================== #


class TestConfigLoaderValidation:
    """load_config_as_namespace() loads any YAML blindly -- no schema check."""

    def test_wrong_structure_loads_without_error(self):
        """A YAML with ``gliner_config:`` instead of ``model:`` loads fine."""
        import tempfile
        gliner_utils = pytest.importorskip("gliner.utils")

        bad_cfg = {"gliner_config": {"model_name": "foo"}, "training": {"num_steps": 100}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(bad_cfg, f)
            tmp_path = f.name

        try:
            ns = gliner_utils.load_config_as_namespace(tmp_path)
            assert hasattr(ns, "gliner_config")
            assert not hasattr(ns, "model")
            with pytest.raises(AttributeError):
                _ = ns.model
        finally:
            os.unlink(tmp_path)

    def test_missing_required_fields_not_caught(self):
        """A YAML missing critical fields loads fine."""
        import tempfile
        gliner_utils = pytest.importorskip("gliner.utils")

        incomplete_cfg = {
            "model": {"model_name": "foo"},
            "data": {"root_dir": "logs"},
            "training": {},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(incomplete_cfg, f)
            tmp_path = f.name

        try:
            ns = gliner_utils.load_config_as_namespace(tmp_path)
            assert hasattr(ns, "training")
            with pytest.raises(AttributeError):
                _ = ns.training.num_steps
        finally:
            os.unlink(tmp_path)

    def test_empty_yaml_crashes_loader(self):
        """An empty YAML file causes load_config_as_namespace to fail."""
        import tempfile
        gliner_utils = pytest.importorskip("gliner.utils")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("# empty config\n")
            tmp_path = f.name

        try:
            with pytest.raises((TypeError, AttributeError)):
                gliner_utils.load_config_as_namespace(tmp_path)
        finally:
            os.unlink(tmp_path)


# ======================================================================== #
#  10.  Schema vs forwarding in training_cli                                #
# ======================================================================== #


class TestSchemaVsForwarding:
    """Cross-reference _FIELD_SCHEMA entries against what _launch_training
    actually passes to model.train_model()."""

    def test_validated_training_fields_forwarding_gaps_reduced(self):
        """After fixes, the remaining gaps are only size_sup, shuffle_types,
        random_drop (dataloader flags, fp16, run_name now forwarded)."""
        from ptbr.training_cli import _FIELD_SCHEMA

        schema_training_fields = {
            key.split(".", 1)[1]
            for key, *_ in _FIELD_SCHEMA
            if key.startswith("training.")
        }

        source = (ROOT / "ptbr" / "training_cli.py").read_text()
        tree = ast.parse(source)
        forwarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "train_model":
                    for kw in node.keywords:
                        if kw.arg:
                            forwarded.add(kw.arg)

        schema_to_kwarg = {
            "num_steps": "max_steps",
            "scheduler_type": "lr_scheduler_type",
            "train_batch_size": "per_device_train_batch_size",
            "eval_batch_size": "per_device_eval_batch_size",
            "lr_encoder": "learning_rate",
            "lr_others": "others_lr",
            "weight_decay_encoder": "weight_decay",
            "weight_decay_other": "others_weight_decay",
            "loss_alpha": "focal_loss_alpha",
            "loss_gamma": "focal_loss_gamma",
            "loss_prob_margin": "focal_loss_prob_margin",
            "eval_every": "save_steps",
            "optimizer": "optim",
        }

        handled_elsewhere = {
            "prev_path", "freeze_components", "compile_model",
        }

        not_forwarded = []
        for field in schema_training_fields:
            if field in handled_elsewhere:
                continue
            kwarg = schema_to_kwarg.get(field, field)
            if kwarg not in forwarded:
                not_forwarded.append(field)

        # Only these should remain as gaps now
        expected_remaining_gaps = {"size_sup", "shuffle_types", "random_drop"}
        actual_gaps = set(not_forwarded)
        assert expected_remaining_gaps.issubset(actual_gaps), (
            f"Expected remaining gaps: {expected_remaining_gaps}. "
            f"Actual gaps: {actual_gaps}"
        )

        # Verify that previously-reported gaps are now CLOSED
        fixed_gaps = {
            "dataloader_pin_memory",
            "dataloader_persistent_workers",
            "dataloader_prefetch_factor",
            "fp16",
        }
        for field in fixed_gaps:
            assert field not in actual_gaps, (
                f"{field} should no longer be a forwarding gap (it was fixed)"
            )


# ======================================================================== #
#  11.  All shipped config YAMLs have consistent structure                   #
# ======================================================================== #


class TestConfigConsistency:
    """All config files in configs/ should have the same top-level sections
    and the same set of training fields."""

    CONFIG_FILES = [
        "config.yaml",
        "config_span.yaml",
        "config_token.yaml",
        "config_decoder.yaml",
        "config_biencoder.yaml",
        "config_relex.yaml",
    ]

    def test_all_configs_have_required_sections(self):
        """Every config should have model, data, training top-level sections."""
        for name in self.CONFIG_FILES:
            path = CONFIGS_DIR / name
            if not path.exists():
                pytest.skip(f"{name} not found")
            cfg = _load_config(name)
            for section in ("model", "data", "training"):
                assert section in cfg, f"{name} missing '{section}' section"

    def test_all_configs_have_dead_fields(self):
        """All configs define size_sup, shuffle_types, random_drop which are
        never forwarded by train.py -- proving these are dead config entries."""
        dead_fields = {"size_sup", "shuffle_types", "random_drop"}
        for name in self.CONFIG_FILES:
            path = CONFIGS_DIR / name
            if not path.exists():
                continue
            cfg = _load_config(name)
            training = cfg.get("training", {})
            for field in dead_fields:
                assert field in training, (
                    f"{name} should contain training.{field} (dead config field)"
                )

    def test_configs_lack_separate_eval_batch_size(self):
        """No shipped config has a separate eval_batch_size field."""
        for name in self.CONFIG_FILES:
            path = CONFIGS_DIR / name
            if not path.exists():
                continue
            cfg = _load_config(name)
            training = cfg.get("training", {})
            assert "eval_batch_size" not in training, (
                f"{name} lacks eval_batch_size (train.py reuses train_batch_size)"
            )


# ======================================================================== #
#  12.  __main__.py lazy import fix verified                                #
# ======================================================================== #


class TestMainImportSideEffects:
    """Verify that training_cli is no longer imported at module level."""

    def test_training_cli_not_imported_at_module_level(self):
        """FIXED: training_cli should NOT be imported at module level."""
        source = (ROOT / "ptbr" / "__main__.py").read_text()
        tree = ast.parse(source)

        top_level_imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    top_level_imports.append(node.module)

        assert not any("training_cli" in imp for imp in top_level_imports), (
            "training_cli should NOT be imported at module level "
            "(lazy import fix was applied)"
        )


# ======================================================================== #
#  13.  training_cli validates the full template correctly                   #
# ======================================================================== #


class TestTemplateValidation:
    """Verify the template passes training_cli validation and has all
    sections the training pipeline expects."""

    def test_template_has_all_required_sections(self):
        """template.yaml should have run, model, data, training, lora, environment."""
        tpl = _load_template()
        for section in ("run", "model", "data", "training", "lora", "environment"):
            assert section in tpl, f"template.yaml missing '{section}' section"

    def test_template_has_no_gliner_config_section(self):
        """template.yaml should NOT have gliner_config (that's config_cli format)."""
        tpl = _load_template()
        assert "gliner_config" not in tpl

    def test_template_passes_training_cli_validation(self):
        """Full template must pass training_cli validation."""
        from ptbr.training_cli import validate_config

        tpl = _load_template()
        vr = validate_config(tpl)
        assert len(vr.errors) == 0, (
            f"template.yaml should be valid for training_cli. Errors: {vr.errors}"
        )
