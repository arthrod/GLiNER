"""Tests for validator integration between ptbr config_cli and training_cli.

These tests verify the integration between the two CLIs and ensure that
validated config fields reach the deep-learning training process.

Tests cover:

1. config_cli.py now accepts ``model:`` as alias for ``gliner_config:``.
2. Parameter forwarding from training_cli to train_model().
3. train.py forwarding gaps (separate from ptbr training_cli).
4. remove_unused_columns defaults to True -- wrong for custom collators.
5. LoRA section naming divergence between config_cli and training_cli.
6. CLI argument style inconsistency (--file vs positional).
"""

from __future__ import annotations

import ast
import copy
import inspect
import os
import textwrap
from pathlib import Path
from typing import Any
from unittest import mock

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
#  1.  CRITICAL – YAML schema incompatibility between the two CLIs         #
# ======================================================================== #


class TestYAMLSchemaCompatibility:
    """Verify config_cli and training_cli accept the same YAML layout
    now that config_cli supports ``model:`` as an alias for ``gliner_config:``."""

    def test_template_yaml_passes_config_cli_validation(self):
        """template.yaml uses ``model:`` which config_cli now accepts as alias.

        Previously this failed with 'Missing gliner_config section'.
        """
        from ptbr.config_cli import load_and_validate_config

        result = load_and_validate_config(
            str(TEMPLATE_YAML), full_or_lora="full", method="span", validate=True,
        )
        assert result.report.is_valid, (
            f"config_cli should accept template.yaml via 'model' alias; "
            f"errors: {[e.message for e in result.report.errors]}"
        )

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

    def test_single_yaml_satisfies_both_clis(self):
        """A single YAML with ``model:`` now works for both CLIs thanks to
        config_cli's alias support."""
        from ptbr.config_cli import load_and_validate_config
        from ptbr.training_cli import validate_config

        # training_cli should pass (it has model:)
        tpl = _load_template()
        vr = validate_config(tpl)
        assert len(vr.errors) == 0, "training_cli should accept template.yaml"

        # config_cli should also pass via alias
        result = load_and_validate_config(
            str(TEMPLATE_YAML), full_or_lora="full", method="span", validate=True,
        )
        assert result.report.is_valid, (
            f"config_cli should accept template.yaml via 'model' alias; "
            f"errors: {[e.message for e in result.report.errors]}"
        )


# ======================================================================== #
#  2.  LoRA section naming divergence                                       #
# ======================================================================== #


class TestLoRASectionNaming:
    """config_cli expects ``lora_config:`` while training_cli expects ``lora:``."""

    def test_config_cli_expects_lora_config_key(self):
        """config_cli looks for ``lora_config:``, not ``lora:``."""
        from ptbr.config_cli import load_and_validate_config

        # Build a valid gliner_config YAML *with* a ``lora:`` section (training_cli style)
        cfg = {
            "gliner_config": {"model_name": "microsoft/deberta-v3-small", "span_mode": "markerV0"},
            "lora": {"enabled": True, "r": 8, "lora_alpha": 16},
        }
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f)
            tmp_path = f.name
        try:
            result = load_and_validate_config(
                tmp_path, full_or_lora="lora", method="span", validate=True,
            )
            # config_cli should warn/error about missing lora_config since we only
            # provided ``lora:`` (training_cli format), not ``lora_config:``.
            warning_fields = [w.field for w in result.report.warnings]
            error_fields = [e.field for e in result.report.errors]
            all_fields = warning_fields + error_fields
            assert "lora_config" in all_fields, (
                "config_cli should complain about missing 'lora_config' when only "
                "'lora' (training_cli format) is provided"
            )
        finally:
            os.unlink(tmp_path)

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
        from ptbr.config_cli import _LORA_RULES
        from ptbr.training_cli import _FIELD_SCHEMA

        config_cli_lora_fields = {rule[0] for rule in _LORA_RULES}
        training_cli_lora_fields = {
            key.split(".", 1)[1] for key, *_ in _FIELD_SCHEMA if key.startswith("lora.")
        }

        # config_cli has fields training_cli lacks
        only_in_config_cli = config_cli_lora_fields - training_cli_lora_fields
        # training_cli has fields config_cli lacks
        only_in_training_cli = training_cli_lora_fields - config_cli_lora_fields

        assert only_in_config_cli or only_in_training_cli, (
            "LoRA field sets should diverge between the two CLIs (this is the bug). "
            f"Only in config_cli: {only_in_config_cli}, "
            f"Only in training_cli: {only_in_training_cli}"
        )


# ======================================================================== #
#  3.  CLI argument style inconsistency                                     #
# ======================================================================== #


class TestCLIArgumentInconsistency:
    """config uses --file (named option), train uses positional argument."""

    def test_config_subcommand_uses_named_option(self):
        """config_cmd takes ``--file`` as a named option."""
        from ptbr.__main__ import config_cmd

        sig = inspect.signature(config_cmd)
        assert "file" in sig.parameters, "config_cmd should have a 'file' parameter"

    def test_train_subcommand_uses_positional_argument(self):
        """training_cli main() takes ``config`` as a positional argument."""
        from ptbr.training_cli import app as train_app

        # Inspect the registered commands to find the main callback
        # The train CLI uses typer.Argument for the config path
        source = Path(ROOT / "ptbr" / "training_cli.py").read_text()
        assert "typer.Argument" in source, (
            "training_cli should use typer.Argument for config path"
        )

    def test_argument_style_divergence(self):
        """Document the inconsistency: config uses --file, train uses positional."""
        main_source = (ROOT / "ptbr" / "__main__.py").read_text()

        # config_cmd uses typer.Option for file
        assert 'typer.Option' in main_source, (
            "__main__.py should use typer.Option for config_cmd's file parameter"
        )

        training_source = (ROOT / "ptbr" / "training_cli.py").read_text()
        assert 'typer.Argument' in training_source, (
            "training_cli.py should use typer.Argument for config path"
        )


# ======================================================================== #
#  4.  Parameters validated by training_cli but NOT forwarded               #
# ======================================================================== #


class TestParameterForwarding:
    """Verify that training_cli forwards validated fields to model.train_model().
    Tests cover both fields that ARE forwarded and remaining gaps."""

    @staticmethod
    def _get_launch_training_source() -> str:
        source = (ROOT / "ptbr" / "training_cli.py").read_text()
        return source

    @staticmethod
    def _extract_train_model_kwargs(tree: ast.AST) -> set[str]:
        """Extract keyword argument names from the model.train_model() call
        inside _launch_training."""
        kwargs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match model.train_model(...)
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "train_model"
                ):
                    for kw in node.keywords:
                        if kw.arg is not None:
                            kwargs.add(kw.arg)
        return kwargs

    # -- Fixed: these are now correctly forwarded --

    def test_dataloader_pin_memory_forwarded(self):
        """training.dataloader_pin_memory is validated AND forwarded."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "dataloader_pin_memory" in forwarded, (
            "dataloader_pin_memory should be forwarded to train_model()"
        )

    def test_dataloader_persistent_workers_forwarded(self):
        """training.dataloader_persistent_workers is validated AND forwarded."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "dataloader_persistent_workers" in forwarded, (
            "dataloader_persistent_workers should be forwarded to train_model()"
        )

    def test_dataloader_prefetch_factor_forwarded(self):
        """training.dataloader_prefetch_factor is validated AND forwarded."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "dataloader_prefetch_factor" in forwarded, (
            "dataloader_prefetch_factor should be forwarded to train_model()"
        )

    def test_run_name_forwarded(self):
        """run.name is forwarded as ``run_name`` to model.train_model()."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "run_name" in forwarded, (
            "run_name should be in the train_model() call for W&B run naming"
        )

    # -- Still outstanding: these remain dead config entries --

    def test_size_sup_not_forwarded(self):
        """training.size_sup validated but never used (dead config)."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "size_sup" not in forwarded, (
            "size_sup should NOT be in the train_model() call (dead config)"
        )

    def test_shuffle_types_not_forwarded(self):
        """training.shuffle_types validated but never forwarded."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "shuffle_types" not in forwarded, (
            "shuffle_types should NOT be in the train_model() call"
        )

    def test_random_drop_not_forwarded(self):
        """training.random_drop validated but never forwarded."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "random_drop" not in forwarded, (
            "random_drop should NOT be in the train_model() call"
        )

    def test_run_tags_not_forwarded(self):
        """run.tags validated but never forwarded to W&B/TrainingArguments."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "run_tags" not in forwarded, (
            "run_tags should NOT be in the train_model() call (documenting the gap)"
        )

    def test_run_description_not_forwarded(self):
        """run.description validated but unused beyond logging."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "run_description" not in forwarded
        assert "description" not in forwarded


# ======================================================================== #
#  5.  train.py hardcoded values override config                            #
# ======================================================================== #


class TestTrainPyForwarding:
    """Verify train.py correctly forwards config values to model.train_model()."""

    @staticmethod
    def _parse_train_py() -> ast.AST:
        return ast.parse(TRAIN_PY.read_text())

    @staticmethod
    def _extract_train_model_kwargs(tree: ast.AST) -> dict[str, Any]:
        """Extract keyword arguments from the model.train_model() call as
        {name: ast_node} pairs."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "train_model":
                    return {kw.arg: kw.value for kw in node.keywords if kw.arg}
        return {}

    def test_output_dir_from_config(self):
        """train.py now reads output_dir from cfg.data.root_dir."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "output_dir" in kwargs, "train.py should pass output_dir"
        node = kwargs["output_dir"]
        # Should NOT be a hardcoded constant "models" anymore
        assert not (isinstance(node, ast.Constant) and node.value == "models"), (
            "output_dir should come from cfg.data.root_dir, not be hardcoded"
        )

    def test_bf16_from_config(self):
        """train.py now reads bf16 from config rather than hardcoding True."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "bf16" in kwargs, "train.py should pass bf16"
        node = kwargs["bf16"]
        # Should NOT be hardcoded to True anymore
        assert not (isinstance(node, ast.Constant) and node.value is True), (
            "bf16 should be read from config, not hardcoded to True"
        )

    def test_eval_batch_size_has_proper_fallback(self):
        """train.py now has a proper eval_batch_size fallback variable."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "per_device_eval_batch_size" in kwargs

    def test_label_smoothing_forwarded_by_train_py(self):
        """train.py now forwards label_smoothing."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "label_smoothing" in kwargs, (
            "train.py should forward label_smoothing to model.train_model()"
        )

    def test_size_sup_not_forwarded_by_train_py(self):
        """train.py does not forward size_sup (dead config field)."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "size_sup" not in kwargs

    def test_shuffle_types_not_forwarded_by_train_py(self):
        """train.py does not forward shuffle_types (dead config field)."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "shuffle_types" not in kwargs

    def test_random_drop_not_forwarded_by_train_py(self):
        """train.py does not forward random_drop (dead config field)."""
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
        # These are used to build the model, not passed to train_model
        "model", "data",
        # These are read separately
        "training.prev_path", "training.freeze_components",
    }

    def test_all_training_fields_in_config_yaml_are_forwarded(self):
        """Every field under ``training:`` in config.yaml should be forwarded
        to model.train_model() by train.py.  Catch fields that are silently
        ignored."""
        cfg = _load_config("config.yaml")
        training_fields = set(cfg.get("training", {}).keys())

        # Parse what train.py actually forwards
        tree = ast.parse(TRAIN_PY.read_text())
        forwarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "train_model":
                    for kw in node.keywords:
                        if kw.arg:
                            forwarded.add(kw.arg)

        # Map config field names to train_model kwarg names
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

        # These fields are IN the config but NOT forwarded (dead config)
        expected_missing = {"size_sup", "shuffle_types", "random_drop"}
        actual_missing = set(not_forwarded)
        assert expected_missing.issubset(actual_missing), (
            f"Expected these config fields to be missing from train.py forwarding: "
            f"{expected_missing}. Actually missing: {actual_missing}"
        )


# ======================================================================== #
#  7.  remove_unused_columns default is wrong for GLiNER                    #
# ======================================================================== #


class TestRemoveUnusedColumns:
    """GLiNER uses custom data collators.  HF TrainingArguments defaults
    remove_unused_columns=True which strips columns the collator needs."""

    def test_default_remove_unused_columns_is_true(self):
        """The HF default for remove_unused_columns is True."""
        from gliner.training.trainer import TrainingArguments

        args = TrainingArguments(output_dir="/tmp/_test_ruc")
        assert args.remove_unused_columns is True, (
            "HF TrainingArguments defaults remove_unused_columns to True"
        )

    def test_create_training_args_does_not_override_remove_unused_columns(self):
        """create_training_args does not set remove_unused_columns=False."""
        from gliner.model import BaseGLiNER

        sig = inspect.signature(BaseGLiNER.create_training_args)
        assert "remove_unused_columns" not in sig.parameters, (
            "create_training_args lacks a named 'remove_unused_columns' parameter"
        )

    def test_training_cli_does_not_pass_remove_unused_columns(self):
        """_launch_training doesn't pass remove_unused_columns to train_model."""
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
        assert "remove_unused_columns" not in forwarded, (
            "_launch_training does not set remove_unused_columns=False "
            "(dangerous for custom collators)"
        )

    def test_train_py_does_not_pass_remove_unused_columns(self):
        """train.py also doesn't pass remove_unused_columns."""
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
#  8.  create_training_args named parameter gaps                            #
# ======================================================================== #


class TestCreateTrainingArgsGaps:
    """create_training_args has named params for some fields but relies on
    **kwargs for others.  This makes the API inconsistent and fragile."""

    def test_label_smoothing_not_named_parameter(self):
        """label_smoothing is a custom TrainingArguments field but not a named
        parameter of create_training_args -- goes through **kwargs."""
        from gliner.model import BaseGLiNER

        sig = inspect.signature(BaseGLiNER.create_training_args)
        # label_smoothing exists on TrainingArguments (custom field)
        from gliner.training.trainer import TrainingArguments
        assert hasattr(TrainingArguments, "label_smoothing"), (
            "TrainingArguments should have label_smoothing"
        )
        # but it's not a named parameter of create_training_args
        assert "label_smoothing" not in sig.parameters, (
            "label_smoothing is NOT a named parameter of create_training_args "
            "(goes through **kwargs)"
        )

    def test_gradient_checkpointing_not_available(self):
        """gradient_checkpointing is important for large models but absent
        from both create_training_args and the training_cli schema."""
        from gliner.model import BaseGLiNER

        sig = inspect.signature(BaseGLiNER.create_training_args)
        assert "gradient_checkpointing" not in sig.parameters

    def test_run_name_not_in_create_training_args(self):
        """run_name is not a parameter of create_training_args."""
        from gliner.model import BaseGLiNER

        sig = inspect.signature(BaseGLiNER.create_training_args)
        assert "run_name" not in sig.parameters


# ======================================================================== #
#  9.  Config loader (gliner/utils.py) lacks schema validation              #
# ======================================================================== #


class TestConfigLoaderValidation:
    """load_config_as_namespace() loads any YAML blindly -- no schema check."""

    def test_wrong_structure_loads_without_error(self):
        """A YAML with ``gliner_config:`` instead of ``model:`` loads fine,
        but will crash at runtime when train.py accesses cfg.model."""
        import tempfile
        from gliner.utils import load_config_as_namespace

        bad_cfg = {"gliner_config": {"model_name": "foo"}, "training": {"num_steps": 100}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(bad_cfg, f)
            tmp_path = f.name

        try:
            ns = load_config_as_namespace(tmp_path)
            # It loads fine -- no validation
            assert hasattr(ns, "gliner_config"), "Should load gliner_config section"
            assert not hasattr(ns, "model"), "Should NOT have a model section"
            # Accessing cfg.model would raise AttributeError at runtime
            with pytest.raises(AttributeError):
                _ = ns.model
        finally:
            os.unlink(tmp_path)

    def test_missing_required_fields_not_caught(self):
        """A YAML missing critical fields like training.num_steps loads fine."""
        import tempfile
        from gliner.utils import load_config_as_namespace

        incomplete_cfg = {
            "model": {"model_name": "foo"},
            "data": {"root_dir": "logs"},
            "training": {},  # missing num_steps, lr_encoder, etc.
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(incomplete_cfg, f)
            tmp_path = f.name

        try:
            ns = load_config_as_namespace(tmp_path)
            # Loads fine with empty training section
            assert hasattr(ns, "training")
            # But accessing required fields crashes at runtime
            with pytest.raises(AttributeError):
                _ = ns.training.num_steps
        finally:
            os.unlink(tmp_path)

    def test_empty_yaml_crashes_loader(self):
        """An empty YAML file causes load_config_as_namespace to fail immediately."""
        import tempfile
        from gliner.utils import load_config_as_namespace

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("# empty config\n")
            tmp_path = f.name

        try:
            # yaml.safe_load returns None for empty file
            with pytest.raises((TypeError, AttributeError)):
                load_config_as_namespace(tmp_path)
        finally:
            os.unlink(tmp_path)


# ======================================================================== #
#  10.  training_cli.py validates but training_cli doesn't forward          #
#       the same set of fields                                               #
# ======================================================================== #


class TestSchemaVsForwarding:
    """Cross-reference _FIELD_SCHEMA entries against what _launch_training
    actually passes to model.train_model()."""

    def test_validated_training_fields_not_all_forwarded(self):
        """Collect training.* fields from the schema and check which ones
        appear in the _launch_training -> model.train_model() call."""
        from ptbr.training_cli import _FIELD_SCHEMA

        # All training.* fields in the schema
        schema_training_fields = {
            key.split(".", 1)[1]
            for key, *_ in _FIELD_SCHEMA
            if key.startswith("training.")
        }

        # Fields forwarded by _launch_training
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

        # Map schema field names to what train_model expects
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

        # Fields used outside of train_model (model build, freeze, compile, etc.)
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

        # These are the remaining documented gaps (dead config fields)
        expected_gaps = {
            "size_sup",
            "shuffle_types",
            "random_drop",
        }
        actual_gaps = set(not_forwarded)
        assert expected_gaps.issubset(actual_gaps), (
            f"Expected forwarding gaps: {expected_gaps}. "
            f"Actual gaps: {actual_gaps}"
        )

        # Verify that dataloader params ARE now forwarded (previously gaps)
        fixed_params = {"dataloader_pin_memory", "dataloader_persistent_workers",
                        "dataloader_prefetch_factor"}
        assert not fixed_params.intersection(actual_gaps), (
            f"Dataloader params should now be forwarded but are still gaps: "
            f"{fixed_params.intersection(actual_gaps)}"
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
        """No shipped config has a separate eval_batch_size field -- train.py
        always reuses train_batch_size for eval."""
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
#  12.  __main__.py top-level import side effects                           #
# ======================================================================== #


class TestMainLazyImport:
    """__main__.py now lazy-loads training_cli to avoid import side effects."""

    def test_training_cli_not_imported_at_module_level(self):
        """Verify that training_cli is NOT imported at module level (lazy)."""
        source = (ROOT / "ptbr" / "__main__.py").read_text()
        tree = ast.parse(source)

        # Find top-level imports (not inside functions)
        top_level_imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    top_level_imports.append(node.module)

        assert not any("training_cli" in imp for imp in top_level_imports), (
            "training_cli should NOT be imported at module level in __main__.py "
            "(should be lazy-loaded to avoid Rich handler side effects)"
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
        assert "gliner_config" not in tpl, (
            "template.yaml should use 'model' not 'gliner_config'"
        )

    def test_template_passes_training_cli_validation(self):
        """Full template must pass training_cli validation."""
        from ptbr.training_cli import validate_config

        tpl = _load_template()
        vr = validate_config(tpl)
        assert len(vr.errors) == 0, (
            f"template.yaml should be valid for training_cli. Errors: {vr.errors}"
        )

    def test_template_passes_config_cli(self):
        """Full template now passes config_cli validation via alias support."""
        from ptbr.config_cli import load_and_validate_config

        result = load_and_validate_config(
            str(TEMPLATE_YAML), full_or_lora="full", method="span", validate=True,
        )
        assert result.report.is_valid, (
            f"template.yaml should pass config_cli validation via 'model' alias; "
            f"errors: {[e.message for e in result.report.errors]}"
        )


# ======================================================================== #
#  14.  E2E workflow: config validate && train                              #
# ======================================================================== #


class TestEndToEndWorkflow:
    """The workflow ``ptbr config --validate && ptbr train`` now works
    with a single YAML file thanks to config_cli's alias support."""

    def test_validate_then_train_works_with_single_yaml(self):
        """A single YAML with ``model:`` now passes both CLIs."""
        from ptbr.config_cli import load_and_validate_config
        from ptbr.training_cli import validate_config

        # training_cli: should pass
        tpl = _load_template()
        vr = validate_config(tpl)
        assert len(vr.errors) == 0, "template format should pass training_cli"

        # config_cli: should also pass via alias
        result = load_and_validate_config(
            str(TEMPLATE_YAML), full_or_lora="full", method="span", validate=True,
        )
        assert result.report.is_valid, (
            f"config_cli should accept template.yaml via 'model' alias; "
            f"errors: {[e.message for e in result.report.errors]}"
        )

    def test_gliner_config_format_still_fails_training_cli(self):
        """training_cli still rejects ``gliner_config:`` format (expected)."""
        from ptbr.training_cli import validate_config

        gliner_format = {
            "gliner_config": {
                "model_name": "microsoft/deberta-v3-small",
                "span_mode": "markerV0",
                "max_len": 384,
            }
        }
        vr = validate_config(gliner_format)
        assert len(vr.errors) > 0, (
            "training_cli should reject the 'gliner_config:' format"
        )
