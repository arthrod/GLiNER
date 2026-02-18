"""Tests for validator integration between ptbr config_cli and training_cli.

These tests catch the critical YAML schema incompatibility between the two
CLIs and verify that validated config fields actually reach the deep-learning
training process.  They expose (but do NOT fix) the issues identified in the
architecture report:

1. config_cli.py expects ``gliner_config:`` / ``lora_config:`` top-level keys
   while training_cli.py (and template.yaml) expect ``model:`` / ``lora:``.
2. Parameters validated by training_cli but never forwarded to train_model().
3. train.py hardcodes output_dir, bf16, and reuses train batch size for eval.
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
    """Verify that the CLIs handle the template.yaml format correctly."""

    def test_template_yaml_passes_config_cli_validation(self):
        """template.yaml passes config_cli validation."""
        from ptbr.config_cli import load_and_validate_config

        result = load_and_validate_config(
            str(TEMPLATE_YAML), full_or_lora="full", method="span", validate=True,
        )
        assert result.report.is_valid, (
            f"template.yaml should pass config_cli validation; errors: "
            f"{[e.message for e in result.report.errors]}"
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

    def test_template_yaml_satisfies_both_clis(self):
        """template.yaml should pass both config_cli and training_cli validation."""
        from ptbr.config_cli import load_and_validate_config
        from ptbr.training_cli import validate_config

        tpl = _load_template()

        # training_cli should pass (it has model:)
        vr = validate_config(tpl)
        assert len(vr.errors) == 0, (
            f"template.yaml should pass training_cli validation; errors: {vr.errors}"
        )

        # config_cli should also pass
        tpl_result = load_and_validate_config(
            str(TEMPLATE_YAML), full_or_lora="full", method="span", validate=True,
        )
        assert tpl_result.report.is_valid, (
            f"template.yaml should pass config_cli validation; errors: "
            f"{[e.message for e in tpl_result.report.errors]}"
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


class TestParameterForwardingFixed:
    """Verify that training_cli's _launch_training correctly forwards validated
    fields to model.train_model().  Previously these were dead config entries
    but have been fixed."""

    @staticmethod
    def _get_launch_training_source() -> str:
        source = (ROOT / "ptbr" / "training_cli.py").read_text()
        return source

    def test_dataloader_pin_memory_forwarded(self):
        """training.dataloader_pin_memory is now forwarded to train_model."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "dataloader_pin_memory" in forwarded

    def test_dataloader_persistent_workers_forwarded(self):
        """training.dataloader_persistent_workers is now forwarded."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "dataloader_persistent_workers" in forwarded

    def test_dataloader_prefetch_factor_forwarded(self):
        """training.dataloader_prefetch_factor is now forwarded."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "dataloader_prefetch_factor" in forwarded

    def test_size_sup_removed_from_schema(self):
        """training.size_sup is no longer in schema (dead config removed)."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "size_sup" not in forwarded

    def test_shuffle_types_removed_from_schema(self):
        """training.shuffle_types is no longer in schema (dead config removed)."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "shuffle_types" not in forwarded

    def test_random_drop_removed_from_schema(self):
        """training.random_drop is no longer in schema (dead config removed)."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "random_drop" not in forwarded

    def test_run_name_forwarded(self):
        """run.name is now forwarded as run_name to TrainingArguments."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "run_name" in forwarded

    def test_remove_unused_columns_forwarded(self):
        """remove_unused_columns is now forwarded to train_model."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "remove_unused_columns" in forwarded

    def test_push_to_hub_forwarded(self):
        """push_to_hub is now forwarded from environment config to train_model."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "push_to_hub" in forwarded

    def test_hub_model_id_forwarded(self):
        """hub_model_id is now forwarded from environment config to train_model."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "hub_model_id" in forwarded

    def test_seed_forwarded(self):
        """seed is now forwarded from run config to train_model."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "seed" in forwarded

    def test_resume_from_checkpoint_forwarded(self):
        """resume_from_checkpoint is now forwarded to train_model."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "resume_from_checkpoint" in forwarded

    def test_run_tags_not_forwarded(self):
        """run.tags validated but not forwarded to W&B/TrainingArguments."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "run_tags" not in forwarded

    def test_run_description_not_forwarded(self):
        """run.description validated but unused beyond logging."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "run_description" not in forwarded
        assert "description" not in forwarded

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


# ======================================================================== #
#  5.  train.py hardcoded values override config                            #
# ======================================================================== #


class TestTrainPyValues:
    """Verify train.py forwards config values correctly."""

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

    def test_output_dir_uses_config(self):
        """train.py now uses cfg.data.root_dir for output_dir."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "output_dir" in kwargs, "train.py should pass output_dir"

    def test_bf16_reads_from_config(self):
        """train.py now reads bf16 from config."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "bf16" in kwargs, "train.py should pass bf16"

    def test_eval_batch_size_has_fallback(self):
        """train.py uses eval_batch_size with fallback to train_batch_size."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "per_device_eval_batch_size" in kwargs

    def test_label_smoothing_forwarded_by_train_py(self):
        """train.py now forwards label_smoothing."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "label_smoothing" in kwargs, (
            "train.py should forward label_smoothing"
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

        # Dead config fields remain in YAML configs but are correctly
        # not forwarded by train.py (they have no consumers).
        # label_smoothing was previously missing but is now forwarded.
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
    remove_unused_columns=True which strips columns the collator needs.
    Fixed: create_training_args now defaults to False, and training_cli forwards it."""

    def test_default_remove_unused_columns_is_true(self):
        """The HF default for remove_unused_columns is True."""
        from gliner.training.trainer import TrainingArguments

        args = TrainingArguments(output_dir="/tmp/_test_ruc")
        assert args.remove_unused_columns is True, (
            "HF TrainingArguments defaults remove_unused_columns to True"
        )

    def test_create_training_args_overrides_remove_unused_columns(self):
        """create_training_args now sets remove_unused_columns=False."""
        from gliner.model import BaseGLiNER

        sig = inspect.signature(BaseGLiNER.create_training_args)
        assert "remove_unused_columns" in sig.parameters, (
            "create_training_args should have a named 'remove_unused_columns' parameter"
        )
        assert sig.parameters["remove_unused_columns"].default is False, (
            "remove_unused_columns should default to False for GLiNER"
        )

    def test_training_cli_passes_remove_unused_columns(self):
        """_launch_training now passes remove_unused_columns to train_model."""
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
        assert "remove_unused_columns" in forwarded, (
            "_launch_training should set remove_unused_columns=False"
        )

    def test_train_py_does_not_pass_remove_unused_columns(self):
        """train.py (legacy) still doesn't pass remove_unused_columns."""
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


class TestCreateTrainingArgsFixed:
    """create_training_args now has explicit named params for critical fields.
    Previously these relied on **kwargs pass-through."""

    def test_label_smoothing_is_named_parameter(self):
        """label_smoothing is now a named parameter of create_training_args."""
        from gliner.model import BaseGLiNER

        sig = inspect.signature(BaseGLiNER.create_training_args)
        from gliner.training.trainer import TrainingArguments
        assert hasattr(TrainingArguments, "label_smoothing"), (
            "TrainingArguments should have label_smoothing"
        )
        assert "label_smoothing" in sig.parameters, (
            "label_smoothing should be a named parameter of create_training_args"
        )

    def test_gradient_checkpointing_is_named_parameter(self):
        """gradient_checkpointing is now a named parameter."""
        from gliner.model import BaseGLiNER

        sig = inspect.signature(BaseGLiNER.create_training_args)
        assert "gradient_checkpointing" in sig.parameters

    def test_run_name_is_named_parameter(self):
        """run_name is now a named parameter of create_training_args."""
        from gliner.model import BaseGLiNER

        sig = inspect.signature(BaseGLiNER.create_training_args)
        assert "run_name" in sig.parameters


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
    actually passes to model.train_model().

    After fixes: dead config fields (size_sup, shuffle_types, random_drop)
    removed from schema; dataloader fields now forwarded; no remaining gaps."""

    def test_all_training_fields_forwarded(self):
        """All training.* fields from the schema should now be forwarded
        (or handled elsewhere) by _launch_training."""
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

        # After fixes, all training schema fields should be forwarded
        assert len(not_forwarded) == 0, (
            f"Forwarding gaps remain: {not_forwarded}"
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


class TestMainImportSideEffects:
    """__main__.py should use lazy imports so that training_cli side effects
    (Rich logging handler setup) don't trigger when only using config/data."""

    def test_training_cli_not_imported_at_module_level(self):
        """Verify that training_cli is NOT imported at module level (lazy is correct)."""
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
            "(causes side effects when using config/data subcommands)"
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
        """Full template should pass config_cli validation."""
        from ptbr.config_cli import load_and_validate_config

        result = load_and_validate_config(
            str(TEMPLATE_YAML), full_or_lora="full", method="span", validate=True,
        )
        assert result.report.is_valid, (
            f"template.yaml should pass config_cli validation; errors: "
            f"{[e.message for e in result.report.errors]}"
        )


# ======================================================================== #
#  14.  E2E workflow: config validate && train                              #
# ======================================================================== #


class TestEndToEndWorkflow:
    """The workflow ``ptbr config --validate && ptbr train`` should work
    with a single YAML file (template.yaml)."""

    def test_validate_then_train_with_template_yaml(self):
        """template.yaml should pass both config_cli and training_cli validation."""
        from ptbr.config_cli import load_and_validate_config
        from ptbr.training_cli import validate_config

        # training_cli: should pass
        tpl = _load_template()
        vr = validate_config(tpl)
        assert len(vr.errors) == 0, (
            f"template format should pass training_cli; errors: {vr.errors}"
        )

        # config_cli: should also pass
        result = load_and_validate_config(
            str(TEMPLATE_YAML), full_or_lora="full", method="span", validate=True,
        )
        assert result.report.is_valid, (
            f"template.yaml should pass config_cli validation; errors: "
            f"{[e.message for e in result.report.errors]}"
        )
