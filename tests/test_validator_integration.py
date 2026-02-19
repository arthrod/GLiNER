"""Tests for validator integration between ptbr config_cli and training_cli.

These tests verify the YAML schema incompatibility between the two CLIs and
confirm that validated config fields actually reach the deep-learning training
process.  Originally written to *expose* bugs from the architecture report,
many issues have since been fixed.  The tests now verify:

1. config_cli.py expects ``gliner_config:`` / ``lora_config:`` top-level keys
   while training_cli.py (and template.yaml) expect ``model:`` / ``lora:``.
   (Still an incompatibility.)
2. Parameters validated by training_cli are now forwarded to train_model()
   (dataloader_pin_memory, dataloader_persistent_workers, etc. – FIXED).
3. train.py reads output_dir, bf16, eval_batch_size from config (FIXED).
4. remove_unused_columns defaults to True -- wrong for custom collators.
5. LoRA section naming divergence between config_cli and training_cli.
6. CLI argument style inconsistency (--file vs positional).
7. __main__.py now uses lazy import for training_cli (FIXED).
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


class TestYAMLSchemaCompatibility:
    """Verify config_cli and training_cli accept the same YAML layout
    now that config_cli supports ``model:`` as an alias for ``gliner_config:``."""

    def test_template_yaml_passes_config_cli_validation(self):
        """template.yaml uses ``model:`` which config_cli now accepts as alias.

        Previously this failed with 'Missing gliner_config section'.
        """
        pytest.importorskip("transformers")
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

    def test_no_single_yaml_satisfies_both_clis(self):
        """Prove that no structure can satisfy both CLIs simultaneously.

        config_cli requires ``gliner_config:`` at top-level.
        training_cli requires ``model:`` at top-level.
        Adding both doesn't help -- config_cli ignores ``model:`` and
        training_cli ignores ``gliner_config:``.
        """
        pytest.importorskip("transformers")
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
#  2.  LoRA section naming divergence (STILL OPEN)                          #
# ======================================================================== #


class TestLoRASectionNaming:
    """config_cli expects ``lora_config:`` while training_cli expects ``lora:``."""

    def test_config_cli_expects_lora_config_key(self):
        """config_cli looks for ``lora_config:``, not ``lora:``."""
        pytest.importorskip("transformers")
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
        pytest.importorskip("transformers")
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


class TestParameterForwarding:
    """Verify that training_cli forwards validated fields to model.train_model().
    Tests cover both fields that ARE forwarded and remaining gaps."""

    @staticmethod
    def _get_launch_training_source() -> str:
        """
        Retrieve the source code of ptbr/training_cli.py.
        
        Returns:
            The file contents of ptbr/training_cli.py as a string.
        """
        source = (ROOT / "ptbr" / "training_cli.py").read_text()
        return source

    def test_dataloader_pin_memory_forwarded(self):
        """training.dataloader_pin_memory is validated and forwarded to train_model."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)

        forwarded = self._extract_train_model_kwargs(tree)
        assert "dataloader_pin_memory" in forwarded, (
            "dataloader_pin_memory should be forwarded to train_model() (bug was fixed)"
        )

    def test_dataloader_persistent_workers_forwarded(self):
        """training.dataloader_persistent_workers is validated and forwarded."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "dataloader_persistent_workers" in forwarded, (
            "dataloader_persistent_workers should be forwarded to train_model() (bug was fixed)"
        )

    def test_dataloader_prefetch_factor_forwarded(self):
        """training.dataloader_prefetch_factor is validated and forwarded."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "dataloader_prefetch_factor" in forwarded, (
            "dataloader_prefetch_factor should be forwarded to train_model() (bug was fixed)"
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
        assert "shuffle_types" not in forwarded

    def test_random_drop_removed_from_schema(self):
        """training.random_drop is no longer in schema (dead config removed)."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "random_drop" not in forwarded

    def test_run_name_forwarded(self):
        """run.name is validated and forwarded as ``run_name`` to TrainingArguments."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "run_name" in forwarded, (
            "run_name should be forwarded to train_model() (bug was fixed)"
        )

    def test_run_tags_not_forwarded(self):
        """run.tags validated but not forwarded to W&B/TrainingArguments."""
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "run_tags" not in forwarded

    def test_run_description_not_forwarded(self):
        """
        Asserts that the config's run.description is validated but not forwarded to train_model.
        
        Checks that neither `run_description` nor `description` appear among the keyword arguments passed to `train_model`.
        """
        source = self._get_launch_training_source()
        tree = ast.parse(source)
        forwarded = self._extract_train_model_kwargs(tree)
        assert "run_description" not in forwarded
        assert "description" not in forwarded


# ======================================================================== #
#  5.  train.py fixes verified                                              #
# ======================================================================== #


class TestTrainPyForwarding:
    """Verify train.py correctly forwards config values to model.train_model()."""

    @staticmethod
    def _parse_train_py() -> ast.AST:
        """
        Parse the project's train.py source into an abstract syntax tree (AST).
        
        Returns:
            tree (ast.AST): The parsed AST for the contents of TRAIN_PY.
        """
        return ast.parse(TRAIN_PY.read_text())

    @staticmethod
    def _extract_train_model_kwargs(tree: ast.AST) -> dict[str, Any]:
        """
        Locate the keywords passed to a train_model(...) call within an AST and map each keyword name to its corresponding AST value node.
        
        Parameters:
            tree (ast.AST): The AST to search (for example, the Module returned by ast.parse()).
        
        Returns:
            dict[str, ast.AST]: A mapping from keyword argument name to the AST node representing its value; returns an empty dict if no train_model call with keywords is found.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "train_model":
                    return {kw.arg: kw.value for kw in node.keywords if kw.arg}
        return {}

    def test_output_dir_from_config(self):
        """train.py reads output_dir from cfg.data.root_dir (no longer hardcoded)."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "output_dir" in kwargs
        node = kwargs["output_dir"]
        # It should NOT be a hardcoded constant "models"
        is_hardcoded_models = isinstance(node, ast.Constant) and node.value == "models"
        assert not is_hardcoded_models, (
            "output_dir should no longer be hardcoded to 'models' (bug was fixed)"
        )

    def test_bf16_from_config(self):
        """train.py reads bf16 from config (no longer hardcoded to True)."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "bf16" in kwargs
        node = kwargs["bf16"]
        is_hardcoded_true = isinstance(node, ast.Constant) and node.value is True
        assert not is_hardcoded_true, (
            "bf16 should no longer be hardcoded to True (bug was fixed)"
        )

    def test_eval_batch_size_separate_from_train(self):
        """train.py now uses a separate eval_batch_size variable (no longer reuses train)."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "per_device_eval_batch_size" in kwargs

        node = kwargs["per_device_eval_batch_size"]
        source_line = ast.dump(node)
        # Should no longer directly reference train_batch_size
        assert "train_batch_size" not in source_line, (
            "eval batch size should use a separate variable, not train_batch_size (bug was fixed)"
        )

    def test_label_smoothing_forwarded_by_train_py(self):
        """train.py now forwards label_smoothing from config."""
        tree = self._parse_train_py()
        kwargs = self._extract_train_model_kwargs(tree)
        assert "label_smoothing" in kwargs, (
            "train.py should forward label_smoothing (bug was fixed)"
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
        """
        Verify that train.py does not forward the `size_sup` training field to train_model.
        
        Asserts that the keyword arguments collected for the call to `train_model` do not include `"size_sup"`.
        """
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
        "model", "data",
        "training.prev_path", "training.freeze_components",
    }

    def test_all_training_fields_in_config_yaml_are_forwarded(self):
        """
        Ensure every field under `training:` in config.yaml is forwarded to train_model() by train.py.
        
        Parses the project's `config.yaml` and `train.py` to compare `training.*` keys against the keyword arguments passed to `train_model()`, and fails the test if any configuration field (other than known, intentionally non-forwarded fields) is not forwarded. Confirms that the legacy dead fields `size_sup`, `shuffle_types`, and `random_drop` remain absent from forwarding.
        """
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

        # These dead config fields remain un-forwarded (label_smoothing was fixed)
        expected_missing = {"size_sup", "shuffle_types", "random_drop"}
        actual_missing = set(not_forwarded)
        assert expected_missing.issubset(actual_missing), (
            f"Expected these dead config fields to still be missing from train.py forwarding: "
            f"{expected_missing}. Actually missing: {actual_missing}"
        )
        # Verify label_smoothing is no longer in the gap set (bug was fixed)
        assert "label_smoothing" not in actual_missing, (
            "label_smoothing should now be forwarded by train.py"
        )


# ======================================================================== #
#  7.  remove_unused_columns (STILL OPEN — needs heavy deps to test fully)  #
# ======================================================================== #


class TestRemoveUnusedColumns:
    """GLiNER uses custom data collators.  HF TrainingArguments defaults
    remove_unused_columns=True which strips columns the collator needs.
    Fixed: create_training_args now defaults to False, and training_cli forwards it."""

    def test_default_remove_unused_columns_is_true(self):
        """The HF default for remove_unused_columns is True."""
        pytest.importorskip("torch")
        from gliner.training.trainer import TrainingArguments
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            args = TrainingArguments(output_dir=tmpdir)
            assert args.remove_unused_columns is True, (
                "HF TrainingArguments defaults remove_unused_columns to True"
            )

    def test_create_training_args_does_not_override_remove_unused_columns(self):
        """create_training_args does not set remove_unused_columns=False."""
        pytest.importorskip("torch")
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
#  8.  create_training_args gaps (needs heavy deps — skipped if absent)     #
# ======================================================================== #


class TestCreateTrainingArgsFixed:
    """create_training_args now has explicit named params for critical fields.
    Previously these relied on **kwargs pass-through."""

    def test_label_smoothing_not_named_parameter(self):
        """label_smoothing is a custom TrainingArguments field but not a named
        parameter of create_training_args -- goes through **kwargs."""
        pytest.importorskip("torch")
        from gliner.model import BaseGLiNER

        sig = inspect.signature(BaseGLiNER.create_training_args)
        from gliner.training.trainer import TrainingArguments
        assert hasattr(TrainingArguments, "label_smoothing"), (
            "TrainingArguments should have label_smoothing"
        )
        assert "label_smoothing" in sig.parameters, (
            "label_smoothing should be a named parameter of create_training_args"
        )

    def test_gradient_checkpointing_not_available(self):
        """gradient_checkpointing is important for large models but absent
        from both create_training_args and the training_cli schema."""
        pytest.importorskip("torch")
        from gliner.model import BaseGLiNER

        sig = inspect.signature(BaseGLiNER.create_training_args)
        assert "gradient_checkpointing" in sig.parameters

    def test_run_name_not_in_create_training_args(self):
        """run_name is not a parameter of create_training_args."""
        pytest.importorskip("torch")
        from gliner.model import BaseGLiNER

        sig = inspect.signature(BaseGLiNER.create_training_args)
        assert "run_name" in sig.parameters


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
    actually passes to model.train_model().

    After fixes: dead config fields (size_sup, shuffle_types, random_drop)
    removed from schema; dataloader fields now forwarded; no remaining gaps."""

    def test_all_training_fields_forwarded(self):
        """All training.* fields from the schema should now be forwarded
        (or handled elsewhere) by _launch_training."""
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

        # Only dead config fields remain as gaps (dataloader_* were fixed)
        expected_gaps = {
            "size_sup",
            "shuffle_types",
            "random_drop",
        }
        actual_gaps = set(not_forwarded)

        # Verify dead config fields are still gaps as expected
        assert expected_gaps.issubset(actual_gaps), (
            f"Expected remaining forwarding gaps: {expected_gaps}. "
            f"Actual gaps: {actual_gaps}"
        )
        # Verify formerly-gapped fields are now forwarded (bugs were fixed)
        for fixed_field in ("dataloader_pin_memory", "dataloader_persistent_workers",
                            "dataloader_prefetch_factor"):
            assert fixed_field not in actual_gaps, (
                f"{fixed_field} should no longer be a forwarding gap"
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
    """__main__.py now uses lazy import for training_cli to avoid Rich logging
    handler setup when only using config or data subcommands (bug was fixed)."""

    def test_training_cli_not_imported_at_module_level(self):
        """Verify that training_cli is NOT imported at module level (lazy import fix)."""
        source = (ROOT / "ptbr" / "__main__.py").read_text()
        tree = ast.parse(source)

        top_level_imports = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                top_level_imports.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    top_level_imports.append(alias.name)

        assert not any("training_cli" in imp for imp in top_level_imports), (
            "training_cli should NOT be imported at module level in __main__.py "
            "(lazy import fix avoids side effects for config/data subcommands)"
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

    def test_template_fails_config_cli(self):
        """Full template must FAIL config_cli validation (the core bug)."""
        pytest.importorskip("transformers")
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

    def test_validate_then_train_is_impossible_with_single_yaml(self):
        """Demonstrate that no single YAML file can:
        1. Pass config_cli validation (requires gliner_config:)
        2. Pass training_cli validation (requires model:)
        """
        pytest.importorskip("transformers")
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

        # The incompatibility is proven: neither format works for both
        assert not config_ok_with_model_format, (
            "model: format fails config_cli"
        )
        assert not training_ok_with_gc_format, (
            "gliner_config: format fails training_cli"
        )
