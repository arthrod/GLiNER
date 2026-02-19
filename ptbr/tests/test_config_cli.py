"""Tests for ptbr.config_cli: YAML validation, CLI, and module API."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

from gliner.config import GLiNERConfig
from ptbr.config_cli import (
    _LORA_RULES,
    _GLINER_RULES,
    ValidationReport,
    _coerce_type,
    _validate_section,
    print_and_log_result,
    load_and_validate_config,
    _validate_cross_constraints,
)

# ============================================================================
# Helpers
# ============================================================================


def _write_yaml(tmp_path: Path, data: dict, filename: str = "cfg.yaml") -> Path:
    """Write a dict as YAML and return the path."""
    p = tmp_path / filename
    with open(p, "w") as f:
        yaml.dump(data, f)
    return p


def _minimal_gliner_config() -> dict:
    """Return the minimal valid gliner_config section."""
    return {
        "model_name": "microsoft/deberta-v3-small",
    }


def _full_gliner_config() -> dict:
    """Return a fully-specified gliner_config section (no defaults needed)."""
    return {
        "model_name": "microsoft/deberta-v3-small",
        "name": "test-model",
        "fine_tune": True,
        "span_mode": "markerV0",
        "max_width": 12,
        "labels_encoder": None,
        "labels_decoder": None,
        "decoder_mode": "span",
        "full_decoder_context": True,
        "blank_entity_prob": 0.1,
        "decoder_loss_coef": 0.5,
        "relations_layer": None,
        "triples_layer": None,
        "embed_rel_token": True,
        "rel_token_index": -1,
        "rel_token": "<<REL>>",
        "adjacency_loss_coef": 1.0,
        "relation_loss_coef": 1.0,
        "hidden_size": 512,
        "dropout": 0.4,
        "subtoken_pooling": "first",
        "words_splitter_type": "whitespace",
        "max_len": 384,
        "max_types": 25,
        "max_neg_type_ratio": 1,
        "post_fusion_schema": "",
        "num_post_fusion_layers": 1,
        "fuse_layers": False,
        "num_rnn_layers": 1,
        "embed_ent_token": True,
        "class_token_index": -1,
        "vocab_size": -1,
        "ent_token": "<<ENT>>",
        "sep_token": "<<SEP>>",
        "token_loss_coef": 1.0,
        "span_loss_coef": 1.0,
        "represent_spans": False,
        "neg_spans_ratio": 1.0,
        "_attn_implementation": None,
    }


def _full_lora_config() -> dict:
    """Return a fully-specified lora_config section."""
    return {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["query_proj", "value_proj"],
        "bias": "none",
        "task_type": "FEATURE_EXTRACTION",
        "modules_to_save": None,
        "fan_in_fan_out": False,
        "use_rslora": False,
        "init_lora_weights": True,
    }


# ============================================================================
# Unit tests: _coerce_type
# ============================================================================


class TestCoerceType:
    def test_none_passthrough(self):
        assert _coerce_type(None, str) is None

    def test_bool_from_bool(self):
        assert _coerce_type(True, bool) is True
        assert _coerce_type(False, bool) is False

    def test_bool_from_string(self):
        assert _coerce_type("true", bool) is True
        assert _coerce_type("yes", bool) is True
        assert _coerce_type("false", bool) is False

    def test_bool_invalid(self):
        with pytest.raises(TypeError):
            _coerce_type("maybe", bool)

    def test_int_from_int(self):
        assert _coerce_type(42, int) == 42

    def test_int_from_float(self):
        assert _coerce_type(3.0, int) == 3

    def test_int_rejects_bool(self):
        with pytest.raises(TypeError):
            _coerce_type(True, int)

    def test_float_from_int(self):
        assert _coerce_type(3, float) == 3.0

    def test_float_rejects_bool(self):
        with pytest.raises(TypeError):
            _coerce_type(False, float)

    def test_str_from_anything(self):
        assert _coerce_type(123, str) == "123"

    def test_list_from_list(self):
        assert _coerce_type([1, 2], list) == [1, 2]

    def test_list_rejects_nonlist(self):
        with pytest.raises(TypeError):
            _coerce_type("not a list", list)


# ============================================================================
# Unit tests: _validate_section
# ============================================================================


class TestValidateSection:
    def test_required_field_missing_errors(self):
        report = ValidationReport()
        _validate_section({}, _GLINER_RULES, "gliner_config", report)
        # model_name is REQUIRED
        errors = [i for i in report.errors if "model_name" in i.field]
        assert len(errors) == 1
        assert "REQUIRED" in errors[0].message

    def test_default_fields_produce_warnings(self):
        report = ValidationReport()
        data = {"model_name": "some-model"}
        _validate_section(data, _GLINER_RULES, "gliner_config", report)
        # Many fields should get defaults → warnings
        warning_fields = {i.field.split(".")[-1] for i in report.warnings}
        assert "name" in warning_fields
        assert "hidden_size" in warning_fields
        assert "dropout" in warning_fields
        assert report.is_valid  # no errors

    def test_fully_specified_no_warnings(self):
        report = ValidationReport()
        data = _full_gliner_config()
        _validate_section(data, _GLINER_RULES, "gliner_config", report)
        # Only _attn_implementation should produce a warning since it's None and default is None
        # (None default → no warning)
        non_attn_warnings = [w for w in report.warnings if "_attn_implementation" not in w.field]
        assert len(non_attn_warnings) == 0
        assert report.is_valid

    def test_range_violation(self):
        report = ValidationReport()
        data = {"model_name": "x", "dropout": 1.5}  # max is 0.9
        _validate_section(data, _GLINER_RULES, "gliner_config", report)
        errors = [i for i in report.errors if "dropout" in i.field]
        assert len(errors) == 1
        assert "range" in errors[0].message.lower() or "outside" in errors[0].message.lower()

    def test_literal_violation(self):
        report = ValidationReport()
        data = {"model_name": "x", "span_mode": "invalid_mode"}
        _validate_section(data, _GLINER_RULES, "gliner_config", report)
        errors = [i for i in report.errors if "span_mode" in i.field]
        assert len(errors) == 1

    def test_type_error(self):
        report = ValidationReport()
        data = {"model_name": "x", "max_width": "not_an_int"}
        _validate_section(data, _GLINER_RULES, "gliner_config", report)
        errors = [i for i in report.errors if "max_width" in i.field]
        assert len(errors) == 1

    def test_unknown_keys_warned(self):
        report = ValidationReport()
        data = {"model_name": "x", "totally_made_up_field": 42}
        _validate_section(data, _GLINER_RULES, "gliner_config", report)
        warnings = [i for i in report.warnings if "totally_made_up_field" in i.field]
        assert len(warnings) == 1
        assert "Unknown" in warnings[0].message

    def test_lora_section_defaults(self):
        report = ValidationReport()
        result = _validate_section({}, _LORA_RULES, "lora_config", report)
        assert report.is_valid
        assert result["r"] == 8
        assert result["lora_alpha"] == 16
        assert result["lora_dropout"] == 0.1

    def test_lora_section_fully_specified(self):
        report = ValidationReport()
        data = _full_lora_config()
        result = _validate_section(data, _LORA_RULES, "lora_config", report)
        assert report.is_valid
        assert result["r"] == 16
        assert result["lora_alpha"] == 32

    def test_lora_range_violation(self):
        report = ValidationReport()
        data = {"r": 999}  # max 256
        _validate_section(data, _LORA_RULES, "lora_config", report)
        errors = [i for i in report.errors if "r" in i.field]
        assert len(errors) == 1


# ============================================================================
# Unit tests: _validate_cross_constraints
# ============================================================================


class TestCrossConstraints:
    def test_biencoder_requires_labels_encoder(self):
        report = ValidationReport()
        data = {"labels_encoder": None, "span_mode": "markerV0"}
        _validate_cross_constraints(data, method="biencoder", full_or_lora="full", report=report)
        errors = [i for i in report.errors if "labels_encoder" in i.field]
        assert len(errors) == 1

    def test_biencoder_with_labels_encoder_ok(self):
        report = ValidationReport()
        data = {"labels_encoder": "some-model", "span_mode": "markerV0"}
        _validate_cross_constraints(data, method="biencoder", full_or_lora="full", report=report)
        assert report.is_valid

    def test_decoder_requires_labels_decoder(self):
        report = ValidationReport()
        data = {"labels_decoder": None, "span_mode": "markerV0"}
        _validate_cross_constraints(data, method="decoder", full_or_lora="full", report=report)
        errors = [i for i in report.errors if "labels_decoder" in i.field]
        assert len(errors) == 1

    def test_decoder_with_labels_decoder_ok(self):
        report = ValidationReport()
        data = {"labels_decoder": "gpt2", "span_mode": "markerV0"}
        _validate_cross_constraints(data, method="decoder", full_or_lora="full", report=report)
        assert report.is_valid

    def test_relex_requires_relations_layer(self):
        report = ValidationReport()
        data = {"relations_layer": None, "span_mode": "markerV0"}
        _validate_cross_constraints(data, method="relex", full_or_lora="full", report=report)
        errors = [i for i in report.errors if "relations_layer" in i.field]
        assert len(errors) == 1

    def test_token_method_forces_span_mode(self):
        report = ValidationReport()
        data = {"span_mode": "markerV0"}
        _validate_cross_constraints(data, method="token", full_or_lora="full", report=report)
        assert data["span_mode"] == "token_level"
        assert len(report.warnings) > 0

    def test_span_method_with_token_level_warns(self):
        report = ValidationReport()
        data = {"span_mode": "token_level"}
        _validate_cross_constraints(data, method="span", full_or_lora="full", report=report)
        warnings = [i for i in report.warnings if "span_mode" in i.field]
        assert len(warnings) == 1

    def test_relex_fields_warn_when_not_relex(self):
        report = ValidationReport()
        data = {"relations_layer": "some_layer", "span_mode": "markerV0"}
        _validate_cross_constraints(data, method="span", full_or_lora="full", report=report)
        warnings = [i for i in report.warnings if "relations_layer" in i.field]
        assert len(warnings) == 1


# ============================================================================
# Integration tests: load_and_validate_config
# ============================================================================


class TestLoadAndValidateConfig:
    def test_minimal_config_loads(self, tmp_path):
        data = {"gliner_config": _minimal_gliner_config()}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="span")
        assert result.report.is_valid
        assert result.gliner_config is not None
        assert isinstance(result.gliner_config, GLiNERConfig)
        assert result.lora_config is None

    def test_full_config_no_warnings(self, tmp_path):
        data = {"gliner_config": _full_gliner_config()}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="span")
        assert result.report.is_valid
        # Only warnings should be for fields with None default that stay None
        real_warnings = [w for w in result.report.warnings if "Not set; using default" in w.message]
        assert len(real_warnings) == 0

    def test_full_config_with_lora(self, tmp_path):
        data = {
            "gliner_config": _minimal_gliner_config(),
            "lora_config": _full_lora_config(),
        }
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="lora", method="span")
        assert result.report.is_valid
        assert result.lora_config is not None
        assert result.lora_config["r"] == 16

    def test_lora_mode_without_section_uses_defaults(self, tmp_path):
        data = {"gliner_config": _minimal_gliner_config()}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="lora", method="span")
        assert result.report.is_valid
        assert result.lora_config is not None
        assert result.lora_config["r"] == 8  # default

    def test_missing_required_field_fails(self, tmp_path):
        data = {"gliner_config": {"name": "test"}}  # missing model_name
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="span")
        assert not result.report.is_valid
        assert result.gliner_config is None

    def test_range_error_fails(self, tmp_path):
        cfg = _minimal_gliner_config()
        cfg["dropout"] = 5.0  # out of range
        data = {"gliner_config": cfg}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="span")
        assert not result.report.is_valid

    def test_biencoder_method_validates(self, tmp_path):
        cfg = _minimal_gliner_config()
        cfg["labels_encoder"] = "microsoft/deberta-v3-small"
        data = {"gliner_config": cfg}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="biencoder")
        assert result.report.is_valid

    def test_biencoder_without_encoder_errors(self, tmp_path):
        data = {"gliner_config": _minimal_gliner_config()}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="biencoder")
        assert not result.report.is_valid
        error_fields = [e.field for e in result.report.errors]
        assert any("labels_encoder" in f for f in error_fields)

    def test_decoder_method_validates(self, tmp_path):
        cfg = _minimal_gliner_config()
        cfg["labels_decoder"] = "openai-community/gpt2"
        data = {"gliner_config": cfg}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="decoder")
        assert result.report.is_valid

    def test_decoder_without_decoder_errors(self, tmp_path):
        data = {"gliner_config": _minimal_gliner_config()}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="decoder")
        assert not result.report.is_valid

    def test_token_method_forces_span_mode(self, tmp_path):
        cfg = _minimal_gliner_config()
        cfg["span_mode"] = "markerV0"
        data = {"gliner_config": cfg}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="token")
        assert result.report.is_valid
        # The config should have had span_mode forced to token_level
        warnings = [w for w in result.report.warnings if "span_mode" in w.field and "forcing" in w.message]
        assert len(warnings) == 1

    def test_relex_without_relations_layer_errors(self, tmp_path):
        data = {"gliner_config": _minimal_gliner_config()}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="relex")
        assert not result.report.is_valid

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_and_validate_config(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_content(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("just a string")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_and_validate_config(p)

    def test_gliner_config_null(self, tmp_path):
        data = {"gliner_config": None}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="span")
        assert not result.report.is_valid
        assert any(e.field == "gliner_config" for e in result.report.errors)

    def test_gliner_config_non_mapping(self, tmp_path):
        for value in (["not", "a", "mapping"], "not-a-mapping"):
            data = {"gliner_config": value}
            cfg_path = _write_yaml(tmp_path, data)
            result = load_and_validate_config(cfg_path, full_or_lora="full", method="span")
            assert not result.report.is_valid
            assert any(e.field == "gliner_config" for e in result.report.errors)

    def test_lora_config_null_when_lora(self, tmp_path):
        data = {
            "gliner_config": _minimal_gliner_config(),
            "lora_config": None,
        }
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="lora", method="span")
        assert not result.report.is_valid
        assert any(e.field == "lora_config" for e in result.report.errors)

    def test_lora_config_non_mapping_when_lora(self, tmp_path):
        for value in (["not", "a", "mapping"], "not-a-mapping"):
            data = {
                "gliner_config": _minimal_gliner_config(),
                "lora_config": value,
            }
            cfg_path = _write_yaml(tmp_path, data)
            result = load_and_validate_config(cfg_path, full_or_lora="lora", method="span")
            assert not result.report.is_valid
            assert any(e.field == "lora_config" for e in result.report.errors)

    def test_missing_gliner_config_section(self, tmp_path):
        data = {"some_other_key": 42}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="span")
        assert not result.report.is_valid

    def test_gliner_config_model_type(self, tmp_path):
        """Verify the returned GLiNERConfig has the correct model_type property."""
        cfg = _minimal_gliner_config()
        data = {"gliner_config": cfg}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="span")
        assert result.report.is_valid
        # span mode with no labels_encoder/decoder/relations → uni_encoder_span
        assert result.gliner_config.model_type == "gliner_uni_encoder_span"

    @pytest.mark.parametrize(
        ("config_kwargs", "expected_model_type"),
        [
            ({"span_mode": "token_level"}, "gliner_uni_encoder_token"),
            (
                {"span_mode": "token_level", "labels_encoder": "sentence-transformers/all-MiniLM-L6-v2"},
                "gliner_bi_encoder_token",
            ),
            ({"span_mode": "token_level", "labels_decoder": "gpt2"}, "gliner_uni_encoder_token_decoder"),
            ({"span_mode": "token_level", "relations_layer": "simple"}, "gliner_uni_encoder_token_relex"),
        ],
    )
    def test_gliner_config_model_type_token_level(self, config_kwargs, expected_model_type):
        """Token-level configs should route to token-level model families."""
        cfg = GLiNERConfig(model_name="microsoft/deberta-v3-small", **config_kwargs)
        assert cfg.model_type == expected_model_type

    def test_config_result_has_raw_yaml(self, tmp_path):
        data = {"gliner_config": _minimal_gliner_config()}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="span")
        assert "gliner_config" in result.raw_yaml

    def test_config_result_attributes(self, tmp_path):
        data = {"gliner_config": _minimal_gliner_config()}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="span")
        assert result.full_or_lora == "full"
        assert result.method == "span"


# ============================================================================
# Tests: print_and_log_result (log file output)
# ============================================================================


class TestPrintAndLogResult:
    def test_log_file_created(self, tmp_path):
        data = {"gliner_config": _minimal_gliner_config()}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="span")
        log_path = print_and_log_result(result, cfg_path, log_dir=tmp_path)
        assert log_path.exists()
        log_data = json.loads(log_path.read_text())
        assert log_data["valid"] is True
        assert "resolved_gliner_config" in log_data

    def test_log_file_records_errors(self, tmp_path):
        data = {"gliner_config": {"name": "bad"}}  # missing model_name
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="span")
        log_path = print_and_log_result(result, cfg_path, log_dir=tmp_path)
        log_data = json.loads(log_path.read_text())
        assert log_data["valid"] is False
        assert log_data["error_count"] > 0

    def test_log_file_records_lora(self, tmp_path):
        data = {
            "gliner_config": _minimal_gliner_config(),
            "lora_config": _full_lora_config(),
        }
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="lora", method="span")
        log_path = print_and_log_result(result, cfg_path, log_dir=tmp_path)
        log_data = json.loads(log_path.read_text())
        assert "resolved_lora_config" in log_data
        assert log_data["resolved_lora_config"]["r"] == 16


# ============================================================================
# Tests: ValidationReport
# ============================================================================


class TestValidationReport:
    def test_empty_report_is_valid(self):
        r = ValidationReport()
        assert r.is_valid
        assert len(r.errors) == 0
        assert len(r.warnings) == 0

    def test_warning_does_not_invalidate(self):
        r = ValidationReport()
        r.add_warning("f", "msg")
        assert r.is_valid

    def test_error_invalidates(self):
        r = ValidationReport()
        r.add_error("f", "msg")
        assert not r.is_valid

    def test_mixed_issues(self):
        r = ValidationReport()
        r.add_warning("f1", "warn")
        r.add_error("f2", "err")
        assert not r.is_valid
        assert len(r.warnings) == 1
        assert len(r.errors) == 1


# ============================================================================
# Tests: Template YAML is itself valid
# ============================================================================


class TestTemplateYaml:
    """Ensure the shipped template.yaml is valid when loaded."""

    @pytest.fixture
    def template_path(self) -> Path:
        return Path(__file__).parent.parent / "template.yaml"

    def test_template_exists(self, template_path):
        assert template_path.exists(), f"template.yaml not found at {template_path}"

    def test_template_parses(self, template_path):
        with open(template_path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "gliner_config" in data or "model" in data

    def test_template_validates_span(self, template_path):
        result = load_and_validate_config(template_path, full_or_lora="full", method="span")
        assert result.report.is_valid, "Template failed validation:\n" + "\n".join(
            f"  {e.field}: {e.message}" for e in result.report.errors
        )

    def test_template_validates_lora(self, template_path):
        result = load_and_validate_config(template_path, full_or_lora="lora", method="span")
        assert result.report.is_valid, "Template failed lora validation:\n" + "\n".join(
            f"  {e.field}: {e.message}" for e in result.report.errors
        )

    def test_template_validates_token(self, template_path):
        result = load_and_validate_config(template_path, full_or_lora="full", method="token")
        assert result.report.is_valid

    def test_template_has_lora_section(self, template_path):
        with open(template_path) as f:
            data = yaml.safe_load(f)
        assert "lora_config" in data or "lora" in data


# ============================================================================
# Tests: CLI (typer app) integration
# ============================================================================


class TestCLI:
    """Test the Typer CLI invocation via CliRunner."""

    @pytest.fixture
    def runner(self):
        from typer.testing import CliRunner

        return CliRunner()

    @pytest.fixture
    def app(self):
        from ptbr.config_cli import _build_app

        return _build_app()

    def test_cli_validate_valid_config(self, runner, app, tmp_path):
        data = {"gliner_config": _minimal_gliner_config()}
        cfg_path = _write_yaml(tmp_path, data)
        result = runner.invoke(
            app,
            [
                "--file",
                str(cfg_path),
                "--validate",
                "--full-or-lora",
                "full",
                "--method",
                "span",
            ],
        )
        assert result.exit_code == 0

    def test_cli_validate_invalid_config(self, runner, app, tmp_path):
        data = {"gliner_config": {"name": "missing-model-name"}}
        cfg_path = _write_yaml(tmp_path, data)
        result = runner.invoke(
            app,
            [
                "--file",
                str(cfg_path),
                "--validate",
                "--full-or-lora",
                "full",
                "--method",
                "span",
            ],
        )
        assert result.exit_code != 0

    def test_cli_invalid_method(self, runner, app, tmp_path):
        data = {"gliner_config": _minimal_gliner_config()}
        cfg_path = _write_yaml(tmp_path, data)
        result = runner.invoke(
            app,
            [
                "--file",
                str(cfg_path),
                "--validate",
                "--full-or-lora",
                "full",
                "--method",
                "invalid_method",
            ],
        )
        assert result.exit_code != 0

    def test_cli_invalid_full_or_lora(self, runner, app, tmp_path):
        data = {"gliner_config": _minimal_gliner_config()}
        cfg_path = _write_yaml(tmp_path, data)
        result = runner.invoke(
            app,
            [
                "--file",
                str(cfg_path),
                "--validate",
                "--full-or-lora",
                "partial",
                "--method",
                "span",
            ],
        )
        assert result.exit_code != 0

    def test_cli_without_validate_flag(self, runner, app, tmp_path):
        data = {"gliner_config": _minimal_gliner_config()}
        cfg_path = _write_yaml(tmp_path, data)
        result = runner.invoke(
            app,
            [
                "--file",
                str(cfg_path),
                "--full-or-lora",
                "full",
                "--method",
                "span",
            ],
        )
        assert result.exit_code == 0
        assert "loaded" in result.output.lower() or "warning" in result.output.lower()

    def test_cli_lora_mode(self, runner, app, tmp_path):
        data = {
            "gliner_config": _minimal_gliner_config(),
            "lora_config": _full_lora_config(),
        }
        cfg_path = _write_yaml(tmp_path, data)
        result = runner.invoke(
            app,
            [
                "--file",
                str(cfg_path),
                "--validate",
                "--full-or-lora",
                "lora",
                "--method",
                "span",
            ],
        )
        assert result.exit_code == 0

    def test_cli_biencoder_method(self, runner, app, tmp_path):
        cfg = _minimal_gliner_config()
        cfg["labels_encoder"] = "microsoft/deberta-v3-small"
        data = {"gliner_config": cfg}
        cfg_path = _write_yaml(tmp_path, data)
        result = runner.invoke(
            app,
            [
                "--file",
                str(cfg_path),
                "--validate",
                "--full-or-lora",
                "full",
                "--method",
                "biencoder",
            ],
        )
        assert result.exit_code == 0

    def test_cli_token_method(self, runner, app, tmp_path):
        data = {"gliner_config": _minimal_gliner_config()}
        cfg_path = _write_yaml(tmp_path, data)
        result = runner.invoke(
            app,
            [
                "--file",
                str(cfg_path),
                "--validate",
                "--full-or-lora",
                "full",
                "--method",
                "token",
            ],
        )
        assert result.exit_code == 0


# ============================================================================
# Edge-case tests
# ============================================================================


class TestEdgeCases:
    def test_boolean_string_coercion_in_yaml(self, tmp_path):
        """YAML 'true'/'false' should parse as bools, but string 'True' might not."""
        cfg = _minimal_gliner_config()
        data = {"gliner_config": cfg}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="span")
        assert result.report.is_valid

    def test_numeric_string_in_int_field(self, tmp_path):
        """String '512' in an int field should coerce."""
        cfg = _minimal_gliner_config()
        cfg["hidden_size"] = "512"
        data = {"gliner_config": cfg}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="span")
        assert result.report.is_valid

    def test_all_methods(self, tmp_path):
        """Verify each method validates with appropriate config."""
        for method, extra in [
            ("span", {}),
            ("token", {}),
            ("biencoder", {"labels_encoder": "model"}),
            ("decoder", {"labels_decoder": "model"}),
            ("relex", {"relations_layer": "layer_name"}),
        ]:
            cfg = _minimal_gliner_config()
            cfg.update(extra)
            data = {"gliner_config": cfg}
            cfg_path = _write_yaml(tmp_path, data, filename=f"cfg_{method}.yaml")
            result = load_and_validate_config(cfg_path, full_or_lora="full", method=method)
            assert result.report.is_valid, f"Method {method} failed:\n" + "\n".join(
                f"  {e.field}: {e.message}" for e in result.report.errors
            )

    def test_multiple_errors_collected(self, tmp_path):
        """Multiple bad fields should all be reported, not just the first."""
        cfg = {
            # Missing model_name (REQUIRED)
            "dropout": 99.0,  # out of range
            "span_mode": "bogus",  # invalid literal
            "max_width": -5,  # out of range
        }
        data = {"gliner_config": cfg}
        cfg_path = _write_yaml(tmp_path, data)
        result = load_and_validate_config(cfg_path, full_or_lora="full", method="span")
        assert not result.report.is_valid
        error_fields = {e.field.split(".")[-1] for e in result.report.errors}
        assert "model_name" in error_fields
        assert "dropout" in error_fields
        assert "span_mode" in error_fields
        assert "max_width" in error_fields
