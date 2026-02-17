"""Tests for ptbr.training_cli -- config validation, CLI flags, API checks."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
import yaml
from typer.testing import CliRunner

from ptbr.training_cli import (
    _check_type,
    _deep_get,
    _deep_set,
    app,
    check_huggingface,
    check_wandb,
    check_resume,
    print_summary,
    semantic_checks,
    validate_config,
    ValidationResult,
)

runner = CliRunner()

# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #

MINIMAL_VALID_CONFIG = {
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


@pytest.fixture()
def valid_cfg() -> dict:
    """Return a deep copy of the minimal valid config."""
    import copy
    return copy.deepcopy(MINIMAL_VALID_CONFIG)


@pytest.fixture()
def cfg_file(tmp_path: Path, valid_cfg: dict) -> Path:
    """Write the valid config to a temp YAML file and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(valid_cfg, default_flow_style=False))
    return p


# ------------------------------------------------------------------ #
# _deep_get / _deep_set                                               #
# ------------------------------------------------------------------ #

class TestDeepGetSet:
    def test_deep_get_found(self) -> None:
        d = {"a": {"b": {"c": 42}}}
        found, val = _deep_get(d, "a.b.c")
        assert found is True
        assert val == 42

    def test_deep_get_missing(self) -> None:
        d = {"a": {"b": 1}}
        found, val = _deep_get(d, "a.x")
        assert found is False
        assert val is None

    def test_deep_set_creates_intermediates(self) -> None:
        d: dict = {}
        _deep_set(d, "a.b.c", 99)
        assert d == {"a": {"b": {"c": 99}}}

    def test_deep_set_raises_when_parent_not_mapping(self) -> None:
        d = {"a": "not-a-dict"}
        with pytest.raises(ValueError):
            _deep_set(d, "a.b.c", 99)


# ------------------------------------------------------------------ #
# _check_type                                                         #
# ------------------------------------------------------------------ #

class TestCheckType:
    def test_int_as_float(self) -> None:
        assert _check_type(3, float) is True

    def test_float_as_float(self) -> None:
        assert _check_type(3.0, float) is True

    def test_bool_not_int(self) -> None:
        # bool is a subclass of int, but we don't want bools matching int
        assert _check_type(True, int) is False

    def test_bool_matches_bool(self) -> None:
        assert _check_type(True, bool) is True

    def test_none_in_union(self) -> None:
        assert _check_type(None, (str, type(None))) is True

    def test_str_for_int_fails(self) -> None:
        assert _check_type("hello", int) is False


# ------------------------------------------------------------------ #
# validate_config                                                      #
# ------------------------------------------------------------------ #

class TestValidateConfig:
    def test_minimal_valid(self, valid_cfg: dict) -> None:
        result = validate_config(valid_cfg)
        assert result.ok, f"Errors: {result.errors}"

    def test_missing_required_field(self, valid_cfg: dict) -> None:
        del valid_cfg["run"]["name"]
        result = validate_config(valid_cfg)
        assert not result.ok
        assert any("run.name" in e for e in result.errors)

    def test_required_field_set_to_none(self, valid_cfg: dict) -> None:
        valid_cfg["run"]["name"] = None
        result = validate_config(valid_cfg)
        assert not result.ok
        assert any("run.name" in e for e in result.errors)

    def test_wrong_type_errors(self, valid_cfg: dict) -> None:
        valid_cfg["training"]["num_steps"] = "not_an_int"
        result = validate_config(valid_cfg)
        assert not result.ok
        assert any("training.num_steps" in e for e in result.errors)

    def test_defaults_applied_with_warning(self, valid_cfg: dict) -> None:
        # scheduler_type is optional with default "cosine"
        assert "scheduler_type" not in valid_cfg.get("training", {})
        result = validate_config(valid_cfg)
        assert result.ok
        # The default should have been applied
        assert valid_cfg["training"]["scheduler_type"] == "cosine"
        # There should be a warning about it
        assert any("scheduler_type" in w for w in result.warnings)

    def test_all_required_missing(self) -> None:
        result = validate_config({})
        assert not result.ok
        required_keys = [
            "run.name", "model.model_name", "model.span_mode", "model.max_len",
            "data.root_dir", "data.train_data",
            "training.num_steps", "training.train_batch_size",
            "training.eval_every", "training.lr_encoder", "training.lr_others",
        ]
        for key in required_keys:
            assert any(key in e for e in result.errors), f"Expected error for {key}"

    def test_extra_keys_warned(self, valid_cfg: dict) -> None:
        valid_cfg["model"]["totally_unknown_param"] = True
        result = validate_config(valid_cfg)
        assert result.ok  # extra keys are warnings, not errors
        assert any("totally_unknown_param" in w for w in result.warnings)

    def test_none_for_optional_is_ok(self, valid_cfg: dict) -> None:
        valid_cfg["model"]["labels_encoder"] = None
        result = validate_config(valid_cfg)
        assert result.ok

    def test_sensitive_values_are_redacted(self, valid_cfg: dict) -> None:
        valid_cfg.setdefault("environment", {})["hf_token"] = "hf_secret_token"
        valid_cfg["environment"]["wandb_api_key"] = "wandb_secret_key"
        result = validate_config(valid_cfg)
        assert result.ok

        info_by_key = {key: value for key, _, value in result.info}
        assert info_by_key["environment.hf_token"] == "***REDACTED***"
        assert info_by_key["environment.wandb_api_key"] == "***REDACTED***"

        summary = print_summary(result)
        assert "hf_secret_token" not in summary
        assert "wandb_secret_key" not in summary


# ------------------------------------------------------------------ #
# semantic_checks                                                      #
# ------------------------------------------------------------------ #

class TestSemanticChecks:
    def test_invalid_span_mode(self, valid_cfg: dict) -> None:
        valid_cfg["model"]["span_mode"] = "invalid"
        result = ValidationResult()
        semantic_checks(valid_cfg, result)
        assert any("span_mode" in e for e in result.errors)

    def test_invalid_scheduler(self, valid_cfg: dict) -> None:
        valid_cfg["training"]["scheduler_type"] = "exponential"
        result = ValidationResult()
        semantic_checks(valid_cfg, result)
        assert any("scheduler_type" in e for e in result.errors)

    def test_invalid_optimizer(self, valid_cfg: dict) -> None:
        valid_cfg["training"]["optimizer"] = "rmsprop"
        result = ValidationResult()
        semantic_checks(valid_cfg, result)
        assert any("optimizer" in e for e in result.errors)

    def test_bf16_fp16_conflict(self, valid_cfg: dict) -> None:
        valid_cfg["training"]["bf16"] = True
        valid_cfg["training"]["fp16"] = True
        result = ValidationResult()
        semantic_checks(valid_cfg, result)
        assert any("bf16" in e and "fp16" in e for e in result.errors)

    def test_wandb_requires_project(self, valid_cfg: dict) -> None:
        valid_cfg.setdefault("environment", {})["report_to"] = "wandb"
        result = ValidationResult()
        semantic_checks(valid_cfg, result)
        assert any("wandb_project" in e for e in result.errors)

    def test_hub_requires_model_id(self, valid_cfg: dict) -> None:
        valid_cfg.setdefault("environment", {})["push_to_hub"] = True
        result = ValidationResult()
        semantic_checks(valid_cfg, result)
        assert any("hub_model_id" in e for e in result.errors)

    def test_positive_num_steps(self, valid_cfg: dict) -> None:
        valid_cfg["training"]["num_steps"] = 0
        result = ValidationResult()
        semantic_checks(valid_cfg, result)
        assert any("num_steps" in e for e in result.errors)

    def test_positive_lr(self, valid_cfg: dict) -> None:
        valid_cfg["training"]["lr_encoder"] = -1e-5
        result = ValidationResult()
        semantic_checks(valid_cfg, result)
        assert any("lr_encoder" in e for e in result.errors)

    def test_valid_enums_pass(self, valid_cfg: dict) -> None:
        valid_cfg["training"]["scheduler_type"] = "cosine"
        valid_cfg["training"]["optimizer"] = "adamw_torch"
        valid_cfg["training"]["loss_reduction"] = "sum"
        valid_cfg["training"]["masking"] = "none"
        valid_cfg.setdefault("environment", {})["report_to"] = "none"
        result = ValidationResult()
        semantic_checks(valid_cfg, result)
        assert len(result.errors) == 0


# ------------------------------------------------------------------ #
# check_huggingface                                                    #
# ------------------------------------------------------------------ #

class TestCheckHuggingFace:
    def test_skip_when_push_disabled(self, valid_cfg: dict) -> None:
        valid_cfg.setdefault("environment", {})["push_to_hub"] = False
        result = ValidationResult()
        check_huggingface(valid_cfg, result)
        assert result.ok

    def test_error_no_token(self, valid_cfg: dict) -> None:
        valid_cfg.setdefault("environment", {})["push_to_hub"] = True
        valid_cfg["environment"]["hf_token"] = None
        result = ValidationResult()
        with mock.patch.dict(os.environ, {}, clear=True):
            check_huggingface(valid_cfg, result)
        assert any("token" in e.lower() for e in result.errors)

    def test_success_with_mock(self, valid_cfg: dict) -> None:
        valid_cfg.setdefault("environment", {})["push_to_hub"] = True
        valid_cfg["environment"]["hf_token"] = "hf_testtoken"
        result = ValidationResult()

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"name": "testuser"}

        with mock.patch("requests.get", return_value=mock_resp):
            check_huggingface(valid_cfg, result)
        assert result.ok

    def test_failure_status(self, valid_cfg: dict) -> None:
        valid_cfg.setdefault("environment", {})["push_to_hub"] = True
        valid_cfg["environment"]["hf_token"] = "hf_badtoken"
        result = ValidationResult()

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"

        with mock.patch("requests.get", return_value=mock_resp):
            check_huggingface(valid_cfg, result)
        assert not result.ok

    def test_network_error(self, valid_cfg: dict) -> None:
        valid_cfg.setdefault("environment", {})["push_to_hub"] = True
        valid_cfg["environment"]["hf_token"] = "hf_token"
        result = ValidationResult()

        with mock.patch("requests.get", side_effect=ConnectionError("no network")):
            check_huggingface(valid_cfg, result)
        assert not result.ok


# ------------------------------------------------------------------ #
# check_wandb                                                         #
# ------------------------------------------------------------------ #

class TestCheckWandB:
    def test_skip_when_disabled(self, valid_cfg: dict) -> None:
        valid_cfg.setdefault("environment", {})["report_to"] = "none"
        result = ValidationResult()
        check_wandb(valid_cfg, result)
        assert result.ok

    def test_error_no_key(self, valid_cfg: dict) -> None:
        valid_cfg.setdefault("environment", {})["report_to"] = "wandb"
        valid_cfg["environment"]["wandb_api_key"] = None
        result = ValidationResult()
        with mock.patch.dict(os.environ, {}, clear=True):
            check_wandb(valid_cfg, result)
        assert any("api key" in e.lower() for e in result.errors)

    def test_success_with_mock(self, valid_cfg: dict) -> None:
        valid_cfg.setdefault("environment", {})["report_to"] = "wandb"
        valid_cfg["environment"]["wandb_api_key"] = "test-key"
        result = ValidationResult()

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {"viewer": {"username": "testuser"}}
        }

        with mock.patch("requests.post", return_value=mock_resp):
            check_wandb(valid_cfg, result)
        assert result.ok

    def test_failure_status(self, valid_cfg: dict) -> None:
        valid_cfg.setdefault("environment", {})["report_to"] = "all"
        valid_cfg["environment"]["wandb_api_key"] = "bad-key"
        result = ValidationResult()

        mock_resp = mock.MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        with mock.patch("requests.post", return_value=mock_resp):
            check_wandb(valid_cfg, result)
        assert not result.ok


# ------------------------------------------------------------------ #
# check_resume                                                         #
# ------------------------------------------------------------------ #

class TestCheckResume:
    def test_missing_run_name_in_current_config(
        self, valid_cfg: dict, tmp_path: Path
    ) -> None:
        valid_cfg["run"].pop("name", None)
        (tmp_path / "checkpoint-100").mkdir()
        result = ValidationResult()
        check_resume(valid_cfg, tmp_path, result)
        assert not result.ok
        assert any("cannot resume: 'run.name' is missing" in e.lower() for e in result.errors)

    def test_no_checkpoints(self, valid_cfg: dict, tmp_path: Path) -> None:
        result = ValidationResult()
        check_resume(valid_cfg, tmp_path, result)
        assert not result.ok
        assert any("no checkpoint" in e.lower() for e in result.errors)

    def test_no_saved_config(self, valid_cfg: dict, tmp_path: Path) -> None:
        (tmp_path / "checkpoint-100").mkdir()
        result = ValidationResult()
        check_resume(valid_cfg, tmp_path, result)
        assert not result.ok
        assert any("config.yaml" in e for e in result.errors)

    def test_name_mismatch(self, valid_cfg: dict, tmp_path: Path) -> None:
        (tmp_path / "checkpoint-100").mkdir()
        saved = {"run": {"name": "different-name"}}
        (tmp_path / "config.yaml").write_text(yaml.dump(saved))
        result = ValidationResult()
        check_resume(valid_cfg, tmp_path, result)
        assert not result.ok
        assert any("mismatch" in e for e in result.errors)

    def test_successful_resume(self, valid_cfg: dict, tmp_path: Path) -> None:
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "checkpoint-200").mkdir()
        saved = {"run": {"name": "test-run"}}
        (tmp_path / "config.yaml").write_text(yaml.dump(saved))
        result = ValidationResult()
        check_resume(valid_cfg, tmp_path, result)
        assert result.ok


# ------------------------------------------------------------------ #
# CLI integration (typer runner)                                       #
# ------------------------------------------------------------------ #

class TestCLI:
    def test_validate_valid_config(self, cfg_file: Path) -> None:
        result = runner.invoke(app, [str(cfg_file), "--validate"])
        assert result.exit_code == 0

    def test_validate_invalid_config(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.dump({"run": {"seed": 42}}))
        result = runner.invoke(app, [str(bad), "--validate"])
        assert result.exit_code == 1

    def test_missing_config_file(self) -> None:
        result = runner.invoke(app, ["/nonexistent/path.yaml", "--validate"])
        assert result.exit_code != 0

    def test_output_folder_must_be_empty(
        self, cfg_file: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        out.mkdir()
        (out / "some_file.txt").write_text("blocker")
        result = runner.invoke(
            app, [str(cfg_file), "--output-folder", str(out)]
        )
        assert result.exit_code == 1

    @mock.patch("ptbr.training_cli._launch_training")
    def test_output_folder_empty_ok(
        self,
        mock_launch: mock.MagicMock,
        cfg_file: Path,
        tmp_path: Path,
    ) -> None:
        """An empty output folder should be accepted for a new training run."""
        out = tmp_path / "output"
        out.mkdir()
        result = runner.invoke(app, [str(cfg_file), "--output-folder", str(out)])
        assert result.exit_code == 0
        mock_launch.assert_called_once()

    @mock.patch("ptbr.training_cli._launch_training")
    def test_output_folder_allows_validation_artifacts(
        self,
        mock_launch: mock.MagicMock,
        cfg_file: Path,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "output"
        out.mkdir()
        (out / "validation_20260101T000000Z.log").write_text("ok")
        (out / "summary_20260101T000000Z.txt").write_text("ok")
        result = runner.invoke(app, [str(cfg_file), "--output-folder", str(out)])
        assert result.exit_code == 0
        mock_launch.assert_called_once()

    def test_validate_writes_summary(self, cfg_file: Path, tmp_path: Path) -> None:
        result = runner.invoke(app, [str(cfg_file), "--validate"])
        assert result.exit_code == 0
        summaries = list(cfg_file.parent.glob("summary_*.txt"))
        assert len(summaries) >= 1

    def test_validate_writes_log(self, cfg_file: Path) -> None:
        result = runner.invoke(app, [str(cfg_file), "--validate"])
        assert result.exit_code == 0
        logs = list(cfg_file.parent.glob("validation_*.log"))
        assert len(logs) >= 1

    def test_resume_without_output_folder(self, cfg_file: Path) -> None:
        """--resume without --output-folder should still work for --validate."""
        result = runner.invoke(
            app, [str(cfg_file), "--validate", "--resume"]
        )
        # validate-only doesn't check resume (no output folder)
        assert result.exit_code == 0


# ------------------------------------------------------------------ #
# Edge cases                                                           #
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_empty_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        result = runner.invoke(app, [str(p), "--validate"])
        assert result.exit_code == 1

    def test_int_accepted_as_float_field(self, valid_cfg: dict) -> None:
        """loss_alpha accepts int or float."""
        valid_cfg["training"]["loss_alpha"] = 1
        result = validate_config(valid_cfg)
        assert result.ok

    def test_full_template_validates(self) -> None:
        """The shipped template.yaml must pass validation."""
        template = Path(__file__).resolve().parent.parent / "template.yaml"
        if not template.exists():
            pytest.skip("template.yaml not found")
        with open(template) as f:
            cfg = yaml.safe_load(f)
        result = validate_config(cfg)
        semantic_checks(cfg, result)
        assert result.ok, f"template.yaml errors: {result.errors}"

    def test_decoder_mode_warning(self, valid_cfg: dict) -> None:
        valid_cfg["model"]["labels_decoder"] = "gpt2"
        valid_cfg["model"]["decoder_mode"] = None
        result = ValidationResult()
        # Need to validate first so all fields exist
        validate_config(valid_cfg)
        semantic_checks(valid_cfg, result)
        assert any("decoder_mode" in w for w in result.warnings)
