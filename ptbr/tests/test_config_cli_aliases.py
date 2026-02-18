"""Regression tests for config_cli section aliases.

These tests avoid importing heavy DL dependencies by stubbing ``gliner.config``
before importing ``ptbr.config_cli``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import yaml


class _FakeGLiNERConfig:
    def __init__(self, **kwargs):
        self._kwargs = dict(kwargs)

    def to_dict(self):
        return dict(self._kwargs)


def _reload_config_cli_with_stub(monkeypatch):
    sys.modules.pop("ptbr.config_cli", None)

    fake_gliner_pkg = ModuleType("gliner")
    fake_gliner_pkg.__path__ = []  # mark as package for ``gliner.config`` imports

    fake_gliner_config = ModuleType("gliner.config")
    fake_gliner_config.GLiNERConfig = _FakeGLiNERConfig
    fake_gliner_pkg.config = fake_gliner_config

    monkeypatch.setitem(sys.modules, "gliner", fake_gliner_pkg)
    monkeypatch.setitem(sys.modules, "gliner.config", fake_gliner_config)
    return importlib.import_module("ptbr.config_cli")


def _write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def test_model_section_alias_is_accepted(tmp_path, monkeypatch):
    config_cli = _reload_config_cli_with_stub(monkeypatch)
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(cfg_path, {"model": {"model_name": "microsoft/deberta-v3-small"}})

    result = config_cli.load_and_validate_config(cfg_path, full_or_lora="full", method="span")

    assert result.report.is_valid
    assert result.validated_gliner["model_name"] == "microsoft/deberta-v3-small"
    assert any(
        "alias for 'gliner_config'" in warning.message for warning in result.report.warnings
    )


def test_lora_section_alias_is_accepted_in_lora_mode(tmp_path, monkeypatch):
    config_cli = _reload_config_cli_with_stub(monkeypatch)
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(
        cfg_path,
        {
            "model": {"model_name": "microsoft/deberta-v3-small"},
            "lora": {"r": 16, "lora_alpha": 32},
        },
    )

    result = config_cli.load_and_validate_config(cfg_path, full_or_lora="lora", method="span")

    assert result.report.is_valid
    assert result.lora_config is not None
    assert result.lora_config["r"] == 16
    assert any(
        "alias for 'lora_config'" in warning.message for warning in result.report.warnings
    )


def test_canonical_sections_take_precedence_over_aliases(tmp_path, monkeypatch):
    config_cli = _reload_config_cli_with_stub(monkeypatch)
    cfg_path = tmp_path / "cfg.yaml"
    _write_yaml(
        cfg_path,
        {
            "gliner_config": {"model_name": "canonical-model"},
            "model": {"model_name": "alias-model"},
            "lora_config": {"r": 12},
            "lora": {"r": 99},
        },
    )

    result = config_cli.load_and_validate_config(cfg_path, full_or_lora="lora", method="span")

    assert result.report.is_valid
    assert result.validated_gliner["model_name"] == "canonical-model"
    assert result.lora_config is not None
    assert result.lora_config["r"] == 12
    assert any(
        "Both 'gliner_config' and 'model' sections found" in warning.message
        for warning in result.report.warnings
    )
    assert any(
        "Both 'lora_config' and 'lora' sections found" in warning.message
        for warning in result.report.warnings
    )
