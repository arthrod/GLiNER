"""Tests for top-level ptbr CLI wiring."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from typer.testing import CliRunner


def _reload_main_module():
    sys.modules.pop("ptbr.__main__", None)
    return importlib.import_module("ptbr.__main__")


def test_config_validate_passes_path_to_summary_printer(tmp_path):
    main_module = _reload_main_module()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("gliner_config: {}\n", encoding="utf-8")

    fake_result = SimpleNamespace(report=SimpleNamespace(is_valid=True))
    fake_config_cli = ModuleType("ptbr.config_cli")
    fake_config_cli.load_and_validate_config = Mock(return_value=fake_result)
    fake_config_cli.print_and_log_result = Mock()

    runner = CliRunner()

    with patch.dict("sys.modules", {"ptbr.config_cli": fake_config_cli}):
        result = runner.invoke(
            main_module.app, ["config", "--file", str(cfg_path), "--validate"]
        )

    assert result.exit_code == 0
    fake_config_cli.print_and_log_result.assert_called_once()
    called_result, called_path = fake_config_cli.print_and_log_result.call_args.args[:2]
    assert called_result is fake_result
    assert isinstance(called_path, Path)
    assert called_path == cfg_path


def test_importing_main_does_not_import_training_cli():
    sys.modules.pop("ptbr.training_cli", None)
    _reload_main_module()

    assert "ptbr.training_cli" not in sys.modules


def test_attach_train_subcommand_is_idempotent():
    main_module = _reload_main_module()
    fake_train_cli = ModuleType("ptbr.training_cli")
    fake_train_app = object()
    fake_train_cli.app = fake_train_app

    with (
        patch.dict("sys.modules", {"ptbr.training_cli": fake_train_cli}),
        patch.object(main_module.app, "add_typer") as add_typer,
    ):
        main_module._attach_train_subcommand()
        main_module._attach_train_subcommand()

    add_typer.assert_called_once_with(fake_train_app, name="train")
