"""Tests for trust_remote_code forwarding in ptbr APIs."""

import json
from unittest.mock import patch

from typer.testing import CliRunner

from ptbr import prepare
from ptbr.__main__ import app


class _FakeModel:
    def encode_labels(self, labels):
        return _FakeEmbeddings(labels)


class _FakeEmbeddings:
    def __init__(self, labels):
        self.labels = labels
        self.shape = (len(labels), 1)


def _write_valid_dataset(path):
    path.write_text(
        json.dumps([{"tokenized_text": ["hello", "world"], "ner": [[0, 1, "GREETING"]]}]),
        encoding="utf-8",
    )


def test_prepare_forwards_trust_remote_code_true(tmp_path):
    data_path = tmp_path / "data.json"
    _write_valid_dataset(data_path)

    with patch("gliner.GLiNER.from_pretrained", return_value=_FakeModel()) as mocked:
        result = prepare(
            str(data_path),
            generate_label_embeddings="dummy/model",
            trust_remote_code=True,
        )

    mocked.assert_called_once_with("dummy/model", trust_remote_code=True)
    assert result.label_embeddings.labels == ["GREETING"]


def test_prepare_defaults_trust_remote_code_false(tmp_path):
    data_path = tmp_path / "data.json"
    _write_valid_dataset(data_path)

    with patch("gliner.GLiNER.from_pretrained", return_value=_FakeModel()) as mocked:
        prepare(str(data_path), generate_label_embeddings="dummy/model")

    mocked.assert_called_once_with("dummy/model", trust_remote_code=False)


def test_cli_forwards_trust_remote_code_flag(tmp_path):
    data_path = tmp_path / "data.json"
    emb_path = tmp_path / "embeddings.pt"
    labels_path = tmp_path / "labels.json"
    _write_valid_dataset(data_path)

    runner = CliRunner()
    with patch("gliner.GLiNER.from_pretrained", return_value=_FakeModel()) as mocked, patch("torch.save") as save_mock:
        result = runner.invoke(
            app,
            [
                "data",
                "--file-or-repo",
                str(data_path),
                "--generate-label-embeddings",
                "dummy/model",
                "--trust-remote-code",
                "--output-embeddings-path",
                str(emb_path),
                "--output-labels-path",
                str(labels_path),
            ],
        )

    assert result.exit_code == 0
    mocked.assert_called_once_with("dummy/model", trust_remote_code=True)
    save_mock.assert_called_once()
    assert labels_path.exists()
