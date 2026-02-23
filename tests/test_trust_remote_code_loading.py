"""Tests for trust_remote_code propagation in GLiNER loaders."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from gliner import GLiNER
from gliner.modeling.decoder import DecoderTransformer
from gliner.modeling.encoder import Transformer


class _FakeConfig:
    """Minimal config object whose class name maps to AutoModel path."""


class _FakeDecoderConfig:
    """Minimal decoder config object for AutoModelForCausalLM path."""


def test_transformer_uses_config_trust_remote_code_flag():
    encoder_config = _FakeConfig()
    config = SimpleNamespace(
        labels_encoder_config=None,
        encoder_config=encoder_config,
        _attn_implementation=None,
        vocab_size=-1,
        fuse_layers=False,
        trust_remote_code=True,
    )
    fake_model = SimpleNamespace(config=SimpleNamespace(hidden_size=16))

    with patch("gliner.modeling.encoder.AutoModel.from_config", return_value=fake_model) as mocked:
        Transformer("dummy-model", config, from_pretrained=False)

    mocked.assert_called_once_with(encoder_config, trust_remote_code=True)


def test_decoder_transformer_uses_config_trust_remote_code_flag():
    decoder_config = _FakeDecoderConfig()
    config = SimpleNamespace(
        labels_decoder_config=decoder_config,
        trust_remote_code=True,
    )
    fake_model = SimpleNamespace(config=SimpleNamespace(hidden_size=16))

    with patch("gliner.modeling.decoder.AutoModelForCausalLM.from_config", return_value=fake_model) as mocked:
        DecoderTransformer("dummy-model", config, from_pretrained=False)

    mocked.assert_called_once_with(decoder_config, trust_remote_code=True)


def test_gliner_from_pretrained_forwards_trust_remote_code(tmp_path, monkeypatch):
    config_path = tmp_path / "gliner_config.json"
    config_path.write_text(json.dumps({}), encoding="utf-8")
    captured = {}

    class _DummyGLiNERType:
        @classmethod
        def from_pretrained(cls, **kwargs):
            captured.update(kwargs)
            return "sentinel"

    monkeypatch.setattr(GLiNER, "_get_gliner_class", staticmethod(lambda _cfg: _DummyGLiNERType))

    out = GLiNER.from_pretrained(str(tmp_path), trust_remote_code=True)

    assert out == "sentinel"
    assert captured["trust_remote_code"] is True
