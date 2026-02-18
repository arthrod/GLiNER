"""Tests for GLiNER configuration validation issues.

These tests deterministically catch issues identified in the GLiNER Model
Configuration Report.  Every test is written to **pass against the current
code** while documenting the incorrect behaviour.  When a bug is fixed the
corresponding test(s) will fail, signalling that the assertion should be
updated to the corrected behaviour.

Bug-detection tests follow a naming convention:
    test_bug_<short_description>

Coverage target: gliner/config.py — all config subclasses and the legacy
GLiNERConfig.model_type routing property.

Issues covered
--------------
1. Critical: GLiNERConfig.model_type uses "token-level" (hyphen) while typed
   subclasses use "token_level" (underscore) → silent architecture misrouting.
2. GLiNERConfig with span_mode="token_level" produces wrong model_type for
   every architecture variant (decoder, bi-encoder, relex, plain).
3. UniEncoderSpanConfig rejects "token_level" (underscore) but silently
   accepts "token-level" (hyphen) — inconsistent guard.
4. represent_spans is hardcoded True in UniEncoderTokenDecoderConfig but
   GLiNERConfig does not enforce the same constraint.
5. UniEncoderTokenDecoderConfig.model_type is "gliner_encoder_token_decoder"
   (missing "uni_") while all sibling configs use "uni_encoder".
6. GLiNERConfig accepts any arbitrary span_mode string without validation.
7. YAML config file config_token.yaml uses "token_level" (underscore) which
   triggers the misrouting bug when loaded through GLiNERConfig.
8. Default value reference: decoder_mode upstream default is None.
"""

import os

import yaml
import pytest

from gliner.config import (
    BaseGLiNERConfig,
    BiEncoderConfig,
    BiEncoderSpanConfig,
    BiEncoderTokenConfig,
    GLiNERConfig,
    UniEncoderConfig,
    UniEncoderRelexConfig,
    UniEncoderSpanConfig,
    UniEncoderSpanDecoderConfig,
    UniEncoderSpanRelexConfig,
    UniEncoderTokenConfig,
    UniEncoderTokenDecoderConfig,
    UniEncoderTokenRelexConfig,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")


# ===================================================================
# Helpers
# ===================================================================

def _load_yaml(filename):
    path = os.path.join(CONFIGS_DIR, filename)
    with open(path) as f:
        return yaml.safe_load(f)


# ===================================================================
# 1. Critical Bug — "token-level" (hyphen) vs "token_level" (underscore)
#
# GLiNERConfig.model_type property (lines 327-338) compares against
# "token-level" (hyphen).  All typed config subclasses set span_mode to
# "token_level" (underscore).  The two conventions never match, so
# GLiNERConfig always falls through to the *span* branch when the
# underscore form is used.
# ===================================================================


class TestTokenLevelHyphenVsUnderscoreBug:
    """Catch the critical hyphen-vs-underscore routing mismatch."""

    # -- typed subclasses: canonical form is underscore ----------------

    def test_typed_subclasses_all_use_underscore(self):
        """All typed token subclasses set span_mode = 'token_level' (underscore).
        This establishes the canonical form used everywhere except
        GLiNERConfig.model_type."""
        for cls in (UniEncoderTokenConfig, UniEncoderTokenDecoderConfig,
                    UniEncoderTokenRelexConfig, BiEncoderTokenConfig):
            cfg = cls()
            assert cfg.span_mode == "token_level", (
                f"{cls.__name__}.span_mode = {cfg.span_mode!r}"
            )

    # -- GLiNERConfig routing uses hyphen (the non-canonical form) -----

    def test_glinerconfig_hyphen_routes_to_token(self):
        """GLiNERConfig with the *hyphen* form routes correctly to a token
        model type.  This proves the property checks for 'token-level'."""
        cfg = GLiNERConfig(span_mode="token-level")
        assert cfg.model_type == "gliner_uni_encoder_token"

    # -- BUG: underscore form misroutes in every branch ----------------

    def test_bug_underscore_plain_misroutes_to_span(self):
        """BUG: GLiNERConfig(span_mode='token_level') returns
        'gliner_uni_encoder_span' instead of 'gliner_uni_encoder_token'.

        Root cause: model_type property line 338 checks
        ``self.span_mode == "token-level"`` (hyphen) which does not match
        "token_level" (underscore), so the else branch returns the span type.

        When this bug is fixed this test will fail — update the assertion to
        expect 'gliner_uni_encoder_token'.
        """
        cfg = GLiNERConfig(span_mode="token_level")
        # Current (wrong) behaviour:
        assert cfg.model_type == "gliner_uni_encoder_span", (
            "If this fails the hyphen/underscore bug may have been fixed. "
            f"Got model_type={cfg.model_type!r}."
        )

    def test_bug_underscore_decoder_misroutes_to_span_decoder(self):
        """BUG: GLiNERConfig(labels_decoder=..., span_mode='token_level')
        returns 'gliner_uni_encoder_span_decoder' instead of the token
        decoder variant.

        When fixed, update to expect the token decoder model_type.
        """
        cfg = GLiNERConfig(labels_decoder="some-decoder", span_mode="token_level")
        assert cfg.model_type == "gliner_uni_encoder_span_decoder", (
            f"Got model_type={cfg.model_type!r}."
        )

    def test_bug_underscore_biencoder_misroutes_to_span(self):
        """BUG: GLiNERConfig(labels_encoder=..., span_mode='token_level')
        returns 'gliner_bi_encoder_span' instead of 'gliner_bi_encoder_token'.
        """
        cfg = GLiNERConfig(labels_encoder="some-encoder", span_mode="token_level")
        assert cfg.model_type == "gliner_bi_encoder_span", (
            f"Got model_type={cfg.model_type!r}."
        )

    def test_bug_underscore_relex_misroutes_to_span_relex(self):
        """BUG: GLiNERConfig(relations_layer=..., span_mode='token_level')
        returns 'gliner_uni_encoder_span_relex' instead of the token relex
        variant.
        """
        cfg = GLiNERConfig(relations_layer="some-layer", span_mode="token_level")
        assert cfg.model_type == "gliner_uni_encoder_span_relex", (
            f"Got model_type={cfg.model_type!r}."
        )

    def test_bug_canonical_form_does_not_route_to_token(self):
        """BUG: The canonical underscore form used by all typed subclasses
        does not route to any token model type through GLiNERConfig.

        This is the summary assertion: the form every typed subclass uses
        ('token_level') does not produce a model_type containing 'token'
        (without 'span') in GLiNERConfig.
        """
        cfg = GLiNERConfig(span_mode="token_level")
        routes_to_token = ("token" in cfg.model_type
                           and "span" not in cfg.model_type)
        # Current (wrong) behaviour — does NOT route to token
        assert not routes_to_token, (
            "If this fails the bug may be fixed. "
            f"model_type={cfg.model_type!r}."
        )


# ===================================================================
# 2. GLiNERConfig vs Typed Subclass — model_type consistency
#
# For every architecture variant we compare the model_type returned by
# GLiNERConfig (legacy auto-detect) to the model_type set by the
# equivalent typed subclass.  Span-mode variants match; token-mode
# variants diverge because of bug #1.
# ===================================================================


class TestModelTypeConsistency:
    """GLiNERConfig should produce the same model_type as the equivalent
    typed config subclass.  Span variants pass; token variants fail due
    to the hyphen/underscore bug."""

    # -- span variants: these DO match (no bug) ------------------------

    def test_span_uni_encoder(self):
        legacy = GLiNERConfig(span_mode="markerV0")
        typed = UniEncoderSpanConfig(span_mode="markerV0")
        assert legacy.model_type == typed.model_type

    def test_span_decoder(self):
        legacy = GLiNERConfig(labels_decoder="d", span_mode="markerV0")
        typed = UniEncoderSpanDecoderConfig(labels_decoder="d", span_mode="markerV0")
        assert legacy.model_type == typed.model_type

    def test_span_biencoder(self):
        legacy = GLiNERConfig(labels_encoder="e", span_mode="markerV0")
        typed = BiEncoderSpanConfig(labels_encoder="e", span_mode="markerV0")
        assert legacy.model_type == typed.model_type

    def test_span_relex(self):
        legacy = GLiNERConfig(relations_layer="r", span_mode="markerV0")
        typed = UniEncoderSpanRelexConfig(relations_layer="r", span_mode="markerV0")
        assert legacy.model_type == typed.model_type

    # -- token variants: these DO NOT match (bug #1) -------------------

    def test_bug_token_uni_encoder_mismatch(self):
        """BUG: GLiNERConfig model_type != UniEncoderTokenConfig model_type
        when using the canonical underscore span_mode."""
        typed = UniEncoderTokenConfig()
        legacy = GLiNERConfig(span_mode=typed.span_mode)
        # They should be equal, but aren't:
        assert legacy.model_type != typed.model_type, (
            "If equal, the bug may be fixed."
        )

    def test_bug_token_decoder_mismatch(self):
        """BUG: GLiNERConfig model_type != UniEncoderTokenDecoderConfig
        model_type when using underscore span_mode."""
        typed = UniEncoderTokenDecoderConfig(labels_decoder="d")
        legacy = GLiNERConfig(labels_decoder="d", span_mode=typed.span_mode)
        assert legacy.model_type != typed.model_type, (
            "If equal, the bug may be fixed."
        )

    def test_bug_token_biencoder_mismatch(self):
        """BUG: GLiNERConfig model_type != BiEncoderTokenConfig model_type."""
        typed = BiEncoderTokenConfig(labels_encoder="e")
        legacy = GLiNERConfig(labels_encoder="e", span_mode=typed.span_mode)
        assert legacy.model_type != typed.model_type, (
            "If equal, the bug may be fixed."
        )

    def test_bug_token_relex_mismatch(self):
        """BUG: GLiNERConfig model_type != UniEncoderTokenRelexConfig model_type."""
        typed = UniEncoderTokenRelexConfig(relations_layer="r")
        legacy = GLiNERConfig(relations_layer="r", span_mode=typed.span_mode)
        assert legacy.model_type != typed.model_type, (
            "If equal, the bug may be fixed."
        )


# ===================================================================
# 3. Span config guard inconsistency — "token_level" rejected but
#    "token-level" silently accepted
# ===================================================================


class TestSpanModeGuardInconsistency:
    """UniEncoderSpanConfig, BiEncoderSpanConfig, and
    UniEncoderSpanRelexConfig reject span_mode='token_level' (underscore)
    with a ValueError.  But they do NOT reject 'token-level' (hyphen).
    Both strings represent the same semantic concept; the guard is
    incomplete.
    """

    @pytest.mark.parametrize("cls", [
        UniEncoderSpanConfig,
        UniEncoderSpanRelexConfig,
        BiEncoderSpanConfig,
    ], ids=lambda c: c.__name__)
    def test_underscore_rejected(self, cls):
        """Underscore form is correctly rejected."""
        with pytest.raises(ValueError, match="token_level"):
            cls(span_mode="token_level")

    @pytest.mark.parametrize("cls", [
        UniEncoderSpanConfig,
        UniEncoderSpanRelexConfig,
        BiEncoderSpanConfig,
    ], ids=lambda c: c.__name__)
    def test_bug_hyphen_not_rejected(self, cls):
        """BUG: Hyphen form bypasses the guard and is silently accepted.

        When fixed, this test will raise ValueError — update to expect
        the exception.
        """
        # Should raise ValueError like the underscore form, but doesn't:
        cfg = cls(span_mode="token-level")  # no error raised
        assert cfg.span_mode == "token-level"


# ===================================================================
# 4. represent_spans — GLiNERConfig vs UniEncoderTokenDecoderConfig
# ===================================================================


class TestRepresentSpans:
    """UniEncoderTokenDecoderConfig hardcodes represent_spans=True.
    GLiNERConfig with equivalent parameters does NOT enforce this."""

    def test_typed_token_decoder_forces_true(self):
        """UniEncoderTokenDecoderConfig always sets represent_spans=True,
        even when the caller explicitly passes False."""
        cfg = UniEncoderTokenDecoderConfig(represent_spans=False)
        assert cfg.represent_spans is True

    def test_base_default_is_false(self):
        """BaseGLiNERConfig defaults represent_spans to False."""
        assert BaseGLiNERConfig().represent_spans is False

    def test_bug_glinerconfig_does_not_enforce(self):
        """BUG: GLiNERConfig with token-decoder-equivalent params keeps
        represent_spans=False — inconsistent with the typed subclass.

        If the DL process relies on represent_spans=True for the token
        decoder architecture, this config will silently produce wrong
        training behaviour.

        When fixed, update to assert represent_spans is True.
        """
        cfg = GLiNERConfig(
            labels_decoder="d",
            span_mode="token_level",
            represent_spans=False,
        )
        # Current (wrong) behaviour — stays False:
        assert cfg.represent_spans is False, (
            "If True, the bug may be fixed."
        )


# ===================================================================
# 5. Naming inconsistency — UniEncoderTokenDecoderConfig.model_type
# ===================================================================


class TestTokenDecoderNaming:
    """UniEncoderTokenDecoderConfig sets model_type to
    'gliner_encoder_token_decoder' (missing 'uni_') while every other
    uni-encoder subclass includes 'uni_encoder' in its model_type."""

    def test_bug_missing_uni_prefix(self):
        """BUG: model_type lacks 'uni_encoder' prefix, breaking the naming
        convention shared by all sibling configs.

        When fixed, update to assert 'uni_encoder' IS in the model_type.
        """
        cfg = UniEncoderTokenDecoderConfig()
        assert "uni_encoder" not in cfg.model_type, (
            f"model_type={cfg.model_type!r} — if it now contains 'uni_encoder' "
            "the naming bug may be fixed."
        )

    def test_all_other_uni_configs_have_uni_prefix(self):
        """All other uni-encoder configs include 'uni_encoder'."""
        for cls in (UniEncoderSpanConfig, UniEncoderTokenConfig,
                    UniEncoderSpanDecoderConfig, UniEncoderSpanRelexConfig,
                    UniEncoderTokenRelexConfig):
            cfg = cls()
            assert "uni_encoder" in cfg.model_type, (
                f"{cls.__name__}.model_type = {cfg.model_type!r}"
            )

    def test_bug_glinerconfig_token_decoder_return_value_differs(self):
        """The model_type returned by GLiNERConfig for the token-decoder
        branch ('gliner_uni_encoder_token_decoder' via hyphen route) does
        NOT match UniEncoderTokenDecoderConfig.model_type
        ('gliner_encoder_token_decoder').

        This is a second consequence of the naming inconsistency: even when
        using the hyphen form to work around bug #1, the returned
        model_type still differs from the typed subclass.
        """
        legacy = GLiNERConfig(labels_decoder="d", span_mode="token-level")
        typed = UniEncoderTokenDecoderConfig(labels_decoder="d")
        # GLiNERConfig returns "gliner_uni_encoder_token_decoder" (via the
        # property string literal on line 328) while the typed subclass has
        # "gliner_encoder_token_decoder" (set at line 186).
        assert legacy.model_type != typed.model_type, (
            "If they match, the naming inconsistency may be fixed."
        )


# ===================================================================
# 6. GLiNERConfig accepts any arbitrary span_mode
# ===================================================================


class TestGLiNERConfigSpanModeValidation:
    """GLiNERConfig does not validate span_mode at all.  Any string is
    accepted and silently falls through to the span model type."""

    def test_bug_arbitrary_span_mode_accepted(self):
        """BUG: Completely invalid span_mode does not raise any error."""
        cfg = GLiNERConfig(span_mode="TOTALLY_INVALID")
        assert cfg.model_type == "gliner_uni_encoder_span"

    def test_bug_empty_string_accepted(self):
        """BUG: Empty string span_mode is accepted."""
        cfg = GLiNERConfig(span_mode="")
        assert cfg.model_type == "gliner_uni_encoder_span"


# ===================================================================
# 7. YAML config_token.yaml misrouting through GLiNERConfig
# ===================================================================


class TestYAMLConfigTokenMisrouting:
    """config_token.yaml uses span_mode: token_level (underscore).
    When loaded through GLiNERConfig the config is misrouted to the
    span architecture instead of the token architecture."""

    def test_yaml_uses_underscore_convention(self):
        """config_token.yaml specifies span_mode with underscore — the
        canonical form used by typed subclasses."""
        data = _load_yaml("config_token.yaml")
        assert data["model"]["span_mode"] == "token_level"

    def test_bug_yaml_token_config_misroutes(self):
        """BUG: config_token.yaml loaded into GLiNERConfig produces a
        *span* model_type, not a token model_type.

        This means that a user following the official config_token.yaml
        template and constructing a GLiNERConfig from it will get the
        wrong model architecture.

        When the bug is fixed, update to assert 'token' is in model_type
        and 'span' is not.
        """
        data = _load_yaml("config_token.yaml")
        cfg = GLiNERConfig(**data["model"])
        # Current (wrong) behaviour:
        assert cfg.model_type == "gliner_uni_encoder_span", (
            f"Got model_type={cfg.model_type!r}."
        )


class TestYAMLConfigsCorrectRouting:
    """Non-token YAML configs should route correctly through GLiNERConfig."""

    def test_config_span_routes_to_span(self):
        data = _load_yaml("config_span.yaml")
        cfg = GLiNERConfig(**data["model"])
        assert cfg.model_type == "gliner_uni_encoder_span"

    def test_config_decoder_routes_to_decoder(self):
        data = _load_yaml("config_decoder.yaml")
        cfg = GLiNERConfig(**data["model"])
        assert "decoder" in cfg.model_type

    def test_config_biencoder_routes_to_biencoder(self):
        data = _load_yaml("config_biencoder.yaml")
        cfg = GLiNERConfig(**data["model"])
        assert "bi_encoder" in cfg.model_type

    def test_all_yaml_configs_use_model_section(self):
        """All YAML configs use 'model:' section key (not 'gliner_config:').
        This documents the canonical section name for any external validator."""
        for fn in ("config.yaml", "config_token.yaml", "config_span.yaml",
                   "config_decoder.yaml", "config_biencoder.yaml",
                   "config_relex.yaml"):
            data = _load_yaml(fn)
            assert "model" in data, f"{fn} missing 'model' section"
            assert "gliner_config" not in data, (
                f"{fn} has 'gliner_config' — should use 'model'"
            )


# ===================================================================
# 8. Default value reference tests
# ===================================================================


class TestDefaultValues:
    """Verify upstream defaults so any external validator can be checked
    against the canonical source of truth."""

    UPSTREAM_BASE_DEFAULTS = {
        "model_name": "microsoft/deberta-v3-small",
        "name": "gliner",
        "max_width": 12,
        "hidden_size": 512,
        "dropout": 0.4,
        "fine_tune": True,
        "subtoken_pooling": "first",
        "span_mode": "markerV0",
        "post_fusion_schema": "",
        "num_post_fusion_layers": 1,
        "vocab_size": -1,
        "max_neg_type_ratio": 1,
        "max_types": 25,
        "max_len": 384,
        "words_splitter_type": "whitespace",
        "num_rnn_layers": 1,
        "fuse_layers": False,
        "embed_ent_token": True,
        "class_token_index": -1,
        "encoder_config": None,
        "ent_token": "<<ENT>>",
        "sep_token": "<<SEP>>",
        "_attn_implementation": None,
        "token_loss_coef": 1.0,
        "span_loss_coef": 1.0,
        "represent_spans": False,
        "neg_spans_ratio": 1.0,
    }

    @pytest.mark.parametrize("field,expected",
                             list(UPSTREAM_BASE_DEFAULTS.items()))
    def test_base_config_default(self, field, expected):
        cfg = BaseGLiNERConfig()
        actual = getattr(cfg, field)
        assert actual == expected, (
            f"BaseGLiNERConfig().{field} = {actual!r}, expected {expected!r}"
        )

    def test_decoder_mode_default_is_none(self):
        """Upstream default for decoder_mode is None, not 'span'.
        Any external validator that defaults to 'span' is wrong."""
        cfg = UniEncoderSpanDecoderConfig()
        assert cfg.decoder_mode is None

    def test_blank_entity_prob_default(self):
        assert UniEncoderSpanDecoderConfig().blank_entity_prob == 0.1

    def test_decoder_loss_coef_default(self):
        assert UniEncoderSpanDecoderConfig().decoder_loss_coef == 0.5

    def test_full_decoder_context_default(self):
        assert UniEncoderSpanDecoderConfig().full_decoder_context is True

    def test_relex_defaults(self):
        cfg = UniEncoderRelexConfig()
        assert cfg.relations_layer is None
        assert cfg.triples_layer is None
        assert cfg.embed_rel_token is True
        assert cfg.rel_token_index == -1
        assert cfg.rel_token == "<<REL>>"
        assert cfg.adjacency_loss_coef == 1.0
        assert cfg.relation_loss_coef == 1.0

    def test_biencoder_defaults(self):
        cfg = BiEncoderConfig()
        assert cfg.labels_encoder is None
        assert cfg.labels_encoder_config is None

    def test_glinerconfig_inherits_base_defaults(self):
        cfg = GLiNERConfig()
        for field, expected in self.UPSTREAM_BASE_DEFAULTS.items():
            actual = getattr(cfg, field)
            assert actual == expected, (
                f"GLiNERConfig().{field} = {actual!r}, expected {expected!r}"
            )


# ===================================================================
# 9. Field coverage — every subclass exposes expected attributes
# ===================================================================


class TestFieldCoverage:
    """Ensure all config subclasses expose the fields that downstream
    code (and validators) need."""

    BASE_FIELDS = [
        "model_name", "name", "max_width", "hidden_size", "dropout",
        "fine_tune", "subtoken_pooling", "span_mode", "post_fusion_schema",
        "num_post_fusion_layers", "vocab_size", "max_neg_type_ratio",
        "max_types", "max_len", "words_splitter_type", "num_rnn_layers",
        "fuse_layers", "embed_ent_token", "class_token_index",
        "encoder_config", "ent_token", "sep_token", "_attn_implementation",
        "token_loss_coef", "span_loss_coef", "represent_spans",
        "neg_spans_ratio",
    ]

    DECODER_FIELDS = [
        "labels_decoder", "decoder_mode", "full_decoder_context",
        "blank_entity_prob", "labels_decoder_config", "decoder_loss_coef",
    ]

    RELEX_FIELDS = [
        "relations_layer", "triples_layer", "embed_rel_token",
        "rel_token_index", "rel_token", "adjacency_loss_coef",
        "relation_loss_coef",
    ]

    def test_base_config_fields(self):
        cfg = BaseGLiNERConfig()
        for f in self.BASE_FIELDS:
            assert hasattr(cfg, f), f"BaseGLiNERConfig missing: {f}"

    def test_decoder_config_fields(self):
        cfg = UniEncoderSpanDecoderConfig()
        for f in self.DECODER_FIELDS:
            assert hasattr(cfg, f), (
                f"UniEncoderSpanDecoderConfig missing: {f}"
            )

    def test_token_decoder_inherits_decoder_fields(self):
        cfg = UniEncoderTokenDecoderConfig()
        for f in self.DECODER_FIELDS:
            assert hasattr(cfg, f), (
                f"UniEncoderTokenDecoderConfig missing: {f}"
            )

    def test_relex_config_fields(self):
        cfg = UniEncoderRelexConfig()
        for f in self.RELEX_FIELDS:
            assert hasattr(cfg, f), (
                f"UniEncoderRelexConfig missing: {f}"
            )

    def test_biencoder_has_labels_encoder_config(self):
        """labels_encoder_config is needed for reproducibility / Hub
        serialization but is missing from external validators."""
        cfg = BiEncoderConfig()
        assert hasattr(cfg, "labels_encoder_config")
        assert hasattr(cfg, "labels_encoder")

    def test_decoder_has_labels_decoder_config(self):
        """labels_decoder_config is needed for reproducibility / Hub
        serialization but is missing from external validators."""
        cfg = UniEncoderSpanDecoderConfig()
        assert hasattr(cfg, "labels_decoder_config")

    def test_glinerconfig_routing_fields(self):
        cfg = GLiNERConfig()
        assert hasattr(cfg, "labels_encoder")
        assert hasattr(cfg, "labels_decoder")
        assert hasattr(cfg, "relations_layer")


# ===================================================================
# 10. Cross-field validation behaviour
# ===================================================================


class TestCrossFieldBehaviour:
    """Document cross-field constraints that exist in typed subclasses
    but are absent from GLiNERConfig."""

    def test_span_configs_reject_token_level(self):
        for cls in (UniEncoderSpanConfig, UniEncoderSpanRelexConfig,
                    BiEncoderSpanConfig):
            with pytest.raises(ValueError):
                cls(span_mode="token_level")

    def test_token_configs_force_token_level(self):
        """Token configs override any provided span_mode to 'token_level'."""
        for cls in (UniEncoderTokenConfig, UniEncoderTokenDecoderConfig,
                    UniEncoderTokenRelexConfig, BiEncoderTokenConfig):
            cfg = cls(span_mode="markerV0")
            assert cfg.span_mode == "token_level"

    def test_routing_priority_decoder_over_encoder(self):
        cfg = GLiNERConfig(labels_decoder="d", labels_encoder="e",
                           relations_layer="r")
        assert "decoder" in cfg.model_type

    def test_routing_priority_encoder_over_relex(self):
        cfg = GLiNERConfig(labels_encoder="e", relations_layer="r")
        assert "bi_encoder" in cfg.model_type


# ===================================================================
# 11. Span mode allowed values
# ===================================================================


class TestSpanModeAllowedValues:
    """BaseGLiNERConfig and UniEncoderSpanConfig accept all non-token
    span modes.  This documents the full set so validators can reference it."""

    NON_TOKEN_MODES = sorted([
        "markerV0", "markerV1", "marker", "query", "mlp", "cat",
        "conv_conv", "conv_max", "conv_mean", "conv_sum", "conv_share",
    ])

    @pytest.mark.parametrize("mode", NON_TOKEN_MODES)
    def test_base_config_accepts(self, mode):
        cfg = BaseGLiNERConfig(span_mode=mode)
        assert cfg.span_mode == mode

    @pytest.mark.parametrize("mode", NON_TOKEN_MODES)
    def test_span_config_accepts(self, mode):
        cfg = UniEncoderSpanConfig(span_mode=mode)
        assert cfg.span_mode == mode

    @pytest.mark.parametrize("mode", NON_TOKEN_MODES)
    def test_glinerconfig_routes_to_span(self, mode):
        cfg = GLiNERConfig(span_mode=mode)
        assert cfg.model_type == "gliner_uni_encoder_span"


# ===================================================================
# 12. Class hierarchy
# ===================================================================


class TestConfigClassHierarchy:

    def test_uni_encoder_inherits_base(self):
        assert issubclass(UniEncoderConfig, BaseGLiNERConfig)

    def test_uni_encoder_span_inherits_uni(self):
        assert issubclass(UniEncoderSpanConfig, UniEncoderConfig)

    def test_uni_encoder_token_inherits_uni(self):
        assert issubclass(UniEncoderTokenConfig, UniEncoderConfig)

    def test_span_decoder_inherits_uni(self):
        assert issubclass(UniEncoderSpanDecoderConfig, UniEncoderConfig)

    def test_token_decoder_inherits_span_decoder(self):
        assert issubclass(UniEncoderTokenDecoderConfig,
                          UniEncoderSpanDecoderConfig)

    def test_relex_inherits_uni(self):
        assert issubclass(UniEncoderRelexConfig, UniEncoderConfig)

    def test_span_relex_inherits_relex(self):
        assert issubclass(UniEncoderSpanRelexConfig, UniEncoderRelexConfig)

    def test_token_relex_inherits_relex(self):
        assert issubclass(UniEncoderTokenRelexConfig, UniEncoderRelexConfig)

    def test_biencoder_inherits_base(self):
        assert issubclass(BiEncoderConfig, BaseGLiNERConfig)

    def test_biencoder_span_inherits_biencoder(self):
        assert issubclass(BiEncoderSpanConfig, BiEncoderConfig)

    def test_biencoder_token_inherits_biencoder(self):
        assert issubclass(BiEncoderTokenConfig, BiEncoderConfig)

    def test_glinerconfig_inherits_base(self):
        assert issubclass(GLiNERConfig, BaseGLiNERConfig)


# ===================================================================
# 13. Smoke test — import and instantiation
# ===================================================================


ALL_CONFIG_CLASSES = [
    BaseGLiNERConfig,
    UniEncoderConfig,
    UniEncoderSpanConfig,
    UniEncoderTokenConfig,
    UniEncoderSpanDecoderConfig,
    UniEncoderTokenDecoderConfig,
    UniEncoderRelexConfig,
    UniEncoderSpanRelexConfig,
    UniEncoderTokenRelexConfig,
    BiEncoderConfig,
    BiEncoderSpanConfig,
    BiEncoderTokenConfig,
    GLiNERConfig,
]


class TestSmokeInstantiation:

    @pytest.mark.parametrize("cls", ALL_CONFIG_CLASSES,
                             ids=lambda c: c.__name__)
    def test_instantiate_defaults(self, cls):
        cfg = cls()
        assert cfg is not None

    def test_config_module_importable(self):
        import gliner.config as mod
        assert hasattr(mod, "GLiNERConfig")
        assert hasattr(mod, "BaseGLiNERConfig")
