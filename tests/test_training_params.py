"""Tests for training parameter validation and forwarding.

These tests catch issues identified in the training parameters audit:

1. Default misalignment between create_training_args() and TrainingArguments
2. Critical parameters missing from create_training_args() explicit signature
3. GLiNER-incompatible HF Trainer defaults (remove_unused_columns, eval_strategy)
4. train.py not forwarding config parameters to model.train_model()
5. Dead config fields that are never consumed by the training pipeline
6. Label smoothing name collision between GLiNER and HF mechanisms
"""

import ast
import inspect

import pytest
import transformers
from pathlib import Path

from gliner.training.trainer import TrainingArguments
from gliner.model import BaseGLiNER


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "train.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _explicit_params(func):
    """Return the set of explicit parameter names (excluding *args / **kwargs)."""
    sig = inspect.signature(func)
    return {
        name
        for name, p in sig.parameters.items()
        if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }


def _get_train_model_call():
    """Parse train.py and return the AST Call node for *.train_model()."""
    source = TRAIN_SCRIPT.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "train_model"
        ):
            return node
    return None


def _get_train_model_kwarg_names():
    """Extract keyword-argument names from the model.train_model() call."""
    call = _get_train_model_call()
    if call is None:
        return set()
    return {kw.arg for kw in call.keywords if kw.arg is not None}


def _make_args(**overrides):
    """Shortcut: create TrainingArguments via the factory with safe defaults."""
    defaults = dict(output_dir="/tmp/test_gliner_params", use_cpu=True, report_to="none")
    defaults.update(overrides)
    return BaseGLiNER.create_training_args(**defaults)


# ---------------------------------------------------------------------------
# 1. Default Alignment
# ---------------------------------------------------------------------------

class TestTrainingArgsDefaultAlignment:
    """Defaults between create_training_args() and TrainingArguments must match.

    When create_training_args() is called without specifying a parameter, the
    resulting TrainingArguments value should equal the dataclass field default.
    A mismatch means the factory silently overrides the upstream default.
    """

    def test_masking_default_alignment(self):
        """create_training_args defaults masking='none'; TrainingArguments defaults 'global'."""
        ta_default = TrainingArguments.__dataclass_fields__["masking"].default

        sig = inspect.signature(BaseGLiNER.create_training_args)
        factory_default = sig.parameters["masking"].default

        assert factory_default == ta_default, (
            f"masking default mismatch: create_training_args='{factory_default}' "
            f"vs TrainingArguments='{ta_default}'"
        )


# ---------------------------------------------------------------------------
# 2. Explicit Parameter Exposure
# ---------------------------------------------------------------------------

_REQUIRED_EXPLICIT_PARAMS = [
    ("label_smoothing", "GLiNER custom loss param forwarded to model.forward()"),
    ("fp16", "Precision flag; bf16 is explicit but fp16 is not"),
    ("seed", "Must reach TrainingArguments for Trainer-level reproducibility"),
    ("run_name", "Required for meaningful experiment-tracking run names"),
    ("eval_steps", "Controls when evaluation runs during training"),
    ("eval_strategy", "Required to enable evaluation during training"),
    ("gradient_checkpointing", "Critical for large-model memory management"),
    ("dataloader_pin_memory", "Dataloader tuning; user overrides silently lost"),
    ("dataloader_persistent_workers", "Dataloader tuning; user overrides silently lost"),
    ("dataloader_prefetch_factor", "Dataloader tuning; user overrides silently lost"),
    ("push_to_hub", "Required for Hugging Face Hub integration"),
    ("hub_model_id", "Required for Hugging Face Hub model identification"),
    ("gradient_accumulation_steps", "Standard param for effective batch-size scaling"),
    ("remove_unused_columns", "Must be False for GLiNER; critical safety parameter"),
]


class TestCreateTrainingArgsExplicitParams:
    """Critical training parameters must be explicit in create_training_args(),
    not hidden behind **kwargs where they are undiscoverable and fragile."""

    @pytest.mark.parametrize(
        "param_name,reason",
        _REQUIRED_EXPLICIT_PARAMS,
        ids=[p[0] for p in _REQUIRED_EXPLICIT_PARAMS],
    )
    def test_param_is_explicit(self, param_name, reason):
        params = _explicit_params(BaseGLiNER.create_training_args)
        assert param_name in params, (
            f"'{param_name}' is not an explicit parameter in "
            f"create_training_args(). Reason it should be: {reason}"
        )


# ---------------------------------------------------------------------------
# 3. GLiNER-Critical Defaults
# ---------------------------------------------------------------------------

class TestGLiNERCriticalDefaults:
    """Defaults produced by create_training_args() must be safe for GLiNER.

    GLiNER uses custom batch dictionaries and needs evaluation enabled; the
    HF Trainer defaults are wrong for both of these.
    """

    def test_remove_unused_columns_is_false(self):
        """GLiNER uses custom batch dicts; HF default True silently drops columns."""
        args = _make_args()
        assert args.remove_unused_columns is False, (
            f"remove_unused_columns={args.remove_unused_columns}; "
            "must be False for GLiNER's custom batch dictionaries"
        )

    def test_evaluation_enabled_when_save_steps_configured(self):
        """Setting save_steps without enabling eval means checkpoints are saved
        but the model is never evaluated during training."""
        args = _make_args(save_steps=500)

        strategy = getattr(args, "eval_strategy", None)
        if strategy is None:
            strategy = getattr(args, "evaluation_strategy", "no")

        strategy_val = strategy.value if hasattr(strategy, "value") else str(strategy)

        assert strategy_val != "no", (
            f"eval_strategy='{strategy_val}'; evaluation never runs "
            "even though save_steps=500 is configured"
        )

    def test_seed_forwarded_via_kwargs(self):
        """seed passed through **kwargs must reach TrainingArguments."""
        args = _make_args(seed=12345)
        assert args.seed == 12345, (
            f"seed={args.seed}; expected 12345. "
            "Seed must be forwarded to TrainingArguments for reproducibility."
        )


# ---------------------------------------------------------------------------
# 4. train.py Parameter Forwarding
# ---------------------------------------------------------------------------

class TestTrainScriptForwarding:
    """train.py must forward all relevant config parameters to model.train_model()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.forwarded = _get_train_model_kwarg_names()

    def test_train_model_call_exists(self):
        """Sanity: train.py must contain a model.train_model() call."""
        assert len(self.forwarded) > 0, (
            "No model.train_model() call found in train.py"
        )

    @pytest.mark.parametrize("param_name", [
        "label_smoothing",
        "remove_unused_columns",
        "dataloader_num_workers",
        "report_to",
        "use_cpu",
    ])
    def test_param_forwarded_to_train_model(self, param_name):
        """Critical parameter must appear as a keyword arg in model.train_model()."""
        assert param_name in self.forwarded, (
            f"'{param_name}' is not passed to model.train_model() in train.py"
        )

    def test_eval_batch_size_independent_of_train_batch_size(self):
        """per_device_eval_batch_size should use a dedicated config field.

        Currently train.py sets per_device_eval_batch_size=cfg.training.train_batch_size
        which prevents independent eval-batch sizing.
        """
        call_node = _get_train_model_call()
        assert call_node is not None

        for kw in call_node.keywords:
            if kw.arg == "per_device_eval_batch_size":
                value_source = ast.dump(kw.value)
                assert "train_batch_size" not in value_source, (
                    "per_device_eval_batch_size references 'train_batch_size' "
                    "instead of a dedicated eval-batch-size config field"
                )
                return

        pytest.fail("per_device_eval_batch_size not found in model.train_model() call")

    def test_eval_strategy_forwarded(self):
        """train.py must forward eval_strategy (or evaluation_strategy) so
        evaluation actually runs during training."""
        assert (
            "eval_strategy" in self.forwarded
            or "evaluation_strategy" in self.forwarded
        ), (
            "Neither eval_strategy nor evaluation_strategy is forwarded "
            "in train.py's model.train_model() call; evaluation never runs"
        )

    def test_eval_steps_forwarded(self):
        """train.py must forward eval_steps so evaluation frequency is controlled."""
        assert "eval_steps" in self.forwarded, (
            "'eval_steps' is not forwarded in train.py; even if eval_strategy "
            "were set, the Trainer wouldn't know when to evaluate"
        )

    def test_seed_forwarded(self):
        """train.py must forward seed for full Trainer-level reproducibility."""
        assert "seed" in self.forwarded, (
            "'seed' is not forwarded in train.py; Trainer shuffle/dropout "
            "seed differs from torch.manual_seed()"
        )


# ---------------------------------------------------------------------------
# 5. Dead Config Fields
# ---------------------------------------------------------------------------

_DEAD_FIELDS = ["size_sup", "shuffle_types", "random_drop"]


class TestDeadConfigFields:
    """Config fields must be consumed by the training pipeline, not silently ignored.

    These fields exist in configs/config.yaml under ``training:`` but are never
    forwarded to model.train_model() and have no corresponding attribute on
    TrainingArguments.
    """

    @pytest.mark.parametrize("field_name", _DEAD_FIELDS)
    def test_field_forwarded_to_train_model(self, field_name):
        """Config training field must be forwarded to model.train_model()."""
        forwarded = _get_train_model_kwarg_names()
        assert field_name in forwarded, (
            f"Config field 'training.{field_name}' is defined in config.yaml "
            f"but never forwarded to model.train_model()"
        )

    @pytest.mark.parametrize("field_name", _DEAD_FIELDS)
    def test_field_is_valid_training_arg(self, field_name):
        """Config training field must be a recognised TrainingArguments attribute."""
        is_gliner_field = field_name in TrainingArguments.__dataclass_fields__
        is_hf_field = hasattr(transformers.TrainingArguments, field_name)

        assert is_gliner_field or is_hf_field, (
            f"'{field_name}' is defined in config.yaml training section but is "
            f"not a field on TrainingArguments — it is a dead configuration option"
        )


# ---------------------------------------------------------------------------
# 6. Label Smoothing
# ---------------------------------------------------------------------------

class TestLabelSmoothing:
    """Label smoothing must be properly wired and guarded against collisions.

    GLiNER defines its own ``label_smoothing`` on TrainingArguments (applied in
    model.forward()), while HF has ``label_smoothing_factor`` (applied in the
    Trainer's loss computation).  Both can be active simultaneously.
    """

    def test_gliner_label_smoothing_field_exists(self):
        """GLiNER's custom TrainingArguments must define label_smoothing."""
        assert "label_smoothing" in TrainingArguments.__dataclass_fields__

    @pytest.mark.skipif(
        not hasattr(transformers.TrainingArguments, "label_smoothing_factor"),
        reason="HF TrainingArguments lacks label_smoothing_factor in this version",
    )
    def test_no_unguarded_dual_smoothing(self):
        """Setting both label_smoothing AND label_smoothing_factor should be
        prevented or produce a warning.  Currently both are silently accepted,
        applying smoothing at two independent levels (model + Trainer).
        """
        args = _make_args(label_smoothing=0.1, label_smoothing_factor=0.2)

        both_active = args.label_smoothing > 0 and args.label_smoothing_factor > 0
        assert not both_active, (
            f"Both label_smoothing={args.label_smoothing} and "
            f"label_smoothing_factor={args.label_smoothing_factor} are active "
            "simultaneously. Dual smoothing applies at model AND Trainer level."
        )


# ---------------------------------------------------------------------------
# Smoke Tests
# ---------------------------------------------------------------------------

class TestSmoke:
    """Smoke tests: verify the training-args API is importable and functional."""

    def test_create_training_args_returns_valid_instance(self):
        args = _make_args()
        assert isinstance(args, TrainingArguments)
        assert isinstance(args, transformers.TrainingArguments)

    def test_training_arguments_has_all_gliner_fields(self):
        """TrainingArguments must expose every GLiNER-specific loss/training field."""
        gliner_fields = [
            "focal_loss_alpha",
            "focal_loss_gamma",
            "focal_loss_prob_margin",
            "label_smoothing",
            "loss_reduction",
            "negatives",
            "masking",
            "others_lr",
            "others_weight_decay",
        ]
        for field in gliner_fields:
            assert field in TrainingArguments.__dataclass_fields__, (
                f"TrainingArguments missing GLiNER-specific field: {field}"
            )

    def test_kwargs_passthrough_works(self):
        """Verify that **kwargs in create_training_args reaches TrainingArguments."""
        args = _make_args(label_smoothing=0.15)
        assert args.label_smoothing == pytest.approx(0.15), (
            "label_smoothing not forwarded through **kwargs"
        )
