from pathlib import Path


TRAINER_SOURCE = Path(__file__).resolve().parents[1] / "gliner" / "training" / "trainer.py"
MODEL_SOURCE = Path(__file__).resolve().parents[1] / "gliner" / "model.py"


def _method_block(source: str, method_name: str, next_method_name: str | None = None) -> str:
    _, after_start = source.split(f"def {method_name}", 1)
    if next_method_name is None:
        return after_start
    method_body, _ = after_start.split(f"def {next_method_name}", 1)
    return method_body


def test_train_dataloader_does_not_call_hf_column_pruning():
    source = TRAINER_SOURCE.read_text(encoding="utf-8")
    method_source = _method_block(source, "get_train_dataloader", "get_eval_dataloader")
    assert "_remove_unused_columns" not in method_source


def test_eval_dataloader_does_not_call_hf_column_pruning():
    source = TRAINER_SOURCE.read_text(encoding="utf-8")
    method_source = _method_block(source, "get_eval_dataloader")
    assert "_remove_unused_columns" not in method_source


def test_train_model_uses_custom_gliner_trainer():
    source = MODEL_SOURCE.read_text(encoding="utf-8")
    assert "trainer = Trainer(**trainer_kwargs)" in source
