
## Integration Plan - Worktrees to Dev

### eager-birch commits to cherry-pick (in order):
1. 667fdb0 - fix: repair noisy-jsonl validation block
2. 244dd09 - fix: pass Path to config summary printer
3. da56320 - fix: clean training_cli imports and handler reuse
4. 7d6f9bb - fix: honor configured output_dir in train.py
5. 9a6a005 - fix: honor eval_batch_size in train.py
6. 8445e6f - fix: honor bf16 config in train.py
7. 4808879 - fix: forward label_smoothing in train.py
8. 1fcf2c1 - fix: forward dataloader runtime kwargs in training_cli
9. 65b130c - fix: lazy-load training_cli in ptbr main

Already in dev: d0f94a0, 5df1e67

## 2026-02-18 - Integration Progress

### Completed: eager-birch
- Integrated noisy-jsonl validation error parsing fix
- Added label_smoothing forwarding in train.py
- Added ptbr/tests/test_train_py.py with regression tests
- Removed eager-birch worktree

### Next: peppy-willow
Commit ea4a760 - Harden noisy JSONL validation index parsing
Already integrated via eager-birch (same fix)

## 2026-02-18 - Integration Complete

### Summary
Integrated changes from 5 worktrees into dev:
1. **eager-birch**: noisy-jsonl fix, label_smoothing forwarding, test_train_py.py
2. **peppy-willow**: already integrated via eager-birch
3. **prime-iris (staging)**: dev was already ahead
4. **savvy-owl**: trainer column pruning tests
5. **sunny-heron**: already integrated (LoRA targeting fix was in dev)

### Commits made:
- 0fd9c9d: fix: repair noisy-jsonl validation error parsing
- 09314e4: feat: forward label_smoothing in train.py and add train.py tests
- 74f3b9a: test: add regression tests for custom trainer column handling

### Branches removed:
- ralph/eager-birch
- ralph/peppy-willow
- ralph/savvy-owl
- ralph/sunny-heron

### Worktrees removed:
- .worktrees/eager-birch
- .worktrees/peppy-willow
- .worktrees/prime-iris
- .worktrees/savvy-owl
- .worktrees/sunny-heron
