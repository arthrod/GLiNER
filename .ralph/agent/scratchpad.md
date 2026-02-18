[2026-02-18T00:00:00Z] Objective focus: investigate reported span_mode bug in GLiNERConfig.model_type where legacy routing checks "token-level" instead of "token_level", confirm whether behavior is unintended via code + git history, and if unintended apply minimal fix with tests.

Plan for this iteration (single atomic task):
1) Inspect gliner/config.py model_type logic and surrounding codepaths/configs to verify the mismatch and behavioral impact.
2) Use git blame/log on the affected lines and related files to determine whether hyphenated value was intentional.
3) If unintended, replace the 4 legacy comparisons in model_type to use "token_level" and add/adjust focused regression tests.
4) Run targeted verification (tests or source-level assertions if dependency-limited), then commit, close task, and record memory.

Decision confidence: 93/100 (proceed autonomously; straightforward bugfix with localized behavior).
[2026-02-18T00:00:00Z] Completed task task-1771372835-9614.

Findings:
- The reported model_type bug is no longer present: GLiNERConfig.model_type (gliner/config.py:327,332,334,338) already checks "token_level".
- Git history confirms intentional fix: commit 6972469 (2026-02-18, "Fix GLiNERConfig token_level model_type routing") changed the four comparisons from "token-level" to "token_level" and added routing regression tests in ptbr/tests/test_config_cli.py.
- Remaining mismatch was documentation-only: UniEncoderSpanDecoderConfig docstring used literal 'token-level' in a ValueError description.

Implemented:
- Updated gliner/config.py docstring at line 165 from 'token-level' to 'token_level' to remove misleading guidance.
- Commit: a4c3f5a "Clarify span_mode docstring uses token_level".

Verification:
- Attempted: python3 -m pytest -q ptbr/tests/test_config_cli.py -k model_type -> failed (No module named pytest).
- Fallback verification: static checks (`python3 -m compileall gliner/config.py`) passed; direct source inspection confirms model_type routing uses token_level consistently.
