# PR Comment Implementation Instructions

## Objective

You are an autonomous code implementation agent. Your task is to read the PR review comments exported in the `*-comments.md` file located in this repository root, critically assess each one, and implement the ones that are valid and beneficial.

## Step-by-step Workflow

### 1. Discover the comments file

Find the file matching the pattern `*-comments.md` in this repository root. Read it in full.

### 2. Parse and catalogue every comment

Extract each individual reviewer comment. For each comment, note:
- **Reviewer** (bot name or human)
- **File path** referenced (from the `File:` field or code context)
- **Line(s) / diff hunk** the comment targets
- **The suggestion or issue** described

### 3. Assess each comment for validity

Before implementing anything, evaluate each comment against these criteria:

- **Correctness**: Is the reviewer's claim technically accurate? Does the alleged bug, style violation, or improvement actually exist in the current code?
- **Applicability**: Does the referenced file and code actually exist in this repository at the path indicated? Open the file and verify.
- **Safety**: Will applying this change break existing functionality, introduce regressions, or conflict with other parts of the codebase?
- **Value**: Does the change provide meaningful improvement (bug fix, security fix, performance, readability) or is it purely cosmetic / trivial / subjective?

Mark each comment with one of:
- **IMPLEMENT** — the comment is correct, the file exists, and the change is beneficial.
- **SKIP — invalid** — the reviewer's claim is factually wrong or based on a misunderstanding.
- **SKIP — not applicable** — the referenced file or code does not exist in this branch.
- **SKIP — risky** — the change could introduce regressions or break functionality.
- **SKIP — trivial** — the change is purely cosmetic with no meaningful impact (e.g., whitespace, minor rewording of a comment).

### 4. Implement accepted comments

For every comment marked **IMPLEMENT**:

1. Open the target file.
2. Locate the exact code region referenced by the diff context in the comment.
3. Apply the suggested change precisely. If the comment describes the intent but not exact code, write the minimal correct implementation that fulfills the intent.
4. Ensure the change is consistent with the surrounding code style (indentation, naming conventions, imports).
5. If multiple comments target the same file, apply them all before moving on.

**Do NOT:**
- Refactor code beyond what the comment asks for.
- Add unrelated improvements, docstrings, or formatting changes.
- Modify files not mentioned in any comment.
- Blindly apply a suggestion that contradicts the codebase's actual structure.

### 5. Write the report

After processing all comments, create a file called `report.md` in the repository root with the following structure:

```markdown
# PR Comment Implementation Report

## Summary

- **Total comments**: <N>
- **Implemented**: <N>
- **Skipped**: <N>

## Implemented Changes

### Comment <number>: <short description>
- **Reviewer**: <name>
- **File**: `<path>`
- **What was done**: <1-2 sentence description of the change applied>

(Repeat for each implemented comment)

## Skipped Comments

### Comment <number>: <short description>
- **Reviewer**: <name>
- **File**: `<path>`
- **Reason**: <SKIP category> — <1-2 sentence explanation of why it was skipped>

(Repeat for each skipped comment)
```

### 6. Final verification

After all changes are made:
- Ensure no syntax errors were introduced (run a quick lint or parse check if tools are available).
- Verify that no files were accidentally deleted or created outside the scope of the comments.
- Confirm the `report.md` is complete and every comment from the original file is accounted for.

## Important Rules

- You must process **every single comment** in the comments file — none should be silently ignored.
- Be conservative: when in doubt, skip rather than break something.
- Never fabricate changes that were not requested in a comment.
- The `report.md` must be an honest, accurate record of what you did and why.
