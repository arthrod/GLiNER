# Worktree Context

- **Loop ID**: prime-iris
- **Workspace**: /home/arthrod/workspace/gliner_review/GliNER/.worktrees/prime-iris
- **Main Repo**: /home/arthrod/workspace/gliner_review/GliNER
- **Branch**: ralph/prime-iris
- **Created**: 2026-02-18T01:34:07.153884994+00:00
- **Prompt**: "fetch this PR, implement the changes, run them, fix and then merge into staging. push after: /home/..."

## Notes

This is a worktree-based parallel loop. The following resources are symlinked
to the main repository:

- `.ralph/agent/memories.md` → shared memories
- `.ralph/specs/` → shared specifications
- `.ralph/tasks/` → shared code task files

Local state (scratchpad, runtime tasks, events) is isolated to this worktree.
