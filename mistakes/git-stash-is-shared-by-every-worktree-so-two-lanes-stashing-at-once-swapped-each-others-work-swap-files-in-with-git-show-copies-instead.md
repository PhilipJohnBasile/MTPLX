# git stash is shared by every worktree, so two lanes stashing at once swapped each other's work; swap files in with `git show` copies instead

**Symptom.** A `git stash push -- <file>; swift test; git stash pop` cycle, used to run a new test against the pre-fix source for a before/after receipt, came back with a file from another lane's worktree modified (`ChatViewModel.swift`) and my own edits to `MTPLXBackendStore.swift` gone. `git stash list` showed neither lane's entry.

**Cause.** `refs/stash` lives in the common git directory and is one stack for all worktrees of a repository. A sibling lane ran its own push/pop in the same seconds: my pop took their stash, their pop took mine. Both commits survived only as dangling objects (`git fsck --unreachable --no-reflogs`).

**Fix / rule.**
- Never use `git stash` in a worktree that shares a repository with other active worktrees or agents.
- For a before/after receipt, keep the working copy in `/tmp` and swap the old file in with `git show HEAD:<path> > <path>`, run, then copy the kept file back. This touches only the working tree.
- If a stash collision has already happened: `git fsck --unreachable --no-reflogs | rg commit`, find the `WIP on <branch>` commits, restore your file with `git show <sha>:<path>`, and pin the other lane's commit under a `recovered/...` branch plus a patch file so it cannot be pruned before its owner is told.
