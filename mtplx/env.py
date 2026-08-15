"""Environment snapshots for reproducible MTPLX measurements."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _run_checked(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return completed.returncode, completed.stdout.strip()
    except Exception as exc:  # pragma: no cover - depends on host tools
        return 1, str(exc)


def _run(args: list[str], cwd: Path | None = None) -> str:
    code, output = _run_checked(args, cwd=cwd)
    if code == 0:
        return output
    return f"ERROR: {output}"


def _inside_git_worktree(root: Path) -> bool:
    """True when ``root`` or a parent carries a ``.git`` entry.

    Checked before any ``git`` subprocess: on a Mac without the Xcode Command
    Line Tools ``/usr/bin/git`` is Apple's install shim, which prints a
    developer-tools notice and opens the CLT install dialog. ``mtplx tune``
    and ``mtplx doctor`` collect this snapshot from an arbitrary user cwd
    (the app runs tune during onboarding), so git must only ever run where a
    repository actually is.
    """

    try:
        candidates = [root.resolve(), *root.resolve().parents]
    except OSError:
        return False
    return any((candidate / ".git").exists() for candidate in candidates)


def _git_snapshot(root: Path) -> tuple[str, str]:
    if not _inside_git_worktree(root):
        return "not a git worktree", "not a git worktree"
    code, inside = _run_checked(["git", "rev-parse", "--is-inside-work-tree"], cwd=root)
    if code != 0 or inside.strip().lower() != "true":
        return "not a git worktree", "not a git worktree"

    branch_code, branch = _run_checked(["git", "branch", "--show-current"], cwd=root)
    if branch_code != 0:
        branch = "unknown"
    elif not branch:
        head_code, head = _run_checked(["git", "rev-parse", "--short", "HEAD"], cwd=root)
        branch = f"detached {head}" if head_code == 0 and head else "detached HEAD"

    status_code, status = _run_checked(["git", "status", "--short", "--branch"], cwd=root)
    if status_code != 0 or not status:
        status = "git status unavailable"
    return branch, status


def _mlx_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import mlx.core as mx

        info["mlx"] = getattr(mx, "__version__", "unknown")
        info["default_device"] = str(mx.default_device())
        for attr in ("get_active_memory", "get_peak_memory"):
            fn = getattr(mx, attr, None)
            if fn is not None:
                try:
                    info[attr] = int(fn())
                except Exception as exc:  # pragma: no cover - host dependent
                    info[attr] = f"ERROR: {exc}"
    except Exception as exc:
        info["mlx_error"] = repr(exc)
    try:
        import mlx_lm

        info["mlx_lm"] = getattr(mlx_lm, "__version__", "unknown")
    except Exception as exc:
        info["mlx_lm_error"] = repr(exc)
    return info


@dataclass(frozen=True)
class EnvironmentSnapshot:
    project_root: str
    python_executable: str
    python_version: str
    platform: str
    git_branch: str
    git_status: str
    hf_path: str | None
    uv_path: str | None
    mlx: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "platform": self.platform,
            "git_branch": self.git_branch,
            "git_status": self.git_status,
            "hf_path": self.hf_path,
            "uv_path": self.uv_path,
            "mlx": self.mlx,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def collect_environment(project_root: Path | str = ".") -> EnvironmentSnapshot:
    root = Path(project_root).resolve()
    git_branch, git_status = _git_snapshot(root)
    return EnvironmentSnapshot(
        project_root=str(root),
        python_executable=sys.executable,
        python_version=sys.version,
        platform=platform.platform(),
        git_branch=git_branch,
        git_status=git_status,
        hf_path=shutil.which("hf"),
        uv_path=shutil.which("uv"),
        mlx=_mlx_info(),
    )
