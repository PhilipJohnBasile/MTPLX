#!/usr/bin/env python3
"""Select a bundled runtime with pip's real Python/ABI/macOS compatibility tags."""

from pathlib import Path
import sys

try:
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
except ImportError:  # A fresh app venv has ensurepip before its dependencies.
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename


def select_runtime_wheel(fallback: Path, native_dir: Path, tags=None) -> Path:
    name, version, _, _ = parse_wheel_filename(fallback.name)
    if name != "mtplx":
        raise ValueError("Expected an MTPLX fallback wheel")
    ranking = {tag: i for i, tag in enumerate(sys_tags() if tags is None else tags)}
    matches = []
    for path in [fallback, *sorted(native_dir.glob("mtplx-*.whl"))]:
        candidate_name, candidate_version, _, candidate_tags = parse_wheel_filename(path.name)
        if candidate_name != name or candidate_version != version:
            raise ValueError(f"Bundled runtime versions disagree: {path.name}")
        ranks = [ranking[tag] for tag in candidate_tags if tag in ranking]
        if ranks:
            matches.append((min(ranks), path))
    if not matches:
        raise ValueError("No bundled runtime matches this Python and macOS")
    return min(matches, key=lambda item: item[0])[1]


if __name__ == "__main__":
    print(select_runtime_wheel(Path(sys.argv[1]), Path(sys.argv[2])))
