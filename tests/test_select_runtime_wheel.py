from pathlib import Path
import importlib.util

import pytest
from packaging.tags import Tag

spec = importlib.util.spec_from_file_location(
    "select_runtime_wheel", Path(__file__).parents[1] / "scripts/select_runtime_wheel.py")
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


@pytest.mark.parametrize("python,os_major,native", [
    ("cp314", 15, True), ("cp314", 26, True),
    ("cp314", 14, False), ("cp313", 26, False),
])
def test_compatible_native_or_pure_fallback(tmp_path, python, os_major, native):
    fallback = tmp_path / "mtplx-2.11.1-py3-none-any.whl"
    binary_dir = tmp_path / "Native"
    binary_dir.mkdir()
    binary = binary_dir / "mtplx-2.11.1-cp314-cp314-macosx_15_0_arm64.whl"
    binary.touch()
    tags = [Tag(python, python, f"macosx_{major}_0_arm64")
            for major in range(os_major, 10, -1)] + [Tag("py3", "none", "any")]
    selected = selector.select_runtime_wheel(fallback, binary_dir, tags)
    assert selected == (binary if native else fallback)


def test_mixed_release_artifacts_fail_instead_of_installing_wrong_version(tmp_path):
    binary = tmp_path / "mtplx-2.10.2-cp314-cp314-macosx_15_0_arm64.whl"
    binary.touch()
    with pytest.raises(ValueError, match="versions disagree"):
        selector.select_runtime_wheel(tmp_path / "mtplx-2.11.1-py3-none-any.whl", tmp_path)
