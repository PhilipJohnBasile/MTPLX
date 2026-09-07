from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
number = int(sys.argv[2])
if number == 219:
    path = root / 'tests/test_request_observability_golden.py'
    source = path.read_text()
    old = '        {"x-mtplx-client-turn-id": "turn-123", "x-mtplx-client-entry-id": "bad value\\nsecret"},\n'
    new = '        headers={"x-mtplx-client-turn-id": "turn-123", "x-mtplx-client-entry-id": "bad value\\nsecret"},\n        metadata={}, session_source=None, request_generation_mode="mtp", request_depth=1,\n'
    assert source.count(old) == 1
    path.write_text(source.replace(old, new))
    # Sort the previously unsorted import block before, not after, validation.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ruff==0.15.12'], check=True)
    subprocess.run([sys.executable, '-m', 'ruff', 'check', '--select', 'I', '--fix', str(path)], check=True)
elif number == 362:
    path = root / 'mtplx/server/runtime_status.py'
    source = path.read_text()
    assert source.count('            except Exception:') == 1
    path.write_text(source.replace('            except Exception:', '            except (AttributeError, RuntimeError, TypeError, ValueError, OSError):'))
elif number == 399:
    path = root / 'tests/test_memory_plan.py'
    source = path.read_text()
    old = '''def test_detect_total_ram_reports_this_machine() -> None:
    detected = detect_total_ram_bytes()
    assert detected is not None and detected > 8 * GIB
'''
    new = '''def test_detect_total_ram_reports_this_machine() -> None:
    import os
    import subprocess
    import sys

    # A hosted Apple Silicon runner can legitimately have 7 GiB. Verify
    # the detected bytes against the OS, not the developer machine's size.
    if sys.platform == "darwin":
        expected = int(subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"], text=True, timeout=5
        ).strip())
    else:
        expected = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    assert expected > 0
    assert detect_total_ram_bytes() == expected
'''
    assert source.count(old) == 1
    path.write_text(source.replace(old, new))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ruff==0.15.12'], check=True)
    subprocess.run([sys.executable, '-m', 'ruff', 'format', str(path)], check=True)
    subprocess.run(['git', '-C', str(root), 'add', 'tests/test_memory_plan.py'], check=True)
