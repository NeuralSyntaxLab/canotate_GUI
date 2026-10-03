"""The double-click launcher must not need a Python of its own.

``launch.py`` finds a suitable interpreter, but something has to run launch.py
in the first place, and ``Cannotate.command`` used whatever ``python3`` it
found on PATH. On the machine this tool is packaged for -- one with no Python
installed, which is the reason setup/install_mac.command exists -- there is no
such interpreter, so the launcher failed before it could reach the environment
the setup script had just built for it.

macOS makes this easy to miss: /usr/bin/python3 exists on every Mac, but it is
a stub that prompts to install the Command Line Tools rather than running
anything.

So the launcher reads launcher.cfg itself, in plain shell, and execs the
interpreter recorded there.

Runs under pytest or directly:  python tests/test_launcher_uses_configured_python.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

MARKER = "CONFIGURED_INTERPRETER_USED"

pytestmark = pytest.mark.skipif(sys.platform != "darwin",
                                reason="Cannotate.command is the macOS launcher")


@pytest.fixture
def launcher_dir(tmp_path):
    """A copy of the launcher beside a stub interpreter and a stub launch.py."""
    shutil.copy(PROJECT_ROOT / "Cannotate.command", tmp_path / "Cannotate.command")
    os.chmod(tmp_path / "Cannotate.command", 0o755)

    # Stands in for the interpreter the setup script builds. It ignores its
    # arguments, so no real Python is involved.
    stub = tmp_path / "stub_python"
    stub.write_text(f'#!/bin/sh\necho "{MARKER}"\n')
    os.chmod(stub, 0o755)

    (tmp_path / "launch.py").write_text("raise SystemExit('stub launch.py ran')\n")
    (tmp_path / "launcher.cfg").write_text(f"[cannotate]\npython = {stub}\n")
    return tmp_path


def _run(launcher_dir, path):
    return subprocess.run(
        [str(launcher_dir / "Cannotate.command")],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PATH": path}, stdin=subprocess.DEVNULL,
    )


def test_the_interpreter_in_launcher_cfg_is_used(launcher_dir):
    result = _run(launcher_dir, os.environ.get("PATH", ""))
    assert MARKER in result.stdout, result.stdout + result.stderr


def test_it_still_launches_when_no_python_is_on_path(launcher_dir):
    """The case the setup script exists for: a Mac with no Python at all."""
    result = _run(launcher_dir, "/nonexistent")
    assert MARKER in result.stdout, result.stdout + result.stderr
    assert result.returncode == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
