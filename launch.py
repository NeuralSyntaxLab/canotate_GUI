#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bootstrap that finds a usable Python and starts the Cannotate GUI.

The point of this file is that nobody should have to open a terminal, activate
a conda environment, or know what a PYTHONPATH is. Double-clicking one of the
wrapper files (Cannotate.command / Cannotate.app on macOS, Cannotate.bat /
Cannotate.vbs on Windows) runs this, and this figures the rest out.

It deliberately uses **only the standard library**, because it has to be able
to run under whatever Python happens to be around -- including one that is
missing every scientific package the GUI needs. Its whole job is to find a
better interpreter than the one running it, if necessary.

Search order:
  1. ``python`` key in launcher.cfg next to this file
  2. CANNOTATE_PYTHON environment variable
  3. the interpreter running this script
  4. conda environments under the usual roots
  5. python3 / python on PATH

The first candidate that can import every hard dependency wins, and its path is
written back to launcher.cfg so the next launch skips the search. If nothing
qualifies, a Tkinter dialog explains what is missing and how to fix it.
"""

import configparser
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "launcher.cfg"
ENTRY_POINT = HERE / "Cannotate_GUI.py"

# Packages the GUI cannot start without. torch, onnxruntime and the rest of the
# deep-learning stack are deliberately absent: the tool disables its NN buttons
# and runs fine without them, so requiring torch here would turn an optional
# feature into a hard blocker on machines that only ever annotate by hand.
#
# A tuple means "any one of these will do" — colors.py computes Lab distances
# with OpenCV when it is available and falls back to scikit-image otherwise.
REQUIRED = [
    "numpy", "scipy", "matplotlib", "soundfile", "sounddevice", "tkinter",
    ("skimage", "cv2"),
]

# Probed only so the failure dialog can mention them; never blocking.
OPTIONAL = ["torch", "onnxruntime", "sklearn", "pandas"]

# Import names that differ from what you type after `pip install`.
PIP_NAMES = {"skimage": "scikit-image", "cv2": "opencv-python",
             "sklearn": "scikit-learn"}

PROBE = (
    "import importlib, json\n"
    "res = {{}}\n"
    "for m in {modules!r}:\n"
    "    try:\n"
    "        importlib.import_module(m)\n"
    "        res[m] = True\n"
    "    except Exception:\n"
    "        res[m] = False\n"
    "print('CANNOTATE_PROBE' + json.dumps(res))\n"
)


def _flatten(requirements):
    """Expand the tuple-means-any-of entries into a flat module list."""
    flat = []
    for item in requirements:
        flat.extend(item if isinstance(item, tuple) else (item,))
    return flat


def _missing_requirements(probe_result):
    """Return the requirements not satisfied by a probe result.

    An alternatives group counts as satisfied when *any* of its members
    imported; it is reported as "skimage or cv2" when none did.
    """
    missing = []
    for item in REQUIRED:
        if isinstance(item, tuple):
            if not any(probe_result.get(m) for m in item):
                missing.append(" or ".join(item))
        elif not probe_result.get(item):
            missing.append(item)
    return missing


def _probe(python_exe):
    """Return {module: bool} for every probed module, or None if unusable."""
    modules = _flatten(REQUIRED)
    try:
        proc = subprocess.run(
            [str(python_exe), "-c", PROBE.format(modules=modules)],
            capture_output=True, text=True, timeout=90,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("CANNOTATE_PROBE"):
            try:
                return json.loads(line[len("CANNOTATE_PROBE"):])
            except ValueError:
                return None
    return None


def _read_configured_python():
    if not CONFIG_PATH.is_file():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(CONFIG_PATH)
    except configparser.Error:
        return None
    value = parser.get("cannotate", "python", fallback="").strip()
    return value or None


def _remember_python(python_exe):
    """Cache the winning interpreter so later launches skip the search."""
    parser = configparser.ConfigParser()
    if CONFIG_PATH.is_file():
        try:
            parser.read(CONFIG_PATH)
        except configparser.Error:
            parser = configparser.ConfigParser()
    if not parser.has_section("cannotate"):
        parser.add_section("cannotate")
    parser.set("cannotate", "python", str(python_exe))
    try:
        with open(CONFIG_PATH, "w") as fh:
            parser.write(fh)
    except OSError:
        pass  # a read-only install directory is not worth failing the launch over


def _env_python(env_dir):
    """Return the interpreter inside a conda env directory, or None."""
    python = (env_dir / "python.exe") if os.name == "nt" else (env_dir / "bin" / "python")
    return python if python.is_file() else None


def _conda_install_roots():
    """Yield plausible conda *installation* roots (the dirs that contain envs/)."""
    for base in (
        os.environ.get("CONDA_ROOT"),
        Path.home() / "anaconda3", Path.home() / "miniconda3",
        Path.home() / "miniforge3", Path.home() / "mambaforge",
        Path("/opt/anaconda3"), Path("/opt/miniconda3"),
        Path("C:/ProgramData/Anaconda3"), Path("C:/ProgramData/miniconda3"),
    ):
        if base:
            yield Path(base)

    # CONDA_PREFIX points at whichever env is active. If that is a named env it
    # sits at <root>/envs/<name>, so the install root is two levels up; if it is
    # the base env, CONDA_PREFIX already *is* the root.
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        prefix = Path(prefix)
        yield prefix.parent.parent if prefix.parent.name == "envs" else prefix


def _conda_env_pythons():
    """Yield the interpreter of every conda env under the usual roots."""
    seen = set()
    for root in _conda_install_roots():
        env_dirs = [root]  # the base env lives directly in the install root
        envs_parent = root / "envs"
        if envs_parent.is_dir():
            try:
                env_dirs.extend(sorted(p for p in envs_parent.iterdir() if p.is_dir()))
            except OSError:
                pass
        for env_dir in env_dirs:
            python = _env_python(env_dir)
            if python and python not in seen:
                seen.add(python)
                yield python


def _candidates():
    """Yield interpreters to try, best guess first, without duplicates."""
    seen = set()

    def offer(path):
        if not path:
            return None
        path = str(path)
        if path in seen:
            return None
        seen.add(path)
        return path

    ordered = [_read_configured_python(), os.environ.get("CANNOTATE_PYTHON"),
               sys.executable]
    ordered += [str(p) for p in _conda_env_pythons()]
    ordered += [shutil.which("python3"), shutil.which("python")]

    for path in ordered:
        resolved = offer(path)
        if resolved:
            yield resolved


def _report_failure(attempts):
    """Explain the problem in a dialog, falling back to stderr."""
    if attempts:
        best_exe, best_missing = min(attempts, key=lambda a: len(a[1]))
        detail = (
            "The closest match was:\n"
            f"    {best_exe}\n\n"
            "but it is missing:\n"
            + "".join(f"    \u2022 {m}\n" for m in best_missing)
            + "\nInstall them with:\n"
            f"    \"{best_exe}\" -m pip install "
            # tkinter has no pip package; for an alternatives group, name the
            # first option rather than the raw "skimage or cv2" string.
            + " ".join(PIP_NAMES.get(m.split(" or ")[0], m.split(" or ")[0])
                       for m in best_missing if m != "tkinter")
        )
        if "tkinter" in best_missing:
            detail += (
                "\n\ntkinter is not a pip package. On macOS install Python from "
                "python.org or run 'brew install python-tk'. On Windows, re-run "
                "the Python installer and tick 'tcl/tk and IDLE'."
            )
    else:
        detail = "No Python interpreter could be found at all."

    message = (
        "Cannotate could not start.\n\n"
        f"It needs a Python with: {', '.join(REQUIRED)}\n\n"
        f"{detail}\n\n"
        "You can also point it at a specific interpreter by editing "
        f"{CONFIG_PATH.name} next to this launcher."
    )

    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("Cannotate — cannot start", message)
        root.destroy()
    except Exception:
        pass  # no display, or no tkinter: stderr is all we have
    print(message, file=sys.stderr)


def main():
    if not ENTRY_POINT.is_file():
        print(f"Cannot find {ENTRY_POINT}", file=sys.stderr)
        return 1

    attempts = []
    for python_exe in _candidates():
        result = _probe(python_exe)
        if result is None:
            continue
        missing = _missing_requirements(result)
        if missing:
            attempts.append((python_exe, missing))
            continue

        _remember_python(python_exe)
        # cwd matters: the GUI resolves models/ and classification_model/
        # relative to the project root.
        return subprocess.call([python_exe, str(ENTRY_POINT)], cwd=str(HERE))

    _report_failure(attempts)
    return 1


if __name__ == "__main__":
    sys.exit(main())
