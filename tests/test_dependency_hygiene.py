"""Nothing imports a package the project never declared.

A stray import is invisible on a developer machine, because the scientific
conda environments people already have happen to carry half of PyPI. It only
surfaces on a freshly built environment -- which is to say, on the machine of
the person the tool was packaged for.

That is exactly how ``from sqlalchemy import exists`` sat unused at the top of
utils.py: every environment it was ever run in had sqlalchemy, so importing
utils always worked, until a clean install didn't.

Declared means: listed in requirements.txt, or the standard library, or part of
this project.

Runs under pytest or directly:  python tests/test_dependency_hygiene.py
"""
import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

# requirements.txt names pip distributions; these are what you type after
# ``import``. tkinter ships with Python and has no pip package.
DECLARED = {
    "numpy", "scipy", "matplotlib", "soundfile", "sounddevice", "tkinter",
    "skimage", "cv2",                       # colors.py uses whichever is present
    "torch", "onnxruntime", "sklearn", "pandas",   # optional: the NN buttons
    "pytest",
}

FIRST_PARTY = {
    "cannotate", "bird_song_detector", "classification_model",
    "utils", "colors", "full_annotation_pipeline", "launch", "smoke_test",
    "Cannotate_GUI",
}

# The sandbox is a scratch area with its own dependencies, and tests may import
# test helpers by bare name.
SKIP_DIRS = {"sandbox", "__pycache__", ".git", "docs", "setup"}


def _source_files():
    for path in sorted(PROJECT_ROOT.rglob("*.py")):
        rel = path.relative_to(PROJECT_ROOT)
        if SKIP_DIRS.intersection(rel.parts):
            continue
        yield path


def _imported_top_level_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, which is first-party by definition
            if node.level == 0 and node.module:
                yield node.module.split(".")[0]


def test_no_module_imports_an_undeclared_package():
    allowed = DECLARED | FIRST_PARTY | set(sys.stdlib_module_names)
    offenders = {}
    for path in _source_files():
        extra = {m for m in _imported_top_level_modules(path) if m not in allowed}
        # A test file importing a sibling test module by bare name is fine.
        extra -= {p.stem for p in (PROJECT_ROOT / "tests").glob("*.py")}
        if extra:
            offenders[str(path.relative_to(PROJECT_ROOT))] = sorted(extra)

    assert not offenders, (
        "these modules import packages the project does not declare:\n"
        + "\n".join(f"  {where}: {', '.join(names)}"
                    for where, names in sorted(offenders.items()))
    )


def test_utils_imports_without_sqlalchemy(monkeypatch):
    """The regression that started this: utils must not need a database library."""
    monkeypatch.setitem(sys.modules, "sqlalchemy", None)  # makes `import` raise
    for name in [n for n in sys.modules if n == "utils" or n.startswith("utils.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    import utils  # noqa: F401


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
