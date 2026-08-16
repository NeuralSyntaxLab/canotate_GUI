"""The templates cache must never be able to break startup.

``_load_templates_cache`` documents that *any* problem with the cache file
yields ``None`` so the caller rebuilds. The interesting case is a cache written
by a different NumPy major version: NumPy 2.x pickles reference ``numpy._core``,
which does not exist under NumPy 1.x, so loading raises ``ModuleNotFoundError``.
That is an ``ImportError`` — not one of the exception types the original
``except`` clause listed — so it escaped and took the GUI down on startup.

Runs under pytest or directly:  python tests/test_templates_cache.py
"""

import importlib
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cannotate import dtw_suggestions as dtw


def _write_pickle_referencing_a_missing_module(cache_path):
    """Write a pickle whose payload needs a module that is gone at load time.

    This is what a NumPy-2.x-written cache looks like to a NumPy-1.x process:
    structurally valid pickle, unreachable module.
    """
    tmp_pkg_dir = tempfile.mkdtemp()
    module_name = "cannotate_vanishing_module"
    module_path = Path(tmp_pkg_dir) / f"{module_name}.py"
    module_path.write_text("class Payload:\n    pass\n")

    sys.path.insert(0, tmp_pkg_dir)
    try:
        mod = importlib.import_module(module_name)
        blob = {
            "cache_version": dtw.TEMPLATE_CACHE_VERSION,
            "fingerprint": "whatever",
            "templates": {"1": mod.Payload()},
        }
        with open(cache_path, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        # Make the module unreachable, exactly like numpy._core under NumPy 1.x.
        sys.path.remove(tmp_pkg_dir)
        sys.modules.pop(module_name, None)
        shutil.rmtree(tmp_pkg_dir, ignore_errors=True)
        importlib.invalidate_caches()


def test_unloadable_cache_is_a_miss_not_a_crash():
    """A cache referencing a missing module must return None, not raise."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = os.path.join(tmp, "x.templates_cache.pkl")
        _write_pickle_referencing_a_missing_module(cache_path)

        result = dtw._load_templates_cache(cache_path, "whatever")

        assert result is None, f"expected a cache miss, got {result!r}"


def test_truncated_cache_is_a_miss():
    """A half-written cache file is a miss too."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = os.path.join(tmp, "x.templates_cache.pkl")
        with open(cache_path, "wb") as f:
            f.write(b"\x80\x05\x95\x00\x00\x00")  # valid header, nothing after

        assert dtw._load_templates_cache(cache_path, "whatever") is None


def test_garbage_cache_is_a_miss():
    """A file that is not a pickle at all is a miss."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = os.path.join(tmp, "x.templates_cache.pkl")
        with open(cache_path, "wb") as f:
            f.write(b"this is not a pickle")

        assert dtw._load_templates_cache(cache_path, "whatever") is None


def test_missing_cache_is_a_miss():
    """No cache file at all is the ordinary first-run case."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = os.path.join(tmp, "does_not_exist.pkl")

        assert dtw._load_templates_cache(cache_path, "whatever") is None


def test_good_cache_still_loads():
    """The fix must not turn valid cache hits into misses."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = os.path.join(tmp, "x.templates_cache.pkl")
        templates = {"1": [[0.0, 1.0], [2.0, 3.0]]}
        dtw._save_templates_cache(cache_path, templates, "fp1", nfft=512,
                                  template_cols=dtw.TEMPLATE_COLS)

        assert dtw._load_templates_cache(cache_path, "fp1") == templates
        # A changed fingerprint must still be a miss.
        assert dtw._load_templates_cache(cache_path, "fp2") is None


def main():
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"FAIL {test.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"ERROR {test.__name__}: {type(e).__name__}: {e}")
            failures += 1
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
