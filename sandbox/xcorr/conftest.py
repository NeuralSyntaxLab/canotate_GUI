"""Let the sandbox tests run from anywhere, not only from this directory.

The modules here import each other by bare name (``import scoring``) because
they are scripts, not a package.  Without this, ``pytest sandbox/xcorr`` from
the repo root fails at collection while ``pytest`` from inside the directory
passes — a trap worth closing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
