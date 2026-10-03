#!/bin/bash
# Cannotate — one-time setup for macOS. Double-click this in Finder.
#
# It builds Cannotate its own private Python in ~/.cannotate and installs
# everything the tool needs, including the deep-learning packages behind the
# NN buttons. Nothing is installed system-wide, nothing is put on PATH, and no
# administrator password is needed.
#
# Safe to run again: it rebuilds the environment from scratch, which is the
# fix if anything ever breaks. Downloaded packages are cached, so a second run
# is much faster than the first.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CANNOTATE_HOME="${CANNOTATE_HOME:-$HOME/.cannotate}"
ENV_DIR="$CANNOTATE_HOME/env"
MICROMAMBA="$CANNOTATE_HOME/bin/micromamba"
export MAMBA_ROOT_PREFIX="$CANNOTATE_HOME/root"

# conda-forge's pytorch is the CPU build: ~26 MB against the ~328 MB the pip
# wheel unpacks to, because it carries no CUDA payload. The annotator's Mac has
# no NVIDIA GPU, so none of that would ever have been used.
CONDA_PACKAGES=(
    "python=3.11" tk numpy scipy matplotlib
    pytorch onnxruntime scikit-learn pandas scikit-image
)
# soundfile and sounddevice come from pip: conda-forge's pysoundfile stopped at
# 0.10.2 in 2019, while the pip wheels bundle an up-to-date libsndfile and
# PortAudio built for this architecture.
PIP_PACKAGES=(soundfile sounddevice)

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '   \033[32m✓\033[0m %s\n' "$*"; }
oops() { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; }

finish() {
    local code=$?
    if [ "$code" -ne 0 ]; then
        oops "Setup did not finish."
        echo
        echo "Nothing on this Mac was changed outside $CANNOTATE_HOME,"
        echo "so it is safe to delete that folder and run this again:"
        echo
        echo "    rm -rf \"$CANNOTATE_HOME\""
        echo
        echo "If it keeps failing, send whatever is printed above to whoever"
        echo "gave you Cannotate."
    fi
    echo
    read -r -p "Press Return to close this window."
    exit "$code"
}
trap finish EXIT

case "$(uname -m)" in
    arm64)  PLATFORM="osx-arm64" ;;
    x86_64) PLATFORM="osx-64" ;;
    *)      oops "Unsupported Mac processor: $(uname -m)"; exit 1 ;;
esac

say "Setting up Cannotate"
echo "   Project:     $PROJECT_DIR"
echo "   Installing:  $CANNOTATE_HOME  (nothing else on this Mac is touched)"
echo "   Processor:   $(uname -m) ($PLATFORM)"

# ── 1. a package manager that needs no Python to run ──────────────────
say "1/5  Fetching the package manager"
if [ -x "$MICROMAMBA" ]; then
    ok "already present"
else
    mkdir -p "$CANNOTATE_HOME/bin"
    if ! curl -fLs --max-time 300 \
         "https://micro.mamba.pm/api/micromamba/$PLATFORM/latest" \
         | tar -xj -C "$CANNOTATE_HOME" bin/micromamba; then
        oops "Could not download the package manager."
        echo "Check that this Mac is online, then run this script again."
        exit 1
    fi
    ok "downloaded ($("$MICROMAMBA" --version))"
fi

# ── 2. a private Python, rebuilt from scratch every run ───────────────
say "2/5  Building Cannotate's own Python (about 250 MB to download)"
echo "     This is the slow part — several minutes on a first run."
if [ -d "$ENV_DIR" ]; then
    echo "     Removing the previous environment first."
    rm -rf "$ENV_DIR"
fi
"$MICROMAMBA" create -y -p "$ENV_DIR" -c conda-forge "${CONDA_PACKAGES[@]}"
PYTHON="$ENV_DIR/bin/python"
[ -x "$PYTHON" ] || { oops "Python was not created at $PYTHON"; exit 1; }
ok "$("$PYTHON" --version)"

# ── 3. the audio libraries ────────────────────────────────────────────
say "3/5  Adding the audio libraries"
"$PYTHON" -m pip install --quiet --disable-pip-version-check "${PIP_PACKAGES[@]}"
ok "soundfile and sounddevice installed"

# ── 4. tell the launcher which Python to use ──────────────────────────
# launch.py reads this and skips searching the machine for an interpreter.
say "4/5  Pointing Cannotate at it"
printf '[cannotate]\npython = %s\n' "$PYTHON" > "$PROJECT_DIR/launcher.cfg"
ok "wrote launcher.cfg"

# ── 5. prove it actually works ────────────────────────────────────────
say "5/5  Checking that everything works"
if ! "$PYTHON" "$PROJECT_DIR/smoke_test.py"; then
    oops "Cannotate is installed but something is not working."
    echo "The failing checks are listed above. Send them to whoever gave you"
    echo "Cannotate — the tool may still open, but do not trust the NN buttons."
    exit 1
fi

say "Done — Cannotate is ready."
printf '   Open it by double-clicking \033[1mCannotate.app\033[0m in:\n'
printf '       %s\n\n' "$PROJECT_DIR"
echo "   You never need to run this setup again unless something breaks."
