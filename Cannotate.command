#!/bin/bash
# macOS: double-click this in Finder to start Cannotate.
#
# A Terminal window opens alongside the GUI and shows any progress or error
# output. If you would rather not see it, double-click Cannotate.app instead --
# it runs exactly this, just without the window.
#
# If Finder opens this in a text editor instead of running it, the executable
# bit is missing. Fix it with:  chmod +x Cannotate.command

cd "$(dirname "$0")" || exit 1

for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
        exec "$py" launch.py
    fi
done

echo "No Python interpreter found on PATH."
echo "Install Python 3 from https://www.python.org/downloads/ and try again."
read -r -p "Press Return to close this window."
exit 1
