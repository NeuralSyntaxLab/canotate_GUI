#!/bin/bash
# macOS: double-click this in Finder to start Cannotate.
#
# A Terminal window opens alongside the GUI and shows any progress or error
# output. If you would rather not see it, double-click Cannotate.app instead --
# it runs exactly this, just without the window.
#
# If Finder opens this in a text editor instead of running it, the executable
# bit is missing. Fix it with:  chmod +x Cannotate.command

# Built-ins only, no external commands: on a Mac with no Python -- the machine
# setup/install_mac.command exists for -- PATH may be of no use to us, and a
# launcher that cannot find `dirname` never gets as far as reporting why.
case "$0" in
    */*) cd "${0%/*}" || exit 1 ;;
esac

# The interpreter setup/install_mac.command recorded. Preferred over anything
# on PATH, and the only thing that works when there is no system Python at all.
# launch.py does its own checking once it is running; this just starts it.
config_python=""
if [ -f launcher.cfg ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            python*=*)
                config_python="${line#*=}"
                config_python="${config_python#"${config_python%%[![:space:]]*}"}"
                config_python="${config_python%"${config_python##*[![:space:]]}"}"
                break
                ;;
        esac
    done < launcher.cfg
fi

if [ -n "$config_python" ] && [ -x "$config_python" ]; then
    exec "$config_python" launch.py
fi

# No launcher.cfg yet: fall back to a Python on PATH, which is enough to run
# launch.py and let it search for a usable interpreter.
#
# Only Python 3 will do. A bare "python" is still Python 2 on some older
# machines, and Python 2 dies on this project's source with a confusing
# "Non-ASCII character" SyntaxError rather than saying it is too old.
for py in python3 python; do
    command -v "$py" >/dev/null 2>&1 || continue
    if "$py" -c 'import sys; sys.exit(0 if sys.version_info[0] >= 3 else 1)' 2>/dev/null; then
        exec "$py" launch.py
    fi
done

echo "Cannotate is not set up on this Mac yet."
echo
echo "Open the 'setup' folder next to this file and double-click"
echo "    install_mac.command"
echo
echo "That installs everything Cannotate needs, including the neural-network"
echo "features. It takes a few minutes and needs no administrator password."
read -r -p "Press Return to close this window."
exit 1
