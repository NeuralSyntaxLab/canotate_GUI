#!/bin/bash
# Rebuild Cannotate.app -- the double-click launcher that opens no Terminal window.
#
# Cannotate.app is committed to the repo, so you only need this if you move the
# project, or if the app bundle gets damaged (macOS occasionally strips the
# quarantine-flagged contents of a downloaded .app).
#
# Run:  ./make_mac_app.sh

set -euo pipefail
cd "$(dirname "$0")"

APP="Cannotate.app"
SCRIPT="$(mktemp -t cannotate_app).applescript"
trap 'rm -f "$SCRIPT"' EXIT

# The bundle resolves its own location at runtime rather than baking in an
# absolute path, so moving or renaming the project folder does not break it.
cat > "$SCRIPT" <<'APPLESCRIPT'
on run
    set bundlePath to POSIX path of (path to me)
    -- path to me is .../<project>/Cannotate.app/ ; the project is its parent
    set projectPath to do shell script "dirname " & quoted form of (text 1 thru -2 of bundlePath)
    try
        do shell script "cd " & quoted form of projectPath & " && /usr/bin/env python3 launch.py > /dev/null 2>&1 &"
    on error errMsg
        display dialog "Cannotate could not start:" & return & return & errMsg ¬
            buttons {"OK"} default button "OK" with icon stop
    end try
end run
APPLESCRIPT

rm -rf "$APP"
osacompile -o "$APP" "$SCRIPT"
echo "Built $APP"
echo "Double-click it in Finder to launch Cannotate without a Terminal window."
