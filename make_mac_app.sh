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
#
# Everything the tool prints goes to cannotate.log (fresh each launch), and a
# non-zero exit puts the tail of that log in a dialog. An earlier version sent
# output to /dev/null, which meant a crash showed the user nothing at all --
# never do that in a launcher whose whole point is that there is no terminal.
#
# It runs Cannotate.command rather than python3 directly, so both launchers
# share one way of finding the interpreter. That matters on a Mac with no
# system Python: Cannotate.command reads launcher.cfg with shell built-ins,
# while "/usr/bin/env python3" would have failed before anything could say why.
cat > "$SCRIPT" <<'APPLESCRIPT'
on run
    set bundlePath to POSIX path of (path to me)
    -- path to me is .../<project>/Cannotate.app/ ; the project is its parent
    set projectPath to do shell script "dirname " & quoted form of (text 1 thru -2 of bundlePath)
    set logPath to projectPath & "/cannotate.log"

    try
        -- do shell script gives up after 120s by default; annotation sessions
        -- run far longer than that, so the timeout has to be raised.
        with timeout of 86400 seconds
            do shell script "cd " & quoted form of projectPath & ¬
                " && ./Cannotate.command > " & quoted form of logPath & " 2>&1"
        end timeout
    on error errMsg
        set logTail to ""
        try
            set logTail to do shell script "tail -n 25 " & quoted form of logPath
        end try
        if logTail is "" then set logTail to errMsg
        display dialog "Cannotate stopped with an error." & return & return & ¬
            logTail & return & return & ¬
            "Full log: " & logPath ¬
            buttons {"OK"} default button "OK" with icon stop
    end try
end run
APPLESCRIPT

rm -rf "$APP"
osacompile -o "$APP" "$SCRIPT"
echo "Built $APP"
echo "Double-click it in Finder to launch Cannotate without a Terminal window."
echo "Output goes to cannotate.log; errors also appear in a dialog."
