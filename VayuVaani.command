#!/usr/bin/env bash
# macOS: double-click to open VayuVaani. The first time, macOS may say it's from an
# unidentified developer: right-click the file, choose Open, then Open again.
cd "$(dirname "$0")" && bash ./vayuvaani.sh
# Close this Terminal window once the app has opened.
osascript -e 'tell application "Terminal" to close (every window whose name contains "VayuVaani.command")' >/dev/null 2>&1 &
