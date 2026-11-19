#!/usr/bin/env bash
# VayuVaani launcher for macOS and Linux.
#   First run: sets up a private Python environment (3 to 10 minutes, needs internet)
#   and adds VayuVaani to your app menu. After that it just opens the app.
# On a Mac, double-click VayuVaani.command instead (it runs this file).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
VENV="$HERE/.venv"
VPY="$VENV/bin/python"
LOG="$HERE/setup_log.txt"
# Same versions as the Windows launcher, so the ready-made model loads everywhere.
PINNED="numpy==2.1.3 scipy==1.14.1 pandas==2.2.3 scikit-learn==1.5.2 lightgbm==4.5.0 requests python-dotenv joblib openpyxl gTTS"
LOOSE="numpy pandas scikit-learn lightgbm requests python-dotenv joblib openpyxl gTTS"

say() { printf '\n  %s\n' "$*"; }

find_python() {
  for c in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

setup() {
  say "VayuVaani: first-time setup (3 to 10 minutes, needs internet). Details go to setup_log.txt."
  PY="$(find_python)" || {
    say "VayuVaani needs Python 3.10 to 3.13."
    if [ "$(uname)" = "Darwin" ]; then
      say "Install it from https://www.python.org/downloads/macos/ (or: brew install python@3.12), then open VayuVaani again."
    else
      say "Install it with your package manager, for example:  sudo apt install python3 python3-venv   (Ubuntu/Debian)"
      say "or  sudo dnf install python3   (Fedora), then run this again."
    fi
    exit 1
  }
  say "Using $($PY --version 2>&1)."
  "$PY" -m venv "$VENV" >>"$LOG" 2>&1 || { say "Could not create the environment. On Ubuntu/Debian run: sudo apt install python3-venv"; exit 1; }
  "$VPY" -m pip install --upgrade pip >>"$LOG" 2>&1
  if ! "$VPY" -m pip install --only-binary=:all: $PINNED >>"$LOG" 2>&1; then
    say "Pinned versions not available for this Python; installing the latest ones instead."
    "$VPY" -m pip install $LOOSE >>"$LOG" 2>&1 || { say "Installing packages failed. See setup_log.txt."; exit 1; }
  fi
  "$VPY" -m pip install --no-deps -e "$HERE" >>"$LOG" 2>&1 || { say "Installing VayuVaani failed. See setup_log.txt."; exit 1; }

  # LightGBM on a Mac needs the OpenMP library.
  if [ "$(uname)" = "Darwin" ] && ! "$VPY" -c "import lightgbm" >>"$LOG" 2>&1; then
    if command -v brew >/dev/null 2>&1; then
      say "Installing OpenMP for the forecast model (brew install libomp)..."
      brew install libomp >>"$LOG" 2>&1
    fi
    "$VPY" -c "import lightgbm" >>"$LOG" 2>&1 || say "Note: the forecast model needs OpenMP. Install Homebrew (https://brew.sh), then run: brew install libomp"
  fi
  "$VPY" -c "import pandas, sklearn, requests" >>"$LOG" 2>&1 || { say "Setup did not finish. See setup_log.txt."; exit 1; }
  add_shortcut
  say "Setup finished."
}

add_shortcut() {
  if [ "$(uname)" = "Linux" ]; then
    APPS="$HOME/.local/share/applications"; mkdir -p "$APPS"
    ICON="$HERE/src/vayuvaani/app/static/icon-512.png"
    cat >"$APPS/vayuvaani.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=VayuVaani
Comment=Bengaluru air quality forecast
Exec="$HERE/vayuvaani.sh"
Path=$HERE
Icon=$ICON
Terminal=false
Categories=Science;Education;
EOF
    chmod +x "$APPS/vayuvaani.desktop"
    if [ -d "$HOME/Desktop" ]; then
      cp "$APPS/vayuvaani.desktop" "$HOME/Desktop/" && chmod +x "$HOME/Desktop/vayuvaani.desktop"
      command -v gio >/dev/null 2>&1 && gio set "$HOME/Desktop/vayuvaani.desktop" metadata::trusted true 2>/dev/null
    fi
    say "Added VayuVaani to your app menu."
  fi
}

if [ ! -x "$VPY" ] || ! "$VPY" -c "import vayuvaani, pandas, sklearn" >/dev/null 2>&1; then
  setup
fi
"$VPY" -c "import gtts" >/dev/null 2>&1 || "$VPY" -m pip install --only-binary=:all: "gTTS>=2.5" >>"$LOG" 2>&1

# Start in the background; the app opens its own window and stops when you close it.
nohup "$VPY" -m vayuvaani.app >>"$HERE/app_log.txt" 2>&1 &
sleep 1
