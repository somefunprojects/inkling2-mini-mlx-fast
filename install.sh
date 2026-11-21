#!/bin/sh
# Overlay the Inkling 2 Mini fast paths onto an mlx-vlm tree.
#   ./install.sh <dir>          <dir> holds mlx_vlm/ : an mlx-vlm source checkout, or a site-packages directory
#   ./install.sh <dir> --force  skip the upstream version check
# The target must be mlx-vlm at Neywa-Labs/mlx-vlm commit fd6404fb3261102600541127d59103dac65c2756 (the code this
# work was built and measured on; it reports version 0.8.0 but differs from the 0.8.0 wheel on PyPI). Overwritten
# files are backed up next to themselves as <file>.orig.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
T=${1:?usage: install.sh <dir containing mlx_vlm/> [--force]}
[ -d "$T/mlx_vlm/models/inkling2" ] || { echo "no mlx_vlm/models/inkling2 under $T" >&2; exit 2; }
# sha256 of the pinned upstream inkling2/language.py
WANT=$(cat "$HERE/UPSTREAM_LANGUAGE_SHA256")
SRC="$T/mlx_vlm/models/inkling2/language.py"; [ -f "$SRC.orig" ] && SRC="$SRC.orig"
GOT=$(shasum -a 256 "$SRC" | cut -d' ' -f1)
if [ "$GOT" != "$WANT" ] && [ "$2" != "--force" ]; then
  echo "upstream inkling2/language.py does not match the pinned mlx-vlm commit (got $GOT); use --force to overlay anyway" >&2; exit 3
fi
cd "$HERE"
find mlx_vlm -type f -name '*.py' | while read f; do
  mkdir -p "$T/$(dirname "$f")"
  [ -f "$T/$f" ] && [ ! -f "$T/$f.orig" ] && cp "$T/$f" "$T/$f.orig"
  cp "$f" "$T/$f"
done
# the tiny-model tests resolve mlx_vlm relative to their own folder, so they go next to mlx_vlm/
mkdir -p "$T/tests_fast" && cp tests_fast/*.py "$T/tests_fast/"
cp run_tests.sh tests.txt "$T/"
echo "overlaid onto $T"
