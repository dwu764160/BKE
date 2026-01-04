#!/usr/bin/env bash
set -euo pipefail

mkdir -p docs
DOT=docs/flow_diagram.dot
OUT=docs/flow_diagram.png

if command -v dot >/dev/null 2>&1; then
  echo "Rendering $DOT → $OUT"
  dot -Tpng "$DOT" -o "$OUT"
  echo "Wrote $OUT"
else
  echo "Graphviz 'dot' not found. Install it (e.g. 'sudo apt install graphviz') and re-run this script." >&2
  exit 2
fi
