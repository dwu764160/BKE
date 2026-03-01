#!/usr/bin/env bash
set -euo pipefail

# Create a dated full data backup similar to data_backup_YYYYMMDD.
# Usage:
#   bash scripts/backup_data_snapshot.sh [YYYYMMDD]

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKSPACE"

STAMP="${1:-$(date +%Y%m%d)}"
TARGET="data_backup_${STAMP}"

echo "[BACKUP] Source: data/"
echo "[BACKUP] Target: ${TARGET}/"

mkdir -p "$TARGET"

for path in nba_headers.json nba_session.json features historical matchup official_stats processed tracking matchup_cache tracking_cache; do
    if [[ -e "data/${path}" ]]; then
        cp -a "data/${path}" "$TARGET/"
        echo "[BACKUP] copied data/${path}"
    else
        echo "[BACKUP] skip missing data/${path}"
    fi
done

echo "[BACKUP] complete: ${TARGET}/"
