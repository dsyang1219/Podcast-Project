#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
echo "[$(date -u +%FT%TZ)] breakdown"; $PY step4_ideology/pairwise_scale.py breakdown --workers 4
echo "[$(date -u +%FT%TZ)] compare";   $PY step4_ideology/pairwise_scale.py compare --workers 4
echo "[$(date -u +%FT%TZ)] fit";       $PY step4_ideology/pairwise_scale.py fit --boot 200
echo "[$(date -u +%FT%TZ)] pilot done"
