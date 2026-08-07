#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
python3 ./d435i_plane_workspace_scan.py "$@"
