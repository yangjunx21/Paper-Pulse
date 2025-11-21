#!/usr/bin/env bash
set -euo pipefail

# Helper script to launch the interactive intent profile builder.
# Usage: ./scripts/build_intent_profile.sh [PROFILE_NAME] [--resume] [extra args...]
# By default it stores configs under <repo>/config/intent_profiles.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -gt 0 ]]; then
    PROFILE_NAME="$1"
    shift
else
    PROFILE_NAME="default"
fi
CONFIG_DIR="${INTENT_CONFIG_DIR:-$PROJECT_ROOT/config/intent_profiles}"

python -m paper_agent.intent_cli \
    --profile-name "$PROFILE_NAME" \
    --config-dir "$CONFIG_DIR" \
    "$@"

