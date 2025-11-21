#!/usr/bin/env bash
set -euo pipefail

# Simple helper to run the paper agent with an intent profile.
# Step 1 (once per profile): ./scripts/build_intent_profile.sh my-profile
# Step 2: run this script (after editing the variables or exporting env overrides).

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROFILE_NAME="${PROFILE_NAME:-default}"
INTENT_CONFIG_DIR="${INTENT_CONFIG_DIR:-$PROJECT_ROOT/config/intent_profiles}"

# Date range (inclusive). Override with DATE_RANGE_START / DATE_RANGE_END env vars.
DATE_RANGE_START="${DATE_RANGE_START:-2025-11-17}"
DATE_RANGE_END="${DATE_RANGE_END:-2025-11-20}"

MAX_RESULTS="${MAX_RESULTS:-10}"
RELEVANCE_THRESHOLD="${RELEVANCE_THRESHOLD:-0.8}"
FALLBACK_LIMIT="${FALLBACK_LIMIT:-20}"
SUMMARY_LANGUAGE="${SUMMARY_LANGUAGE:-English}"
LLM_WORKERS="${LLM_WORKERS:-4}"

# Comma-separated sources, e.g., "arxiv,huggingface_daily"
SOURCES_CSV="${SOURCES_CSV:-arxiv,huggingface_daily}"
IFS=',' read -r -a SOURCES <<< "$SOURCES_CSV"

RECEIVER="${RECEIVER:-yangjunx21@gmail.com}"

python -m paper_agent.cli \
    --intent-profile "$PROFILE_NAME" \
    --intent-config-dir "$INTENT_CONFIG_DIR" \
    --date-range "$DATE_RANGE_START" "$DATE_RANGE_END" \
    --max-results "$MAX_RESULTS" \
    --relevance-threshold "$RELEVANCE_THRESHOLD" \
    --fallback-report-limit "$FALLBACK_LIMIT" \
    --llm-workers "$LLM_WORKERS" \
    --summary-language "$SUMMARY_LANGUAGE" \
    --send-email \
    --receiver "$RECEIVER" \
    --sources "${SOURCES[@]}"


