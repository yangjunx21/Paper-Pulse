#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Paper Pulse: Pipeline Runner
# =============================================================================
# This script executes the paper discovery pipeline using a stored intent profile.
#
# Usage:
#   [ENV_VARS] ./scripts/run_with_intent.sh
#
# Configuration (Environment Variables):
#   PROFILE_NAME       : Name of the profile to use (default: "default")
#   DATE_RANGE_START   : Start date YYYY-MM-DD (default: today - 3 days)
#   DATE_RANGE_END     : End date YYYY-MM-DD (default: today)
#   MAX_RESULTS        : Max papers to keep after keyword filtering (default: 20)
#   SOURCES_CSV        : Comma-separated sources (default: arxiv,huggingface_daily)
#   RECEIVER           : Email receiver address (optional if EMAIL_* not set)
#   RELEVANCE_THRESHOLD: Min score (0.0-1.0) to include paper (default: 0.6)
#   ENABLE_PDF_ANALYSIS: Set to "true" to enable PDF downloading & full-text analysis (default: false)
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Setup & Defaults
# -----------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Determine dates (Linux/BSD compatible)
if date -v -3d >/dev/null 2>&1; then
    # BSD/MacOS
    DEFAULT_START=$(date -v -3d +%Y-%m-%d)
    DEFAULT_END=$(date +%Y-%m-%d)
else
    # GNU/Linux
    DEFAULT_START=$(date -d "3 days ago" +%Y-%m-%d)
    DEFAULT_END=$(date +%Y-%m-%d)
fi

# -----------------------------------------------------------------------------
# 2. User Configuration (Edit these or override via ENV)
# -----------------------------------------------------------------------------
PROFILE_NAME="${PROFILE_NAME:-default}"
INTENT_CONFIG_DIR="${INTENT_CONFIG_DIR:-$PROJECT_ROOT/config/intent_profiles}"

# Search Range
DATE_RANGE_START="${DATE_RANGE_START:-$DEFAULT_START}"
DATE_RANGE_END="${DATE_RANGE_END:-$DEFAULT_END}"

# Pipeline Tuning
MAX_RESULTS="${MAX_RESULTS:-20}"
RELEVANCE_THRESHOLD="${RELEVANCE_THRESHOLD:-0.6}"
FALLBACK_LIMIT="${FALLBACK_LIMIT:-10}"
SUMMARY_LANGUAGE="${SUMMARY_LANGUAGE:-English}"
LLM_WORKERS="${LLM_WORKERS:-4}"
ENABLE_PDF_ANALYSIS="${ENABLE_PDF_ANALYSIS:-false}"

# Sources & Output
SOURCES_CSV="${SOURCES_CSV:-arxiv,huggingface_daily}"
IFS=',' read -r -a SOURCES <<< "$SOURCES_CSV"
RECEIVER="${RECEIVER:-}"  # Leave empty if just testing local generation

# -----------------------------------------------------------------------------
# 3. Execution
# -----------------------------------------------------------------------------
echo "----------------------------------------------------------------"
echo "🚀 Starting Paper Pulse Pipeline"
echo "----------------------------------------------------------------"
echo "📄 Profile:      $PROFILE_NAME"
echo "📅 Date Range:   $DATE_RANGE_START to $DATE_RANGE_END"
echo "🔍 Sources:      ${SOURCES[*]}"
echo "🎯 Max Results:  $MAX_RESULTS (Threshold: $RELEVANCE_THRESHOLD)"
echo "📚 PDF Analysis: ${ENABLE_PDF_ANALYSIS}"
echo "----------------------------------------------------------------"

# Build arguments array
ARGS=(
    --intent-profile "$PROFILE_NAME"
    --intent-config-dir "$INTENT_CONFIG_DIR"
    --date-range "$DATE_RANGE_START" "$DATE_RANGE_END"
    --max-results "$MAX_RESULTS"
    --relevance-threshold "$RELEVANCE_THRESHOLD"
    --fallback-report-limit "$FALLBACK_LIMIT"
    --llm-workers "$LLM_WORKERS"
    --summary-language "$SUMMARY_LANGUAGE"
    --sources "${SOURCES[@]}"
)

# Enable PDF analysis if requested
if [[ "${ENABLE_PDF_ANALYSIS,,}" == "true" ]]; then
    ARGS+=(--enable-pdf-analysis)
fi

# Add optional email arguments if configured
if [[ -n "$RECEIVER" ]] || [[ -n "${EMAIL_RECEIVER:-}" ]]; then
    ARGS+=(--send-email)
    if [[ -n "$RECEIVER" ]]; then
        ARGS+=(--receiver "$RECEIVER")
    fi
else
    echo "ℹ️  No receiver specified; skipping email (local report only)."
fi

# Run
python -m paper_agent.cli "${ARGS[@]}"
