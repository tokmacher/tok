#!/usr/bin/env bash
# tok_claude.sh — Shell integration for Tok + Claude Code
#
# This repo copy is for development. Public installs should prefer `tok install`,
# which sources the packaged script from the installed tok-protocol path.
#
# If you need to source this file manually during local development:
#   source /path/to/tok/scripts/tok_claude.sh
#
# This defines only a `claude` function:
#   1. Starts the current Tok bridge if it is not already running
#   2. Routes Claude traffic through the bridge
#   3. Leaves the real `tok` CLI untouched

# Path to the tok repo (auto-detected from this script's location)
TOK_DIR="${TOK_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}"
TOK_BRIDGE_PORT="${TOK_BRIDGE_PORT:-9090}"
TOK_BRIDGE_HOST="${TOK_BRIDGE_HOST:-localhost}"
_TOK_STARTUP_TRIES="${_TOK_STARTUP_TRIES:-15}"

_tok_bridge_running() {
    curl -s --connect-timeout 1 "http://${TOK_BRIDGE_HOST}:${TOK_BRIDGE_PORT}/health" \
        -o /dev/null 2>&1
}

_tok_start_bridge() {
    if _tok_bridge_running; then
        return 0
    fi

    (
        cd "$TOK_DIR" || return 1
        command tok bridge start --port "$TOK_BRIDGE_PORT"
    )

    local tries=0
    while ! _tok_bridge_running && [ "$tries" -lt "$_TOK_STARTUP_TRIES" ]; do
        sleep 0.2
        tries=$((tries + 1))
    done

    if _tok_bridge_running; then
        echo "[tok] bridge ready on :${TOK_BRIDGE_PORT}" >&2
        return 0
    fi

    echo "[tok] bridge did not start in time" >&2
    return 1
}

claude() {
    _tok_start_bridge || return 1
    ANTHROPIC_BASE_URL="http://${TOK_BRIDGE_HOST}:${TOK_BRIDGE_PORT}" command claude "$@"
}
