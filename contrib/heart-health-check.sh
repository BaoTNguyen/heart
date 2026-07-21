#!/bin/sh
# Runs `heart pulse health`; on a WARN (non-zero exit) sends a desktop
# notification via notify-send and, if NTFY_TOPIC is set, a push to
# https://ntfy.sh/$NTFY_TOPIC. Invoked every 10 min by heart-health.timer.
set -u

OUT=$(heart pulse health --hours 1 2>&1)
CODE=$?

if [ "$CODE" -ne 0 ]; then
    FIRST=$(printf '%s\n' "$OUT" | grep -m1 '^WARN' || printf '%s\n' "$OUT" | head -n1)
    notify-send "heart health" "$FIRST" 2>/dev/null || true
    if [ -n "${NTFY_TOPIC:-}" ]; then
        curl -fsS -d "$FIRST" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
    fi
fi

exit "$CODE"
