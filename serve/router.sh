#!/usr/bin/env bash
# One llama-server (router mode) on 127.0.0.1:8080. A model loads when a request names its preset, the previous one is
# unloaded (--models-max 1), and the server sleeps after 30 idle minutes.
#   LLAMA_SERVER=~/llama.cpp/build/bin/llama-server ./serve/router.sh
set -euo pipefail
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
PRESET="${PRESET:-$(dirname "$0")/models.ini}"
PIDFILE="${PIDFILE:-/tmp/llm-router.pid}"
export PATH="/usr/local/cuda/bin:$PATH"
[ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null && sleep 4 || true   # exact PID, never pkill -f
nohup "$LLAMA_SERVER" --models-preset "$PRESET" --models-max 1 --sleep-idle-seconds 1800 \
  --host 127.0.0.1 --port 8080 > /tmp/llm-router.log 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "router started, pid $(cat "$PIDFILE"), log /tmp/llm-router.log"
