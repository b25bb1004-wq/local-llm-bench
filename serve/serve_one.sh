#!/usr/bin/env bash
# One model, no router: serve_one.sh MODEL_FILE [N_CPU_MOE] [CTX] [PORT]
set -euo pipefail
MODEL="$1"; NCMOE="${2:-0}"; CTX="${3:-8192}"; PORT="${4:-8080}"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
PIDFILE="${PIDFILE:-/tmp/llm-server.pid}"
export PATH="/usr/local/cuda/bin:$PATH"
[ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null && sleep 3 || true
nohup "$LLAMA_SERVER" -m "models/$MODEL" -ngl 99 --n-cpu-moe "$NCMOE" -c "$CTX" -fa on -ctk q8_0 -ctv q8_0 \
  --host 127.0.0.1 --port "$PORT" --jinja > /tmp/llm-server.log 2>&1 < /dev/null &
echo $! > "$PIDFILE"
