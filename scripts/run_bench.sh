#!/usr/bin/env bash
# Full benchmark for one model behind the router: warm-up (speed sample), MMLU-300, HumanEval-164.
#   scripts/run_bench.sh PRESET_NAME TAG        e.g. scripts/run_bench.sh qwen35-35b 35b
# Raw outputs land in results/raw/ (JSON + log); the warm-up reply, with llama.cpp timings, in results/speed/.
set -uo pipefail
MODEL="${1:?preset name}"; TAG="${2:?short tag}"
URL="${URL:-http://127.0.0.1:8080}"
mkdir -p results/raw results/speed
# the first request loads the model; its reply carries prompt and generation speed
curl -s -m 1200 "$URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" > "results/speed/warm_$TAG.json"
python eval/mmlu.py --url "$URL" --model "$MODEL" --n 300 --out "results/raw/mmlu_$TAG.json" > "results/raw/mmlu_$TAG.log" 2>&1
python eval/humaneval.py --url "$URL" --model "$MODEL" --out "results/raw/humaneval_$TAG.json" > "results/raw/humaneval_$TAG.log" 2>&1
echo "done $TAG"
