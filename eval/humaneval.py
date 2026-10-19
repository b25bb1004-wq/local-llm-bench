"""HumanEval pass@1 (greedy) through a running llama-server.

  python eval/humaneval.py --url http://127.0.0.1:8080 --model qwen3-coder --out results/raw/humaneval_qwen3-coder.json

WARNING: this executes model-written code. Each program runs in its own temporary directory with a 10 s timeout, which
is NOT a sandbox. Run it in a container or a throwaway VM if you do not trust the model.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8080")
p.add_argument("--model", default=os.environ.get("MODEL", ""), help="preset name when the server runs in router mode")
p.add_argument("--out", default="humaneval.json")
p.add_argument("--limit", type=int, default=164)
p.add_argument("--max_tokens", type=int, default=512)
a = p.parse_args()

ds = load_dataset("openai/openai_humaneval", split="test")
STOP = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n```"]


def complete(prompt):
    body = json.dumps({"model": a.model, "prompt": prompt, "n_predict": a.max_tokens, "temperature": 0, "stop": STOP,
                       "cache_prompt": True}).encode()
    req = urllib.request.Request(a.url + "/completion", body, {"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    return d["content"], d.get("timings", {})


passed, rows, t0, toks = 0, [], time.time(), 0
for k, ex in enumerate(list(ds)[: a.limit]):
    completion, timings = complete(ex["prompt"])
    toks += timings.get("predicted_n", 0)
    program = ex["prompt"] + completion + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.py")
        open(path, "w").write(program)
        try:
            ok = subprocess.run([sys.executable, path], cwd=d, capture_output=True, timeout=10).returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
    passed += ok
    rows.append({"task": ex["task_id"], "pass": ok})
    if k % 10 == 0 or k == a.limit - 1:
        print(f"{k+1}/{a.limit} pass@1 {passed/(k+1):.3f} {time.time()-t0:.0f}s gen_tok {toks}", flush=True)
json.dump({"pass1": passed / min(a.limit, len(ds)), "secs": time.time() - t0, "gen_tokens": toks, "rows": rows},
          open(a.out, "w"))
print("FINAL pass@1", passed / min(a.limit, len(ds)))
