"""MMLU accuracy through a running llama-server (zero-shot, letter log-probability).

  python eval/mmlu.py --url http://127.0.0.1:8080 --model qwen35-35b --n 300 --out results/raw/mmlu_qwen35-35b.json

Method: a zero-shot prompt that ends in "Answer:", one generated token, and the prediction is the letter A-D with the
highest log-probability. The same 300 questions (random seed 0 over the full test split) are used for every model, so
scores are comparable across models; with n = 300 the 95% interval is about +/- 5 points.
"""
import argparse
import json
import os
import random
import time
import urllib.request

from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8080")
p.add_argument("--model", default=os.environ.get("MODEL", ""), help="preset name when the server runs in router mode")
p.add_argument("--n", type=int, default=300)
p.add_argument("--out", default="mmlu.json")
a = p.parse_args()

ds = load_dataset("cais/mmlu", "all", split="test")
idx = random.Random(0).sample(range(len(ds)), a.n)
LETTERS = "ABCD"


def ask(prompt):
    body = json.dumps({"model": a.model, "prompt": prompt, "n_predict": 1, "temperature": 0, "n_probs": 300,
                       "cache_prompt": True}).encode()
    req = urllib.request.Request(a.url + "/completion", body, {"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    probs = (d.get("completion_probabilities") or [{}])[0]
    top = probs.get("top_logprobs") or probs.get("top_probs") or []
    best, best_lp = None, -1e9
    for t in top:
        tok = (t.get("token") or "").strip()
        lp = t.get("logprob", t.get("prob", -1e9))
        if tok and tok in LETTERS and lp > best_lp:
            best, best_lp = tok, lp
    return best


correct, t0, rows = 0, time.time(), []
for k, i in enumerate(idx):
    ex = ds[i]
    q = (f"The following is a multiple choice question about {ex['subject'].replace('_', ' ')}.\n\n{ex['question']}\n"
         + "".join(f"{LETTERS[j]}. {c}\n" for j, c in enumerate(ex["choices"])) + "Answer:")
    pred = ask(q)
    correct += pred == LETTERS[ex["answer"]]
    rows.append({"i": i, "subject": ex["subject"], "pred": pred, "gold": LETTERS[ex["answer"]]})
    if k % 25 == 0 or k == len(idx) - 1:
        print(f"{k+1}/{len(idx)} acc {correct/(k+1):.3f} {time.time()-t0:.0f}s", flush=True)
json.dump({"n": a.n, "acc": correct / len(idx), "secs": time.time() - t0, "rows": rows}, open(a.out, "w"))
print("FINAL acc", correct / len(idx))
