"""Build the results table from the raw JSON files (so the table is always derived from the data).

  python scripts/summarize.py > results/RESULTS.md
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results"
MODELS = [  # preset, display name, quantisation, file size, tag in the speed folder
    ("qwen35-35b", "Qwen3.5-35B-A3B (MoE)", "UD-IQ4_XS", "17.5 GB"),
    ("qwen25-32b", "Qwen2.5-32B-Instruct (dense)", "IQ4_XS", "17.7 GB"),
    ("qwen3-coder", "Qwen3-Coder-30B-A3B (MoE)", "IQ4_XS", "16.4 GB"),
    ("llama33-70b", "Llama-3.3-70B-Instruct (dense)", "IQ2_XXS", "19.1 GB"),
    ("llama31-8b", "Llama-3.1-8B-Instruct (dense)", "Q4_K_M", "4.9 GB"),
]


def load(path):
    return json.load(open(path)) if path.exists() else None


def speed(preset):
    warm = load(ROOT / "speed" / f"warm_{preset}.json")
    if warm:
        return f"{warm['timings']['predicted_per_second']:.1f} tok/s (warm-up reply)"
    bench = ROOT / "speed" / f"llama-bench_{preset}.txt"
    if bench.exists():
        m = re.search(r"tg128\s*\|\s*([\d.]+)", bench.read_text())
        if m:
            return f"{float(m.group(1)):.0f} tok/s (llama-bench tg128)"
    return "not saved"


print("| Model | Quant | File | MMLU-300 | HumanEval-164 | Generation speed |")
print("|---|---|---|---|---|---|")
for preset, name, quant, size in MODELS:
    mm = load(ROOT / "raw" / f"mmlu_{preset}.json")
    he = load(ROOT / "raw" / f"humaneval_{preset}.json")
    print(f"| {name} | {quant} | {size} | {mm['acc']*100:.1f}% | "
          f"{(str(round(he['pass1']*100, 1)) + '%') if he else 'not run'} | {speed(preset)} |")
