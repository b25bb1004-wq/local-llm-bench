# local-llm-bench

Five open LLMs on one 12 GB gaming GPU: MMLU and HumanEval scores, plus a llama.cpp router that loads each model on
demand. The scripts are small on purpose (two evaluators, one driver) and the results table is generated from the raw
JSON in this repo.

| Model | Quant | File | MMLU-300 | HumanEval-164 | Generation speed |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B (MoE) | UD-IQ4_XS | 17.5 GB | 83.7% | 75.6% | 16 to 37 tok/s, unsettled |
| Qwen2.5-32B-Instruct (dense) | IQ4_XS | 17.7 GB | 82.3% | 70.1% | 5.1 tok/s |
| Qwen3-Coder-30B-A3B (MoE) | IQ4_XS | 16.4 GB | 74.7% | 85.4% | 34 to 40 tok/s |
| Llama-3.3-70B-Instruct (dense) | IQ2_XXS | 19.1 GB | 75.0% | not run | 2.3 tok/s |
| Llama-3.1-8B-Instruct (dense) | Q4_K_M | 4.9 GB | 65.7% | 61.0% | 92 tok/s |

Hardware: Ryzen 5 7600X, RTX 5070 (12 GB), 32 GB RAM, WSL2. Details, caveats and what is not settled:
[docs/METHOD.md](docs/METHOD.md). The table is regenerated from `results/raw/` by `scripts/summarize.py`
([results/RESULTS.md](results/RESULTS.md)).

## What the numbers say
- **A mixture-of-experts model is the right shape for this card.** The 35B model with about 3B active parameters scored
  best on MMLU and ran several times faster than the dense 32B, which has to be half-offloaded to the CPU.
- **The coding specialist wins at code** (85.4% HumanEval) and gives up about 9 points of MMLU for it.
- **The 70B did not earn its place.** At 2-bit it scored 8.7 points below the 35B on MMLU at about a seventeenth of the
  speed, so it was dropped.
- **With 300 questions, differences under about 5 MMLU points are noise** (see the method notes).

## Reproduce
```bash
pip install -r requirements.txt                       # datasets
ln -s /path/to/gguf/folder models                      # presets in serve/models.ini point at models/
LLAMA_SERVER=~/llama.cpp/build/bin/llama-server ./serve/router.sh
scripts/run_bench.sh qwen35-35b 35b                    # warm-up speed sample, MMLU-300, HumanEval-164
python scripts/summarize.py > results/RESULTS.md
```
`eval/humaneval.py` runs model-written code: use a container or a throwaway machine.

## Layout
`eval/` the two evaluators · `serve/` router presets and launch scripts · `scripts/` driver and table builder ·
`results/raw/` per-question outputs · `results/speed/` the speed samples that were saved · `docs/METHOD.md`.

MIT licence.
