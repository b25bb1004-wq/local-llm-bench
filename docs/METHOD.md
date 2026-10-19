# Method and caveats

## Hardware and software
Ryzen 5 7600X, 32 GB DDR5-4800 (WSL capped at 15 GB), NVIDIA RTX 5070 with 12 GB, WSL2 Ubuntu. llama.cpp
(build `b1-797de63`, CUDA). One model on the GPU at a time.

## MMLU (accuracy)
- 300 questions drawn once with `random.Random(0).sample` from the full `cais/mmlu` "all" test split (14,042 questions),
  the same 300 for every model.
- Zero-shot prompt: "The following is a multiple choice question about <subject>.", the question, four lettered
  choices, then `Answer:`. One token is generated with `n_probs = 300`; the prediction is the letter A-D with the highest
  log-probability among the returned candidates.
- With n = 300 the 95% interval on an accuracy near 80% is about +/- 4.5 points, so differences under about 5 points
  (for example 83.7% vs 82.3%) are not significant.

## HumanEval (pass@1)
- All 164 problems of `openai/openai_humaneval`, greedy decoding (temperature 0), 512 new tokens at most, stop strings
  `\ndef `, `\nclass `, `\nif __name__`, `\nprint(`, `\n#`, ```` \n``` ````.
- The completion is appended to the prompt, the problem's tests are run in a fresh temporary directory with a 10 s
  timeout; pass = exit code 0. This is not a sandbox: run it somewhere disposable.
- The 70B was not run: at 2.3 tok/s the 164 problems would take about 1.8 hours.

## Serving
- `serve/router.sh` starts one `llama-server` with `--models-preset serve/models.ini --models-max 1`: a request that
  names a preset loads that model and unloads the previous one, which is how five models share 12 GB.
- All presets: all layers on the GPU (`ngl = 99`), flash attention, q8_0 KV cache, 8192 context.
- Mixture-of-experts models (Qwen3.5-35B-A3B, Qwen3-Coder-30B-A3B) keep the expert tensors of the first 20 layers on
  the CPU (`n-cpu-moe = 20`). They read about 3B active parameters per token, which is why they are fast despite the size.
- Dense 32B and 70B models do not fit: 34 and 36 layers are offloaded and the context is cut to 4096, so they run partly
  on the CPU and are 7x to 17x slower than the MoE models.

## What is not settled
- **Speed.** Only some speeds were saved (`results/speed/`). The 35B was first measured at 37 tok/s and later at
  16 to 18 tok/s with the same files; the cause (warm file cache, a background job, or clocks) was not found, so no
  single figure is quoted for it. The Coder model measured 34 to 40 tok/s. Re-measure with `llama-bench` on a warm
  system before relying on any speed number here.
- One machine, one quantisation per model, 300 MMLU questions: read the table as a ranking with error bars, not as
  published scores.
