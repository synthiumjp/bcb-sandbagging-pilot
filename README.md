# Below-Chance Blindness: Prompted Underperformance in Small LLMs Produces Positional Bias Rather than Answer Avoidance

**Pre-registered at OSF:** [https://osf.io/6zftv/](https://osf.io/6zftv/)

**Author:** Jon-Paul Cacioli | [ORCID 0009-0000-7054-2014](https://orcid.org/0009-0000-7054-2014)

---

## Overview

This repository contains all code, data, and analysis for a pre-registered pilot study testing whether symptom validity testing (SVT) logic from clinical malingering detection can identify LLM sandbagging through below-chance performance (BCB) on forced-choice items.

**Key finding:** The plausibility gate failed — zero of 12 model–domain cells showed below-chance performance under sandbagging instruction. Exploratory analyses revealed that compliant models implement underperformance as positional bias (collapsing responses onto preferred option positions) rather than answer-aware avoidance. An explicit anti-task instruction ("pick the least likely answer") drove two of three models below chance, confirming the capability exists but is not activated by "deliberately underperform."

## Setup

### Requirements

- Python 3.11
- AMD GPU with Vulkan support (tested on RX 7900 GRE 16GB)
- ~18GB disk space for GGUF models

### Installation

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install llama-cpp-python datasets scipy huggingface-hub

# For Vulkan GPU acceleration:
# CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python
```

### Models

Three GGUF models at Q5_K_M quantisation:

| Model | Source |
|-------|--------|
| Qwen-2.5-7B-Instruct | `bartowski/Qwen2.5-7B-Instruct-GGUF` |
| Llama-3-8B-Instruct | `bartowski/Meta-Llama-3-8B-Instruct-GGUF` |
| Phi-3.5-mini-instruct | `bartowski/Phi-3.5-mini-instruct-GGUF` |

Download via:
```bash
hf download bartowski/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q5_K_M.gguf --local-dir ./models
hf download bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q5_K_M.gguf --local-dir ./models
hf download bartowski/Phi-3.5-mini-instruct-GGUF Phi-3.5-mini-instruct-Q5_K_M.gguf --local-dir ./models
```

Then edit `configs/local_config.py` with your model paths.

## Execution

**All steps must be run in order. Condition A must run before B, C, D.**

```bash
# Step 1: Sample items (run once)
python sample_items.py

# Step 2: Run inference (one model at a time)
python run_inference.py --model llama3-8b --condition all
python run_inference.py --model qwen2.5-7b --condition all
python run_inference.py --model phi3.5-mini --condition all

# Step 3: Pre-registered analysis
python run_analysis.py

# Step 4: Exploratory post-hoc analyses
python run_posthoc.py
python run_position_analysis.py
```

Inference supports **resume** — if interrupted, re-run the same command and it skips completed trials.

## File Structure

```
bcb-sandbagging-pilot/
├── configs/
│   ├── config.py              # Pre-registered parameters (DO NOT MODIFY)
│   └── local_config.py        # Your GGUF paths (git-ignored)
├── data/
│   ├── sampled_items.jsonl    # 2,000 sampled items with C sub-prompt assignments
│   └── sampling_stats.json    # Sampling audit trail
├── results/
│   └── {model}_{condition}.jsonl  # Trial-level results (24,000 trials)
├── analysis/
│   ├── analysis_report.txt        # Pre-registered gate decision + full report
│   ├── full_results.json          # Machine-readable results
│   ├── posthoc_report.txt         # Exploratory: compliance, sub-prompts, distributions
│   └── position_analysis_report.txt  # Exploratory: positional bias analysis
├── sample_items.py            # Step 1: item sampling
├── run_inference.py           # Step 2: model inference
├── run_analysis.py            # Step 3: pre-registered analysis
├── run_posthoc.py             # Step 4a: exploratory post-hoc
├── run_position_analysis.py   # Step 4b: positional bias analysis
└── README.md
```

## Pre-registered Design

All parameters locked per [OSF pre-registration](https://osf.io/6zftv/):

- **Seed:** 42
- **Items:** 10-option MMLU-Pro items only (chance = 0.10)
- **Domains:** Physics, Law, Psychology, Economics (500 items each)
- **Models:** Qwen-2.5-7B, Llama-3-8B, Phi-3.5-mini (Q5_K_M, greedy T=0)
- **Gate:** ≥3 of 12 cells significant below chance (α = 0.01, Bonferroni)
- **Gate outcome: FAIL** (0 significant cells)

## Key Results

| Analysis | Outcome |
|----------|---------|
| H1 (plausibility gate) | **FAIL** — 0/12 cells below chance |
| H2 (B vs A) | Significant (W=3.0, p=.002) — B suppresses accuracy |
| H2 (B vs C) | Significant (W=4.0, p=.003) — C suppresses more than B |
| H3 (null prediction) | Confirmed — 0 cells below chance in A or D |
| C3 below chance | 8/12 cells for Qwen+Phi — capability exists |
| Positional bias (Llama) | E: 31.8%, F: 26.1% under B (vs ~10% under A) |

## Citation

```bibtex
@misc{cacioli2026bcb,
  title={Below-Chance Blindness: Prompted Underperformance in Small LLMs
         Produces Positional Bias Rather than Answer Avoidance},
  author={Cacioli, Jon-Paul},
  year={2026},
  note={Pre-registered at OSF: https://osf.io/6zftv/}
}
```

## License

Code: MIT | Data: CC-BY-4.0
