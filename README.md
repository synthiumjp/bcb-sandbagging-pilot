# BCB Sandbagging Detection Programme

A pre-registered research programme investigating behavioural signatures of sandbagging (deliberate underperformance) in small instruction-tuned LLMs.

## Studies

### Study 1: Below-Chance Blindness Pilot

**Paper:** [Below-Chance Blindness: Prompted Underperformance in Small LLMs Produces Positional Bias Rather than Answer Avoidance](link-to-arxiv)

**Pre-registration:** [OSF](https://osf.io/6zftv/)

Tested whether symptom validity testing (SVT) logic from clinical malingering detection could identify sandbagging through below-chance performance (BCB) on forced-choice items. 3 models (Qwen-2.5-7B, Llama-3-8B, Phi-3.5-mini), 4 MMLU-Pro domains, 4 conditions, 24,000 trials.

The plausibility gate failed. Zero cells showed below-chance performance. Exploratory analysis revealed that Llama-3-8B complied with sandbagging by adopting a position-dominant response policy (E: 31.8%, F: 26.1%) rather than answer avoidance. The critical limitation: fixed option ordering in MMLU-Pro could not distinguish model-level positional preferences from dataset-level distractor structure.

### Study 2: Option-Order Randomisation (Positional Attractor)

**Paper:** Option-Order Randomisation Reveals a Distributional Position Attractor in Prompted Sandbagging (in preparation)

**Pre-registration:** [OSF](https://osf.io/efr6s)

Confirmatory follow-up adding cyclic option-order randomisation as the critical control. Same 3 models, same 2,000 items, 4 conditions (honest/sandbagging x original/permuted order), 24,000 primary trials plus 20,000 stochastic robustness check trials.

Key findings:
- The pre-registered item-level same-letter diagnostic returned ambiguous (37.3% same-letter rate, below the 50% threshold)
- Aggregate distributional evidence was overwhelming: Pearson r = 0.9994, JS divergence = 0.027 between original and permuted sandbagging distributions (vs 0.386 between honest and sandbagging)
- Accuracy spiked to 72.1% when the correct answer coincidentally occupied the model's preferred position E
- The data reveal a distributional position attractor: under sandbagging, the model enters a low-entropy response-position basin that is invariant under content rotation
- Qwen-2.5-7B ignored the sandbagging instruction (negative control). Phi-3.5-mini showed weak compliance and its pilot J-bias did not replicate
- No model showed significant accuracy change under honest option permutation

## Repository structure

```
bcb-sandbagging-pilot/
  configs/             # Model and experiment configuration
  data/                # Sampled items and permutation assignments
  results/             # Raw response data (JSONL)
    positional_bias/   # Study 2 results
      robustness/      # T=0.7 stochastic robustness check
      analysis/        # Analysis outputs and item classifications
  generate_permutations.py   # Study 2: cyclic shift assignments (seed 84)
  run_inference.py           # Study 1: inference harness
  run_positional_bias.py     # Study 2: inference harness with permutation
  run_robustness_check.py    # Study 2: T=0.7, K=5 robustness check
  run_analysis_local.py      # Study 2: pre-registered analysis pipeline
  sample_items.py            # Item sampling from MMLU-Pro (seed 42)
```

## Hardware

All inference: AMD Radeon RX 7900 GRE (16 GB VRAM), Q5_K_M quantisation, llama-cpp-python 0.3.16, Vulkan backend. Python 3.11.

## Models

- Qwen-2.5-7B-Instruct (Q5_K_M)
- Meta-Llama-3-8B-Instruct (Q5_K_M)
- Phi-3.5-mini-instruct (Q5_K_M)

## Citation

```bibtex
@article{cacioli2026bcb,
  title={Below-Chance Blindness: Prompted Underperformance in Small LLMs
         Produces Positional Bias Rather than Answer Avoidance},
  author={Cacioli, Jon-Paul},
  year={2026},
  note={arXiv preprint}
}
```

## License

Code: MIT. Data: CC-BY 4.0.
