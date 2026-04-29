# BCB Sandbagging Detection Programme

A pre-registered research programme investigating behavioural signatures of sandbagging (deliberate underperformance) in small instruction-tuned LLMs, applying psychometric measurement principles to LLM evaluation.

## Studies

### Study 1: Below-Chance Blindness Pilot

**Paper:** [Below-Chance Blindness: Prompted Underperformance in Small LLMs Produces Positional Bias Rather than Answer Avoidance](https://arxiv.org/abs/2604.25249)

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

### Study 3: Instruction Complexity and Positional Collapse

**Paper:** Instruction Complexity Induces Positional Collapse in Adversarial LLM Evaluation (in preparation)

**Pre-registration:** [OSF](https://osf.io/7p64)

Maps the boundary between content-blind positional shortcuts and content-aware sandbagging strategies using a six-condition adversarial instruction-specificity gradient. 2 models (Llama-3-8B, Llama-3.1-8B), 7 primary conditions, 4 null baselines, 29,600 trials.

Key findings:
- Three response regimes identified: content-engaged compliance, shortcut with residual engagement, and extreme positional collapse
- The only multi-step instruction (S5: "identify correct, then choose different") produced near-total positional collapse (99.9% at one position for Llama-3-8B), matching the model's content-absent null-prompt default
- Distributional collapse and content engagement co-occur (50% concordance between screening criteria), revealing that entropy-based screening and difficulty-based content assessment measure partially independent dimensions of response validity
- Effect replicated across both models and four academic domains

## Repository structure

```
bcb-sandbagging-pilot/
  configs/                    # Model and experiment configuration
  data/                       # Sampled items and permutation assignments
  results/
    positional_bias/          # Study 2 results
      robustness/             # T=0.7 stochastic robustness check
      analysis/               # Analysis outputs
    study3/                   # Study 3 results
      analysis/               # Study 3 analysis outputs and figures

  # Study 1
  run_inference.py            # Study 1: inference harness
  run_analysis.py             # Study 1: analysis pipeline
  sample_items.py             # Item sampling from MMLU-Pro (seed 42)

  # Study 2
  generate_permutations.py    # Cyclic shift assignments (seed 84)
  run_positional_bias.py      # Inference harness with permutation
  run_robustness_check.py     # T=0.7, K=5 robustness check
  run_analysis_local.py       # Pre-registered analysis pipeline

  # Study 3
  run_study3.py               # Inference harness (7 primary + 4 null conditions)
  run_study3_analysis.py      # Pre-registered analysis pipeline
```

## Hardware

All inference: AMD Radeon RX 7900 GRE (16 GB VRAM), Q5_K_M quantisation, llama-cpp-python 0.3.16, Vulkan backend. Python 3.11.

## Models

Studies 1 and 2:
- Qwen-2.5-7B-Instruct (Q5_K_M)
- Meta-Llama-3-8B-Instruct (Q5_K_M)
- Phi-3.5-mini-instruct (Q5_K_M)

Study 3:
- Meta-Llama-3-8B-Instruct (Q5_K_M)
- Meta-Llama-3.1-8B-Instruct (Q5_K_M)

## Citation

```bibtex
@article{cacioli2026bcb,
  title={Below-Chance Blindness: Prompted Underperformance in Small LLMs
         Produces Positional Bias Rather than Answer Avoidance},
  author={Cacioli, Jon-Paul},
  year={2026},
  eprint={2604.25249},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

@article{cacioli2026positional,
  title={Option-Order Randomisation Reveals a Distributional Position
         Attractor in Prompted Sandbagging},
  author={Cacioli, Jon-Paul},
  year={2026},
  note={OSF: osf.io/efr6s}
}

@article{cacioli2026complexity,
  title={Instruction Complexity Induces Positional Collapse in
         Adversarial LLM Evaluation},
  author={Cacioli, Jon-Paul},
  year={2026},
  note={OSF: osf.io/7p64}
}
```

## License

Code: MIT. Data: CC-BY 4.0.
