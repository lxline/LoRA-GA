Separate LoRA-GA from the PEFT library.

# [LoRA-GA: Low-Rank Adaptation with Gradient Approximation](https://arxiv.org/abs/2407.05000)

- [LoRA-GA: Low-Rank Adaptation with Gradient Approximation](#lora-ga-low-rank-adaptation-with-gradient-approximation)
  - [Overview](#overview)
  - [Quick start](#quick-start)
    - [1. Install custom peft](#1-install-custom-peft)
    - [2. Use LoRA-GA in peft](#2-use-lora-ga-in-peft)
    - [3. Explanation](#3-explanation)
  - [Examples](#examples)
    - [Multi-card training example](#multi-card-training-example)
  - [Note on Usage](#note-on-usage)
  - [Citation](#citation)

## Overview

We introduce a novel initialization method, LoRA-GA (Low Rank Adaptation with Gradient Approximation), which aligns the gradients of low-rank matrix product with those of full fine-tuning at the first step. Our extensive experiments demonstrate that LoRA-GA achieves a convergence rate comparable to that of full fine-tuning (hence being significantly faster than vanilla LoRA as well as various recent improvements) while simultaneously attaining comparable or even better performance.
![](./resource/pic/lora_ga_exp_pic.png)
(Left) Training loss curves of Llama 2-7B on MetaMathQA to training steps. LoRA-GA
converges as quickly as full fine-tuning and outperforms LoRA. (Right) Initialization procedures
used in LoRA and LoRA-GA. The key difference is that LoRA-GA initializes adapters using the
eigenvectors of the gradient matrix, as opposed to random initialization with a scaling factor.

## Citation

```
@misc{wang2024loragalowrankadaptationgradient,
    title={LoRA-GA: Low-Rank Adaptation with Gradient Approximation},
    author={Shaowen Wang and Linxi Yu and Jian Li},
    year={2024},
    eprint={2407.05000},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2407.05000},
}
```
