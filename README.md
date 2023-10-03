# CARE: Certifiably Robust Learning with Reasoning Pipeline

## Introduction

Deep Neural Networks (DNNs) have revolutionized a multitude of machine learning applications but are notorious for their susceptibility to adversarial attacks. Our project introduces [**CARE (Certifiably Robust leArning with REasoning)**](https://arxiv.org/abs/2209.05055), aiming to enhance the robustness of DNNs by integrating them with reasoning abilities. This pipeline consists of two primary components:

- **Learning Component**: Utilizes **CLIP** model for different semantic predictions, e.g., recognizing if an input image contains something furry.

- **Reasoning Component**: Employs probabilistic graphical models like Markov Logic Networks (MLN) to apply domain-specific knowledge and logic reasoning to the learning process.

## Getting Started

### Prerequisites

`conda create -n name care python=3.9`

`conda activate care && pip install -r requirements.txt`

### Load the model:

```
from defense import load_care_model
# the input is a tensor with shape N * 3 * 32 * 32 in range [0,1]
# for adv attack （although our model is trained with Gaussian Augmentation instead of adversarial training）
model = load_care_model().cuda()

# for ceritification with randomized smoothing, you need to specify the noise_sd
model = load_care_model(noise_sd = 0.25).cuda()
model = model.eval()

# And you can use model.predict(x: torch.tensor, n: int, alpha: float, batch_size: int)
# or model.certify(x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int)
# to do the predicition or certification
```

## Citation & Further Reading

If you find our work useful, please consider citing our [paper](https://arxiv.org/abs/2209.05055):

```
@inproceedings{zhang2023care,
  title={CARE: Certifiably Robust Learning with Reasoning via Variational Inference},
  author={Zhang, Jiawei and Li, Linyi and Zhang, Ce and Li, Bo},
  booktitle={2023 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML)},
  pages={554--574},
  year={2023},
  organization={IEEE}
}
```

## Contact

If you have any questions or encounter any errors while running the code, feel free to contact [jiaweiz@illinois.edu](mailto:jiaweiz@illinois.edu)!