# ROSE
| <img src="https://github.com/jiangllan/ROSE/blob/main/docs/raw.pdf" alt="space-1.jpg" style="zoom: 33%;" /> | <img src="https://github.com/jiangllan/ROSE/blob/main/docs/overfit.pdf" alt="space-1.jpg" style="zoom:33%;" /> | <img src="https://github.com/jiangllan/ROSE/blob/main/docs/vanilla.pdf" alt="space-1.jpg" style="zoom:33%;" /> | <img src="https://github.com/jiangllan/ROSE/blob/main/docs/ensemble.pdf" alt="space-1.jpg" style="zoom: 33%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                       w/o Fine-Tuning                        |                           Overfit                            |                     Vanilla Fine-Tuning                      |                        ROSE-Ensemble                         |

Codes for paper "ROSE: Robust Selective Fine-tuning for Pre-trained Language Models".

## Contents

1. Introduction
2. Usage
3. Contacts
4. Acknowledgements

## Introduction

We present an ***attack-agnostic*** and ***model-agnostic*** defense method called **RO**bust **SE**letive fine-tuning (**ROSE**).
ROSE conducts selective updates when adapting pre-trained models to downstream tasks, filtering out invaluable and unrobust updates of parameters.
Specifically, we propose two strategies: the first-order and second-order ROSE for selecting target robust parameters.
The experimental results show that ROSE achieves significant improvements in adversarial robustness on various downstream NLP tasks, and the ensemble method even surpasses both variants above.
Furthermore, ROSE can be easily incorporated into existing fine-tuning methods to improve their adversarial robustness further.
The empirical analysis confirms that ROSE eliminates unrobust spurious updates during fine-tuning, leading to solutions corresponding to flatter and wider optima than the conventional method. For more details. please refer to our paper.

## Usage



### Requirements

Install dependencies and apex:

```

```

### Training and Evaluation

1. Training with ROSE 

2. Evaluation on 

   Please refer to "runs/tr"

## Contacts

Jiangl20 at mails dot tsinghua dot edu dot cn

## Acknowledgements

This codebase is built on Huggingface's Transformers. Thanks to them!