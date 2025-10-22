---
library_name: transformers
license: apache-2.0
base_model: google-t5/t5-small
tags:
- translation
- generated_from_trainer
metrics:
- bleu
model-index:
- name: marian-finetuned-opusbooks-en-to-fr-graph
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# marian-finetuned-opusbooks-en-to-fr-graph

This model is a fine-tuned version of [google-t5/t5-small](https://huggingface.co/google-t5/t5-small) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.6308
- Model Preparation Time: 0.0016
- Bleu: 5.9861

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 32
- eval_batch_size: 64
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Model Preparation Time | Bleu   |
|:-------------:|:-----:|:----:|:---------------:|:----------------------:|:------:|
| 1.9345        | 1.0   | 2780 | 1.6793          | 0.0016                 | 5.5751 |
| 1.9325        | 2.0   | 5560 | 1.6405          | 0.0016                 | 5.8815 |
| 1.9334        | 3.0   | 8340 | 1.6308          | 0.0016                 | 5.9861 |


### Framework versions

- Transformers 4.56.2
- Pytorch 2.8.0+cu128
- Datasets 4.1.1
- Tokenizers 0.22.1
