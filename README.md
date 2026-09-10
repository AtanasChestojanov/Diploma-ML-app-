# Detecting hate speech using deep learning on smartphones

This repository was developed as part of the diploma thesis *"Detecting hate speech using deep
learning on smartphones"*. It contains the code and the Android application for base-model
training, conversion to TensorFlow Lite, and personalization by on-device training of the classifier head.

Dataset: Dynahate (Dynamically Generated Hate Speech Dataset), Vidgen et al. 2021 —
https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset
(downloaded automatically by the training pipeline).

## ml/

Python scripts for training, evaluating, and exporting the model.

- **core.py** — shared library: the MobileBERT encoder and the classifier head, tokenization,
  TensorFlow Lite conversion, and evaluation metrics.
- **train_base.py** — the base-model pipeline (data preparation, training, evaluation, and
  TensorFlow Lite export).
- **personalize.py** — exports the trainable classifier head with the four on-device training
  signatures (`train` / `infer` / `save` / `restore`).
- **config/config.yaml** — all hyperparameters and file paths.

## App/

Android application that runs the exported model on-device to classify
typed text and incoming SMS messages, and performs on-device personalization: the small classifier
head is fine-tuned directly on the phone from a set of examples.
