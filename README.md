# 🏥 Skin Disease Classification — EfficientNetB0 + Transfer Learning

Classifies 7 types of skin lesions from dermoscopy images using EfficientNetB0 with transfer learning and Grad-CAM explainability. Trained on the HAM10000 dataset.

## Results

- Test Accuracy: 74.15%
- Vascular Lesion detection: 92.9%
- Melanoma detection: 75.0%
- Training: 10 epochs on T4 GPU — Free Google Colab

## Classes

1. Melanoma (MEL) — Malignant
2. Melanocytic Nevi (NV) — Benign mole
3. Basal Cell Carcinoma (BCC) — Malignant
4. Actinic Keratosis (AKIEC) — Precancerous
5. Benign Keratosis (BKL) — Benign
6. Dermatofibroma (DF) — Benign
7. Vascular Lesion (VASC) — Benign

## Architecture

- Base: EfficientNetB0 pretrained on ImageNet
- Fine-tuned: Last 3 blocks + custom classifier head
- Trainable params: 3.49M / 4.34M (80.36%)
- Loss: CrossEntropyLoss with class weights for imbalance
- Optimizer: AdamW with CosineAnnealingLR

## Dataset

- HAM10000 — 10,015 dermoscopy images
- Split: 8,012 train / 1,001 val / 1,002 test
- Heavy class imbalance handled with weighted loss

## Key Features

- Transfer learning from ImageNet weights
- Class-weighted loss for imbalanced dataset
- Data augmentation: flip, rotate, color jitter
- Grad-CAM explainability — visualizes model attention
- Full training curves and confusion matrix

## Tech Stack

- PyTorch + torchvision
- EfficientNetB0
- Grad-CAM
- scikit-learn
- Matplotlib + Seaborn
- Google Colab T4 GPU

## Portfolio Links

- Project 4 LLM Eval Framework: https://github.com/Boatengs/llm-evaluation-framework
- Project 3 Medical QA LoRA: https://github.com/Boatengs/medical-qa-lora
- Project 2 SPORTZBOT RAG: https://github.com/Boatengs/sports-rag-chatbot-
- Project 1 Sentiment Analyzer: https://github.com/Boatengs/sentiment-analyzer
