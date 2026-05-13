# 📰 Fake News Detection Using NLP & Machine Learning

## Overview
This project builds a machine learning pipeline to automatically classify news articles as real or fake based on their text content. With misinformation spreading faster than ever online, scalable automated detection is a critical tool for platforms, search engines, and fact-checkers.

## Research Problem
Can machine learning models reliably detect fake news articles based on text content alone? Manual fact-checking is too slow and expensive to keep up with the volume of content published daily. This project explores whether NLP-based models can flag misleading articles in real time.

## Hypothesis
Transformer-based models (e.g. DistilBERT) will significantly outperform traditional TF-IDF classifiers at detecting fake news, but even simple baseline models will perform above chance due to strong linguistic signals present in misleading content.

## Dataset
**WELFake Dataset** — 72,134 labeled news articles (real/fake), aggregated from four independent sources: Kaggle, McIntire, Reuters, and BuzzFeed.
- Source: [Kaggle – WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- Format: CSV with `title`, `text`, and `label` columns (0 = fake, 1 = real)

<!-- ## Models
We compare three tiers of models to measure the performance gap between traditional and modern NLP approaches:

| Model | Type |
|---|---|
| TF-IDF + Logistic Regression | Baseline |
| XGBoost + engineered features | Intermediate |
| Fine-tuned DistilBERT | Advanced |

## Tech Stack
- **Data:** `pandas`, `scikit-learn`
- **NLP:** `nltk`, `spaCy`, HuggingFace `datasets`
- **Modeling:** `scikit-learn`, `xgboost`, `transformers`, `torch`
- **Evaluation:** `matplotlib`, `seaborn`
- **Compute:** Google Colab (T4 GPU)

## Project Structure
```
fake-news-detection/
├── data/               # Raw and processed datasets
├── notebooks/          # Exploration and model training notebooks
├── models/             # Saved model checkpoints
├── src/                # Reusable scripts (preprocessing, training, eval)
└── README.md
``` -->