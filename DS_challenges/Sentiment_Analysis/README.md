### Sentiment Analysis — Movie Review Classification
#### End-to-End NLP Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![NLTK](https://img.shields.io/badge/NLTK-3.9%2B-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Description** — this project demonstrates the complete NLP machine learning lifecycle from raw text to production-ready predictions.

---

### Table of Contents
1. [Project Overview](#1-project-overview)
2. [Why This Project](#2-why-this-project)
3. [Dataset](#3-dataset)
4. [Project Structure](#4-project-structure)
5. [Setup & Installation](#5-setup--installation)
6. [How to Run](#6-how-to-run)
7. [Methodology](#7-methodology)
8. [Model Results](#8-model-results)
9. [Key Findings](#9-key-findings)
10. [Interview Talking Points](#10-interview-talking-points)
11. [Future Improvements](#11-future-improvements)
12. [References](#12-references)

---

#### 1. Project Overview

**Task:** Binary sentiment classification — predict whether a movie review is **Positive** or **Negative**.

**Problem type:** Supervised binary text classification

**Tech stack:** Python · scikit-learn · NLTK · TF-IDF · Logistic Regression · Naive Bayes · SVM · Random Forest

**What it produces:**
- A trained model that classifies any text review in milliseconds
- Confidence scores alongside each prediction
- Interpretable feature weights showing *which words* drive predictions

---

#### 2. The Project

Sentiment analysis is one of the most commercially valuable NLP applications:

| Industry | Use Case |
|---|---|
| E-commerce | Automatically tag product reviews; flag negative feedback for customer service |
| Finance | Classify news headlines as positive/negative for trading signals |
| Healthcare | Analyse patient feedback to identify service quality issues |
| SaaS / Tech | Monitor app store reviews at scale without manual reading |
| Marketing | Brand monitoring — track sentiment on social media mentions |



---

#### 3. Dataset

### Description
A **movie review dataset** with binary labels: Positive (1) / Negative (0).

The notebook uses a rich synthetic dataset that mirrors the widely-used [IMDb 50K Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). To use real IMDb data, simply replace the data-generation block with:

```python
df = pd.read_csv('IMDB Dataset.csv')
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
```

### Dataset Statistics

| Property | Value |
|---|---|
| Total reviews | 2,000 |
| Positive reviews | 1,000 (50%) |
| Negative reviews | 1,000 (50%) |
| Average words per review | ~30 words |
| Class balance | Perfectly balanced |
| Train / Test split | 80% / 20% (stratified) |

### Sample Reviews

**Positive:**
> *"This film was absolutely brilliant! The acting was superb and the storyline kept me engaged throughout."*

**Negative:**
> *"What a terrible disappointment. The plot made no sense and the acting was wooden throughout."*

---
