# Law Enforcement Public Sentiment Analysis

This project presents an NLP sentiment analysis pipeline designed to quantify public sentiment in complex linguistic environments across Kenyan civic discourse. It combines a research notebook workflow and a production-ready Python package so the same work can be inspected academically and deployed operationally.

## Research Framing

This work is positioned as a public-interest NLP study with three goals:

1. Measure sentiment dynamics in law-enforcement-related civic discourse at scale.
2. Improve model behavior on multilingual and code-switched expressions, including Sheng-coded markers.
3. Generate interpretable outputs that can support transparent public-service monitoring.

The research narrative targets analysis across 10,000+ civic discourse data points, while the current reproducible labeled release in this repository contains 5,500 records for model training and evaluation.

## Why This Is Hard

Standard sentiment tooling often underperforms in Kenyan urban language settings where English, Swahili, and Sheng are blended in a single utterance. The core technical challenge is to reduce false polarity assignments caused by:

- Non-standard vocabulary and slang drift.
- Context-dependent sentiment markers.
- Lexicon and model bias from tools tuned to Western corpora.

This project addresses those limitations through explicit preprocessing, feature engineering, and iterative model tuning in Scikit-learn, with notebook experiments that can be extended with NLTK-based linguistic normalization.

## Notebook Workflow (Matches `Law_Enforcement_Public_Sentiment_Analysis(3).ipynb`)

The notebook demonstrates a staged research workflow:

1. Filter the Africa-wide event dataset to Kenya and law-enforcement-relevant records.
2. Clean and preprocess `notes` text using regex normalization and NLTK lemmatization.
3. Run multilingual transformer sentiment inference on a subset (`nlptown/bert-base-multilingual-uncased-sentiment`) to estimate runtime and bootstrap labels.
4. Apply domain-aware rule refinement (including Sheng-relevant markers) to produce `flagged_sentiment`.
5. Explore sentiment distributions across classes, time, and geography.
6. Train TF-IDF + Logistic Regression baselines with class imbalance handling.
7. Tune hyperparameters and persist the trained model artifact.

The notebook emits intermediate files such as:

- `kenya_law_enforcement_data.csv`
- `preprocessed_kenya_acled_data.csv`
- `kenya_labeled_subset.csv`
- `kenya_labeled_data.csv`
- `kenya_labeled_with_flagged_sentiment.csv`
- `sentiment_analysis_model.joblib`

## Package Pipeline (Reproducible Demo Path)

The `src/sentiment_demo` package provides a reproducible execution path for demos and deployment:

1. Data loading and validation.
2. Text column resolution (`preprocessed_text` preferred, `notes` fallback).
3. TF-IDF feature extraction with uni- and bi-grams.
4. Balanced Logistic Regression training.
5. Hyperparameter search (or direct-fit fallback for very small samples).
6. Artifact export (`model.joblib`, metrics, confusion matrix, report).
7. FastAPI serving for live inference.

## Current Reproducible Results

From the current training run in this repository:

- Dataset rows: 5,500
- Accuracy: 0.9782
- Weighted F1: 0.9770
- Best `C`: 2.0 (grid search mode)

Artifacts are written to the `artifacts/` directory:

- `model.joblib`
- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.csv`

## SDG 16 Alignment

This work aligns with **UN Sustainable Development Goal 16 (Peace, Justice and Strong Institutions)** by enabling data-driven visibility into public service outcomes and civic trust signals. A robust sentiment pipeline can help teams:

- Detect perception shifts around institutional performance.
- Triangulate policy interventions with community response trends.
- Build more accountable and evidence-based monitoring frameworks.

## Repository Structure

```text
.
|-- kenya_labeled_with_flagged_sentiment.csv
|-- Law_Enforcement_Public_Sentiment_Analysis(3).ipynb
|-- src/
|   `-- sentiment_demo/
|       |-- __init__.py
|       |-- api.py
|       |-- config.py
|       |-- data.py
|       |-- model.py
|       |-- portfolio.py
|       `-- train.py
|-- tests/
|   |-- test_data.py
|   |-- test_portfolio.py
|   `-- test_train_smoke.py
|-- examples/
|   `-- admissions_profile.example.json
|-- artifacts/
|-- pyproject.toml
|-- requirements.txt
|-- requirements-dev.txt
`-- README.md
```

## Quick Start

### 1. Create and activate a virtual environment

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

### 3. Train and generate artifacts (package pipeline)

```powershell
sentiment-train --data-path kenya_labeled_with_flagged_sentiment.csv --output-dir artifacts
```

### 4. Run API demo

```powershell
sentiment-api
```

Health endpoint:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

Prediction example:

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/predict" -ContentType "application/json" -Body '{"text":"Police officers restored order after clashes in Nairobi."}'
```

### 5. Generate admissions portfolio narrative

```powershell
sentiment-portfolio --profile-path examples/admissions_profile.example.json --metrics-path artifacts/metrics.json --output-path artifacts/admissions_portfolio.md
```

This command creates a 250-word response scaffold that integrates technical evidence and live references for publications, posters, and entrepreneurship records.

## Testing

```powershell
pytest
```

## Notebook Positioning

Use the notebook as the research narrative and exploratory analysis layer:

- Document language-specific feature design for Sheng-coded sentiment markers.
- Compare transformer-assisted labeling and classical baseline behavior.
- Discuss bias risks and mitigation decisions.
- Connect findings to governance and SDG 16 outcomes.

Use the package under `src/sentiment_demo` as the production-style execution layer for repeatable demos, evaluations, and API deployment.

## Dataset Credit

- ACLED data context: https://acleddata.com
