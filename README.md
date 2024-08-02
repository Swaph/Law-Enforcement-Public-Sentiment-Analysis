# Sentiment Analysis Notebook

This Jupyter notebook is part of the Kenya SentimentMonitor project, which aims to analyze public sentiment towards Kenya law enforcement activities. The notebook demonstrates the entire process from data preprocessing to model training, evaluation, and feature engineering.

## Notebook Overview

The notebook covers the following steps:

1. **Data Loading**: Importing the dataset `kenya_labeled_with_flagged_sentiment.csv`.
2. **Data Preprocessing**: Cleaning and preparing the data for analysis.
3. **Feature Engineering**: Applying TF-IDF vectorization and chi-square feature selection.
4. **Handling Class Imbalance**: Using SMOTE to balance the dataset.
5. **Model Training**: Training a Logistic Regression model with hyperparameter tuning using GridSearchCV.
6. **Model Evaluation**: Evaluating the model's performance using various metrics.
7. **Model Saving**: Saving the trained model using joblib.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Jupyter Notebook
- Required Python packages (listed in `requirements.txt`)

Install the required Python packages:

```bash
pip install -r requirements.txt
