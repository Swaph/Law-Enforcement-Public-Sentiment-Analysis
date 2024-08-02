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
```

### Running the Notebook

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Swaph/kenya-sentimentmonitor.git
    cd kenya-sentimentmonitor
    ```

2. **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

3. **Open the Notebook**:
    In the Jupyter Notebook interface, navigate to the `notebooks` directory and open `Law-Enforcement-Public-Sentiment_Analysis.ipynb`.

### Notebook Sections

1. **Data Loading**:
    - Load the dataset from the `data` directory.
    - Display basic statistics and data overview.

2. **Data Preprocessing**:
    - Clean the data by handling missing values and normalizing text.
    - Tokenize text data and remove stop words.
    - Apply stemming or lemmatization.

3. **Feature Engineering**:
    - Convert text data into numerical features using TF-IDF vectorization.
    - Select the most relevant features using chi-square feature selection.

4. **Handling Class Imbalance**:
    - Use SMOTE to generate synthetic samples for the minority class, balancing the dataset.

5. **Model Training**:
    - Split the data into training and testing sets.
    - Train a Logistic Regression model with hyperparameter tuning using GridSearchCV.

6. **Model Evaluation**:
    - Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1-score.
    - Display confusion matrix and classification report.

7. **Model Saving**:
    - Save the trained model using joblib for future use.

## Accessing Project Files

All project files, including datasets, models, and additional documentation, can be found in the following Google Drive folder:
[Google Drive Link](https://drive.google.com/drive/folders/1TgQhWarpPPWu6E6_4uXA1eOqEFnDxjGm?usp=sharing)

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.

## Acknowledgements

- Thanks to [ACLED](https://acleddata.com) for providing the dataset used in this project.
- Thanks to the open-source community for the libraries and tools used in this project.
```
