from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split

from sentiment_demo.config import Settings
from sentiment_demo.data import load_dataset, resolve_text_column
from sentiment_demo.model import build_pipeline


def train_and_evaluate(
    data_path: Path,
    output_dir: Path,
    test_size: float,
    random_state: int,
    max_features: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    settings = Settings()
    text_col = resolve_text_column(df, settings.text_column, settings.fallback_text_column)

    working_df = df[[text_col, settings.target_column]].dropna().copy()
    working_df[text_col] = working_df[text_col].astype(str)

    class_counts = working_df[settings.target_column].value_counts()
    stratify_target = working_df[settings.target_column] if class_counts.min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        working_df[text_col],
        working_df[settings.target_column],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    model = build_pipeline(max_features=max_features)

    if len(x_train) >= 30 and y_train.value_counts().min() >= 3:
        grid = GridSearchCV(
            estimator=model,
            param_grid={
                "classifier__C": [0.5, 1.0, 2.0],
            },
            scoring="f1_weighted",
            cv=3,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(x_train, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        training_mode = "grid_search"
    else:
        best_model = model.fit(x_train, y_train)
        best_params = {"classifier__C": 1.0}
        training_mode = "direct_fit"

    predictions = best_model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted")),
        "best_params": best_params,
        "training_mode": training_mode,
        "label_distribution": working_df[settings.target_column].value_counts().to_dict(),
        "dataset_rows": int(working_df.shape[0]),
        "text_column_used": text_col,
    }

    report = classification_report(y_test, predictions, zero_division=0)
    labels = sorted(working_df[settings.target_column].unique())
    confusion = confusion_matrix(y_test, predictions, labels=labels)

    model_path = output_dir / "model.joblib"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "classification_report.txt"
    confusion_path = output_dir / "confusion_matrix.csv"

    joblib.dump(best_model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")

    confusion_df = pd.DataFrame(confusion, index=labels, columns=labels)
    confusion_df.to_csv(confusion_path, index=True)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "confusion_path": str(confusion_path),
        "metrics": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train sentiment classification model")
    parser.add_argument("--data-path", type=Path, default=Settings().data_path)
    parser.add_argument("--output-dir", type=Path, default=Settings().output_dir)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=Settings().random_state)
    parser.add_argument("--max-features", type=int, default=20000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_and_evaluate(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
    )

    print("Training complete.")
    print(f"Model saved to: {result['model_path']}")
    print(f"Metrics saved to: {result['metrics_path']}")
    print(f"Report saved to: {result['report_path']}")
    print(f"Confusion matrix saved to: {result['confusion_path']}")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
