from pathlib import Path

import pandas as pd

from sentiment_demo.train import train_and_evaluate


def test_train_and_evaluate_smoke(tmp_path: Path):
    data = pd.DataFrame(
        {
            "preprocessed_text": [
                "police used force during protest",
                "community praised police rescue",
                "officers monitored peaceful event",
                "reported abduction by security team",
                "citizens thanked police support",
                "protesters clashed with police",
            ],
            "flagged_sentiment": [
                "negative",
                "positive",
                "neutral",
                "negative",
                "positive",
                "negative",
            ],
        }
    )

    csv_path = tmp_path / "sample.csv"
    out_dir = tmp_path / "artifacts"
    data.to_csv(csv_path, index=False)

    result = train_and_evaluate(
        data_path=csv_path,
        output_dir=out_dir,
        test_size=0.33,
        random_state=7,
        max_features=500,
    )

    assert Path(result["model_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    assert "accuracy" in result["metrics"]
