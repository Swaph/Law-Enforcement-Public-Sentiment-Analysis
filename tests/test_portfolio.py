from pathlib import Path

from sentiment_demo.portfolio import build_admissions_answer, build_markdown_report, load_profile


def test_build_admissions_answer_mentions_links_and_metrics():
    profile = load_profile(Path("examples/admissions_profile.example.json"))
    metrics = {
        "accuracy": 0.978,
        "f1_weighted": 0.977,
        "dataset_rows": 5500,
        "training_mode": "grid_search",
        "text_column_used": "preprocessed_text",
    }

    answer = build_admissions_answer(profile, metrics)

    assert "Google Scholar" in answer
    assert "0.978" in answer
    assert len(answer.split()) <= 250


def test_build_markdown_report_includes_sections():
    profile = load_profile(Path("examples/admissions_profile.example.json"))
    metrics = {
        "accuracy": 0.978,
        "f1_weighted": 0.977,
        "dataset_rows": 5500,
        "training_mode": "grid_search",
        "text_column_used": "preprocessed_text",
    }

    report = build_markdown_report(profile, metrics)

    assert "## Publications" in report
    assert "## Entrepreneurship" in report
    assert "Verified Project Metrics" in report