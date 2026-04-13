from sentiment_demo.data import resolve_text_column


def test_resolve_text_column_prefers_preprocessed():
    class DummyDf:
        columns = ["preprocessed_text", "notes", "flagged_sentiment"]

    assert resolve_text_column(DummyDf, "preprocessed_text", "notes") == "preprocessed_text"


def test_resolve_text_column_uses_fallback():
    class DummyDf:
        columns = ["notes", "flagged_sentiment"]

    assert resolve_text_column(DummyDf, "preprocessed_text", "notes") == "notes"
