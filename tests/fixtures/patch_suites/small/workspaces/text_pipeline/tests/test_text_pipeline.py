from src.normalize import normalize_text
from src.pipeline import prepare_for_index
from src.tokenize import tokenize


def test_normalize_text_lowercases_and_trims() -> None:
    assert normalize_text("  Hello World  ") == "hello world"


def test_tokenize_splits_on_whitespace() -> None:
    assert tokenize("  Hello   world  from Tok ") == ["hello", "world", "from", "tok"]


def test_prepare_for_index_removes_stopwords_and_dedupes_in_order() -> None:
    text = "The tok and tok system and tests"
    assert prepare_for_index(text) == ["tok", "system", "tests"]
