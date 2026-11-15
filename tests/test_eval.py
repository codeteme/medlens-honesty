import pytest

from medlens.data.schema import MCQItem
from medlens.eval.answers import cot_messages, extract_answer
from medlens.eval.stats import wilson_interval


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Reasoning...\nFinal answer: B", "B"),
        ("final answer: (c)", "C"),
        ("**Final Answer:** **D**", "D"),
        ("**Final answer**: B.", "B"),
        ("Maybe Final answer: A ... actually Final answer: C", "C"),
        ("B", "B"),
        ("(A)", "A"),
        ("I think it is B because...", None),
        ("Final answer: E", None),  # not a valid option for a 4-option question
        ("", None),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_cot_messages_contains_options_and_instruction():
    item = MCQItem("i", "s", "Q?", {"A": "x", "B": "y"}, answer="A")
    content = cot_messages(item)[0]["content"]
    assert "A. x" in content and "Final answer: X" in content


def test_wilson_interval():
    lo, hi = wilson_interval(8, 10)
    assert 0.44 < lo < 0.50 and 0.94 < hi < 0.97
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and hi < 0.2


def test_cohens_kappa():
    from medlens.eval.stats import cohens_kappa

    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0.0
    assert cohens_kappa([1, 1], [1, 1]) is None


def test_auroc():
    from medlens.eval.stats import auroc

    assert auroc([3, 4], [1, 2]) == 1.0
    assert auroc([1, 2], [3, 4]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    assert auroc([2, 0], [1]) == 0.5
    assert auroc([], [1]) is None
