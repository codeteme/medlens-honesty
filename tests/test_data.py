import numpy as np
import pytest

from medlens.config import deep_merge, load_config
from medlens.data.loaders import normalize_medmcqa_row, normalize_medqa_row, question_overlap
from medlens.data.schema import MCQItem, format_mcq


def medqa_row(**overrides):
    row = {
        "question": "A 45-year-old man presents with chest pain. Next step?",
        # Parquet stores options as an array of {"key", "value"} dicts, possibly unsorted.
        "options": np.array(
            [
                {"key": "B", "value": "ECG"},
                {"key": "A", "value": "Aspirin"},
                {"key": "D", "value": "CT"},
                {"key": "C", "value": "Troponin"},
            ],
            dtype=object,
        ),
        "answer_idx": "B",
        "answer": "ECG",
        "meta_info": "step2&3",
    }
    row.update(overrides)
    return row


def test_medqa_options_sorted_and_answer_kept():
    item = normalize_medqa_row(medqa_row(), split="test")
    assert list(item.options) == ["A", "B", "C", "D"]
    assert item.options[item.answer] == "ECG"
    assert item.meta == {"split": "test", "step": "step2&3"}


def test_medqa_ids_are_stable_and_content_based():
    a = normalize_medqa_row(medqa_row(), "test")
    b = normalize_medqa_row(medqa_row(), "train")
    c = normalize_medqa_row(medqa_row(question="Different question?"), "test")
    assert a.id == b.id != c.id


def test_medmcqa_cop_maps_to_letter_and_nan_fields_become_none():
    row = {
        "id": "x1",
        "question": "Q?",
        "opa": "a",
        "opb": "b",
        "opc": "c",
        "opd": "d",
        "cop": 2,
        "choice_type": "single",
        "subject_name": "Anatomy",
        "topic_name": "nan",
    }
    item = normalize_medmcqa_row(row, "validation")
    assert item.answer == "C" and item.options["C"] == "c"
    assert item.meta["subject"] == "Anatomy" and item.meta["topic"] is None


def test_medmcqa_rejects_unanswered_rows():
    row = {"id": "x", "question": "Q", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": -1}
    with pytest.raises(ValueError):
        normalize_medmcqa_row(row, "test")


def test_mcqitem_validates_answer_and_keys():
    with pytest.raises(ValueError):
        MCQItem("i", "s", "q", {"A": "x", "B": "y"}, answer="C")
    with pytest.raises(ValueError):
        MCQItem("i", "s", "q", {"B": "x", "A": "y"}, answer="A")


def test_format_mcq():
    item = MCQItem("i", "s", " Which? ", {"A": "one", "B": "two"}, answer="A")
    assert format_mcq(item) == "Which?\n\nA. one\nB. two"


def test_question_overlap_ignores_case_and_whitespace():
    assert question_overlap(["What  is X?"], ["what is x?", "other"]) == {"what is x?"}


def test_debug_block_overrides_only_in_debug(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "n: 500\nmodel: {id: big, layers: [1, 2]}\ndebug:\n  n: 20\n  model: {id: small}\n"
    )
    full, dbg = load_config(p), load_config(p, debug=True)
    assert full["n"] == 500 and full["model"]["id"] == "big" and full["debug"] is False
    assert dbg["n"] == 20 and dbg["model"] == {"id": "small", "layers": [1, 2]}
    assert full["seed"] == 0


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}
