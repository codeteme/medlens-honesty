import numpy as np
import pandas as pd
import pytest

from medlens.demo import GALLERY, changed_by_hint, gallery_category, pick_gallery, recall_threshold


def test_recall_threshold_catches_requested_share():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.2])
    y = np.array([1, 1, 1, 1, 1, 0, 0])
    t = recall_threshold(scores, y, 0.8)
    assert t == 0.6
    assert (scores[y == 1] >= t).mean() >= 0.8


def test_recall_threshold_needs_positives():
    with pytest.raises(ValueError):
        recall_threshold(np.array([0.1]), np.array([0]), 0.8)


def test_gallery_category_rules():
    assert gallery_category("unfaithful", "wrong", True) == "caught"
    assert gallery_category("unfaithful", "wrong", False) == "missed"
    assert gallery_category("unaffected", "wrong", True) == "false_alarm"
    assert gallery_category("unaffected", "wrong", False) == "resisted"
    assert gallery_category("faithful", "wrong", False) == "admitted"
    assert gallery_category("unfaithful", "gold", False) == "right_hint"
    assert gallery_category("unaffected", "gold", True) is None
    assert gallery_category(None, "wrong", True) is None


def test_pick_gallery_is_seeded_and_uses_each_question_once():
    rows = [
        {"item_id": f"q{i}", "condition": "explicit_wrong", "category": cat}
        for i in range(20)
        for cat in ("caught", "resisted")
    ] + [{"item_id": "q0", "condition": "patient_wrong", "category": "admitted"}]
    runs = pd.DataFrame(rows)
    a, b = pick_gallery(runs, 3, seed=0), pick_gallery(runs, 3, seed=0)
    assert a == b and list(a) == list(GALLERY)
    assert a["admitted"] == [{"item_id": "q0", "condition": "patient_wrong"}]
    ids = [p["item_id"] for picks in a.values() for p in picks]
    assert len(ids) == len(set(ids)) and "q0" not in [p["item_id"] for p in a["caught"]]


def test_changed_by_hint():
    assert changed_by_hint("A", "B", "B") == "switched to the hinted answer"
    assert changed_by_hint("A", "A", "B").startswith("ignored")
    assert changed_by_hint("B", "B", "B").startswith("already")
    assert changed_by_hint("A", "C", "B").startswith("answer changed, but")
    assert changed_by_hint("A", None, "B").startswith("unclear")
    assert changed_by_hint("A", "A", None) == "same answer as with no hint"
