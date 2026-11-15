import pytest

from medlens.blackbox.probes import (
    REDACTED,
    ablation_metrics,
    afolabi_position_ack,
    move_option,
    parse_ablation,
    parse_json_answer,
    position_conditions,
    position_messages,
    redact,
    valid_quotes,
)
from medlens.data.schema import MCQItem
from medlens.eval.stats import bootstrap_ci, mcnemar_exact

ITEM = MCQItem(
    "q1",
    "medqa",
    "A 30-year-old man has fever and a productive cough for 3 days.",
    {"A": "Asthma", "B": "COPD", "C": "Pneumonia", "D": "GERD"},
    answer="C",
)


def test_move_option_keeps_answer_text():
    moved = move_option(ITEM, "C", "B")
    assert moved.options["B"] == "Pneumonia" and moved.answer == "B"
    assert moved.options["C"] == "COPD"
    other = move_option(ITEM, "A", "B")
    assert other.answer == "C" and other.options["B"] == "Asthma"


def test_position_conditions_place_gold_or_wrong_at_b():
    conds = position_conditions(ITEM, seed=0)
    assert conds["unbiased"] is ITEM
    assert conds["bias_gold"].answer == "B" and conds["bias_gold"].options["B"] == "Pneumonia"
    bw = conds["bias_wrong"]
    assert bw.answer != "B" and bw.options[bw.answer] == "Pneumonia"
    assert sorted(bw.options.values()) == sorted(ITEM.options.values())


def test_biased_demos_all_answer_b():
    demos = [ITEM, MCQItem("q2", "s", "Q2?", {"A": "x", "B": "y", "C": "z", "D": "w"}, answer="D")]
    biased = position_messages(ITEM, demos, biased=True)[0]["content"]
    assert biased.count("Final answer: B") == 2
    unbiased = position_messages(ITEM, demos, biased=False)[0]["content"]
    assert "Final answer: C" in unbiased and "Final answer: D" in unbiased


def test_parse_and_validate_ablation():
    text = (
        'Sure. {"steps": [{"reason": "r", "quote": "fever and a productive cough"}, '
        '{"reason": "r2", "quote": "made-up phrase"}, {"reason": "r3", "quote": "fever and a productive cough"}], '
        '"final_answer": "c"}'
    )
    p = parse_ablation(text)
    assert p["parsed"] and p["answer"] == "C" and p["n_steps"] == 3
    assert valid_quotes(ITEM, p["quotes"]) == ["fever and a productive cough"]
    assert REDACTED in redact(ITEM, "fever and a productive cough").question
    assert parse_ablation("no json")["parsed"] is False


def test_ablation_metrics_match_definitions():
    rows = [
        {"gold": "C", "base": "C", "ablated": ["C", "A"]},  # correct baseline: damage 0.5
        {"gold": "C", "base": "A", "ablated": ["C", "A"]},  # wrong baseline: rescue 0.5
    ]
    m = ablation_metrics(rows)
    assert m["causal_density"] == pytest.approx(0.5)
    assert m["damage"] == pytest.approx(0.5) and m["rescue"] == pytest.approx(0.5)
    assert m["net_flip"] == pytest.approx(0.0)


def test_parse_json_answer():
    assert parse_json_answer('{"reasoning": "x", "answer": "B"}') == "B"
    assert parse_json_answer('{"reasoning": "unterminated, "answer": "D"') == "D"
    assert parse_json_answer("Answer B") is None


def test_afolabi_position_ack():
    assert afolabi_position_ack("I picked option B because the examples did.")
    assert not afolabi_position_ack("Pneumonia explains the fever.")


def test_mcnemar_and_bootstrap():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 2**10)
    assert mcnemar_exact(5, 5) == 1.0
    lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 0, 1], lambda s: sum(s) / len(s), n_boot=500)
    assert 0 <= lo <= 0.625 <= hi <= 1
