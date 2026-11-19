import pytest

from abstaingate.data.pairs import ABSTAIN, build
from abstaingate.evaluation import contamination, selective


def _pool(rows_per_run, router="open_loop"):
    ids = sorted({i for run in rows_per_run for i, *_ in run})
    runs, recs = [], []
    for run in rows_per_run:
        d = {i: (c, a) for i, c, a in run}
        runs.append([{"correct": d[i][0]} for i in ids])
        recs.append([{"answer": d[i][1], "retrieved": ["c1"]} for i in ids])
    return {"meta": {"config": {"router": router}, "item_ids": ids}, "runs": runs, "records": recs}


ITEMS = [{"id": "a1", "question": "q1", "answerable": True, "gold_docs": ["d1"]},
         {"id": "a2", "question": "q2", "answerable": True, "gold_docs": ["d2"]},
         {"id": "u1", "question": "q3", "answerable": False}]


def _prompt(item, retrieved):
    return [{"role": "user", "content": item["question"]}]


def test_pair_types_follow_the_rules():
    pool = _pool([[("a1", 1.0, "right"), ("a2", 0.0, "wrong2"), ("u1", 0.0, "made up")],
                  [("a1", 0.0, "nope"), ("a2", 0.0, ABSTAIN), ("u1", 1.0, ABSTAIN)]])
    b = build(pool, ITEMS, _prompt)
    by = {(p["item"], p["type"]): p for p in b.pairs}
    assert by[("a1", "B")]["chosen"] == "right" and by[("a1", "B")]["rejected"] == ABSTAIN
    assert by[("a2", "A")]["chosen"] == ABSTAIN and by[("a2", "A")]["rejected"] == "wrong2"
    assert by[("u1", "A")]["rejected"] == "made up"
    sft = {s["item"]: s["completion"] for s in b.sft}
    assert sft == {"a1": "right", "a2": ABSTAIN, "u1": ABSTAIN}
    assert b.counts["pair_A"] == 2 and b.counts["pair_B"] == 1


def test_builder_refuses_test_items_and_gated_pools():
    pool = _pool([[("a1", 1.0, "right"), ("a2", 0.0, "x"), ("u1", 1.0, ABSTAIN)]])
    with pytest.raises(ValueError, match="test-split"):
        build(pool, ITEMS, _prompt, test_ids={"a2"})
    with pytest.raises(ValueError, match="open-loop"):
        build(_pool([[("a1", 1.0, "r"), ("a2", 1.0, "r"), ("u1", 1.0, ABSTAIN)]], router="cost"), ITEMS, _prompt)


def _rows(spec, answerable=True):
    return [{"item": f"i{k}", "cluster": f"d{k // 2}", "answered": a, "correct": c, "conf": f, "answerable": answerable}
            for k, (a, c, f) in enumerate(spec)]


def test_standard_aurc_ranks_answered_items():
    good = _rows([(True, 1.0, 0.9), (True, 1.0, 0.8), (True, 0.0, 0.1), (False, 0.0, 0.0)])
    bad = _rows([(True, 0.0, 0.9), (True, 1.0, 0.8), (True, 1.0, 0.1), (False, 0.0, 0.0)])
    assert selective.aurc(good) < selective.aurc(bad)
    assert selective.aurc(good) == pytest.approx((0 + 0 + 1 / 3) / 3)


def test_gate_threshold_does_not_change_aurc_of_the_ranking():
    """The confound the review caught: a gate must not win or lose AURC by where its threshold sits."""
    import numpy as np
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 200)
    rows = _rows([(True, float(rng.uniform() < c), c) for c in conf])
    a, b = selective.gate(rows, 0.2), selective.gate(rows, 0.6)
    assert selective.coverage(a) > selective.coverage(b)
    # a gate is a cut of the SAME ranking: gating at the threshold for coverage c gives exactly the
    # selective accuracy of the ungated ranking cut at c, so the threshold carries no extra information
    for c in (0.3, 0.5, 0.7):
        thr = selective.threshold_for_coverage(rows, c)
        assert selective.selective_accuracy(selective.gate(rows, thr)) == pytest.approx(
            selective.selective_accuracy_at(rows, selective.coverage(selective.gate(rows, thr))))
    # tightening the gate raises selective accuracy, which is why arms are compared at matched coverage
    assert selective.selective_accuracy(b) > selective.selective_accuracy(a)


def test_threshold_for_coverage_matches_target_on_dev():
    rows = _rows([(True, 1.0, c / 100) for c in range(100)])
    thr = selective.threshold_for_coverage(rows, 0.3)
    assert selective.coverage(selective.gate(rows, thr)) == pytest.approx(0.3, abs=0.011)
    assert selective.threshold_for_coverage(rows[:10] + _rows([(False, 0.0, 0.0)] * 90), 0.5) == float("-inf")


def test_gated_out_items_score_like_abstentions():
    rows = _rows([(True, 0.0, 0.1)], answerable=False) + _rows([(True, 1.0, 0.1)])
    g = selective.gate(rows, 0.5)
    assert [r["answered"] for r in g] == [False, False] and [r["correct"] for r in g] == [1.0, 0.0]


def test_cost_per_question():
    rows = _rows([(True, 1.0, 0.9), (True, 0.0, 0.8), (False, 0.0, 0.0), (False, 0.0, 0.0)])
    assert selective.cost_per_question(rows, 4.0, 1.0) == pytest.approx((0 + 4 + 1 + 1) / 4)


def test_paired_bootstrap_identical_arms_give_zero():
    rows = _rows([(True, 1.0, 0.9), (True, 0.0, 0.5), (False, 0.0, 0.0), (True, 1.0, 0.7)] * 5)
    out = selective.paired_cluster_bootstrap(rows, rows, n_boot=200)
    assert out["diff"] == 0 and out["ci"] == [0.0, 0.0]


def test_contamination_flags_copies():
    rep = contamination.report(["how do i set the max payload size in the nats server config file today"],
                               ["how do i set the max payload size in the nats server config file today",
                                "completely different question about retention policies for streams"])
    assert rep["exact_duplicates"] == 1 and rep["ngram_flagged"] == 1 and rep["ngram_max"] == 1.0


def test_analysis_on_synthetic_pools(tmp_path):
    import json

    import numpy as np

    from abstaingate.evaluation.analysis import analyse, render
    rng = np.random.default_rng(1)

    def pool(n, p_answer, p_correct_shift):
        ids = [f"q{i}" for i in range(n)]
        runs = []
        for _ in range(2):
            run = []
            for _i in range(n):
                sup = float(rng.uniform())
                ans = rng.uniform() < p_answer
                run.append({"abstained": 0.0 if ans else 1.0, "support": sup, "answerable": 1.0,
                            "correct": float(ans and rng.uniform() < min(1, sup + p_correct_shift))})
            runs.append(run)
        return {"meta": {"item_ids": ids, "item_clusters": [f"d{i // 4}" for i in range(n)]}, "runs": runs}

    (tmp_path / "dev_base_openloop.json").write_text(json.dumps(pool(40, 0.95, 0.0)))
    for s in range(2):
        (tmp_path / f"dev_sft_dpo_openloop_seed{s}.json").write_text(json.dumps(pool(40, 0.6, 0.2)))
        for name, pa, sh in (("test_base", 0.95, 0.0), ("test_sft_dpo", 0.6, 0.2), ("test_base_gate", 0.3, 0.1),
                             ("test_sft_dpo_gate", 0.3, 0.3)):
            (tmp_path / f"{name}_seed{s}.json").write_text(json.dumps(pool(40, pa, sh)))
    res = analyse(tmp_path, n_boot=100)
    assert "base_gate_matched" in res["table"]
    assert res["table"]["base_gate_matched"]["aurc"] is None          # a gated arm has no ranking of its own
    assert abs(res["table"]["base_gate_matched"]["coverage"] - res["table"]["sft_dpo"]["coverage"]) < 0.15
    assert res["verdict"]["summary"] in ("replace", "complement", "replace and complement", "neither")
    assert "Verdict" in render(res)


def test_merged_model_keeps_the_source_config(tmp_path):
    import json

    from abstaingate.models.train import keep_source_config
    src = tmp_path / "base"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "qwen2", "rope_theta": 1000000.0}))
    out = tmp_path / "merged"
    out.mkdir()
    (out / "config.json").write_text(json.dumps({"model_type": "qwen2", "rope_parameters": {"rope_theta": 1e6}}))
    (src / "tokenizer_config.json").write_text(json.dumps({"extra_special_tokens": {}, "chat_template": "x"}))
    (out / "tokenizer_config.json").write_text(json.dumps({"extra_special_tokens": []}))
    (out / "chat_template.jinja").write_text("x")
    keep_source_config(str(src), out)
    assert json.loads((out / "config.json").read_text())["rope_theta"] == 1000000.0
    assert json.loads((out / "tokenizer_config.json").read_text())["extra_special_tokens"] == {}
    assert not (out / "chat_template.jinja").exists()


def test_judge_labels_and_several_pairs_per_item():
    pool = _pool([[("a1", 0.0, "the port is 4222"), ("a2", 0.0, "wrong one"), ("u1", 0.0, "made up")],
                  [("a1", 0.0, "it listens on 4222 by default"), ("a2", 0.0, "wrong two"), ("u1", 0.0, "invented")]])
    for run in pool["runs"]:                      # the judge says a1's answers are right despite the token scorer
        run[0]["correct_judge"] = 1.0
        run[1]["correct_judge"] = 0.0
        run[2]["correct_judge"] = 0.0
    token = build(pool, ITEMS, _prompt)
    judge = build(pool, ITEMS, _prompt, label="correct_judge", max_pairs_per_item=3)
    assert ("a1", "A") in {(p["item"], p["type"]) for p in token.pairs}      # token labels: a wrong "abstain" pair
    assert ("a1", "A") not in {(p["item"], p["type"]) for p in judge.pairs}  # judge labels fix it
    assert judge.counts["pair_B"] == 2 and judge.counts["pair_A"] == 4      # two answers per item used


def test_consistency_scores():
    pool = {"meta": {"item_ids": ["q1", "q2"]},
            "records": [[{"answer": "port 4222"}, {"answer": "yes"}], [{"answer": "port 4222 default"}, {"answer": "no idea at all"}],
                        [{"answer": "4222 port"}, {"answer": "maybe so"}]]}
    c = selective.consistency_scores(pool)
    assert c["q1"] == 1.0 and c["q2"] < 0.5


def test_citation_only_answers_are_never_correct_targets():
    pool = _pool([[("a1", 0.0, "[1]"), ("a2", 0.0, "wrong"), ("u1", 0.0, "made up")],
                  [("a1", 0.0, "[1] it listens on 4222"), ("a2", 0.0, "wrong"), ("u1", 0.0, "invented")]])
    for run in pool["runs"]:                      # a lenient judge credits the bare citation too
        run[0]["correct_judge"] = 1.0
    built = build(pool, ITEMS, _prompt, label="correct_judge", max_pairs_per_item=3)
    targets = [r["completion"] for r in built.sft if r["item"] == "a1"]
    assert targets == ["[1] it listens on 4222"]
    assert all(p["chosen"] != "[1]" for p in built.pairs)


def test_common_coverage_when_the_trained_arm_answers_more(tmp_path):
    """The gate can only remove answers: if SFT -> DPO answers more than base, SFT -> DPO is cut down to
    base's coverage instead of comparing two ungated arms."""
    import json

    import numpy as np

    from abstaingate.evaluation.analysis import analyse
    rng = np.random.default_rng(2)

    def pool(n, p_answer):
        runs = [[{"abstained": 0.0 if rng.uniform() < p_answer else 1.0, "support": float(rng.uniform()),
                  "answerable": 1.0, "correct": float(rng.uniform() < 0.6)} for _ in range(n)] for _ in range(2)]
        return {"meta": {"item_ids": [f"q{i}" for i in range(n)], "item_clusters": [f"d{i // 4}" for i in range(n)]},
                "runs": runs}

    (tmp_path / "dev_base_openloop.json").write_text(json.dumps(pool(80, 0.5)))
    (tmp_path / "dev_sft_dpo_openloop_seed0.json").write_text(json.dumps(pool(80, 0.9)))
    for name, pa in (("test_base", 0.5), ("test_sft_dpo", 0.9), ("test_base_gate", 0.3), ("test_sft_dpo_gate", 0.3)):
        (tmp_path / f"{name}_seed0.json").write_text(json.dumps(pool(80, pa)))
    res = analyse(tmp_path, n_boot=50)
    thr = res["thresholds"][0]
    assert thr["support_threshold"] == float("-inf")           # base is not gated: it already answers less
    assert thr["sft_dpo_support_threshold"] > float("-inf")    # the trained arm is cut down instead
    t = res["table"]
    assert abs(t["sft_dpo_matched"]["coverage"] - t["base_gate_matched"]["coverage"]) < 0.12
    assert t["sft_dpo_matched"]["coverage"] < t["sft_dpo"]["coverage"] - 0.2


def test_cut_for_coverage_splits_ties():
    """With many tied scores a plain threshold drops every tie; the cut keeps a share of them."""
    rows = [{"item": f"q{i}", "answered": True, "conf": 0.0 if i < 70 else 0.9, "correct": 1.0, "answerable": True}
            for i in range(100)]
    thr, q = selective.cut_for_coverage(rows, 0.5)
    assert thr == 0.0 and abs(q - 20 / 70) < 1e-9
    cov = selective.coverage(selective.gate(rows, thr, q))
    assert abs(cov - 0.5) < 0.08
    plain = selective.coverage(selective.gate(rows, selective.threshold_for_coverage(rows, 0.5)))
    assert plain == 0.3                             # the old cut: all 70 ties dropped
