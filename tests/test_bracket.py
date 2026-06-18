"""Bracket logic: standings, best-thirds, same-group avoidance, resolution."""
from worldcup2026 import tournament as T
from worldcup2026.model import bracket as bk


def test_group_table_orders_by_points_then_gd():
    tbl = bk.GroupTable(["A", "B", "C", "D"])
    tbl.add("A", "B", 2, 0)   # A win
    tbl.add("C", "D", 1, 1)   # draw
    tbl.add("A", "C", 3, 0)   # A win big
    tbl.add("B", "D", 0, 0)   # draw
    tbl.add("A", "D", 1, 0)   # A win
    tbl.add("B", "C", 2, 1)   # B win
    order = tbl.standings(strength={})
    assert order[0] == "A"  # 9 pts, dominant
    assert tbl.stats("A")[0] == 9


def test_select_best_thirds_takes_top_n():
    stats = {f"T{i}": (i % 4, i, i) for i in range(12)}  # varied
    best = bk.select_best_thirds(stats, strength={}, n=8)
    assert len(best) == 8
    # highest points first
    assert stats[best[0]][0] >= stats[best[-1]][0]


def test_assign_thirds_avoids_same_group():
    third_slots = ["T1", "T2"]
    slot_groups = {"T1": "A", "T2": "B"}      # T1 faces winner of A, T2 winner of B
    thirds_ranked = ["x", "y"]
    team_group = {"x": "A", "y": "C"}          # x is from group A -> clashes with T1
    assign = bk.assign_thirds(third_slots, thirds_ranked, slot_groups, team_group)
    assert assign["T1"] != "x"                 # swapped to avoid A-vs-A
    assert set(assign.values()) == {"x", "y"}  # same teams, reordered


def test_resolve_tournament_produces_single_champion():
    cfg = T.bracket()
    # deterministic teams for every first-round slot token
    tokens = [s for m in cfg["round_of_32"] for s in (m["top"], m["bottom"])]
    slot_to_team = {tok: f"Team_{tok}" for tok in tokens}

    def play(a, b, ko):
        return (a, b, 1, 0)  # 'top' always wins

    res = bk.resolve_tournament(cfg, slot_to_team, play)
    assert res["champion"] is not None
    assert res["runner_up"] is not None
    assert res["champion"] != res["runner_up"]
    # round sizes
    assert len(res["rounds"]["round_of_32"]) == 16
    assert len(res["rounds"]["final"]) == 1
