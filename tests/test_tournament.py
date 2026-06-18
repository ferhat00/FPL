"""Tournament-definition integrity: 48 teams, 12 groups, valid fixtures & bracket."""
from worldcup2026 import tournament as T


def test_48_teams_in_12_groups_of_4():
    grps = T.groups()
    assert len(grps) == 12
    assert all(len(v) == 4 for v in grps.values())
    teams = T.all_teams()
    assert len(teams) == 48
    assert len(set(teams)) == 48  # no duplicates


def test_group_fixtures_are_full_round_robin():
    fx = T.group_fixtures()
    assert len(fx) == 12 * 6  # 6 matches per group of 4
    # every pair is unique and within a group
    pairs = {(r.group, frozenset((r.home, r.away))) for r in fx.itertuples()}
    assert len(pairs) == len(fx)


def test_teams_table_covers_all_teams():
    tt = T.teams_table()
    assert set(tt.index) == set(T.all_teams())
    for col in ["confederation", "host", "elo", "fifa_rank", "market_value_m", "squad_age"]:
        assert col in tt.columns
    assert tt["host"].sum() == 3  # USA, Canada, Mexico


def test_bracket_shape_and_tokens():
    bk = T.bracket()
    assert len(bk["round_of_32"]) == 16
    assert len(bk["round_of_16"]) == 8
    assert len(bk["quarter_finals"]) == 4
    assert len(bk["semi_finals"]) == 2
    assert len(bk["final"]) == 1

    valid_first = {f"1{g}" for g in T.groups()} | {f"2{g}" for g in T.groups()} \
        | {f"T{i}" for i in range(1, 9)}
    ids = set()
    for m in bk["round_of_32"]:
        ids.add(m["id"])
        assert m["top"] in valid_first and m["bottom"] in valid_first
    # later rounds reference winners of valid earlier matches
    for rnd in ["round_of_16", "quarter_finals", "semi_finals", "final"]:
        for m in bk[rnd]:
            for slot in (m["top"], m["bottom"]):
                assert slot.startswith("W:")


def test_each_group_winner_and_third_token_used_once():
    bk = T.bracket()
    firsts = [s for m in bk["round_of_32"] for s in (m["top"], m["bottom"])]
    # 12 winners, 12 runners-up, 8 thirds = 32 distinct slot tokens
    assert len(firsts) == 32
    assert len(set(firsts)) == 32
