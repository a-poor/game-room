from collections import Counter

import numpy as np
import pytest
from pettingzoo.test import api_test, seed_test

from game_room import kingdomino_v0 as game


def position(domino=1, **kwargs):
    env = game.raw_env(**kwargs)
    env.reset(seed=1)
    env.phase = game.PLACE
    env.cursor = 0
    env.current[0] = domino
    env.current_owners[0] = 0
    env.agent_selection = "player_0"
    return env


def test_inventory():
    assert len(game.DOMINOES) == 48
    squares = Counter(t for d in game.DOMINOES for t, _ in d)
    crowns = Counter()
    for d in game.DOMINOES:
        for t, c in d:
            crowns[t] += c
    assert [squares[t] for t in range(1, 7)] == [26, 22, 18, 14, 10, 6]
    # Mines: one single, three doubles, one triple = ten crowns.
    assert [crowns[t] for t in range(1, 7)] == [5, 6, 6, 6, 6, 10]
    assert game.DOMINOES[18] == ((game.WHEAT, 1), (game.FOREST, 0))
    assert game.DOMINOES[47] == ((game.WHEAT, 0), (game.MINE, 3))


@pytest.mark.parametrize("players,count,kings", [(2, 24, 4), (3, 36, 3), (4, 48, 4)])
def test_setup_and_initial_draft(players, count, kings):
    env = game.raw_env(num_players=players)
    env.reset(seed=1)
    assert len(env.deck) + len(env.current) == count
    assert sorted(env.deck + env.current + env.removed) == list(range(1, 49))
    assert env.current == sorted(env.current)
    assert len(env.initial_order) == kings
    assert Counter(env.initial_order) == Counter(
        {p: 2 if players == 2 else 1 for p in range(players)}
    )
    order = env.initial_order[:]
    for i in range(kings):
        assert env.agent_selection == f"player_{order[i]}"
        env.step(kings - 1 - i)
    assert env.phase == game.PLACE
    assert env.agent_selection == f"player_{order[-1]}"
    assert env.next_row == sorted(env.next_row)
    assert all(owner == -1 for owner in env.next_owners)


def test_placement_orientation_and_draft_order():
    env = position(19)
    env.next_row = [1, 2, 3, 4]
    env.next_owners = [-1] * 4
    env.current_owners = [0, 1, 2, 3]
    env.step(game.placement_action(4, 5, 1))
    assert tuple(env.boards[0, 4, 5]) == (game.WHEAT, 1)
    assert tuple(env.boards[0, 5, 5]) == (game.FOREST, 0)
    assert env.phase == game.DRAFT and env.agent_selection == "player_0"
    env.step(3)
    assert env.next_owners == [-1, -1, -1, 0]
    assert env.agent_selection == "player_1" and env.phase == game.PLACE


def test_matching_connection_overlap_and_diagonal():
    env = position(3)  # forest/forest
    mask = env.observe("player_0")["action_mask"]
    assert mask[game.placement_action(4, 5, 0)]  # castle is wild
    assert not mask[game.placement_action(4, 4, 0)]  # occupied
    assert not mask[game.placement_action(5, 5, 0)]  # diagonal only
    assert not mask[game.placement_action(0, 0, 2)]  # off board
    env.boards[0, 4, 5] = (game.WHEAT, 0)
    mask = env.observe("player_0")["action_mask"]
    assert not mask[game.placement_action(4, 6, 0)]  # mismatch
    env.boards[0, 4, 5] = (game.FOREST, 0)
    assert env.observe("player_0")["action_mask"][game.placement_action(4, 6, 0)]


def test_only_one_half_must_match_existing_board():
    env = position(13)  # wheat/forest
    env.boards[0, 4, 5] = (game.FOREST, 0)
    # Forest second half touches existing forest; wheat touches nothing.
    assert env.observe("player_0")["action_mask"][game.placement_action(5, 6, 2)]


def test_five_by_five_bounds_allow_off_center_castle():
    env = position()
    env.boards[0, 4, 5:8, 0] = game.WHEAT
    mask = env.observe("player_0")["action_mask"]
    assert mask[game.placement_action(4, 8, 1)]  # width five, castle on edge
    assert not mask[game.placement_action(4, 3, 2)]  # expands width to six
    env.boards[0, 0:4, 4, 0] = game.WHEAT
    assert not env.observe("player_0")["action_mask"][game.placement_action(5, 4, 0)]


def test_discard_only_if_no_placement_and_invalid_action_atomic():
    env = position()
    before = env.observe("player_0")["observation"]
    for a in (game.DISCARD, -1, game.ACTION_COUNT, None, 0):
        with pytest.raises(ValueError):
            env.step(a)
        np.testing.assert_array_equal(before, env.observe("player_0")["observation"])
    env.boards[0, 2:7, 2:7, 0] = game.WHEAT
    env.boards[0, 4, 4, 0] = game.CASTLE
    assert np.flatnonzero(env.observe("player_0")["action_mask"]).tolist() == [
        game.DISCARD
    ]
    env.next_row = [5, 6, 7, 8]
    env.next_owners = [-1] * 4
    env.step(game.DISCARD)
    assert env.discarded == [1] and env.phase == game.DRAFT


def test_scoring_regions_crowns_and_castle_not_bridge():
    env = position()
    env.boards[0, 4, 2:4] = [(game.WHEAT, 1), (game.WHEAT, 2)]
    env.boards[0, 4, 5] = (game.WHEAT, 1)
    env.boards[0, 5, 5] = (game.FOREST, 0)
    env.boards[0, 5, 2] = (game.MINE, 3)
    assert env.score_details(0) == (10, 2, 7)


@pytest.mark.parametrize(
    "details,winners",
    [
        ([(10, 3, 2), (9, 5, 9), (8, 5, 9), (0, 0, 0)], [0]),
        ([(10, 3, 2), (10, 4, 1), (8, 5, 9), (0, 0, 0)], [1]),
        ([(10, 3, 2), (10, 3, 3), (8, 5, 9), (0, 0, 0)], [1]),
        ([(10, 3, 2), (10, 3, 2), (8, 5, 9), (0, 0, 0)], [0, 1]),
    ],
)
def test_tiebreaks(details, winners, monkeypatch):
    env = position()
    monkeypatch.setattr(env, "score_details", lambda p: details[p])
    env._finish()
    assert env.winners == [f"player_{p}" for p in winners]


def test_hidden_future_tiles_observation_copies_and_render():
    env = game.raw_env(render_mode="ansi")
    env.reset(seed=42)
    a = env.agent_selection
    before = env.observe(a)["observation"]
    rendered = env.render()
    env.deck.reverse()
    np.testing.assert_array_equal(before, env.observe(a)["observation"])
    assert rendered == env.render()
    before[:] = 99
    assert not np.all(env.observe(a)["observation"] == 99)
    for other in env.agents:
        if other != a:
            assert not env.observe(other)["action_mask"].any()
    assert env.infos == {a: {} for a in env.agents}


@pytest.mark.parametrize("players", [2, 3, 4])
def test_complete_games_conserve_tiles_and_rewards(players):
    for seed in range(5):
        env = game.raw_env(num_players=players, max_steps=None)
        env.reset(seed=seed)
        returns = dict.fromkeys(env.agents, 0)
        for a in env.agent_iter(200):
            obs, reward, terminated, truncated, _ = env.last()
            returns[a] += reward
            assert env.observation_space(a).contains(obs)
            tiles = (
                env.deck
                + env.removed
                + [t for t in env.current if t]
                + env.next_row
                + env.discarded
                + [t for hand in env.placed for t in hand]
            )
            assert Counter(tiles) == Counter(range(1, 49))
            for p in range(players):
                board = env.boards[p]
                occupied = np.argwhere(board[:, :, 0])
                assert np.all(occupied.max(axis=0) - occupied.min(axis=0) < 5)
                assert len(occupied) == 1 + 2 * len(env.placed[p])
                assert int(board[:, :, 1].sum()) == sum(
                    c for t in env.placed[p] for _, c in game.DOMINOES[t - 1]
                )
            env.step(
                None
                if terminated or truncated
                else env.action_space(a).sample(obs["action_mask"])
            )
        assert not env.agents
        assert env.winners
        assert env.steps == players * 24
        assert env.round_number == (6 if players == 2 else 12)
        assert all(returns[a] == (1 if a in env.winners else -1) for a in returns)
        assert sum(map(len, env.placed)) + len(env.discarded) == players * 12


def test_truncation():
    env = game.raw_env(max_steps=1)
    env.reset(seed=42)
    env.step(0)
    assert all(env.truncations.values())
    assert not any(env.terminations.values())
    assert not any(env.rewards.values())


@pytest.mark.parametrize("players", [2, 3, 4])
def test_api(players):
    api_test(game.env(num_players=players), num_cycles=200)


def test_seed():
    seed_test(game.env, num_cycles=200)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_players": 1},
        {"num_players": 5},
        {"num_players": True},
        {"num_players": 2.5},
        {"max_steps": 0},
        {"max_steps": True},
        {"render_mode": "rgb_array"},
    ],
)
def test_config(kwargs):
    with pytest.raises(ValueError):
        game.raw_env(**kwargs)


def test_action_helpers_and_render(capsys):
    for r, c, d in [(0, 0, 0), (8, 8, 3), (4, 6, 2)]:
        assert game.decode_placement(game.placement_action(r, c, d)) == (r, c, d)
    with pytest.raises(ValueError):
        game.placement_action(9, 0, 0)
    with pytest.raises(ValueError):
        game.decode_placement(0)
    env = position(render_mode="human")
    env.render()
    assert "Kingdomino" in capsys.readouterr().out
    env.close()
    assert position().render() is None


def test_next_round_follows_draft_order():
    env = game.raw_env()
    env.reset(seed=10)
    for slot in range(4):
        env.step(slot)
    old_owners = env.current_owners[:]
    claimed_row = env.next_row[:]
    for slot in range(4):
        assert env.agent_selection == f"player_{old_owners[slot]}"
        legal = np.flatnonzero(env.observe(env.agent_selection)["action_mask"])
        env.step(int(legal[0]))
        assert env.phase == game.DRAFT
        env.step(3 - slot)
    assert env.round_number == 2
    assert env.current == claimed_row
    assert env.current_owners == list(reversed(old_owners))
    assert env.agent_selection == f"player_{old_owners[-1]}"


def test_final_row_has_no_claims_and_completion_precedes_limit():
    env = game.raw_env(num_players=2, max_steps=48)
    env.reset(seed=20)
    final_actions = 0
    while not any(env.terminations.values()):
        a = env.agent_selection
        if env.round_number == 6:
            assert env.phase == game.PLACE
            assert not env.next_row
            final_actions += 1
        env.step(env.action_space(a).sample(env.observe(a)["action_mask"]))
    assert final_actions == 4
    assert env.steps == 48
    assert not any(env.truncations.values())
    assert env.observe(env.agent_selection)["observation"][8] == 0
    for a in env.agent_iter():
        assert not env.observe(a)["action_mask"].any()
        env.step(None)
    assert not env.agents
