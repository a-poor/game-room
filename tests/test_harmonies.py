from collections import Counter

import numpy as np
import pytest
from pettingzoo.test import api_test, seed_test

from game_room import harmonies_v0 as game


def position(**kwargs):
    e = game.raw_env(**kwargs)
    e.reset(seed=42)
    e.first_player = e.turn_player = 0
    e.agent_selection = "player_0"
    return e


def c(q, r):
    return game.CELL_INDEX[q, r]


def mask(e):
    return e.observe(e.agent_selection)["action_mask"]


def test_inventory_and_geometry():
    e = position()
    assert len(game.CELLS) == len(set(game.CELLS)) == 23
    assert [sum(q == col for q, _ in game.CELLS) for col in range(5)] == [5, 4, 5, 4, 5]
    assert len(game.NEIGHBORS[c(2, 1)]) == 6
    assert all(
        i in game.NEIGHBORS[j] for i, ns in enumerate(game.NEIGHBORS) for j in ns
    )
    assert len(game.ANIMALS) == 42
    assert sum(card.spirit for card in game.ANIMALS) == 10
    assert all(2 <= len(card.pattern) <= 4 for card in game.ANIMALS)
    tokens = e.bag + [t for group in e.market for t in group]
    assert Counter(tokens) == Counter(dict(enumerate((23, 23, 21, 19, 19, 15), 1)))
    assert sorted(e.deck + e.cards) == list(range(32))
    assert all(len(group) == 3 for group in e.market)
    assert game.ANIMALS[22].name == "Fennec Fox"
    assert game.ANIMALS[22].points == (4, 9, 16)


@pytest.mark.parametrize(
    "base,color,allowed",
    [
        ((), game.BLUE, True),
        ((), game.GREEN, True),
        ((game.GRAY,), game.GRAY, True),
        ((game.GRAY, game.GRAY), game.GRAY, True),
        ((game.GRAY,) * 3, game.GRAY, False),
        ((game.BROWN,), game.BROWN, True),
        ((game.BROWN,) * 2, game.BROWN, False),
        ((game.BROWN,), game.GREEN, True),
        ((game.BROWN,) * 2, game.GREEN, True),
        ((game.GREEN,), game.BROWN, False),
        ((game.RED,), game.RED, True),
        ((game.BROWN,), game.RED, True),
        ((game.GRAY,), game.RED, True),
        ((game.GRAY,) * 2, game.RED, False),
        ((game.RED,) * 2, game.RED, False),
        ((game.BLUE,), game.BLUE, False),
        ((game.YELLOW,), game.YELLOW, False),
        ((game.GREEN,), game.GREEN, False),
    ],
)
def test_stack_rules(base, color, allowed):
    e = position()
    e.boards[0][0] = base
    assert e.can_place(0, color, 0) == allowed
    e.cubes[0][0] = 1
    assert not e.can_place(0, color, 0)


def test_interleaving_draft_tokens_and_cubes():
    e = position()
    e.cards[0] = 25  # Meerkat: field next to one-high mountain, cube on mountain.
    e.market[0] = [game.YELLOW, game.GRAY, game.GREEN]
    assert not mask(e)[game.END_TURN]
    e.step(game.GROUP_OFFSET)
    assert not mask(e)[game.GROUP_OFFSET]
    e.step(game.place_action(game.YELLOW, c(2, 0)))
    e.step(game.DRAFT_OFFSET)
    assert e.owners[25] == 0 and e.drafted
    assert not mask(e)[game.DRAFT_OFFSET + 1]
    e.step(game.place_action(game.GRAY, c(2, 1)))
    e.step(game.settle_action(25, c(2, 1)))
    assert e.progress[25] == 1
    assert e.cubes[0][c(2, 1)] == 26
    assert not e.can_place(0, game.GRAY, c(2, 1))
    assert not mask(e)[game.END_TURN]
    e.step(game.place_action(game.GREEN, c(0, 0)))
    assert mask(e)[game.END_TURN]
    e.step(game.END_TURN)
    assert e.turn_player == 1 and not e.taken and not e.drafted
    assert len(e.market[0]) == 3 and e.cards[0] >= 0


def test_animals_survive_changes_to_other_habitat_cells():
    e = position()
    e.owners[25] = 0  # Field + mountain, cube mountain.
    e.boards[0][c(2, 0)] = (game.YELLOW,)
    e.boards[0][c(2, 1)] = (game.GRAY,)
    e.step(game.settle_action(25, c(2, 1)))
    # A bat uses a three-high tree and one-high mountain; cube stays when tree
    # is part of another habitat. More directly, fennec support mountain may grow.
    e = position()
    e.owners[22] = 0
    e.boards[0][c(0, 0)] = (game.YELLOW,)
    e.boards[0][c(0, 1)] = (game.GRAY,)
    e.boards[0][c(0, 2)] = (game.GRAY,)
    e.step(game.settle_action(22, c(0, 2)))
    e.taken = True
    e.remaining = Counter({game.GRAY: 1})
    e.step(game.place_action(game.GRAY, c(0, 1)))
    assert e.cubes[0][c(0, 2)] == 23 and e.progress[22] == 1
    assert e.score_details(0)["animals"] == 4


@pytest.mark.parametrize("card", range(42))
def test_every_card_pattern_rotates_and_requires_exact_heights(card):
    # Materialize the component's axial offsets in each of six orientations.
    # This checks anchors, edge handling, building wildcards, and cube locking.
    e = position(nature_spirits=False)
    e.owners[card] = 0
    offsets = game.ANIMALS[card].pattern
    for rotation in range(6):
        e.boards[0] = [() for _ in game.CELLS]
        e.cubes[0] = [0] * 23
        e.progress[card] = 0
        transformed = []
        for q, r, stack in offsets:
            for _ in range(rotation):
                q, r = -r, q + r
            transformed.append((c(2 + q, 1 + r), stack))
        for cell, stack in transformed:
            e.boards[0][cell] = (
                (game.BROWN, game.RED) if stack == (0, game.RED) else stack
            )
        anchor = c(2, 1)
        assert e.can_settle(0, card, anchor)
        assert mask(e)[game.settle_action(card, anchor)]
        e.step(game.settle_action(card, anchor))
        assert e.progress[card] == 1
        assert not e.can_settle(0, card, anchor)
        # Removing a required supporting stack invalidates that habitat.
        e.cubes[0][anchor] = 0
        e.progress[card] = 0
        e.boards[0][transformed[0][0]] = ()
        assert not e.can_settle(0, card, anchor)


def test_buildings_need_two_tokens_but_accept_any_allowed_base():
    e = position()
    e.owners[5] = 0  # Duck on water next to building.
    e.boards[0][c(2, 1)] = (game.BLUE,)
    for base in (game.BROWN, game.GRAY, game.RED):
        e.boards[0][c(2, 0)] = (base, game.RED)
        assert e.can_settle(0, 5, c(2, 1))
    e.boards[0][c(2, 0)] = (game.RED,)
    assert not e.can_settle(0, 5, c(2, 1))


def test_mountain_and_tree_habitat_heights_exact():
    e = position()
    e.owners[2] = 0  # Hippo on water beside a height-three mountain.
    e.boards[0][c(2, 1)] = (game.BLUE,)
    e.boards[0][c(2, 0)] = (game.GRAY, game.GRAY)
    assert not e.can_settle(0, 2, c(2, 1))
    e.boards[0][c(2, 0)] += (game.GRAY,)
    assert e.can_settle(0, 2, c(2, 1))
    e.owners[4] = 0  # Frog needs a height-one tree.
    e.boards[0][c(2, 0)] = (game.BROWN, game.GREEN)
    assert not e.can_settle(0, 4, c(2, 1))
    e.boards[0][c(2, 0)] = (game.GREEN,)
    assert e.can_settle(0, 4, c(2, 1))


def test_capacity_completion_and_partial_scoring():
    e = position()
    for card in (4, 6, 8, 10):
        e.owners[card] = 0
    assert not e._capacity(0)
    assert not mask(e)[game.DRAFT_OFFSET : game.PLACE_OFFSET].any()
    e.progress[4] = len(game.ANIMALS[4].points)
    assert e._capacity(0)
    assert mask(e)[game.DRAFT_OFFSET : game.PLACE_OFFSET].any()
    e.progress[6] = 2
    assert e.score_details(0)["animals"] == 15 + 10
    assert not any(e.can_settle(0, 4, cell) for cell in range(23))


def test_tree_and_mountain_scores():
    e = position()
    for i, stack in enumerate(
        (
            (game.GREEN,),
            (game.BROWN, game.GREEN),
            (game.BROWN, game.BROWN, game.GREEN),
            (game.BROWN,),
        )
    ):
        e.boards[0][i] = stack
    assert e.score_details(0)["trees"] == 11
    e.boards[0] = [() for _ in game.CELLS]
    e.boards[0][c(2, 0)] = (game.GRAY,)
    e.boards[0][c(2, 1)] = (game.GRAY,) * 2
    e.boards[0][c(2, 2)] = (game.GRAY,) * 3
    e.boards[0][c(0, 4)] = (game.GRAY,) * 3
    assert e.score_details(0)["mountains"] == 11  # Isolated mountain scores zero.


def test_fields_score_per_group_not_per_pair():
    e = position()
    for coord in ((0, 0), (0, 1), (0, 2), (0, 3), (4, -2), (4, -1), (2, 3)):
        e.boards[0][c(*coord)] = (game.YELLOW,)
    assert e.score_details(0)["fields"] == 10


def test_building_score_uses_distinct_top_colors():
    e = position()
    center = c(2, 1)
    e.boards[0][center] = (game.BROWN, game.RED)
    neighbors = game.NEIGHBORS[center]
    for i, stack in enumerate(((game.GRAY, game.GRAY), (game.GRAY,), (game.BLUE,))):
        e.boards[0][neighbors[i]] = stack
    assert e.score_details(0)["buildings"] == 0
    e.boards[0][neighbors[3]] = (game.RED,)
    assert e.score_details(0)["buildings"] == 5
    e.boards[0][neighbors[3]] = (game.BROWN, game.GREEN)
    assert e.score_details(0)["buildings"] == 5


def test_rivers_use_shortest_paths_not_loops_or_all_tokens():
    e = position()
    center = c(2, 1)
    for i in game.NEIGHBORS[center]:
        e.boards[0][i] = (game.BLUE,)
    assert e.score_details(0)["water"] == 8  # Six-cell ring, diameter four cells.
    e.boards[0][center] = (game.BLUE,)
    assert e.score_details(0)["water"] == 5  # Shortcut reduces diameter to three.
    e.boards[0] = [(game.BLUE,)] * 23
    assert (
        e.score_details(0)["water"] == 19
    )  # Seven-cell shortest path corner to corner.


def test_islands_include_empty_cells():
    e = position(board_side="B")
    assert e.score_details(0)["water"] == 5
    for r in range(-1, 4):
        e.boards[0][c(2, r)] = (game.BLUE,)
    assert e.score_details(0)["water"] == 10
    e.boards[0][c(0, 0)] = (game.GREEN,)
    assert e.score_details(0)["water"] == 10


@pytest.mark.parametrize(
    "spirit,expected",
    [
        (32, 14),
        (33, 15),
        (34, 8),
        (35, 7),
        (36, 8),
        (37, 6),
        (38, 8),
        (39, 7),
        (40, 7),
        (41, 6),
    ],
)
def test_all_spirit_scoring(spirit, expected):
    e = position()
    e.spirit_choices[0] = spirit
    e.owners[spirit] = 0
    e.progress[spirit] = 1
    if spirit in (32, 33):
        for coord in ((0, 0), (0, 1), (0, 2), (4, -2), (4, -1), (2, 3)):
            e.boards[0][c(*coord)] = (game.YELLOW,)
    elif spirit in (34, 35):
        e.boards[0][:3] = [
            (game.GREEN,),
            (game.BROWN, game.GREEN),
            (game.BROWN, game.BROWN, game.GREEN),
        ]
    elif spirit in (36, 37):
        for coord in ((0, 0), (0, 1), (4, 2)):
            e.boards[0][c(*coord)] = (game.RED, game.RED)
    elif spirit in (38, 39):
        e.boards[0][:3] = [(game.GRAY,), (game.GRAY,) * 2, (game.GRAY,) * 3]
    else:
        for coord in ((0, 0), (0, 1), (4, 2)):
            e.boards[0][c(*coord)] = (game.BLUE,)
    assert e.score_details(0)["spirits"] == expected
    e.progress[spirit] = 0
    assert e.score_details(0)["spirits"] == 0


def test_spirit_choice_is_private_mandatory_and_counts_against_capacity():
    e = position(nature_spirits=True)
    assert np.flatnonzero(mask(e)).tolist() == [
        game.SPIRIT_OFFSET,
        game.SPIRIT_OFFSET + 1,
    ]
    other = e.observe("player_1")["observation"]
    e.spirit_offers[0].reverse()
    np.testing.assert_array_equal(other, e.observe("player_1")["observation"])
    choice, rejected = e.spirit_offers[0]
    e.step(game.SPIRIT_OFFSET)
    assert e.owners[choice] == 0 and e.spirit_choices[0] == choice
    assert rejected in e.unused_spirits
    for card in (0, 1, 2):
        e.owners[card] = 0
    assert not e._capacity(0)
    e.progress[choice] = 1
    assert e._capacity(0)


def test_solo_discard_and_refill():
    e = position(num_players=1)
    assert len(e.market) == len(e.cards) == 3
    offers = [row[:] for row in e.market]
    e.step(game.GROUP_OFFSET)
    for color in list(offers[0]):
        legal = [cell for cell in range(23) if e.can_place(0, color, cell)]
        e.step(game.place_action(color, legal[0]))
    old = e.cards[1]
    assert mask(e)[game.SOLO_DISCARD_OFFSET + 1]
    e.step(game.SOLO_DISCARD_OFFSET + 1)
    assert old in e.card_discards
    assert Counter(e.token_discards) == Counter(offers[1] + offers[2])
    assert len(e.bag) == 102
    assert all(len(offer) == 3 for offer in e.market)
    assert e.cards[1] >= 0 and e.cards[1] != old
    e.step(game.DRAFT_OFFSET)
    e.taken = True
    e.remaining.clear()
    assert not mask(e)[game.SOLO_DISCARD_OFFSET :].any()


def test_end_round_equal_turns_and_end_condition_at_turn_end_only():
    e = position(num_players=3)
    e.boards[0][:21] = [(game.BLUE,)] * 21
    assert not e.final_round
    e.taken = True
    e.step(game.END_TURN)
    assert e.final_round and not any(e.terminations.values())
    assert e.turn_player == 1
    for p in (1, 2):
        e.taken = True
        e.step(game.END_TURN)
    assert all(e.terminations.values()) and e.turns == 3


def test_bag_exhaustion_completes_round():
    e = position(num_players=4)
    e.bag = []
    e.market[0] = []
    for p in range(4):
        e.taken = True
        e.step(game.END_TURN)
        assert e.final_round
        if p < 3:
            assert not any(e.terminations.values())
    assert all(e.terminations.values())


def test_final_refill_does_not_end_until_an_empty_bag_is_needed():
    e = position()
    e.bag = [game.BLUE] * 3
    e.market[0] = []
    e.taken = True
    e.step(game.END_TURN)
    assert not e.final_round
    assert not e.bag
    e.market[0] = []
    e.taken = True
    e.step(game.END_TURN)
    assert all(e.terminations.values())


def test_ties_and_solo_reward(monkeypatch):
    e = position(num_players=3)
    monkeypatch.setattr(e, "score_details", lambda p: {"animals": 50})
    e.cubes[0][0] = 1
    e.cubes[1][0] = 2
    e._finish()
    assert e.winners == ["player_0", "player_1"]
    assert list(e.rewards.values()) == [1, 1, -1]
    solo = position(num_players=1)
    monkeypatch.setattr(solo, "score_details", lambda p: {"animals": 130})
    solo._finish()
    assert solo.rewards["player_0"] == 130
    assert solo.infos["player_0"]["suns"] == 8  # Five thresholds + side A + no spirit.


def test_hidden_draw_order_observation_copies_and_illegal_atomicity():
    e = position(nature_spirits=True, render_mode="ansi")
    before = e.observe("player_0")["observation"]
    rendered = e.render()
    e.bag.reverse()
    e.deck.reverse()
    e.spirit_offers[1].reverse()
    np.testing.assert_array_equal(before, e.observe("player_0")["observation"])
    assert rendered == e.render()
    assert not e.observe("player_1")["action_mask"].any()
    for action in (-1, game.ACTION_COUNT, None, game.END_TURN, game.PLACE_OFFSET):
        with pytest.raises(ValueError):
            e.step(action)
        np.testing.assert_array_equal(before, e.observe("player_0")["observation"])
    before[:] = 99
    assert not np.all(e.observe("player_0")["observation"] == 99)
    assert e.infos == {a: {} for a in e.agents}


@pytest.mark.parametrize(
    "players,side,spirits",
    [
        (1, "A", False),
        (1, "B", True),
        (2, "A", True),
        (3, "B", False),
        (4, "A", True),
        (4, "B", True),
    ],
)
def test_complete_games_conserve_components(players, side, spirits):
    for seed in range(4):
        e = game.raw_env(
            num_players=players, board_side=side, nature_spirits=spirits, max_steps=None
        )
        e.reset(seed=seed)
        returns = dict.fromkeys(e.agents, 0)
        for a in e.agent_iter(1500):
            obs, reward, done, truncated, _ = e.last()
            returns[a] += reward
            assert e.observation_space(a).contains(obs)
            tokens = e.bag + e.token_discards + [t for offer in e.market for t in offer]
            tokens += [t for board in e.boards for stack in board for t in stack]
            tokens += [t for t, n in e.remaining.items() for _ in range(n)]
            assert Counter(tokens) == Counter(dict(enumerate(game.TOKEN_COUNTS, 1)))
            cards = e.deck + e.card_discards + [c for c in e.cards if c != -1]
            cards += [c for c in range(32) if e.owners[c] >= 0]
            assert Counter(cards) == Counter(range(32))
            spirit_cards = e.unused_spirits + [
                c for offer in e.spirit_offers for c in offer
            ]
            spirit_cards += [c for c in e.spirit_choices if c != -1]
            assert Counter(spirit_cards) == Counter(range(32, 42))
            for card in range(42):
                assert (
                    sum(c == card + 1 for board in e.cubes for c in board)
                    == e.progress[card]
                )
                assert e.progress[card] <= len(game.ANIMALS[card].points)
            for p in range(players):
                assert (
                    sum(
                        e.owners[i] == p and e.progress[i] < len(card.points)
                        for i, card in enumerate(game.ANIMALS)
                    )
                    <= 4
                )
            if done or truncated:
                action = None
            else:
                legal = np.flatnonzero(obs["action_mask"]).tolist()
                settlements = [
                    x for x in legal if game.SETTLE_OFFSET <= x < game.SPIRIT_OFFSET
                ]
                drafts = [
                    x for x in legal if game.DRAFT_OFFSET <= x < game.PLACE_OFFSET
                ]
                action = int(e.np_random.choice(settlements or drafts or legal))
            e.step(action)
        assert not e.agents and e.winners
        assert e.turns % players == 0
        for a in returns:
            assert returns[a] == (
                e.scores[0] if players == 1 else 1 if a in e.winners else -1
            )


@pytest.mark.parametrize(
    "kwargs",
    [{"num_players": 1}, {"num_players": 4, "nature_spirits": True, "board_side": "B"}],
)
def test_api(kwargs):
    api_test(game.env(**kwargs), num_cycles=500)


def test_seed():
    seed_test(lambda: game.env(nature_spirits=True), num_cycles=500)


def test_truncation():
    e = position(max_steps=1)
    e.step(game.GROUP_OFFSET)
    assert all(e.truncations.values()) and not e.winners
    assert not any(e.rewards.values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_players": 0},
        {"num_players": 5},
        {"num_players": True},
        {"board_side": "C"},
        {"nature_spirits": 1},
        {"max_steps": 0},
        {"max_steps": True},
        {"render_mode": "rgb_array"},
    ],
)
def test_config(kwargs):
    with pytest.raises(ValueError):
        game.raw_env(**kwargs)


def test_render_and_helpers(capsys):
    e = position(render_mode="human")
    e.render()
    assert "Harmonies" in capsys.readouterr().out
    e.close()
    assert position().render() is None
    with pytest.raises(ValueError):
        game.place_action(0, 0)
    with pytest.raises(ValueError):
        game.settle_action(42, 0)


def test_support_cells_can_already_hold_cubes():
    e = position()
    e.owners[25] = 0
    e.boards[0][c(2, 0)] = (game.YELLOW,)
    e.boards[0][c(2, 1)] = (game.GRAY,)
    e.cubes[0][c(2, 0)] = 31  # An earlier animal on the supporting field.
    assert e.can_settle(0, 25, c(2, 1))
    e.step(game.settle_action(25, c(2, 1)))
    assert e.cubes[0][c(2, 0)] == 31
    assert e.cubes[0][c(2, 1)] == 26


def test_completion_takes_precedence_over_step_limit():
    e = position(num_players=1)
    e.boards[0][:21] = [(game.BLUE,)] * 21
    e.taken = True
    e.max_steps = 1
    e.step(game.END_TURN)
    assert all(e.terminations.values())
    assert not any(e.truncations.values())
    for a in e.agent_iter():
        assert not e.observe(a)["action_mask"].any()
        e.step(None)
    assert not e.agents
