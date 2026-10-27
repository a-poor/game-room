from collections import Counter
from copy import deepcopy

import numpy as np
import pytest
from pettingzoo.test import api_test, seed_test

from game_room import sasquatch_v0 as g
from game_room.sasquatch_v0 import Card as C


def position(hands=None, collections=None, deals=(), points=None, **kwargs):
    e = g.raw_env(**kwargs)
    e.reset(seed=3)
    e.hands = hands or [[] for _ in e.agents]
    e.collections = collections or [[] for _ in e.agents]
    e.points = points or [0] * e.num_players
    e.discard = []
    e.deals = [None] * g.DEAL_SLOTS
    for i, card in enumerate(deals):
        e.deals[i] = card
    used = [c for hand in e.hands + e.collections for c in hand]
    used += [c.kind for c in deals]
    e.deck = g.make_deck()
    for card in used:
        e.deck.remove(card)
    e.np_random.shuffle(e.deck)
    return e


def deal(kind, owner, contributor=None, face_up=False):
    return g.DealCard(
        kind, owner, owner if contributor is None else contributor, face_up
    )


def things(e, p=0):
    e.phase = "things"
    e._select(p)


def choose(e):
    e.phase = "choose"
    e._select(1 - e.turn if e.num_players == 2 else e.turn)


def sets(e, first=0):
    e.set_order = [(first + i) % e.num_players for i in range(e.num_players)]
    e._advance_sets()


def assert_conservation(e):
    cards = e.deck + e.discard
    cards += [c for row in e.hands + e.collections for c in row]
    cards += [c.kind for c in e.deals if c is not None]
    assert Counter(cards) == Counter(g.make_deck())
    assert all(0 <= len(hand) <= 5 for hand in e.hands)
    assert all(score >= 0 for score in e.points)


def test_inventory():
    assert Counter(g.make_deck()) == dict(enumerate(g.CARD_COUNTS))
    assert sum(g.CARD_COUNTS) == 120
    assert sum(g.CARD_COUNTS[:4]) == 65
    assert sum(g.CARD_COUNTS[4:7]) == 32
    assert sum(g.CARD_COUNTS[7:]) == 23


@pytest.mark.parametrize("players,target", [(2, 5), (3, 4), (4, 4), (5, 5), (6, 5)])
def test_initial_setup(players, target):
    e = g.raw_env(num_players=players)
    e.reset(seed=9)
    assert e.target_score == target
    assert list(map(len, e.hands)) == [5] * players
    assert len(e.deck) == 120 - 5 * players
    assert e.agent_selection == ("player_0" if players == 2 else "player_1")
    assert_conservation(e)


@pytest.mark.parametrize("players", [2, 3, 6])
def test_api(players):
    api_test(g.env(num_players=players, max_steps=500), num_cycles=1000)


@pytest.mark.parametrize("players", [2, 5])
def test_seed(players):
    seed_test(lambda: g.env(num_players=players, max_steps=600), num_cycles=1000)


@pytest.mark.parametrize("players", [2, 3, 4, 5, 6])
def test_complete_random_games(players):
    for seed in range(5):
        e = g.raw_env(num_players=players, max_steps=5000)
        e.reset(seed=seed)
        returns = dict.fromkeys(e.agents, 0.0)
        for agent in e.agent_iter():
            obs, reward, terminated, truncated, _ = e.last()
            returns[agent] += reward
            assert e.observation_space(agent).contains(obs)
            if terminated or truncated:
                e.step(None)
            else:
                assert obs["action_mask"].any()
                e.step(e.action_space(agent).sample(obs["action_mask"]))
                assert_conservation(e)
                if (
                    e.phase == "offer_card"
                    and e.offered_count == 0
                    and e.offer_index == 0
                ):
                    assert list(map(len, e.hands)) == [5] * players
                    assert all(
                        Counter(row)[c] < g.SET_SIZES[c]
                        for row in e.collections
                        for c in range(7)
                    )
        assert len(e.winners) == 1
        winner = e._indices[e.winners[0]]
        assert e.points[winner] >= e.target_score
        assert all(
            e.points[winner] > e.points[q] for q in range(players) if q != winner
        )
        assert returns == {a: 1 if a in e.winners else -1 for a in e.possible_agents}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_players": 1},
        {"num_players": 7},
        {"num_players": True},
        {"max_steps": 0},
        {"max_steps": True},
        {"render_mode": "bad"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        g.raw_env(**kwargs)


def test_invalid_actions_are_atomic():
    e = position(hands=[[], [C.TINY], []])
    before = deepcopy(e.__dict__)
    with pytest.raises(ValueError):
        e.step(g.PLAY + C.CRYPTOZOOPTIC_EXPANDER)
    for key in (
        "hands",
        "deck",
        "discard",
        "points",
        "collections",
        "deals",
        "phase",
        "steps",
    ):
        assert getattr(e, key) == before[key]
    assert e.np_random.bit_generator.state == before["np_random"].bit_generator.state


def test_initial_offers_are_sealed_and_buyer_reveals_only_one():
    e = position(
        hands=[[C.GIANT] * 2, [C.TINY, C.MEDIUM, C.BIG], [C.TINY, C.MEDIUM, C.BIG]]
    )
    for c in (C.TINY, C.MEDIUM, C.BIG):
        e.step(c)
    e.step(g.SELECT_SLOT)
    assert not e.deals[0].face_up
    assert e.agent_selection == "player_2"
    assert all(
        row[-1] == 0
        for row in e.observe("player_2")["observation"][-600:].reshape(120, 5)[:3]
    )
    for c in (C.TINY, C.MEDIUM, C.BIG):
        e.step(c)
    e.step(g.SELECT_SLOT + 3)
    assert e.phase == "buyer_reveal"
    assert sum(c.face_up for c in e.deals if c) == 2
    e.step(g.SELECT_SLOT + 1)
    assert sum(c.face_up for c in e.deals if c) == 3
    assert e.phase == "things" and e.agent_selection == "player_0"
    for _ in range(3):
        e.step(g.PASS)
    assert e.phase == "choose"
    e.step(g.SELECT_DEAL + 1)
    assert e.buyer == e.turn == 1
    assert Counter(e.collections[0]) == Counter([C.TINY, C.MEDIUM, C.BIG])
    assert not e.collections[1]
    assert Counter(e.collections[2]) == Counter([C.TINY, C.MEDIUM, C.BIG])
    assert e.agent_selection == "player_2" and e.turn_number == 2
    assert_conservation(e)


def test_hidden_hands_deck_and_deals():
    e = position(
        hands=[[C.TINY], [C.MEDIUM], [C.BIG]],
        deals=[deal(C.TINY, 1), deal(C.MEDIUM, 2)],
    )
    e.phase = "buyer_reveal"
    e._select(0)
    before = e.observe("player_0")
    e.hands[1], e.hands[2] = e.hands[2], e.hands[1]
    e.deals[0].kind, e.deals[1].kind = e.deals[1].kind, e.deals[0].kind
    e.deck.reverse()
    for key in before:
        np.testing.assert_array_equal(before[key], e.observe("player_0")[key])
    rows = e.observe("player_1")["observation"][-600:].reshape(120, 5)
    assert rows[0, 4] == C.MEDIUM + 1 and rows[1, 4] == 0
    assert not e.observe("player_1")["action_mask"].any()


@pytest.mark.parametrize("self_cards", [0, 1, 2, 3])
@pytest.mark.parametrize("decision", [g.ACCEPT, g.REVERSE])
def test_two_player_splits_accept_reverse_and_alternation(self_cards, decision):
    cards = [C.TINY, C.MEDIUM, C.BIG]
    e = position(hands=[cards[:], []], num_players=2)
    for i, c in enumerate(cards):
        e.step(c)
        assert e.phase == "offer_place"
        assert e.hands[0].count(c) == 1  # Pending card stays in hand until placed.
        e.step(g.SELECT_DEAL + (0 if i < self_cards else 1))
    e.step(g.SELECT_SLOT)
    assert e.phase == "things"  # No extra reveal in two-player games.
    assert sum(c.face_up for c in e.deals if c) == 1
    e.step(g.PASS)
    e.step(g.PASS)
    assert e.agent_selection == "player_1" and e.phase == "choose"
    e.step(decision)
    expected = [cards[:self_cards], cards[self_cards:]]
    if decision == g.REVERSE:
        expected.reverse()
    assert e.collections == expected
    assert e.turn == 1 and e.buyer == -1
    assert e.agent_selection == "player_1"
    assert_conservation(e)


def test_only_collection_thingamabobs_play_and_pass_window_restarts():
    e = position(
        hands=[[C.DETRITAL_REPOSITIONER], [], []],
        collections=[[], [C.SPECTROELECTRIC_OPTIMETER], []],
        deals=[deal(C.DETRITAL_REPOSITIONER, 1), deal(C.BIG, 2)],
    )
    things(e)
    assert np.flatnonzero(e.observe("player_0")["action_mask"]).tolist() == [g.PASS]
    e.step(g.PASS)
    e.step(g.PLAY + C.SPECTROELECTRIC_OPTIMETER)
    assert e.passes == 0
    e.step(g.SELECT_SLOT)
    assert e.deals[0].face_up
    assert e.agent_selection == "player_1"
    for expected in ("player_1", "player_2", "player_0"):
        assert e.agent_selection == expected
        e.step(g.PASS)
    assert e.phase == "choose"
    assert C.SPECTROELECTRIC_OPTIMETER in e.discard
    assert_conservation(e)


def test_multiple_tools_by_same_player_and_removal_reveals_discard():
    e = position(
        collections=[[C.DETRITAL_REPOSITIONER, C.SPECTROELECTRIC_OPTIMETER], [], []],
        deals=[deal(C.GIANT, 1), deal(C.TINY, 2)],
    )
    things(e)
    e.step(g.PLAY + C.SPECTROELECTRIC_OPTIMETER)
    e.step(g.SELECT_SLOT)
    e.step(g.PLAY + C.DETRITAL_REPOSITIONER)
    e.step(g.SELECT_SLOT)
    assert e.deals[0] is None
    assert C.GIANT in e.discard
    assert e.phase == "things" and e.agent_selection == "player_0"
    assert_conservation(e)


@pytest.mark.parametrize("owners", [(1, 1), (1, 2)])
def test_super_removes_two_from_one_or_two_deals(owners):
    e = position(
        collections=[[C.SUPER_DETRITAL_REPOSITIONER], [], []],
        deals=[deal(C.GIANT, owners[0]), deal(C.BIG, owners[1])],
    )
    things(e)
    e.step(g.PLAY + C.SUPER_DETRITAL_REPOSITIONER)
    e.step(g.SELECT_SLOT)
    assert e.remaining == 1 and e.phase == "remove"
    assert not e.observe("player_0")["action_mask"][g.SELECT_SLOT]
    e.step(g.SELECT_SLOT + 1)
    assert not e._slots() and e.phase == "things"
    assert_conservation(e)


def test_tools_without_legal_targets_are_masked():
    e = position(
        collections=[
            [
                C.PLATONIC_ISOLATOR,
                C.SUPER_DETRITAL_REPOSITIONER,
                C.DETRITAL_REPOSITIONER,
                C.SPECTROELECTRIC_OPTIMETER,
                C.CRYPTOZOOPTIC_EXPANDER,
            ],
            [],
            [],
        ]
    )
    things(e)
    assert np.flatnonzero(e.observe("player_0")["action_mask"]).tolist() == [g.PASS]
    e.deals[0] = deal(C.TINY, 1, face_up=True)
    mask = e.observe("player_0")["action_mask"]
    assert mask[g.PLAY + C.DETRITAL_REPOSITIONER]
    assert not mask[g.PLAY + C.SUPER_DETRITAL_REPOSITIONER]
    assert not mask[g.PLAY + C.SPECTROELECTRIC_OPTIMETER]


def test_expansion_to_another_deal_stays_known_only_to_contributor():
    e = position(
        hands=[[C.GIANT], [], []], collections=[[C.CRYPTOZOOPTIC_EXPANDER], [], []]
    )
    things(e)
    e.step(g.PLAY + C.CRYPTOZOOPTIC_EXPANDER)
    e.step(C.GIANT)
    assert e.observe("player_0")["observation"][17] == C.GIANT + 1
    assert e.observe("player_1")["observation"][17] == 0
    e.step(g.SELECT_DEAL + 1)
    assert e.deals[0] == deal(C.GIANT, 1, contributor=0)
    assert (
        e.observe("player_0")["observation"][-600:].reshape(120, 5)[0, 4] == C.GIANT + 1
    )
    assert e.observe("player_1")["observation"][-600:].reshape(120, 5)[0, 4] == 0
    assert not e.hands[0]
    assert_conservation(e)


def test_isolator_only_steals_from_higher_score_and_does_not_end_turn():
    e = position(collections=[[C.PLATONIC_ISOLATOR], [], []], points=[3, 4, 3])
    things(e)
    e.step(g.PLAY + C.PLATONIC_ISOLATOR)
    assert np.flatnonzero(e.observe("player_0")["action_mask"]).tolist() == [
        g.SELECT_PLAYER + 1
    ]
    e.step(g.SELECT_PLAYER + 1)
    assert e.points == [4, 3, 3] and not e.winners
    assert e.phase == "things"


def test_new_buyer_steals_from_outgoing_buyer_before_creature_scoring():
    e = position(
        collections=[[C.POISON_PILL_BUG], [], []],
        deals=[deal(C.POISON_PILL_BUG, 1), deal(C.GIANT, 1), deal(C.TINY, 2)],
    )
    choose(e)
    e.step(g.SELECT_DEAL + 1)
    assert e.buyer == 1 and e.turn == 0
    assert e.phase == "nasty" and e.agent_selection == "player_0"
    e.step(C.POISON_PILL_BUG)
    assert e.agent_selection == "player_1" and e.phase == "steal"
    e.step(C.GIANT)
    assert e.points == [0, 1, 0]
    assert not e.collections[0] and not e.collections[1]
    assert e.turn == 1
    assert_conservation(e)


@pytest.mark.parametrize("taken", [0, 1, 2])
def test_loan_shark_allows_zero_one_or_two_cards(taken):
    e = position(collections=[[C.LOAN_SHARK] * 2 + [C.TINY, C.MEDIUM], [], []])
    e.buyer = 1
    sets(e)
    e.step(C.LOAN_SHARK)
    for card in [C.TINY, C.MEDIUM][:taken]:
        e.step(card)
    if taken < 2:
        e.step(g.STOP)
    assert e.collections[1] == [C.TINY, C.MEDIUM][:taken]
    assert e.collections[0] == [C.TINY, C.MEDIUM][taken:]
    assert_conservation(e)


@pytest.mark.parametrize("points", [0, 2])
def test_trojan_horse_only_steals_existing_points(points):
    e = position(
        collections=[[C.TROJAN_HORSE] * 3 + [C.GIANT], [], []], points=[points, 0, 0]
    )
    e.buyer = 1
    sets(e)
    e.step(C.TROJAN_HORSE)
    assert e.points == [points - (1 if points else 0) + 1, 1 if points else 0, 0]
    assert_conservation(e)


def test_buyer_own_nasty_is_discarded_without_self_transfer():
    e = position(collections=[[C.LOAN_SHARK] * 2 + [C.GIANT], [], []])
    sets(e)
    e.step(C.LOAN_SHARK)
    assert e.points == [1, 0, 0]
    assert not e.collections[0]
    assert e.turn_number == 2


def test_owners_choose_nasty_order_and_all_nasties_precede_creatures():
    e = position(
        collections=[[C.TROJAN_HORSE] * 3 + [C.POISON_PILL_BUG] * 2 + [C.GIANT], [], []]
    )
    e.buyer = 1
    sets(e)
    mask = e.observe("player_0")["action_mask"]
    assert mask[C.TROJAN_HORSE] and mask[C.POISON_PILL_BUG]
    e.step(C.TROJAN_HORSE)
    assert e.points == [0, 0, 0]
    e.step(C.POISON_PILL_BUG)
    e.step(g.STOP)
    assert e.points == [1, 0, 0]


def test_two_player_nasty_theft_cascades():
    e = position(
        num_players=2,
        collections=[[C.POISON_PILL_BUG] * 2 + [C.LOAN_SHARK], [C.LOAN_SHARK, C.GIANT]],
    )
    sets(e)
    e.step(C.POISON_PILL_BUG)
    assert e.agent_selection == "player_1"
    e.step(C.LOAN_SHARK)
    assert e.phase == "nasty" and e.agent_selection == "player_1"
    assert e.points == [0, 0]
    e.step(C.LOAN_SHARK)
    assert e.phase == "steal" and e.agent_selection == "player_0"
    e.step(C.GIANT)
    assert e.points == [1, 0]
    assert not any(e.collections)
    assert_conservation(e)


def test_all_creature_sets_score_and_leave_only_incomplete_remainders():
    e = position(
        num_players=2,
        collections=[[C.TINY] * 9 + [C.MEDIUM] * 4 + [C.BIG] * 3 + [C.GIANT], []],
    )
    sets(e)
    assert e.points == [5, 0]
    assert Counter(e.collections[0]) == Counter([C.TINY, C.MEDIUM, C.BIG])
    assert e.winners == ["player_0"]
    assert list(map(len, e.hands)) == [5, 5]
    assert_conservation(e)


def test_tie_continues_until_unique_leader_and_win_is_at_turn_end():
    e = position(num_players=2, points=[4, 4], collections=[[C.GIANT], [C.GIANT]])
    sets(e)
    assert e.points == [5, 5] and not e.winners
    assert e.turn_number == 2
    # A later completed turn resolves the tie.
    e.collections[0].append(C.GIANT)
    e.deck.remove(C.GIANT)
    sets(e)
    assert e.points == [6, 5] and e.winners == ["player_0"]
    for agent in e.agent_iter():
        assert e.last()[2]
        assert not e.observe(agent)["action_mask"].any()
        e.step(None)
    assert not e.agents


def test_midturn_lead_can_be_lost_before_scoring():
    e = position(points=[4, 3, 0], collections=[[C.TROJAN_HORSE] * 3, [], []])
    e.buyer = 1
    sets(e)
    assert not e.winners
    e.step(C.TROJAN_HORSE)
    assert e.points == [3, 4, 0]
    assert e.winners == ["player_1"]


def test_empty_deal_remains_selectable_and_changes_buyer():
    e = position(deals=[deal(C.TINY, 2)])
    choose(e)
    assert e.observe("player_0")["action_mask"][g.SELECT_DEAL + 1]
    e.step(g.SELECT_DEAL + 1)
    assert e.buyer == e.turn == 1
    assert e.collections[2] == [C.TINY]


def test_draw_recycling_and_hand_refill():
    e = position(hands=[[C.TINY], [], []])
    e.discard, e.deck = e.deck, []
    e._refill(0)
    assert len(e.hands[0]) == 5 and not e.discard
    assert_conservation(e)


def test_truncation_reset_and_public_render(capsys):
    e = position(hands=[[], [C.GIANT], []], max_steps=1, render_mode="ansi")
    assert "GIANT" not in e.render()
    e.step(C.GIANT)
    assert all(e.truncations.values()) and not any(e.terminations.values())
    assert "GIANT" not in e.render()
    e.reset(seed=1)
    assert not any(e.truncations.values())
    assert e.action_label(g.SELECT_SLOT + 4) == "Deal slot 4"
    assert e.action_label(g.PLAY + C.PLATONIC_ISOLATOR) == "Play PLATONIC_ISOLATOR"
    e.render_mode = "human"
    assert e.render() is None
    assert "Sell Me a Sasquatch" in capsys.readouterr().out


def test_victory_takes_priority_over_step_limit():
    e = position(deals=[deal(C.GIANT, 1)], points=[3, 0, 0], max_steps=1)
    choose(e)
    e.step(g.SELECT_DEAL + 1)
    assert e.winners == ["player_0"]
    assert not any(e.truncations.values())
