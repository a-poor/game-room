from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from itertools import permutations

import numpy as np
import pytest
from pettingzoo.test import api_test, seed_test

from game_room import mahjong_v0 as game
from game_room.mahjong_cards import (
    BAMS,
    CRAKS,
    DOTS,
    FLOWER,
    GREEN,
    JOKER,
    NMJL_2026,
    NORTH,
    RED,
    TILE_COUNTS,
    WHITE,
    Card,
    Exposure,
    Group,
    HandPattern,
    suited,
)


def card(*groups, concealed=False, bonus=True, value=25):
    return Card(
        "Test card",
        (
            HandPattern(
                "Test hand", tuple(Group(*g) for g in groups), value, concealed, bonus
            ),
        ),
    )


STANDARD = card((0, 3), (1, 3), (2, 4), (3, 4))
PAIRS = card(*[(t, 2) for t in range(7)], concealed=True, bonus=False, value=50)


def position(**kwargs):
    env = game.raw_env(**kwargs)
    env.reset(seed=42)
    env.hands = [env.hands[(env.dealer + i) % 4] for i in range(4)]
    env.dealer = env.turn_player = 0
    env._select(0)
    return env


def arrange(env, hands, exposures=None, discards=()):
    """Make a physically possible fixture and put all remaining tiles in wall."""
    env.hands = [Counter(h) for h in hands]
    env.exposures = exposures or [[] for _ in range(4)]
    env.discards = [list(d) for d in discards]
    used = Counter()
    for h in env.hands:
        used.update(h)
    for es in env.exposures:
        for e in es:
            used[e.tile] += e.size - e.jokers
            used[JOKER] += e.jokers
    used.update(t for _, t, claimed in env.discards if not claimed)
    supply = Counter(game.make_wall())
    assert not used - supply
    env.wall = list((supply - used).elements())
    env.phase = game.TURN


def inventory(env):
    total = Counter(env.wall)
    for h, es, pending in zip(env.hands, env.exposures, env.pass_tiles):
        total.update(h)
        total.update(t for t in pending if t >= 0)
        for e in es:
            total[e.tile] += e.size - e.jokers
            total[JOKER] += e.jokers
    total.update(t for _, t, claimed in env.discards if not claimed)
    return +total


def mask(env):
    return env.observe(env.agent_selection)["action_mask"]


def pass_responses(env):
    while env.phase == game.RESPOND:
        env.step(game.PASS)


def test_setup():
    env = position()
    assert env.card is NMJL_2026
    assert sorted(h.total() for h in env.hands) == [13, 13, 13, 14]
    assert len(env.wall) == 99
    assert inventory(env) == Counter(game.make_wall())
    assert len(game.make_wall()) == 152
    assert mask(env)[game.CONTINUE]
    assert all(not env.observe(a)["action_mask"].any() for a in env.agents[1:])


def test_nmjl_card_inventory_and_every_pattern_is_winnable():
    assert len(NMJL_2026.patterns) == 1077
    assert Counter(p.name.split(" #")[0] for p in NMJL_2026.patterns) == {
        "2026": 30,
        "2468": 66,
        "Any Like Numbers": 135,
        "Quints": 264,
        "Consecutive Run": 315,
        "13579": 81,
        "Winds and Dragons": 107,
        "369": 48,
        "Singles and Pairs": 31,
    }
    assert len({p.name.split(" / ")[0] for p in NMJL_2026.patterns}) == 55
    for i, pattern in enumerate(NMJL_2026.patterns):
        hand = Counter()
        for g in pattern.groups:
            hand[g.tile] += g.size
        for tile in tuple(hand):
            excess = max(0, hand[tile] - TILE_COUNTS[tile])
            hand[tile] -= excess
            hand[JOKER] += excess
        assert i in NMJL_2026.matching_hands(hand), pattern.name
        if pattern.concealed:
            continue
        # An exposed group must consume a whole printed group, even when
        # another group of the same tile is present on this card line.
        for g in pattern.groups:
            if g.size < 3 or hand[g.tile] < g.size:
                continue
            remainder = hand.copy()
            remainder[g.tile] -= g.size
            assert i in NMJL_2026.matching_hands(remainder, [Exposure(g.tile, g.size)])


def line_card(name):
    return Card(
        name, tuple(p for p in NMJL_2026.patterns if p.name.startswith(name + " /"))
    )


def test_2026_any_dragon_and_flower_transcription_regressions():
    for a, b, c in permutations((BAMS, CRAKS, DOTS)):
        for dragon in (GREEN, RED, WHITE):
            h = Counter(
                {
                    FLOWER: 2,
                    suited(a, 5): 4,
                    suited(b, 5): 2,
                    suited(c, 5): 4,
                    dragon: 2,
                }
            )
            assert line_card("Any Like Numbers #3").matching_hands(h)
        h = Counter(
            {FLOWER: 3, suited(a, 1): 2, suited(b, 2): 2, suited(a, 3): 3, 31 + b: 4}
        )
        assert line_card("Consecutive Run #5").matching_hands(h)
        h[FLOWER] -= 1
        assert not line_card("Consecutive Run #5").matching_hands(h)
        h = Counter(
            {
                suited(a, 2): 1,
                suited(a, 4): 1,
                suited(a, 6): 2,
                suited(a, 8): 2,
                suited(b, 2): 1,
                suited(b, 4): 1,
                suited(b, 6): 2,
                suited(b, 8): 2,
                suited(c, 8): 2,
            }
        )
        assert line_card("Singles and Pairs #2").matching_hands(h)


def test_flower_pungs_cannot_be_combined_into_sextet():
    c = line_card("Winds and Dragons #5")
    assert c.matching_hands(
        Counter({NORTH: 4, GREEN: 4}), [Exposure(FLOWER, 3), Exposure(FLOWER, 3)]
    )
    assert not c.matching_hands(Counter({NORTH: 4, GREEN: 4}), [Exposure(FLOWER, 6)])
    env = position(card=c)
    arrange(env, [[FLOWER] * 5, [FLOWER], [], []])
    env.turn_player = 1
    env._select(1)
    env.step(FLOWER)
    env.step(game.PASS)
    env.step(game.PASS)
    assert env.agent_selection == "player_0"
    assert mask(env)[game.claim_action(3)]
    assert not mask(env)[game.claim_action(6)]


def test_fixed_ranks_and_dragon_relationships():
    c = line_card("Consecutive Run #1")
    assert c.matching_hands(Counter({0: 2, 1: 3, 2: 2, 3: 3, 4: 4}))
    assert not c.matching_hands(Counter({1: 2, 2: 3, 3: 2, 4: 3, 5: 4}))
    c = line_card("2468 #4")
    assert c.matching_hands(Counter({1: 4, GREEN: 3, 16: 4, RED: 3}))
    assert not c.matching_hands(Counter({1: 4, GREEN: 3, 16: 4, WHITE: 3}))
    c = line_card("Quints #3")
    assert c.matching_hands(Counter({0: 4, 8: 4, JOKER: 2, RED: 4}))
    assert not c.matching_hands(Counter({0: 4, 8: 4, JOKER: 2, GREEN: 4}))


def test_jokers_only_replace_groups_of_three_or_more():
    assert STANDARD.matching_hands(Counter({JOKER: 3, 1: 3, 2: 4, 3: 4})) == (0,)
    c = card((0, 2), (1, 4), (2, 4), (3, 4))
    assert not c.matching_hands(Counter({0: 1, JOKER: 1, 1: 4, 2: 4, 3: 4}))
    c = card((0, 1), (1, 1), (2, 4), (3, 4), (4, 4))
    assert not c.matching_hands(Counter({1: 1, JOKER: 1, 2: 4, 3: 4, 4: 4}))
    c = card((0, 1), (0, 1), (0, 1), (1, 3), (2, 4), (3, 4))
    assert not c.matching_hands(Counter({0: 2, JOKER: 1, 1: 3, 2: 4, 3: 4}))


def test_duplicate_groups_require_distinct_exposures_and_naturals_for_pair():
    c = card((FLOWER, 3), (FLOWER, 3), (0, 4), (1, 4))
    assert c.matching_hands(
        Counter({0: 4, 1: 4}), [Exposure(FLOWER, 3), Exposure(FLOWER, 3)]
    )
    assert not c.matching_hands(
        Counter({FLOWER: 3, 0: 4, 1: 4}), [Exposure(FLOWER, 3), Exposure(FLOWER, 3)]
    )
    c = card((0, 2), (0, 3), (1, 3), (2, 3), (3, 3))
    assert c.matching_hands(Counter({0: 4, JOKER: 1, 1: 3, 2: 3, 3: 3}))
    assert not c.matching_hands(Counter({0: 1, JOKER: 4, 1: 3, 2: 3, 3: 3}))


def test_concealed_pattern_allows_final_discard_but_no_exposures():
    c = Card("Concealed", (replace(STANDARD.patterns[0], concealed=True),))
    assert not c.matching_hands(Counter({1: 3, 2: 4, 3: 4}), [Exposure(0, 3)])
    env = position(card=c)
    arrange(env, [[0], {0: 2, 1: 3, 2: 4, 3: 4}, [], []])
    env.step(0)
    assert mask(env)[game.MAHJONG]
    assert not mask(env)[game.CLAIM_OFFSET : game.EXCHANGE_OFFSET].any()
    env.step(game.MAHJONG)
    pass_responses(env)
    assert env.winners == ["player_1"]


def test_charleston_directions_and_dealer_does_not_draw():
    env = position()
    env.step(game.CONTINUE)
    for index, direction in enumerate((1, 2, -1, -1, 2, 1)):
        assert env.phase == game.CHARLESTON and env.pass_index == index
        before = deepcopy(env.hands)
        passed = [[] for _ in range(4)]
        for p in range(4):
            for _ in range(3):
                tile = next(t for t in range(JOKER) if env.hands[p][t])
                passed[p].append(tile)
                env.step(tile)
        for p in range(4):
            expected = before[p] - Counter(passed[p])
            expected.update(passed[(p - direction) % 4])
            assert +env.hands[p] == expected
        assert inventory(env) == Counter(game.make_wall())
        if index == 2:
            for _ in range(4):
                env.step(game.CONTINUE)
    for _ in range(4):
        env.step(game.COURTESY_OFFSET)
    assert env.phase == game.TURN and env.agent_selection == "player_0"
    assert len(env.wall) == 99
    assert [h.total() for h in env.hands] == [14, 13, 13, 13]


def test_stop_second_charleston_and_courtesy_minimum():
    env = position()
    env._begin_pass(2)
    for _ in range(12):
        env.step(int(np.flatnonzero(mask(env)[:JOKER])[0]))
    env.step(game.STOP)
    for n in (3, 2, 1, 0):
        env.step(game.COURTESY_OFFSET + n)
    assert env.pass_counts == [1, 0, 1, 0]
    before = deepcopy(env.hands)
    a = int(np.flatnonzero(mask(env)[:JOKER])[0])
    env.step(a)
    assert env.agent_selection == "player_2"
    b = int(np.flatnonzero(mask(env)[:JOKER])[0])
    env.step(b)
    before[0][a] -= 1
    before[0][b] += 1
    before[2][b] -= 1
    before[2][a] += 1
    assert env.hands == before
    assert env.phase == game.TURN


def test_blind_pass_routes_unseen_tiles_without_jokers():
    env = position()
    env._begin_pass(2)
    before = deepcopy(env.hands)
    sent = {}
    # Seat 1's three tiles pass through seat 0 unseen to seat 3.
    for _ in range(3):
        env.step(game.BLIND)
    for p in (1, 2, 3):
        sent[p] = []
        for _ in range(3):
            assert not mask(env)[JOKER]
            tile = int(np.flatnonzero(mask(env)[:JOKER])[0])
            sent[p].append(tile)
            env.step(tile)
    assert env.hands[0] == before[0]
    assert +env.hands[3] == before[3] - Counter(sent[3]) + Counter(sent[1])
    assert inventory(env) == Counter(game.make_wall())


@pytest.mark.parametrize("index", [2, 5])
def test_all_blind_cycle(index):
    env = position()
    before = deepcopy(env.hands)
    env._begin_pass(index)
    for _ in range(12):
        env.step(game.BLIND)
    if index == 2:
        assert env.blind_recovery
        assert not mask(env)[game.BLIND]
        env.step(int(np.flatnonzero(mask(env)[:JOKER])[0]))
        assert env.phase == game.SECOND_VOTE
    else:
        assert env.phase == game.TURN
    assert env.hands == before
    assert inventory(env) == Counter(game.make_wall())


@pytest.mark.parametrize("index", [0, 1, 3, 4, 6])
def test_blind_disallowed_on_other_passes(index):
    env = position()
    env._begin_pass(index)
    assert not mask(env)[game.BLIND]
    with pytest.raises(ValueError):
        env.step(game.BLIND)


@pytest.mark.parametrize("size,jokers", [(3, 0), (4, 1), (5, 2), (6, 3)])
def test_claims_no_replacement_draw_and_turn_skips(size, jokers):
    # Flowers can naturally form a sextet; the other groups total 14-size.
    groups = {
        3: [(0, 3), (1, 4), (2, 4)],
        4: [(0, 3), (1, 3), (2, 4)],
        5: [(0, 3), (1, 3), (2, 3)],
        6: [(0, 4), (1, 4)],
    }
    env = position(card=card((FLOWER, size), *groups[size]))
    arrange(env, [[FLOWER], [], {FLOWER: size - jokers - 1, JOKER: jokers, 9: 1}, []])
    wall_size = len(env.wall)
    env.step(FLOWER)
    env.step(game.PASS)
    env.step(game.claim_action(size, jokers))
    env.step(game.PASS)
    assert env.agent_selection == "player_2"
    assert env.exposures[2] == [Exposure(FLOWER, size, jokers)]
    assert len(env.wall) == wall_size
    assert inventory(env) == Counter(game.make_wall())
    env.step(9)
    pass_responses(env)
    assert env.agent_selection == "player_3"


def test_mahjong_priority_over_exposure_and_nearest_winner():
    # Players 2 and 3 both wait for one flower. Player 1 can expose a pung.
    c = card((FLOWER, 2), (0, 4), (1, 4), (2, 4))
    alt = card((FLOWER, 3), (4, 3), (5, 4), (6, 4)).patterns[0]
    p3 = card((FLOWER, 2), (9, 4), (10, 4), (11, 4)).patterns[0]
    c = Card(
        "Priority",
        (c.patterns[0], replace(alt, name="pung"), replace(p3, name="other win")),
    )
    env = position(card=c)
    arrange(
        env,
        [
            [FLOWER],
            [FLOWER, FLOWER, 20],
            {FLOWER: 1, 0: 4, 1: 4, 2: 4},
            {FLOWER: 1, 9: 4, 10: 4, 11: 4},
        ],
    )
    env.step(FLOWER)
    env.step(game.claim_action(3))
    env.step(game.MAHJONG)
    assert not any(env.terminations.values())
    env.step(game.MAHJONG)
    assert env.winners == ["player_2"]
    assert not env.exposures[1]
    assert inventory(env) == Counter(game.make_wall())


def test_nearest_exposure_wins():
    env = position(card=card((FLOWER, 3), (0, 3), (1, 4), (2, 4)))
    arrange(env, [[FLOWER], [FLOWER, FLOWER, 20], [FLOWER, FLOWER, 21], []])
    env.step(FLOWER)
    env.step(game.claim_action(3))
    env.step(game.claim_action(3))
    env.step(game.PASS)
    assert env.agent_selection == "player_1"
    assert not env.exposures[2]


def test_discarded_joker_cannot_win_call_or_redeem():
    env = position(card=STANDARD)
    arrange(env, [[JOKER], {0: 2, 1: 3, 2: 4, 3: 4}, [], []])
    env.step(JOKER)
    for _ in range(3):
        assert np.flatnonzero(mask(env)).tolist() == [game.PASS]
        env.step(game.PASS)


def test_joker_exchange_after_claim_and_multiple_redemptions():
    env = position(card=STANDARD)
    arrange(env, [[0], {0: 2, 1: 2, 9: 1}, [], []], [[], [], [Exposure(1, 3, 2)], []])
    env.step(0)
    assert not mask(env)[game.exchange_action(2, 0)]
    env.step(game.claim_action(3))
    pass_responses(env)
    assert env.agent_selection == "player_1"
    for _ in range(2):
        env.step(game.exchange_action(2, 0))
    assert env.hands[1][JOKER] == 2 and env.hands[1][1] == 0
    assert env.exposures[2][0].jokers == 0
    assert not mask(env)[game.exchange_action(2, 0)]
    assert inventory(env) == Counter(game.make_wall())


def test_own_joker_exchange_and_finesse_scores_self_pick():
    env = position(card=STANDARD)
    arrange(
        env, [[0], {0: 2, 1: 1, 2: 4, 3: 3}, [], []], [[], [Exposure(1, 3, 1)], [], []]
    )
    env.step(0)
    env.step(game.claim_action(3))
    pass_responses(env)
    assert not mask(env)[game.MAHJONG]
    env.step(game.exchange_action(1, 0))
    assert mask(env)[game.MAHJONG]
    env.step(game.MAHJONG)
    assert env.scores == [-50, 150, -50, -50]
    assert env.infos["player_0"]["self_picked"]


@pytest.mark.parametrize(
    "self_pick,jokerless", [(True, True), (True, False), (False, True), (False, False)]
)
def test_scoring_and_terminal_rewards(self_pick, jokerless):
    env = position(card=STANDARD)
    hand = Counter({0: 3, 1: 3, 2: 4, 3: 4})
    if not jokerless:
        hand[3] -= 1
        hand[JOKER] += 1
    if self_pick:
        arrange(env, [hand, [], [], []])
        env.step(game.MAHJONG)
        unit = 50 if jokerless else 25
        expected = [6 * unit, -2 * unit, -2 * unit, -2 * unit]
    else:
        hand[0] -= 1
        arrange(env, [[0], hand, [], []])
        env.step(0)
        env.step(game.MAHJONG)
        pass_responses(env)
        unit = 50 if jokerless else 25
        expected = [-2 * unit, 4 * unit, -unit, -unit]
    assert env.scores == expected
    assert sum(env.rewards.values()) == 0
    returns = {}
    for a in env.agent_iter():
        _, reward, terminated, truncated, _ = env.last()
        assert terminated and not truncated
        returns[a] = reward
        assert not mask(env).any()
        env.step(None)
    assert returns == dict(zip(env.possible_agents, expected))


def test_singles_pairs_no_jokerless_bonus_and_best_scoring_match():
    env = position(card=PAIRS)
    arrange(env, [{t: 2 for t in range(7)}, [], [], []])
    env.step(game.MAHJONG)
    assert env.scores == [300, -100, -100, -100]
    better = replace(STANDARD.patterns[0], name="Better", value=40)
    env = position(card=Card("Overlap", (STANDARD.patterns[0], better)))
    arrange(env, [{0: 3, 1: 3, 2: 4, 3: 4}, [], [], []])
    env.step(game.MAHJONG)
    assert env.infos["player_0"]["pattern"] == "Better"
    assert env.scores == [480, -160, -160, -160]


def test_heavenly_hand_and_terminal_precedes_max_steps():
    env = position(card=STANDARD, max_steps=1)
    arrange(env, [{0: 3, 1: 3, 2: 4, 3: 4}, [], [], []])
    env.phase = game.START
    assert np.flatnonzero(mask(env)).tolist() == [game.MAHJONG]
    env.step(game.MAHJONG)
    assert all(env.terminations.values()) and not any(env.truncations.values())


def test_last_discard_can_win_before_wall_game():
    env = position(card=PAIRS)
    arrange(env, [[0], {0: 1, **{t: 2 for t in range(1, 7)}}, [], []])
    env.wall = []
    env.step(0)
    env.step(game.MAHJONG)
    pass_responses(env)
    assert env.winners == ["player_1"]
    env = position()
    env.phase = game.TURN
    env.wall = []
    env.step(int(np.flatnonzero(mask(env)[:36])[0]))
    pass_responses(env)
    assert all(env.terminations.values())
    assert env.winners == [] and env.scores == [0] * 4
    assert env.infos["player_0"]["end_reason"] == "wall_game"


def test_dead_challenge_uses_public_information_and_skips_turns():
    c = card((0, 2), (1, 4), (2, 4), (3, 4))
    env = position(card=c)
    arrange(
        env,
        [[20], [21], [22], [23]],
        [[], [Exposure(1, 4)], [], []],
        [(0, 0, False)] * 3,
    )
    assert env._publicly_dead(1)
    assert mask(env)[game.DEAD_OFFSET + 1]
    env.step(game.DEAD_OFFSET + 1)
    assert env.dead[1]
    env.step(20)
    assert env.agent_selection == "player_2"
    pass_responses(env)
    assert env.agent_selection == "player_2"


def test_dead_hands_keep_exchangeable_jokers_and_pay():
    env = position(card=STANDARD)
    arrange(env, [{0: 3, 1: 3, 2: 4, 3: 4}, [], [], []])
    env.dead[1] = True
    env.step(game.MAHJONG)
    assert env.scores == [300, -100, -100, -100]
    env = position(card=STANDARD)
    arrange(env, [[0, 20], [], [], []], [[], [Exposure(0, 3, 1)], [], []])
    env.dead[1] = True
    assert mask(env)[game.exchange_action(1, 0)]
    env.step(game.exchange_action(1, 0))
    assert env.hands[0][JOKER] == 1


def test_public_dead_check_accounts_for_possible_joker_redemption():
    env = position(card=STANDARD)
    arrange(
        env, [[], [], [], []], [[], [Exposure(0, 3, 1)], [], []], [(0, 3, False)] * 4
    )
    # Four missing tiles can still be jokers; don't use anyone's hidden hand.
    assert not env._publicly_dead(1)
    env.discards += [[0, JOKER, False]] * 7
    assert env._publicly_dead(1)
    # One missing tile and one spare natural allow a joker redemption.
    env.discards = [[0, 3, False]] + [[0, JOKER, False]] * 7
    assert not env._publicly_dead(1)


def test_three_dead_hands_end_without_a_winner():
    env = position(card=PAIRS)
    arrange(env, [[20], [], [], []], discards=[(0, 0, False)] * 3)
    for p in (1, 2, 3):
        env.step(game.DEAD_OFFSET + p)
    assert all(env.terminations.values())
    assert env.scores == [0] * 4 and env.winners == []


def test_hidden_opponent_hands_wall_and_sealed_passes():
    env = position(render_mode="ansi")
    env.step(game.CONTINUE)
    before, rendered = env.observe("player_0"), env.render()
    env.hands[1], env.hands[2] = env.hands[2], env.hands[1]
    env.wall.reverse()
    np.testing.assert_array_equal(
        before["observation"], env.observe("player_0")["observation"]
    )
    np.testing.assert_array_equal(
        before["action_mask"], env.observe("player_0")["action_mask"]
    )
    assert rendered == env.render()
    env._select(1)
    tile = int(np.flatnonzero(mask(env)[:JOKER])[0])
    before = env.observe("player_0")["observation"]
    env.step(tile)
    after = env.observe("player_0")["observation"]
    # Only the public decision count changes while an opponent reserves a tile.
    before[5] += 1
    np.testing.assert_array_equal(before, after)
    assert env.render() == rendered
    before[:] = 999
    assert not np.all(env.observe("player_0")["observation"] == 999)


def test_sealed_claim_does_not_leak_before_resolution():
    env = position(card=STANDARD)
    arrange(env, [[0], [0, 0, 9], [], []])
    env.step(0)
    env.step(game.claim_action(3))
    before = env.observe("player_2")["observation"]
    rendered = env.render()
    env.responses[1] = game.PASS
    np.testing.assert_array_equal(before, env.observe("player_2")["observation"])
    assert env.render() == rendered


def test_illegal_action_is_atomic():
    env = position()
    for action in (-1, game.ACTION_COUNT, None, game.MAHJONG, 0):
        state = deepcopy(env.__dict__)
        before = env.observe("player_0")
        with pytest.raises(ValueError):
            env.step(action)
        assert env.steps == state["steps"] and env.hands == state["hands"]
        assert env.wall == state["wall"] and env.rewards == state["rewards"]
        assert (
            env.np_random.bit_generator.state == state["np_random"].bit_generator.state
        )
        np.testing.assert_array_equal(
            before["observation"], env.observe("player_0")["observation"]
        )


def test_truncation_and_reset():
    env = position(max_steps=1)
    env.step(game.CONTINUE)
    assert all(env.truncations.values()) and not any(env.terminations.values())
    assert not any(env.rewards.values())
    assert env.infos["player_0"]["truncation_reason"] == "max_steps"
    env.reset(seed=42)
    assert not any(env.truncations.values()) and not any(env.terminations.values())
    assert env.steps == 0 and not any(env.dead)
    assert inventory(env) == Counter(game.make_wall())


@pytest.mark.parametrize("seed", range(8))
def test_complete_games_conserve_tiles_private_observations_and_rewards(seed):
    env = game.raw_env(max_steps=None)
    env.reset(seed=seed)
    returns = dict.fromkeys(env.agents, 0)
    for a in env.agent_iter(3000):
        obs, reward, terminated, truncated, _ = env.last()
        returns[a] += reward
        assert inventory(env) == Counter(game.make_wall())
        assert all(n >= 0 for h in env.hands for n in h.values())
        assert env.observation_space(a).contains(obs)
        assert not truncated
        if not terminated:
            assert obs["action_mask"].any()
        env.step(None if terminated else env.action_space(a).sample(obs["action_mask"]))
    assert not env.agents
    assert returns == dict(zip(env.possible_agents, env.scores))
    assert sum(returns.values()) == 0


def test_api():
    api_test(game.env(), num_cycles=1000)


def test_seed():
    seed_test(game.env, num_cycles=1000)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_players": 3},
        {"num_players": True},
        {"num_players": 4.0},
        {"max_steps": 0},
        {"max_steps": True},
        {"render_mode": "rgb_array"},
    ],
)
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        game.raw_env(**kwargs)


def test_card_validation_and_immutability():
    for args in ((JOKER, 3), (-1, 3), (0, 0), (0, 7), (True, 3)):
        with pytest.raises(ValueError):
            Group(*args)
    with pytest.raises(ValueError):
        card((0, 3))
    with pytest.raises(ValueError):
        card((0, 2), (0, 2), (0, 2), (1, 4), (2, 4))
    with pytest.raises(ValueError):
        Card("Empty", ())
    with pytest.raises(ValueError):
        Card("Duplicates", STANDARD.patterns * 2)
    with pytest.raises(FrozenInstanceError):
        STANDARD.patterns[0].value = 100
    with pytest.raises(TypeError):
        game.raw_env(card="2026")
    for args in ((2, 0), (3, 3), (6, -1)):
        with pytest.raises(ValueError):
            game.claim_action(*args)
    with pytest.raises(ValueError):
        game.exchange_action(4, 0)
    with pytest.raises(ValueError):
        suited(BAMS, 0)


def test_render(capsys):
    env = position(render_mode="human")
    env.render()
    assert "American Mahjong" in capsys.readouterr().out
    assert "NMJL 2026" in position(render_mode="ansi").render()
    assert position().render() is None
    env.close()
