from collections import Counter
from copy import deepcopy

import numpy as np
import pytest
from pettingzoo.test import api_test, seed_test

from game_room import monopoly_deal_v0 as game


def ids(kind, color=None):
    return [
        i
        for i, c in enumerate(game.CARDS)
        if c.kind == kind and (color is None or c.colors == (color,))
    ]


def position(hands=None, banks=None, groups=None, **kwargs):
    env = game.raw_env(**kwargs)
    env.reset(seed=4)
    env.hands = hands or [[] for _ in env.agents]
    env.banks = banks or [[] for _ in env.agents]
    env.groups = [[None] * game.GROUP_COUNT for _ in env.agents]
    for p, row in enumerate(groups or []):
        env.groups[p][: len(row)] = row
    env.discard = []
    used = [c for row in env.hands + env.banks for c in row]
    used += [c for row in env.groups for g in row if g for c in g.cards + g.buildings]
    assert len(used) == len(set(used))
    env.deck = [c for c in game.make_deck() if c not in used]
    return env


def group(color, count=None, buildings=()):
    return game.PropertyGroup(color, ids("property", color)[:count], list(buildings))


def play(env, card):
    env.step(game.PLAY + card)


def accept(env):
    env.step(game.ACCEPT)


def debt(env, card, target=1):
    play(env, card)
    env.step(game.SELECT_PLAYER + target)


def assert_conservation(env):
    cards = env.deck + env.discard
    cards += [c for row in env.hands + env.banks for c in row]
    cards += [c for row in env.groups for g in row if g for c in g.cards + g.buildings]
    assert Counter(cards) == Counter(game.make_deck())
    for row in env.groups:
        for g in row:
            if g:
                assert len(g.cards) <= game.SET_SIZES[g.color]
                assert all(g.color in game.CARDS[c].colors for c in g.cards)


def test_deck_and_initial_deal():
    assert game.CARD_COUNT == 106
    counts = Counter(c.kind for c in game.CARDS)
    assert counts == {
        "property": 39,
        "deal_breaker": 2,
        "forced_deal": 3,
        "sly_deal": 3,
        "just_say_no": 3,
        "debt_collector": 3,
        "birthday": 3,
        "double_rent": 2,
        "house": 3,
        "hotel": 2,
        "pass_go": 10,
        "rent": 10,
        "rent_any": 3,
        "money": 20,
    }
    env = game.raw_env(num_players=5)
    env.reset(seed=42)
    assert list(map(len, env.hands)) == [7, 5, 5, 5, 5]
    assert len(env.deck) == 79
    assert_conservation(env)


def test_api():
    api_test(game.env(max_steps=400), num_cycles=500)


def test_seed():
    seed_test(lambda: game.env(max_steps=500), num_cycles=500)


@pytest.mark.parametrize("num_players", [2, 3, 4, 5])
def test_random_games(num_players):
    env = game.raw_env(num_players=num_players, max_steps=700)
    for seed in range(3):
        env.reset(seed=seed)
        returns = dict.fromkeys(env.agents, 0.0)
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, _ = env.last()
            returns[agent] += reward
            assert env.observation_space(agent).contains(obs)
            if terminated or truncated:
                env.step(None)
                continue
            assert obs["action_mask"].sum() > 0
            action = env.action_space(agent).sample(obs["action_mask"])
            env.step(action)
            assert_conservation(env)
        assert returns == {
            a: (1.0 if a in env.winners else -1.0) if env.winners else 0.0
            for a in env.possible_agents
        }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_players": 1},
        {"num_players": 6},
        {"num_players": True},
        {"max_steps": 0},
        {"max_steps": True},
        {"render_mode": "bad"},
    ],
)
def test_configuration(kwargs):
    with pytest.raises(ValueError):
        game.raw_env(**kwargs)


def test_hidden_hands_and_deck():
    env = game.raw_env()
    env.reset(seed=2)
    before = env.observe("player_0")
    env.hands[1][0], env.deck[0] = env.deck[0], env.hands[1][0]
    env.deck.reverse()
    after = env.observe("player_0")
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert not env.observe("player_1")["action_mask"].any()


def test_invalid_action_is_atomic():
    env = game.raw_env()
    env.reset(seed=1)
    before = deepcopy(env.__dict__)
    illegal = np.flatnonzero(env.observe("player_0")["action_mask"] == 0)[0]
    with pytest.raises(ValueError):
        env.step(int(illegal))
    for key in (
        "hands",
        "deck",
        "discard",
        "groups",
        "phase",
        "steps",
        "plays",
        "rewards",
    ):
        assert env.__dict__[key] == before[key]


def test_three_plays_and_bank_is_permanent():
    cards = ids("pass_go")[:4]
    env = position([cards[:], [], []])
    for c in cards[:3]:
        env.step(game.BANK + c)
    assert env.plays == 3
    mask = env.observe("player_0")["action_mask"]
    assert np.flatnonzero(mask).tolist() == [game.END_TURN]
    assert env.banks[0] == cards[:3]
    assert not mask[game.PLAY + cards[0]]
    assert not mask[game.BANK + cards[3]]


def test_pass_go_and_empty_hand_draw():
    c = ids("pass_go")[0]
    env = position([[c], [], []])
    play(env, c)
    assert len(env.hands[0]) == 2
    assert env.plays == 1
    env.step(game.END_TURN)
    assert len(env.hands[1]) == 5


def test_discard_to_bottom_and_recycling():
    cards = ids("money")[:9]
    env = position([cards[:], [ids("pass_go")[0]], []])
    top = env.deck[-2:]
    env.step(game.END_TURN)
    assert env.phase == "discard"
    env.step(cards[0])
    env.step(cards[1])
    assert env.deck[:2] == [cards[1], cards[0]]
    assert env.hands[1][1:] == top[::-1]
    env.discard = env.deck
    env.deck = []
    before = len(env.discard)
    env._draw(0, 2)
    assert len(env.deck) == before - 2 and not env.discard
    assert_conservation(env)


def test_property_and_free_reorganization():
    wild = 28
    env = position([[wild], [], []], groups=[[group(0, 1)], [], []])
    assert not env.observe("player_0")["action_mask"][game.BANK + wild]
    play(env, wild)
    env.step(game.SELECT_COLOR)
    env.step(game.SELECT_GROUP)
    assert env.groups[0][0].complete
    env.step(game.MOVE + wild)
    env.step(game.SELECT_COLOR + 1)
    env.step(game.SELECT_GROUP + 1)
    assert env.groups[0][1].cards == [wild]
    assert not env.groups[0][0].complete
    assert env.plays == 1
    assert_conservation(env)


def test_payment_choice_overpay_and_bank_action_stays_money():
    c = ids("debt_collector")[0]
    money = ids("money")[-1]
    action_money = ids("deal_breaker")[0]
    env = position(
        [[c], [], []], [[], [money, action_money], []], [[], [group(0, 1)], []]
    )
    debt(env, c)
    accept(env)
    assert env.phase == "pay"
    assert env.observe("player_1")["action_mask"][0]
    env.step(action_money)
    assert env.banks[0] == [action_money]
    assert money in env.banks[1]
    assert env.phase == "play"
    assert not env.observe("player_0")["action_mask"][game.PLAY + action_money]


def test_partial_payment_and_unpayable_rainbow():
    c = ids("debt_collector")[0]
    env = position([[c], [], []], groups=[[], [game.PropertyGroup(0, [0, 37])], []])
    debt(env, c)
    accept(env)
    assert not env.observe("player_1")["action_mask"][37]
    env.step(0)
    assert env.phase == "play"
    assert env.groups[1][0].cards == [37]
    assert env.groups[0][0].cards == [0]
    assert_conservation(env)


def test_no_assets_still_offers_response():
    c = ids("debt_collector")[0]
    env = position([[c], [], []])
    debt(env, c)
    assert env.agent_selection == "player_1"
    assert env.phase == "response"
    accept(env)
    assert env.phase == "play"


@pytest.mark.parametrize("count", [1, 2, 3])
def test_just_say_no_chains(count):
    c = ids("debt_collector")[0]
    no = ids("just_say_no")
    money = ids("money")[-1]
    env = position([[c, no[1]], [no[0], no[2]], []], [[], [money], []])
    debt(env, c)
    env.step(no[0])
    if count >= 2:
        env.step(no[1])
    if count == 3:
        env.step(no[2])
    accept(env)
    if count == 2:
        assert env.phase == "pay"
        env.step(money)
        assert money in env.banks[0]
    else:
        assert money in env.banks[1]
    assert env.plays == 1
    assert env.phase == "play"
    assert_conservation(env)


def test_birthday_cancellation_only_protects_one_player():
    c = ids("birthday")[0]
    no = ids("just_say_no")[0]
    money = ids("money")[:2]
    env = position([[c], [no], []], [[], [money[0]], [money[1]]])
    play(env, c)
    env.step(no)
    accept(env)
    assert env.agent_selection == "player_2"
    accept(env)
    env.step(money[1])
    assert env.banks == [[money[1]], [money[0]], []]
    assert env.phase == "play"


def test_sly_deal_protects_full_sets():
    c = ids("sly_deal")[0]
    env = position([[c], [], []], groups=[[], [group(0), group(1, 1)], []])
    debt(env, c)
    mask = env.observe("player_0")["action_mask"]
    assert not mask[0] and mask[2]
    env.step(2)
    accept(env)
    assert env.groups[0][0].cards == [2]
    assert env.groups[1][1] is None
    assert_conservation(env)


def test_forced_deal_and_cancelled_exchange():
    c = ids("forced_deal")[0]
    no = ids("just_say_no")[0]
    for cancel in [False, True]:
        env = position([[c], [no], []], groups=[[group(0, 1)], [group(1, 1)], []])
        debt(env, c)
        env.step(2)
        env.step(0)
        if cancel:
            env.step(no)
        accept(env)
        assert env.groups[0][0].cards == ([0] if cancel else [2])
        assert env.groups[1][0].cards == ([2] if cancel else [0])
        assert_conservation(env)


def test_deal_breaker_moves_buildings_and_preserves_duplicate_groups():
    c = ids("deal_breaker")[0]
    buildings = [ids("house")[0], ids("hotel")[0]]
    env = position(
        [[c], [], []],
        groups=[[game.PropertyGroup(0, [28])], [group(0, buildings=buildings)], []],
    )
    debt(env, c)
    env.step(game.SELECT_GROUP)
    accept(env)
    assert env.groups[1][0] is None
    assert env.groups[0][1].cards == [0, 1]
    assert env.groups[0][1].buildings == buildings
    assert env._complete_colors(0) == {0}
    assert_conservation(env)


def test_buildings_require_house_and_eligible_full_set():
    house, hotel = ids("house")[0], ids("hotel")[0]
    env = position([[house, hotel], [], []], groups=[[group(0), group(8)], [], []])
    mask = env.observe("player_0")["action_mask"]
    assert mask[game.PLAY + house] and not mask[game.PLAY + hotel]
    play(env, house)
    assert not env.observe("player_0")["action_mask"][game.SELECT_GROUP + 1]
    env.step(game.SELECT_GROUP)
    play(env, hotel)
    env.step(game.SELECT_GROUP)
    assert env.groups[0][0].buildings == [house, hotel]
    assert env.plays == 2


def test_rent_building_bonus_and_double_twice():
    rent = ids("rent")[0]
    doubles = ids("double_rent")
    env = position(
        [[rent, *doubles], [], []],
        groups=[[group(0, buildings=[ids("house")[0], ids("hotel")[0]])], [], []],
    )
    play(env, rent)
    env.step(game.SELECT_GROUP)
    assert env.amount == 9
    env.step(doubles[0])
    env.step(doubles[1])
    assert env.amount == 36 and env.plays == 3
    accept(env)
    accept(env)
    assert env.agent_selection == "player_2"
    accept(env)
    assert env.phase == "play"
    assert_conservation(env)


def test_any_rent_one_target_and_no_standard_double():
    rent = ids("rent_any")[0]
    env = position([[rent, ids("double_rent")[0]], [], []], groups=[[group(0)], [], []])
    play(env, rent)
    env.step(game.SELECT_GROUP)
    assert np.flatnonzero(env.observe("player_0")["action_mask"]).tolist() == [
        game.ACCEPT
    ]
    accept(env)
    env.step(game.SELECT_PLAYER + 2)
    accept(env)
    assert env.phase == "play"


def test_payment_can_break_built_set_and_buildings_pay_as_cash():
    c = ids("debt_collector")[0]
    house = ids("house")[0]
    env = position([[c], [], []], groups=[[], [group(0, buildings=[house])], []])
    debt(env, c)
    accept(env)
    env.step(0)
    env.step(house)
    env.step(1)
    assert house in env.banks[0]
    assert env.groups[1][0] is None
    assert env.groups[0][0].complete
    assert_conservation(env)


def test_win_on_own_turn_and_dead_steps():
    last = ids("property", 2)[-1]
    env = position([[last], [], []], groups=[[group(0), group(1), group(2, 2)], [], []])
    play(env, last)
    env.step(game.SELECT_COLOR + 2)
    env.step(game.SELECT_GROUP + 2)
    assert env.winners == ["player_0"]
    assert env.rewards == {"player_0": 1, "player_1": -1, "player_2": -1}
    for agent in env.agent_iter():
        assert env.last()[2]
        assert not env.observe(agent)["action_mask"].any()
        env.step(None)
    assert not env.agents


def test_off_turn_completion_waits_for_turn():
    env = position(groups=[[], [group(0), group(1), group(2)], []])
    assert not env._check_win()
    env.step(game.END_TURN)
    assert env.winners == ["player_1"]


def test_truncation_reset_and_render(capsys):
    env = game.raw_env(max_steps=1, render_mode="ansi")
    env.reset(seed=1)
    assert "Monopoly Deal" in env.render()
    assert env.action_label(game.END_TURN) == "End turn"
    env.step(game.END_TURN)
    assert all(env.truncations.values())
    assert not any(env.terminations.values())
    env.reset(seed=1)
    assert not any(env.truncations.values())
    env.render_mode = "human"
    assert env.render() is None
    assert "Monopoly Deal" in capsys.readouterr().out
