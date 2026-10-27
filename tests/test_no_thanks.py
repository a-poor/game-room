from collections import Counter

import numpy as np
import pytest
from pettingzoo.test import api_test, seed_test

from game_room import no_thanks_v0 as game


def position(**kwargs):
    env = game.raw_env(**kwargs)
    env.reset(seed=42)
    env.agent_selection = "player_0"
    return env


@pytest.mark.parametrize("players,chips", [(3, 11), (4, 11), (5, 11), (6, 9), (7, 7)])
def test_setup(players, chips):
    env = position(num_players=players)
    assert env.chips == [chips] * players
    assert len(env.removed) == 9
    assert len(env.deck) == 23
    assert sorted(env.removed + env.deck + [env.card]) == list(range(3, 36))


def test_pass_take_and_same_player_continues():
    env = position()
    card, following = env.card, env.deck[-1]
    env.step(game.PASS)
    env.step(game.PASS)
    assert env.pot == 2 and env.chips == [10, 10, 11]
    env.step(game.TAKE)
    assert env.cards[2] == {card}
    assert env.chips == [10, 10, 13]
    assert env.pot == 0 and env.card == following
    assert env.agent_selection == "player_2"


def test_empty_chips_forces_take_and_invalid_is_atomic():
    env = position()
    env.chips[0] = 0
    before = env.observe("player_0")["observation"]
    assert env.observe("player_0")["action_mask"].tolist() == [1, 0]
    for action in (game.PASS, -1, 2, None):
        with pytest.raises(ValueError):
            env.step(action)
        np.testing.assert_array_equal(before, env.observe("player_0")["observation"])
    env.step(game.TAKE)


def test_chains_bridge_and_negative_scores():
    env = position()
    env.cards[0] = {3, 4, 6, 7, 20, 21}
    env.chips[0] = 10
    assert env.score(0) == 19
    env.cards[0].add(5)
    assert env.score(0) == 13
    env.chips[0] = 30
    assert env.score(0) == -7


def test_shared_winners_and_terminal_precedes_truncation():
    env = position(max_steps=1)
    env.deck = []
    env.card = 3
    env.cards = [set(), {3}, {20}]
    env.chips = [5, 5, 1]
    env.step(game.TAKE)
    assert env.scores == [-2, -2, 19]
    assert env.winners == ["player_0", "player_1"]
    assert list(env.rewards.values()) == [1, 1, -1]
    assert all(env.terminations.values())
    assert not any(env.truncations.values())
    for a in env.agent_iter():
        assert not env.observe(a)["action_mask"].any()
        env.step(None)
    assert not env.agents


def test_private_chips_and_hidden_deck():
    env = position(render_mode="ansi")
    before = env.observe("player_0")["observation"]
    rendered = env.render()
    env.chips[1] -= 1
    env.chips[2] += 1
    env.deck.reverse()
    env.removed.reverse()
    np.testing.assert_array_equal(before, env.observe("player_0")["observation"])
    assert env.render() == rendered
    assert env.infos == {a: {} for a in env.agents}
    before[:] = 99
    assert not np.all(env.observe("player_0")["observation"] == 99)
    assert not env.observe("player_1")["action_mask"].any()


def test_truncation():
    env = position(max_steps=1)
    env.step(game.PASS)
    assert all(env.truncations.values())
    assert not any(env.terminations.values())
    assert not any(env.rewards.values())
    assert env.winners == []


@pytest.mark.parametrize("players", range(3, 8))
def test_complete_games_conserve_cards_chips_and_rewards(players):
    for seed in range(5):
        env = game.raw_env(num_players=players, max_steps=None)
        env.reset(seed=seed)
        chips = sum(env.chips)
        returns = dict.fromkeys(env.agents, 0)
        for a in env.agent_iter(2000):
            obs, reward, terminated, truncated, _ = env.last()
            returns[a] += reward
            assert env.observation_space(a).contains(obs)
            assert sum(env.chips) + env.pot == chips
            cards = env.removed + env.deck + [c for hand in env.cards for c in hand]
            if env.card is not None:
                cards.append(env.card)
            assert Counter(cards) == Counter(range(3, 36))
            env.step(
                None
                if terminated or truncated
                else env.action_space(a).sample(obs["action_mask"])
            )
        assert not env.agents
        assert env.winners
        assert all(returns[a] == (1 if a in env.winners else -1) for a in returns)


@pytest.mark.parametrize("players", [3, 7])
def test_api(players):
    api_test(game.env(num_players=players), num_cycles=2000)


def test_seed():
    seed_test(game.env, num_cycles=2000)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_players": 2},
        {"num_players": 8},
        {"num_players": True},
        {"num_players": 3.5},
        {"max_steps": 0},
        {"max_steps": True},
        {"render_mode": "rgb_array"},
    ],
)
def test_config(kwargs):
    with pytest.raises(ValueError):
        game.raw_env(**kwargs)


def test_render(capsys):
    env = position(render_mode="human")
    env.render()
    assert "No Thanks!" in capsys.readouterr().out
    env.close()
    assert position().render() is None
