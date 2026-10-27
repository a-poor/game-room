from collections import Counter, deque

import numpy as np
import pytest
from pettingzoo.test import api_test, seed_test

from game_room import flip7_v0 as game
from game_room.flip7_v0 import Card as C


def position(hands=None, draws=(), **kwargs):
    """Build a controlled public position with a physically consistent deck."""
    env = game.raw_env(**kwargs)
    env.reset(seed=1)
    env.scores = [0] * env.num_players
    env.hands = hands or [[] for _ in env.possible_agents]
    env.status = [game.ACTIVE] * env.num_players
    env.discard = []
    env.pending_card = None
    env.deferred = []
    env.flip7_player = None
    env.round_number = 1
    env.tasks = deque([(game.TURN, 1, 0)])
    env.agent_selection = "player_0"
    deck = game.make_deck()
    for card in [c for hand in env.hands for c in hand] + list(draws):
        deck.remove(card)
    env.deck = deck + list(reversed(draws))
    return env


def test_deck():
    counts = Counter(game.make_deck())
    assert sum(counts.values()) == 94
    assert counts[0] == 1
    assert all(counts[i] == i for i in range(1, 13))
    assert all(counts[i] == 1 for i in range(13, 19))
    assert all(counts[i] == 3 for i in range(19, 22))


def test_api():
    api_test(game.env(max_rounds=10), num_cycles=2000)


def test_seed():
    seed_test(lambda: game.env(max_rounds=10), num_cycles=1000)


def test_duplicate_bust_and_retained_cards():
    env = position([[12], [], []], [12])
    env.step(game.HIT)
    assert env.status[0] == game.BUSTED
    assert env.hands[0] == [12, 12]
    assert env._round_score(0) == 0
    assert env.agent_selection == "player_1"


def test_second_chance_consumed():
    env = position([[12, C.SECOND_CHANCE], [], []], [12])
    env.step(game.HIT)
    assert env.status[0] == game.ACTIVE
    assert env.hands[0] == [12]
    assert Counter(env.discard) == Counter([12, C.SECOND_CHANCE])


def test_second_chance_gift_and_ineligible_players():
    env = position([[C.SECOND_CHANCE], [C.SECOND_CHANCE], []], [C.SECOND_CHANCE])
    env.step(game.HIT)
    assert env.observe("player_0")["action_mask"].tolist() == [0, 0, 0, 0, 1]
    env.step(4)
    assert C.SECOND_CHANCE in env.hands[2]
    assert env.agent_selection == "player_1"


def test_second_chance_discard_without_recipient():
    env = position([[C.SECOND_CHANCE], [], []], [C.SECOND_CHANCE])
    env.status[1:] = [game.STAYED, game.STAYED]
    env.step(game.HIT)
    assert env.discard == [C.SECOND_CHANCE]


def test_stay_requires_any_card_and_invalid_action_is_atomic():
    env = position()
    before = env.observe("player_0")["observation"].copy()
    assert env.observe("player_0")["action_mask"].tolist() == [1, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        env.step(game.STAY)
    np.testing.assert_array_equal(before, env.observe("player_0")["observation"])
    env.hands[0] = [C.PLUS_2]
    assert env.observe("player_0")["action_mask"][game.STAY]


def test_freeze_and_turn_resume():
    env = position([[10], [11], []], [C.FREEZE])
    env.step(game.HIT)
    env.step(3)
    assert env.status[1] == game.FROZEN
    assert env._round_score(1) == 11
    assert env.agent_selection == "player_2"
    assert env.observe("player_1")["action_mask"].sum() == 0


def test_flip_three_deferred_freeze():
    env = position([[1], [], []], [C.FLIP_THREE, C.FREEZE, 2, 3])
    env.step(game.HIT)
    env.step(2)
    assert env.pending_card == C.FREEZE
    assert env.hands[0] == [1, C.FLIP_THREE, 2, 3]
    env.step(2)
    assert env.status[0] == game.FROZEN
    assert env._round_score(0) == 6
    assert env.agent_selection == "player_1"


def test_flip_three_bust_cancels_deferred_and_remaining_draws():
    env = position([[12], [], []], [C.FLIP_THREE, C.FREEZE, 12, 5])
    env.step(game.HIT)
    env.step(2)
    assert env.status[0] == game.BUSTED
    assert env.pending_card is None
    assert C.FREEZE in env.hands[0]
    assert env.deck[-1] == 5


def test_second_chance_is_usable_during_flip_three():
    env = position([[12], [], []], [C.FLIP_THREE, C.SECOND_CHANCE, 12, 2])
    env.step(game.HIT)
    env.step(2)
    assert env.status[0] == game.ACTIVE
    assert env.hands[0] == [12, C.FLIP_THREE, 2]


def test_chained_flip_three():
    env = position([[0], [], []], [C.FLIP_THREE, C.FLIP_THREE, 1, 2, 3, 4, 5])
    env.step(game.HIT)
    env.step(2)
    assert env.hands[0] == [0, C.FLIP_THREE, 1, 2]
    assert env.pending_card == C.FLIP_THREE
    env.step(3)
    assert env.hands[1] == [C.FLIP_THREE, 3, 4, 5]
    assert env.agent_selection == "player_1"


def test_flip7_ends_round_and_bonus_not_doubled():
    env = position(
        [[0, 1, 2, 3, 4, 5, C.TIMES_2, C.PLUS_10], [12], [11]], [6], target_score=1
    )
    env.step(game.HIT)
    assert env.scores == [67, 12, 11]
    assert env.rewards == {"player_0": 67, "player_1": 12, "player_2": 11}
    assert all(env.terminations.values())
    assert env.winners == ["player_0"]
    for agent in env.agent_iter():
        assert env.last()[2]
        env.step(None)
    assert not env.agents


def test_flip7_interrupts_forced_draws():
    env = position(
        [[0, 1, 2, 3, 4, 5], [], []], [C.FLIP_THREE, 6, 7, 8], target_score=1
    )
    env.step(game.HIT)
    env.step(2)
    assert env.scores[0] == 36
    assert env.deck[-1] == 7


def test_winner_only_at_round_end_and_shared_tie():
    env = position([[12], [12], []], target_score=12)
    env.step(game.STAY)
    assert not any(env.terminations.values())
    env.step(game.STAY)
    env.hands[2] = [C.TIMES_2]
    env.step(game.STAY)
    assert env.winners == ["player_0", "player_1"]


def test_modifier_only_score():
    env = position([[C.PLUS_10, C.TIMES_2], [], []])
    assert env._round_score(0) == 10


def test_next_round_keeps_deck_and_rotates_dealer():
    env = position([[1], [2], [3]], [4, 5, 6])
    before = len(env.deck)
    for _ in range(3):
        env.step(game.STAY)
    assert env.round_number == 2
    assert env.dealer == 1
    assert env.agent_selection == "player_2"
    assert env.hands == [[5], [6], [4]]
    assert len(env.deck) == before - 3
    assert env.discard == [1, 2, 3]


def test_reshuffle_excludes_table():
    env = position([[12], [], []])
    env.deck = []
    env.discard = [3]
    env.step(game.HIT)
    assert env.hands[0] == [12, 3]
    assert env.discard == []


def test_exhausted_deck_banks_round():
    env = position([[1], [2], [3]], max_rounds=1)
    env.deck = []
    env.discard = []
    env.step(game.HIT)
    assert env.scores == [1, 2, 3]
    assert all(env.truncations.values())


def test_second_chance_gift_resumes_remaining_forced_draws():
    env = position(
        [[12, C.SECOND_CHANCE], [], []],
        [C.FLIP_THREE, C.SECOND_CHANCE, 12, 3],
    )
    env.step(game.HIT)
    env.step(2)
    assert env.pending_card == C.SECOND_CHANCE
    env.step(3)
    assert C.SECOND_CHANCE in env.hands[1]
    assert env.hands[0] == [12, C.FLIP_THREE, 3]
    assert env.agent_selection == "player_1"


def test_deferred_actions_keep_draw_order_after_self_freeze():
    env = position([[1], [], []], [C.FLIP_THREE, C.FREEZE, C.FLIP_THREE, 2, 3, 4, 5])
    env.step(game.HIT)
    env.step(2)
    assert env.pending_card == C.FREEZE
    env.step(2)
    assert env.status[0] == game.FROZEN
    assert env.agent_selection == "player_0"
    assert env.pending_card == C.FLIP_THREE
    assert env.observe("player_0")["action_mask"].tolist() == [0, 0, 0, 1, 1]
    env.step(3)
    assert env.hands[1] == [C.FLIP_THREE, 3, 4, 5]


def test_opening_deal_pauses_for_actions(monkeypatch):
    # A no-op shuffle makes the rigged deck deterministic.
    class RNG:
        def shuffle(self, deck):
            pass

    monkeypatch.setattr(game.seeding, "np_random", lambda seed=None: (RNG(), 1))
    monkeypatch.setattr(game, "make_deck", lambda: [7, 6, 5, C.FREEZE])
    env = game.raw_env()
    env.reset()
    assert env.agent_selection == "player_1"
    assert env.pending_card == C.FREEZE
    env.step(4)  # Freeze player 2 before their opening deal; skip them.
    assert env.hands[0] == [5]
    assert env.hands[1] == []
    assert env.hands[2] == [C.FREEZE]
    assert env.agent_selection == "player_1"
    assert not env.observe("player_1")["action_mask"][game.STAY]


def test_truncation():
    env = position([[1], [2], [3]], max_rounds=1)
    for _ in range(3):
        env.step(game.STAY)
    assert all(env.truncations.values())
    assert not any(env.terminations.values())
    assert env.scores == [1, 2, 3]


@pytest.mark.parametrize("num_players", [3, 6, 18])
def test_random_games_conserve_cards_and_rewards(num_players):
    for seed in range(10):
        env = game.raw_env(num_players=num_players, max_rounds=100)
        env.reset(seed=seed)
        totals = dict.fromkeys(env.possible_agents, 0)
        for agent in env.agent_iter(20000):
            observation, reward, terminated, truncated, _ = env.last()
            totals[agent] += reward
            assert env.observation_space(agent).contains(observation)
            cards = env.deck + env.discard + [c for h in env.hands for c in h]
            cards += env.deferred
            cards += [c for k, _, c in env.tasks if k == game.RESOLVE]
            if env.pending_card is not None:
                cards.append(env.pending_card)
            assert Counter(cards) == Counter(game.make_deck())
            action = (
                None
                if terminated or truncated
                else env.action_space(agent).sample(observation["action_mask"])
            )
            env.step(action)
        assert not env.agents
        assert list(totals.values()) == env.scores


def test_hidden_deck_order_not_observed_and_observations_are_copies():
    env = position([[1], [2], [3]])
    before = env.observe("player_0")["observation"]
    env.deck.reverse()
    np.testing.assert_array_equal(before, env.observe("player_0")["observation"])
    before[:] = 99
    assert not np.all(env.observe("player_0")["observation"] == 99)


def test_render():
    env = game.env(render_mode="ansi")
    env.reset(seed=5)
    assert "Flip 7 | round 1" in env.render()
    env.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_players": 2},
        {"num_players": 19},
        {"target_score": 0},
        {"max_rounds": 0},
        {"render_mode": "rgb_array"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        game.raw_env(**kwargs)
