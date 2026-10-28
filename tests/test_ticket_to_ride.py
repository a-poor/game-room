from collections import Counter

import numpy as np
import pytest
from pettingzoo.test import api_test, seed_test

from game_room import ticket_to_ride_v0 as game


def position(**kwargs):
    e = game.raw_env(**kwargs)
    e.reset(seed=42)
    while e.phase == game.SETUP:
        e.step(game.KEEP_OFFSET + 15)
    e.turn_player = 0
    e.agent_selection = "player_0"
    return e


def route(a, b, color=None):
    endpoints = {game.CITY_INDEX[a], game.CITY_INDEX[b]}
    return next(
        i
        for i, r in enumerate(game.ROUTES)
        if set(r.cities) == endpoints and (color is None or r.color == color)
    )


def mask(e):
    return e.observe(e.agent_selection)["action_mask"]


def test_inventory_and_map():
    assert len(game.CITIES) == 36
    assert len(game.ROUTES) == 100
    assert len(game.TICKETS) == 33
    assert Counter(game.make_deck()) == Counter({**dict.fromkeys(range(8), 12), 8: 14})
    assert sum(r.length for r in game.ROUTES) == 309
    assert sum(bool(p) for p in game.PARALLELS) == 44
    assert all(1 <= r.length <= 6 and -1 <= r.color < 8 for r in game.ROUTES)
    assert len({tuple(sorted(t.cities)) for t in game.TICKETS}) == 33
    assert game.TICKETS[-3].points == 4
    assert game.TICKETS[-2].points == 8
    assert game.TICKETS[-1].points == 21
    assert game.ROUTES[route("Seattle", "Helena")].length == 6
    assert game.ROUTES[route("Los Angeles", "El Paso")].color == game.BLACK
    assert len(game.PARALLELS[route("Vancouver", "Seattle")]) == 1
    assert all(
        game.CITY_INDEX[c] in {x for r in game.ROUTES for x in r.cities}
        for c in game.CITIES
    )


@pytest.mark.parametrize("players", [2, 3, 4, 5])
def test_setup_minimum_two_and_sealed_offers(players):
    e = game.raw_env(num_players=players)
    e.reset(seed=1)
    first = e.first_player
    assert all(sum(h.values()) == 4 for h in e.hands)
    assert all(len(o) == 4 for o in e.offers)
    assert e.market.count(game.LOCOMOTIVE) < 3
    original_deck = e.ticket_deck[:]
    returned = []
    for _ in range(players):
        p = e.turn_player
        assert not mask(e)[game.KEEP_OFFSET + 1]
        assert mask(e)[game.KEEP_OFFSET + 3]
        kept = e.pending[:2]
        returned.extend(e.pending[2:])
        e.step(game.KEEP_OFFSET + 3)
        assert e.tickets[p] == kept
    assert e.turn_player == first and e.phase == game.TURN
    assert e.ticket_deck[-len(original_deck) :] == original_deck
    assert sorted(e.ticket_deck[: len(returned)]) == sorted(returned)
    assert not e.setup_returns and all(not o for o in e.offers)


def test_faceup_locomotive_ends_turn_and_blind_one_does_not():
    e = position()
    e.market = [game.LOCOMOTIVE, game.RED, game.BLUE, game.GREEN, game.BLACK]
    e.deck = [game.YELLOW] * 10
    before = e.hands[0][game.LOCOMOTIVE]
    e.step(game.MARKET_OFFSET)
    assert e.hands[0][game.LOCOMOTIVE] == before + 1
    assert e.turn_player == 1
    e.deck.append(game.LOCOMOTIVE)
    e.step(game.DRAW_BLIND)
    assert e.phase == game.SECOND_DRAW and e.turn_player == 1
    e.market[0] = game.LOCOMOTIVE
    assert not mask(e)[game.MARKET_OFFSET]
    e.step(game.DRAW_BLIND)
    assert e.turn_player == 2


def test_market_replacement_and_three_wild_sweep():
    e = position()
    e.market = [game.LOCOMOTIVE, game.LOCOMOTIVE, game.RED, game.BLUE, game.GREEN]
    e.deck = [game.YELLOW] * 5 + [game.LOCOMOTIVE]
    e.discard = []
    e.step(game.MARKET_OFFSET + 2)
    assert e.market == [game.YELLOW] * 5
    assert Counter(e.discard) == Counter(
        {game.LOCOMOTIVE: 3, game.BLUE: 1, game.GREEN: 1}
    )
    assert e.phase == game.SECOND_DRAW


def test_impossible_sweep_is_finite_and_resumes_when_payments_allow():
    e = position()
    e.market = [game.LOCOMOTIVE] * 5
    e.deck = [game.LOCOMOTIVE] * 3
    e.discard = []
    e._refill_market()
    assert e.market == [game.LOCOMOTIVE] * 5
    e.discard = [game.RED, game.GREEN, game.BLUE]
    e._refill_market()
    assert e.market.count(game.LOCOMOTIVE) < 3
    assert Counter(e.deck + e.market + e.discard) == Counter(
        {game.LOCOMOTIVE: 8, game.RED: 1, game.GREEN: 1, game.BLUE: 1}
    )


def test_recycling_and_second_draw_shortage():
    e = position()
    e.market = [-1] * 5
    e.deck = [game.RED]
    e.discard = []
    e.step(game.DRAW_BLIND)
    assert e.turn_player == 1 and e.phase == game.TURN
    e.discard = [game.BLUE]
    e.step(game.DRAW_BLIND)
    assert e.hands[1][game.BLUE] >= 1 and not e.discard
    assert e.turn_player == 2


def test_second_draw_cannot_take_only_remaining_faceup_wilds():
    e = position()
    e.market = [game.RED, game.LOCOMOTIVE, -1, -1, -1]
    e.deck = e.discard = []
    e.step(game.MARKET_OFFSET)
    assert e.turn_player == 1


def test_colored_route_payment_choices_score_and_train_cost():
    e = position()
    i = route("Los Angeles", "El Paso")
    e.hands[0] = Counter({game.BLACK: 6, game.LOCOMOTIVE: 3, game.RED: 12})
    e.step(game.CLAIM_OFFSET + i)
    assert e.phase == game.PAYMENT
    assert {a for a in np.flatnonzero(mask(e))} == {
        game.payment_action(game.BLACK, w) for w in range(4)
    }
    before = e.observe("player_0")["observation"].copy()
    with pytest.raises(ValueError):
        e.step(game.payment_action(game.RED, 0))
    np.testing.assert_array_equal(before, e.observe("player_0")["observation"])
    e.step(game.payment_action(game.BLACK, 2))
    assert e.owners[i] == 0 and e.trains[0] == 39
    assert e.route_scores[0] == 15 and e.hands[0][game.BLACK] == 2
    assert e.hands[0][game.LOCOMOTIVE] == 1
    assert e.turn_player == 1


def test_gray_routes_require_one_color_and_can_use_all_wild():
    e = position()
    i = route("Los Angeles", "Phoenix")
    e.hands[0] = Counter({game.RED: 2, game.BLUE: 1})
    assert not mask(e)[game.CLAIM_OFFSET + i]
    e.hands[0][game.LOCOMOTIVE] = 3
    e.step(game.CLAIM_OFFSET + i)
    assert mask(e)[game.payment_action(game.RED, 1)]
    assert mask(e)[game.payment_action(game.BLUE, 2)]
    assert mask(e)[game.payment_action(0, 3)]
    assert not mask(e)[game.payment_action(game.BLUE, 3)]
    e.step(game.payment_action(0, 3))
    assert e.hands[0][game.RED] == 2 and e.hands[0][game.BLUE] == 1


@pytest.mark.parametrize("players", [2, 3, 4, 5])
def test_double_routes(players):
    e = position(num_players=players)
    first = route("Seattle", "Portland")
    second = game.PARALLELS[first][0]
    e.owners[first] = 0
    for p in (0, 1):
        e.turn_player = p
        e.agent_selection = f"player_{p}"
        e.hands[p] = Counter({game.RED: 10})
        assert bool(mask(e)[game.CLAIM_OFFSET + second]) == (players >= 4 and p == 1)
    e.trains[1] = 0
    assert not mask(e)[game.CLAIM_OFFSET + second]


def test_ticket_draw_keep_and_return_order():
    e = position()
    old = e.tickets[0][:]
    e.ticket_deck = [0, 1, 2, 3, 4]
    e.step(game.DRAW_TICKETS)
    assert e.pending == [4, 3, 2]
    assert not mask(e)[game.KEEP_OFFSET]
    e.step(game.KEEP_OFFSET + 2)  # Keep only middle card.
    assert e.tickets[0] == old + [3]
    assert e.phase == game.RETURN_TICKETS
    e.step(game.RETURN_OFFSET + 2)  # Put 2 nearer the top than 4.
    e.step(game.RETURN_OFFSET)
    assert e.ticket_deck == [4, 2, 0, 1]
    assert e.turn_player == 1


def test_short_ticket_deck():
    e = position()
    e.ticket_deck = [5]
    e.step(game.DRAW_TICKETS)
    assert np.flatnonzero(mask(e)).tolist() == [game.KEEP_OFFSET + 1]
    e.step(game.KEEP_OFFSET + 1)
    assert not mask(e)[game.DRAW_TICKETS]


def test_connectivity_and_longest_trail_with_loop_and_branch():
    e = position()
    # Triangle through Seattle, Vancouver and Calgary, plus a Portland branch.
    for a, b in [
        ("Seattle", "Vancouver"),
        ("Vancouver", "Calgary"),
        ("Calgary", "Seattle"),
        ("Seattle", "Portland"),
    ]:
        e.owners[route(a, b)] = 0
    assert e.longest_path(0) == 9  # Revisits Seattle, never reuses an edge.
    assert e.connected(0, (game.CITY_INDEX["Portland"], game.CITY_INDEX["Calgary"]))
    assert not e.connected(1, (game.CITY_INDEX["Portland"], game.CITY_INDEX["Calgary"]))
    e.owners[route("Calgary", "Helena")] = 0
    # Two branches at different vertices cannot both join the full cycle in a trail.
    assert e.longest_path(0) == 12


def test_final_round_includes_trigger_player_once_more():
    e = position(num_players=3)
    i = route("Seattle", "Portland")
    e.trains[0] = 3
    e.hands[0] = Counter({game.RED: 1})
    e.step(game.CLAIM_OFFSET + i)
    e.step(game.payment_action(game.RED, 0))
    assert e.final_turns == 3 and e.turn_player == 1
    for p in (1, 2, 0):
        assert e.turn_player == p
        # A faceup locomotive is one complete turn, convenient for this scenario.
        e.market = [game.LOCOMOTIVE, game.RED, game.BLUE, game.GREEN, game.BLACK]
        e.deck = [game.YELLOW] * 10
        e.step(game.MARKET_OFFSET)
        if p != 0:
            assert not any(e.terminations.values())
    assert all(e.terminations.values()) and e.final_turns == 0


def test_ticket_scoring_and_shared_longest_bonus():
    e = position(num_players=2)
    e.tickets = [[25], [25]]  # Helena - Los Angeles, 8 points
    e.route_scores = [10, 10]
    e._finish()
    assert e.scores == [12, 12]  # 10 - 8 + tied longest bonus 10
    assert e.winners == e.possible_agents


@pytest.mark.parametrize(
    "lengths,completed,winners",
    [
        ([4, 4, 2], [2, 1, 0], [0]),
        ([4, 2, 2], [2, 2, 0], [0]),
        ([4, 4, 2], [2, 2, 0], [0, 1]),
    ],
)
def test_tiebreaks(lengths, completed, winners, monkeypatch):
    e = position(num_players=3)
    e.tickets = [[0] * count for count in completed]
    monkeypatch.setattr(e, "connected", lambda p, cities: True)
    monkeypatch.setattr(e, "longest_path", lambda p: lengths[p])
    e.route_scores = [
        100 - 21 * completed[p] - 10 * (lengths[p] == max(lengths)) for p in range(3)
    ]
    e.route_scores[2] -= 1
    e._finish()
    assert e.winners == [f"player_{p}" for p in winners]


def test_hidden_information_and_copies():
    e = position(render_mode="ansi")
    before = e.observe("player_0")["observation"]
    rendered = e.render()
    e.deck.reverse()
    e.ticket_deck.reverse()
    e.hands[1] = Counter({game.BLUE: sum(e.hands[1].values())})
    e.tickets[1] = list(reversed(e.tickets[1]))
    np.testing.assert_array_equal(before, e.observe("player_0")["observation"])
    assert rendered == e.render()
    assert not e.observe("player_1")["action_mask"].any()
    before[:] = 99
    assert not np.all(e.observe("player_0")["observation"] == 99)
    e.step(game.DRAW_TICKETS)
    other = e.observe("player_1")["observation"]
    e.pending.reverse()
    np.testing.assert_array_equal(other, e.observe("player_1")["observation"])
    assert e.infos == {a: {} for a in e.agents}


def test_truncation_and_forced_pass_stalemate():
    e = position(num_players=2)
    e.deck = e.discard = e.ticket_deck = []
    e.market = [-1] * 5
    e.hands = [Counter(), Counter()]
    for _ in range(2):
        assert np.flatnonzero(mask(e)).tolist() == [game.PASS]
        e.step(game.PASS)
    assert all(e.truncations.values()) and not e.winners
    assert not any(e.rewards.values())
    assert e.infos["player_0"]["truncation_reason"] == "no_legal_moves"
    e = game.raw_env(max_steps=1)
    e.reset(seed=1)
    e.step(game.KEEP_OFFSET + 15)
    assert all(e.truncations.values())


@pytest.mark.parametrize("players", [2, 3, 4, 5])
def test_complete_builder_games_conserve_resources(players):
    for seed in range(3):
        e = game.raw_env(num_players=players)
        e.reset(seed=seed)
        returns = dict.fromkeys(e.agents, 0)
        for a in e.agent_iter(4000):
            obs, reward, done, truncated, _ = e.last()
            returns[a] += reward
            assert e.observation_space(a).contains(obs)
            cards = e.deck + e.discard + [c for c in e.market if c >= 0]
            cards += [c for hand in e.hands for c, n in hand.items() for _ in range(n)]
            assert Counter(cards) == Counter(game.make_deck())
            tickets = (
                e.ticket_deck
                + e.setup_returns
                + [t for hand in e.tickets for t in hand]
            )
            tickets += [t for offer in e.offers for t in offer]
            if e.phase == game.CHOOSE_TICKETS:
                tickets += e.pending
            elif e.phase == game.RETURN_TICKETS:
                tickets += [e.pending[i] for i in e.return_slots]
            assert Counter(tickets) == Counter(range(33))
            for p in range(players):
                claimed = [r for owner, r in zip(e.owners, game.ROUTES) if owner == p]
                assert e.trains[p] + sum(r.length for r in claimed) == 45
                assert e.route_scores[p] == sum(
                    game.ROUTE_POINTS[r.length] for r in claimed
                )
            if done or truncated:
                action = None
            else:
                legal = np.flatnonzero(obs["action_mask"]).tolist()
                claims = [x for x in legal if game.CLAIM_OFFSET <= x < game.PAY_OFFSET]
                draws = [x for x in legal if x < game.DRAW_TICKETS]
                candidates = claims or draws or legal
                if e.phase == game.SETUP:
                    action = game.KEEP_OFFSET + 3
                else:
                    action = int(e.np_random.choice(candidates))
            e.step(action)
        assert not e.agents and e.winners
        assert all(returns[a] == (1 if a in e.winners else -1) for a in returns)


@pytest.mark.parametrize("players", [2, 5])
def test_api(players):
    api_test(game.env(num_players=players), num_cycles=1000)


def test_seed():
    seed_test(game.env, num_cycles=1000)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_players": 1},
        {"num_players": 6},
        {"num_players": True},
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
    assert "Ticket to Ride" in capsys.readouterr().out
    e.close()
    assert position().render() is None
    with pytest.raises(ValueError):
        game.payment_action(8, 0)


def test_each_player_sees_only_their_own_initial_ticket_offer():
    e = game.raw_env(num_players=3)
    e.reset(seed=9)
    offer_start = 171 + 4 * e.num_players
    for p, a in enumerate(e.possible_agents):
        obs = e.observe(a)["observation"]
        assert obs[offer_start : offer_start + 4].tolist() == [
            t + 1 for t in e.offers[p]
        ]
        other = (p + 1) % e.num_players
        e.offers[other].reverse()
        np.testing.assert_array_equal(obs, e.observe(a)["observation"])
        e.offers[other].reverse()


def test_completion_takes_precedence_over_step_limit():
    e = position(num_players=2)
    e.final_turns = 1
    e.max_steps = e.steps + 1
    e.market = [game.LOCOMOTIVE, game.RED, game.BLUE, game.GREEN, game.BLACK]
    e.deck = [game.YELLOW] * 10
    e.step(game.MARKET_OFFSET)
    assert all(e.terminations.values())
    assert not any(e.truncations.values())
    for a in e.agent_iter():
        assert not e.observe(a)["action_mask"].any()
        e.step(None)
    assert not e.agents
