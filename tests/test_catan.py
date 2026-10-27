import itertools
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest
from pettingzoo.test import api_test, seed_test

from game_room import catan_v0 as g


def position(**kwargs):
    e = g.raw_env(**kwargs)
    e.reset(seed=4)
    e.turn = 0
    e.phase = "main"
    e.resume_phase = "main"
    e.turn_number = 1
    e._select(0)
    return e


def grant(e, p, resources):
    values = np.array(resources)
    e.resources[p] += values
    e.bank -= values
    assert np.all(e.bank >= 0)


def develop(e, p, kind, count=1, new=False):
    for _ in range(count):
        e.development_deck.remove(kind)
    e.development[p, kind] += count
    if new:
        e.new_development[p, kind] += count


def building(e, p, vertex, size=1):
    e.owners[vertex], e.buildings[vertex] = p, size


def edge(a, b):
    return g.EDGES.index(tuple(sorted((a, b))))


def path(length, excluded=()):
    def walk(v, vertices):
        if len(vertices) == length + 1:
            return vertices
        for n in g.NEIGHBORS[v]:
            if n not in vertices and n not in excluded:
                result = walk(n, vertices + [n])
                if result:
                    return result
        return None

    for v in range(54):
        if v not in excluded:
            result = walk(v, [v])
            if result:
                return result
    raise AssertionError("No path found")


def road_path(e, p, vertices):
    for a, b in itertools.pairwise(vertices):
        e.roads[edge(a, b)] = p


def invariant(e):
    assert np.all(e.bank >= 0)
    assert np.all(e.resources >= 0)
    np.testing.assert_array_equal(e.bank + e.resources.sum(axis=0), [19] * 5)
    counts = Counter(e.development_deck)
    for d in range(5):
        counts[d] += int(e.development[:, d].sum())
        if d < 4:
            counts[d] += int(e.played[:, d].sum())
    assert counts == Counter(g.make_development_deck())
    assert np.all(e.new_development <= e.development)
    for p in range(e.num_players):
        a, b, c = e._counts(p)
        assert a <= 5 and b <= 4 and c <= 15
    for v in np.flatnonzero(e.owners >= 0):
        assert all(e.owners[n] == -1 for n in g.NEIGHBORS[v])


def test_topology_and_random_map():
    assert (len(g.HEX_COORDS), len(g.VERTEX_COORDS), len(g.EDGES), len(g.COAST)) == (
        19,
        54,
        72,
        30,
    )
    assert all(len(v) == 6 for v in g.HEX_VERTICES)
    assert all(len(n) in (2, 3) for n in g.NEIGHBORS)
    assert len({v for e in g.PORT_EDGES for v in g.EDGES[e]}) == 18
    for seed in range(20):
        e = g.raw_env()
        e.reset(seed=seed)
        assert Counter(e.terrain) == {0: 3, 1: 4, 2: 4, 3: 4, 4: 3, 5: 1}
        assert Counter(e.numbers) == {
            0: 1,
            2: 1,
            3: 2,
            4: 2,
            5: 2,
            6: 2,
            8: 2,
            9: 2,
            10: 2,
            11: 2,
            12: 1,
        }
        assert e.terrain[e.robber] == g.DESERT and e.numbers[e.robber] == 0
        assert Counter(e.ports[e.ports >= 0]) == {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 8}
        for h in np.flatnonzero(np.isin(e.numbers, [6, 8])):
            assert all(e.numbers[n] not in (6, 8) for n in g.HEX_NEIGHBORS[h])


@pytest.mark.parametrize("players", [3, 4])
def test_setup_snake_resources_and_adjoining_roads(players):
    e = g.raw_env(num_players=players)
    e.reset(seed=3)
    expected = e.setup_order[:]
    seen = []
    for i in range(2 * players):
        p = e._indices[e.agent_selection]
        seen.append(p)
        mask = e.observe(e.agent_selection)["action_mask"]
        v = int(np.flatnonzero(mask)[0]) - g.BUILD_SETTLEMENT
        before = e.resources[p].copy()
        e.step(g.BUILD_SETTLEMENT + v)
        gains = np.zeros(5, dtype=int)
        if i >= players:
            for h in g.VERTEX_HEXES[v]:
                if e.terrain[h] != g.DESERT:
                    gains[e.terrain[h]] += 1
        np.testing.assert_array_equal(e.resources[p], before + gains)
        actions = np.flatnonzero(e.observe(e.agent_selection)["action_mask"])
        assert all(v in g.EDGES[a - g.BUILD_ROAD] for a in actions)
        e.step(int(actions[0]))
        invariant(e)
    assert seen == expected
    assert e.phase == "roll" and e.turn == e.first_player
    assert all(e._counts(p) == (2, 0, 2) for p in range(players))


def test_api():
    api_test(g.env(max_steps=500), num_cycles=600)


def test_seed():
    seed_test(lambda: g.env(max_steps=500), num_cycles=600)


@pytest.mark.parametrize("players", [3, 4])
def test_random_episodes(players):
    for seed in range(4):
        e = g.raw_env(num_players=players, max_steps=1500)
        e.reset(seed=seed)
        returns = dict.fromkeys(e.agents, 0.0)
        for a in e.agent_iter():
            obs, reward, terminated, truncated, _ = e.last()
            returns[a] += reward
            assert e.observation_space(a).contains(obs)
            if terminated or truncated:
                e.step(None)
            else:
                assert obs["action_mask"].any()
                e.step(e.action_space(a).sample(obs["action_mask"]))
                invariant(e)
        expected = {
            a: (1 if a in e.winners else -1) if e.winners else 0
            for a in e.possible_agents
        }
        assert returns == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_players": 2},
        {"num_players": 5},
        {"num_players": True},
        {"max_steps": 0},
        {"max_steps": True},
        {"render_mode": "bad"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        g.raw_env(**kwargs)


def test_invalid_action_is_atomic():
    e = position()
    before = deepcopy(e.__dict__)
    with pytest.raises(ValueError):
        e.step(g.BUY_DEVELOPMENT)
    for key in ("resources", "bank", "owners", "buildings", "roads", "development"):
        np.testing.assert_array_equal(getattr(e, key), before[key])
    assert e.steps == before["steps"]
    assert e.np_random.bit_generator.state == before["np_random"].bit_generator.state


def test_hidden_hands_development_and_draft():
    e = position()
    grant(e, 1, [2, 0, 0, 0, 0])
    grant(e, 2, [0, 2, 0, 0, 0])
    develop(e, 1, g.Development.KNIGHT)
    develop(e, 2, g.Development.VICTORY_POINT)
    before = e.observe("player_0")
    e.resources[[1, 2]] = e.resources[[2, 1]]
    e.development[[1, 2]] = e.development[[2, 1]]
    e.development_deck.reverse()
    for key, value in before.items():
        np.testing.assert_array_equal(e.observe("player_0")[key], value)
    assert not e.observe("player_1")["action_mask"].any()
    grant(e, 0, [0, 0, 1, 0, 0])
    e.step(g.TRADE)
    e.step(g.SELECT_PLAYER + 1)
    before = e.observe("player_2")["observation"]
    e.step(g.OFFER_GIVE + 2)
    after = e.observe("player_2")["observation"]
    # Only the public decision count changed while the offer is being drafted.
    before[5] += 1
    np.testing.assert_array_equal(before, after)


def test_production_city_robber_and_shortages():
    e = position()
    e.numbers[:] = 0
    e.numbers[0] = 6
    e.terrain[0] = 0
    vs = g.HEX_VERTICES[0]
    building(e, 0, vs[0], 2)
    building(e, 1, vs[2])
    e.robber = 18
    e._produce(6)
    assert e.resources[0, 0] == 2 and e.resources[1, 0] == 1
    e.robber = 0
    before = e.resources.copy()
    e._produce(6)
    np.testing.assert_array_equal(e.resources, before)
    e.robber = 18
    grant(e, 2, [15, 0, 0, 0, 0])
    assert e.bank[0] == 1
    e._produce(6)
    assert e.bank[0] == 1  # Multiple claimants: nobody receives this resource.
    e.owners[vs[2]] = -1
    e.buildings[vs[2]] = 0
    e._produce(6)
    assert e.bank[0] == 0 and e.resources[0, 0] == 3


def test_seven_discards_rounded_down_then_moves_robber():
    e = position()
    grant(e, 0, [7, 0, 0, 0, 0])
    grant(e, 1, [0, 9, 0, 0, 0])
    grant(e, 2, [0, 0, 8, 0, 0])
    e._resolve_roll(7)
    assert e.discards.tolist() == [0, 4, 4, 0]
    for _ in range(4):
        assert e.agent_selection == "player_1"
        e.step(g.SELECT_RESOURCE + 1)
    for _ in range(4):
        assert e.agent_selection == "player_2"
        e.step(g.SELECT_RESOURCE + 2)
    assert e.phase == "robber" and e.agent_selection == "player_0"
    assert not e.observe("player_0")["action_mask"][g.MOVE_ROBBER + e.robber]
    h = (e.robber + 1) % 19
    e.step(g.MOVE_ROBBER + h)
    assert e.phase == "main"
    assert e.resources.sum(axis=1).tolist() == [7, 5, 4, 0]
    invariant(e)


def test_robbery_victim_choice_and_private_stolen_type():
    e = position()
    h = (e.robber + 1) % 19
    building(e, 1, g.HEX_VERTICES[h][0])
    building(e, 2, g.HEX_VERTICES[h][2])
    grant(e, 1, [0, 0, 3, 0, 0])
    e.phase = "robber"
    e.step(g.MOVE_ROBBER + h)
    mask = e.observe("player_0")["action_mask"]
    assert (
        mask[g.SELECT_PLAYER + 1] and mask[g.SELECT_PLAYER + 2]
    )  # Empty victim is legal.
    e.step(g.SELECT_PLAYER + 1)
    assert e.resources[0, 2] == 1 and e.resources[1, 2] == 2
    assert e.observe("player_0")["observation"][-1] == 3
    assert e.observe("player_1")["observation"][-1] == 3
    assert e.observe("player_2")["observation"][-1] == 0
    assert e.phase == "main"


def test_road_connectivity_and_opponent_blocking():
    e = position()
    vs = path(3)
    building(e, 0, vs[0])
    grant(e, 0, [4, 4, 0, 0, 0])
    e.step(g.BUILD_ROAD + edge(*vs[:2]))
    assert edge(vs[1], vs[2]) in e._road_sites(0)
    building(e, 1, vs[1])
    assert edge(vs[1], vs[2]) not in e._road_sites(0)
    assert edge(vs[2], vs[3]) not in e._road_sites(0)


def test_settlement_distance_and_city_returns_piece():
    e = position()
    vs = path(3)
    building(e, 0, vs[0])
    road_path(e, 0, vs)
    assert vs[1] not in e._settlement_sites(0)
    assert vs[2] in e._settlement_sites(0)
    grant(e, 0, [2, 2, 2, 4, 3])
    e.step(g.BUILD_SETTLEMENT + vs[2])
    assert e._counts(0) == (2, 0, 3)
    e.step(g.BUILD_CITY + vs[0])
    assert e._counts(0) == (1, 1, 3)
    assert e._score(0) == 3
    assert e.resources[0].tolist() == [1, 1, 1, 1, 0]


def test_piece_limits():
    e = position()
    for v in range(5):
        building(e, 0, v)
    assert not e._settlement_sites(0, True)
    e.roads[:15] = 0
    assert not e._road_sites(0)
    for v in range(5, 9):
        building(e, 0, v, 2)
    grant(e, 0, [0, 0, 0, 2, 3])
    assert not e.observe("player_0")["action_mask"][g.BUILD_CITY : g.MOVE_ROBBER].any()


def test_longest_road_cycle_branch_and_block():
    e = position()
    cycle = list(g.HEX_VERTICES[9])
    road_path(e, 0, cycle + [cycle[0]])
    assert e._longest(0) == 6
    branch = next(n for n in g.NEIGHBORS[cycle[0]] if n not in cycle)
    e.roads[edge(cycle[0], branch)] = 0
    assert e._longest(0) == 7
    building(e, 1, cycle[0])
    assert e._longest(0) == 6  # Can start/end at the block; cannot pass through it.
    building(e, 2, cycle[3])
    assert e._longest(0) == 3


def test_road_award_ties_and_interruption():
    e = position()
    a = path(6)
    b = path(5, excluded=a)
    road_path(e, 0, a)
    road_path(e, 1, b)
    e._update_awards()
    assert e.longest_road == 0
    building(e, 2, a[3])
    e._update_awards()
    assert e.road_lengths[0] == 3 and e.longest_road == 1
    assert e._award([5, 5, 3], 0, 5) == 0
    assert e._award([5, 5, 3], 2, 5) == -1
    assert e._award([4, 4, 3], 0, 5) == -1


def test_maritime_ports_and_bank_availability():
    e = position()
    grant(e, 0, [8, 6, 0, 0, 0])
    assert e._trade_rates(0) == [4] * 5
    e.step(g.MARITIME + 2)
    assert e.resources[0].tolist() == [4, 6, 1, 0, 0]
    generic = int(np.flatnonzero(e.ports == g.GENERIC_PORT)[0])
    building(e, 0, generic)
    assert e._trade_rates(0) == [3] * 5
    specific = int(np.flatnonzero(e.ports == 1)[0])
    building(e, 0, specific)
    assert e._trade_rates(0) == [3, 2, 3, 3, 3]
    e.step(g.MARITIME + 1 * 5 + 4)
    assert e.resources[0, 1] == 4 and e.resources[0, 4] == 1
    grant(e, 1, [0, 0, 0, 19, 0])
    assert not e.observe("player_0")["action_mask"][g.MARITIME + 3]


def offer(e, partner=1):
    e.step(g.TRADE)
    e.step(g.SELECT_PLAYER + partner)


def test_domestic_bundles_and_counteroffer():
    e = position()
    grant(e, 0, [2, 1, 0, 0, 0])
    grant(e, 1, [0, 0, 2, 0, 1])
    before = e.bank.copy()
    offer(e)
    e.step(g.OFFER_GIVE)
    e.step(g.OFFER_GIVE + 1)
    e.step(g.OFFER_WANT + 2)
    assert not e.observe("player_0")["action_mask"][g.OFFER_WANT]
    e.step(g.SUBMIT)
    assert e.agent_selection == "player_1"
    e.step(g.COUNTER)
    e.step(g.OFFER_GIVE + 4)
    e.step(g.OFFER_WANT)
    e.step(g.OFFER_WANT)
    e.step(g.SUBMIT)
    e.step(g.ACCEPT)
    assert e.resources[0].tolist() == [0, 1, 0, 0, 1]
    assert e.resources[1].tolist() == [2, 0, 2, 0, 0]
    np.testing.assert_array_equal(e.bank, before)
    assert e.phase == "main" and e.agent_selection == "player_0"


def test_offer_requests_do_not_inspect_opponent_resources():
    e = position()
    grant(e, 0, [1, 0, 0, 0, 0])
    grant(e, 1, [0, 1, 0, 0, 0])
    offer(e)
    e.step(g.OFFER_GIVE)
    e.step(g.OFFER_WANT + 4)  # Ask for ore even though opponent has only lumber.
    e.step(g.SUBMIT)
    assert not e.observe("player_1")["action_mask"][g.ACCEPT]
    e.step(g.REJECT)
    assert e.resources[0, 0] == 1 and e.resources[1, 1] == 1


def test_invited_offer_and_cancellation():
    e = position()
    grant(e, 0, [1, 0, 0, 0, 0])
    grant(e, 1, [0, 1, 0, 0, 0])
    offer(e)
    e.step(g.INVITE_OFFER)
    assert e.agent_selection == "player_1"
    e.step(g.OFFER_GIVE + 1)
    e.step(g.OFFER_WANT)
    e.step(g.REMOVE_WANT)
    assert not e.observe("player_1")["action_mask"][g.SUBMIT]
    e.step(g.CANCEL)
    assert e.agent_selection == "player_0" and e.phase == "main"
    assert e.resources[0, 0] == 1


def test_development_purchase_age_and_one_per_turn():
    e = position()
    grant(e, 0, [0, 0, 2, 2, 2])
    e.development_deck.remove(g.Development.MONOPOLY)
    e.development_deck.append(g.Development.MONOPOLY)
    e.step(g.BUY_DEVELOPMENT)
    assert not e.observe("player_0")["action_mask"][g.PLAY_DEVELOPMENT + 3]
    develop(e, 0, g.Development.MONOPOLY)
    e.step(g.PLAY_DEVELOPMENT + 3)
    e.step(g.SELECT_RESOURCE)
    assert e.development[0, 3] == e.new_development[0, 3] == 1
    assert not e.observe("player_0")["action_mask"][
        g.PLAY_DEVELOPMENT : g.BUILD_ROAD
    ].any()
    e._start_turn()
    assert e.observe("player_0")["action_mask"][g.PLAY_DEVELOPMENT + 3]
    invariant(e)


def test_knight_before_roll_and_largest_army():
    e = position()
    e.phase = "roll"
    develop(e, 0, g.Development.KNIGHT, 3)
    # Account for two previously played cards.
    e.development[0, 0] -= 2
    e.played[0, 0] = 2
    grant(e, 0, [9, 0, 0, 0, 0])
    e.step(g.PLAY_DEVELOPMENT)
    assert e.largest_army == 0 and e.phase == "robber"
    e.step(g.MOVE_ROBBER + (e.robber + 1) % 19)
    assert e.phase == "roll" and e.resources[0].sum() == 9
    assert e.observe("player_0")["action_mask"][g.ROLL]
    assert e._award([3, 3, 2], 0, 3) == 0
    assert e._award([3, 4, 2], 0, 3) == 1


def test_road_building_two_free_and_one_piece_left():
    for count in [0, 14]:
        e = position()
        vs = path(15)
        if count:
            road_path(e, 0, vs[:15])
        else:
            building(e, 0, vs[0])
        develop(e, 0, g.Development.ROAD_BUILDING)
        e.step(g.PLAY_DEVELOPMENT + 1)
        for _ in range(1 if count else 2):
            action = int(np.flatnonzero(e.observe("player_0")["action_mask"])[-1])
            assert g.BUILD_ROAD <= action < g.BUILD_SETTLEMENT
            e.step(action)
        assert e.phase == "main"
        assert e._counts(0)[2] == (15 if count else 2)
        assert e.resources[0].sum() == 0


def test_year_of_plenty_and_monopoly():
    e = position()
    develop(e, 0, g.Development.YEAR_OF_PLENTY)
    e.step(g.PLAY_DEVELOPMENT + 2)
    e.step(g.SELECT_RESOURCE + 3)
    e.step(g.SELECT_RESOURCE + 3)
    assert e.resources[0, 3] == 2 and e.phase == "main"
    e._start_turn()
    develop(e, 0, g.Development.MONOPOLY)
    grant(e, 1, [0, 0, 0, 3, 0])
    grant(e, 2, [0, 0, 0, 4, 0])
    e.step(g.PLAY_DEVELOPMENT + 3)
    bank = e.bank.copy()
    e.step(g.SELECT_RESOURCE + 3)
    assert e.resources[:, 3].tolist() == [9, 0, 0, 0]
    np.testing.assert_array_equal(e.bank, bank)
    assert e.phase == "roll"
    invariant(e)


def test_year_of_plenty_short_bank():
    e = position()
    grant(e, 1, [19, 19, 19, 19, 18])
    develop(e, 0, g.Development.YEAR_OF_PLENTY)
    e.step(g.PLAY_DEVELOPMENT + 2)
    assert np.flatnonzero(e.observe("player_0")["action_mask"]).tolist() == [
        g.SELECT_RESOURCE + 4
    ]
    e.step(g.SELECT_RESOURCE + 4)
    assert e.phase == "main" and e.resources[0, 4] == 1


def test_victory_point_purchase_wins_immediately_and_dead_steps():
    e = position(max_steps=1)
    for v in (0, 2, 4, 6):
        building(e, 0, v, 2)
    building(e, 0, 8)
    grant(e, 0, [0, 0, 1, 1, 1])
    e.development_deck.remove(4)
    e.development_deck.append(4)
    e.step(g.BUY_DEVELOPMENT)
    assert e.winners == ["player_0"]
    assert all(e.terminations.values()) and not any(e.truncations.values())
    assert e.rewards == {"player_0": 1, "player_1": -1, "player_2": -1, "player_3": -1}
    for a in e.agent_iter():
        assert not e.observe(a)["action_mask"].any()
        e.step(None)
    assert not e.agents


def test_off_turn_victory_waits_until_turn():
    e = position()
    for v in (0, 2, 4, 6):
        building(e, 1, v, 2)
    develop(e, 1, 4, 2)
    assert not e._check_win()
    e.step(g.END_TURN)
    assert e.winners == ["player_1"] and e.dice == 0


def test_truncation_reset_render_and_labels(capsys):
    e = position(max_steps=1, render_mode="ansi")
    assert "Catan" in e.render()
    assert e.action_label(g.ROLL) == "Roll dice"
    assert "Road" in e.action_label(g.BUILD_ROAD)
    assert "Maritime" in e.action_label(g.MARITIME + 1)
    e.step(g.END_TURN)
    assert all(e.truncations.values()) and not e.winners
    e.reset(seed=2)
    assert not any(e.truncations.values())
    e.render_mode = "human"
    assert e.render() is None
    assert "Catan" in capsys.readouterr().out


def builder_action(e, agent, mask, rng):
    """Small test policy that pursues buildings and cards without opponent secrets."""
    legal = np.flatnonzero(mask)
    p = e._indices[agent]
    if e.phase != "main":
        return int(rng.choice(legal))
    for lo, hi in [
        (g.BUILD_CITY, g.MOVE_ROBBER),
        (g.BUILD_SETTLEMENT, g.BUILD_CITY),
        (g.PLAY_DEVELOPMENT, g.BUILD_ROAD),
    ]:
        choices = legal[(legal >= lo) & (legal < hi)]
        if len(choices):
            return int(rng.choice(choices))
    if mask[g.BUY_DEVELOPMENT]:
        return g.BUY_DEVELOPMENT
    costs = []
    if e._counts(p)[0] and e._counts(p)[1] < 4:
        costs.append(g.CITY_COST)
    if e._settlement_sites(p):
        costs.append(g.SETTLEMENT_COST)
    if e.development_deck:
        costs.append(g.DEVELOPMENT_COST)
    if e._road_sites(p):
        costs.append(g.ROAD_COST)
    for cost in costs:
        for action in legal[(legal >= g.MARITIME) & (legal < g.OFFER_GIVE)]:
            give, want = divmod(int(action) - g.MARITIME, 5)
            if (
                e.resources[p, want] < cost[want]
                and e.resources[p, give] - e._trade_rates(p)[give] >= cost[give]
            ):
                return int(action)
    roads = legal[(legal >= g.BUILD_ROAD) & (legal < g.BUILD_SETTLEMENT)]
    return int(rng.choice(roads)) if len(roads) else g.END_TURN


@pytest.mark.parametrize("seed,players", [(0, 3), (1, 4)])
def test_complete_games_reach_winner_with_conservation(seed, players):
    e = g.raw_env(num_players=players, max_steps=2000)
    e.reset(seed=seed)
    rng = np.random.default_rng(seed)
    returns = dict.fromkeys(e.agents, 0.0)
    for agent in e.agent_iter():
        obs, reward, terminated, truncated, _ = e.last()
        returns[agent] += reward
        if terminated or truncated:
            e.step(None)
        else:
            e.step(builder_action(e, agent, obs["action_mask"], rng))
            invariant(e)
    assert len(e.winners) == 1
    assert e._score(e._indices[e.winners[0]]) >= 10
    assert returns == {a: 1 if a in e.winners else -1 for a in e.possible_agents}


def test_worse_maritime_rates_remain_legal_with_harbors():
    e = position()
    specific = int(np.flatnonzero(e.ports == 0)[0])
    building(e, 0, specific)
    grant(e, 0, [12, 0, 0, 0, 0])
    mask = e.observe("player_0")["action_mask"]
    assert mask[g.MARITIME + 1] and mask[g.MARITIME_4 + 1]
    assert not mask[g.MARITIME_3 + 1]
    generic = int(np.flatnonzero(e.ports == g.GENERIC_PORT)[0])
    building(e, 0, generic)
    assert e.observe("player_0")["action_mask"][g.MARITIME_3 + 1]
    e.step(g.MARITIME_3 + 1)
    assert e.resources[0].tolist() == [9, 1, 0, 0, 0]
    e.step(g.MARITIME_4 + 1)
    assert e.resources[0].tolist() == [5, 2, 0, 0, 0]
    e.step(g.MARITIME + 1)
    assert e.resources[0].tolist() == [3, 3, 0, 0, 0]
    assert "4:1" in e.action_label(g.MARITIME_4 + 1)
