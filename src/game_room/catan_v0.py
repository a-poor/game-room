"""Catan base game, sequential AEC API. See docs/rules-catan.md."""

from collections import Counter
from enum import IntEnum
from functools import lru_cache
from typing import ClassVar

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers


class Resource(IntEnum):
    BRICK = 0
    LUMBER = 1
    WOOL = 2
    GRAIN = 3
    ORE = 4


class Development(IntEnum):
    KNIGHT = 0
    ROAD_BUILDING = 1
    YEAR_OF_PLENTY = 2
    MONOPOLY = 3
    VICTORY_POINT = 4


DESERT = 5
GENERIC_PORT = 5
ROAD_COST = np.array([1, 1, 0, 0, 0])
SETTLEMENT_COST = np.array([1, 1, 1, 1, 0])
CITY_COST = np.array([0, 0, 0, 2, 3])
DEVELOPMENT_COST = np.array([0, 0, 1, 1, 1])


def _topology():
    hexes = tuple((q, r) for r in range(-2, 3) for q in range(-2, 3) if abs(q + r) <= 2)
    corners = ((1, 1), (0, 2), (-1, 1), (-1, -1), (0, -2), (1, -1))
    hex_points = [
        tuple((2 * q + r + x, 3 * r + y) for x, y in corners) for q, r in hexes
    ]
    vertices = tuple(
        sorted({v for h in hex_points for v in h}, key=lambda v: (v[1], v[0]))
    )
    lookup = {v: i for i, v in enumerate(vertices)}
    hex_vertices = tuple(tuple(lookup[v] for v in h) for h in hex_points)
    edge_counts = Counter(
        tuple(sorted((h[i], h[(i + 1) % 6]))) for h in hex_vertices for i in range(6)
    )
    edges = tuple(sorted(edge_counts))
    vertex_edges = tuple(
        tuple(i for i, e in enumerate(edges) if v in e) for v in range(len(vertices))
    )
    neighbors = tuple(
        tuple(b if a == v else a for a, b in (edges[i] for i in vertex_edges[v]))
        for v in range(len(vertices))
    )
    boundary = {e for e, count in edge_counts.items() if count == 1}
    coast = []
    v = min(a for e in boundary for a in e)
    while boundary:
        edge = min(e for e in boundary if v in e)
        boundary.remove(edge)
        coast.append(edges.index(edge))
        v = edge[1] if edge[0] == v else edge[0]
    return hexes, vertices, hex_vertices, edges, vertex_edges, neighbors, tuple(coast)


HEX_COORDS, VERTEX_COORDS, HEX_VERTICES, EDGES, VERTEX_EDGES, NEIGHBORS, COAST = (
    _topology()
)
VERTEX_HEXES = tuple(
    tuple(h for h, vs in enumerate(HEX_VERTICES) if v in vs) for v in range(54)
)
HEX_NEIGHBORS = tuple(
    tuple(
        j
        for j, other in enumerate(HEX_VERTICES)
        if j != i and len(set(vs) & set(other)) == 2
    )
    for i, vs in enumerate(HEX_VERTICES)
)
PORT_EDGES = tuple(COAST[i] for i in (0, 3, 6, 10, 13, 16, 20, 23, 26))

# Fixed action IDs; smaller decisions reuse the resource and player ranges.
ROLL, END_TURN, BUY_DEVELOPMENT, TRADE, ACCEPT, REJECT, COUNTER, SUBMIT, CANCEL = range(
    9
)
PLAY_DEVELOPMENT = 9  # + 0..3
BUILD_ROAD = 13  # + edge 0..71
BUILD_SETTLEMENT = 85  # + vertex 0..53
BUILD_CITY = 139  # + vertex 0..53
MOVE_ROBBER = 193  # + hex 0..18
SELECT_PLAYER = 212  # + seat 0..3
SELECT_RESOURCE = 216  # + resource 0..4
MARITIME = 221  # + give * 5 + receive
OFFER_GIVE = 246  # + resource: add one to offered bundle
OFFER_WANT = 251  # + resource: add one to requested bundle
REMOVE_GIVE = 256
REMOVE_WANT = 261
INVITE_OFFER = 266  # ask the selected partner to make the first proposal
MARITIME_3 = 267  # optional 3:1 even when a specialized harbor offers 2:1
MARITIME_4 = 292  # 4:1 remains legal even when a better harbor is available
ACTION_COUNT = 317
PHASES = (
    "setup_settlement",
    "setup_road",
    "roll",
    "main",
    "discard",
    "robber",
    "victim",
    "free_roads",
    "plenty",
    "monopoly",
    "trade_partner",
    "trade_draft",
    "trade_response",
)


def make_development_deck():
    return [0] * 14 + [1] * 2 + [2] * 2 + [3] * 2 + [4] * 5


def env(**kwargs):
    return wrappers.OrderEnforcingWrapper(
        wrappers.AssertOutOfBoundsWrapper(raw_env(**kwargs))
    )


class raw_env(AECEnv):
    metadata: ClassVar[dict] = {
        "name": "catan_v0",
        "render_modes": ["ansi", "human"],
        "render_fps": 1,
        "is_parallelizable": False,
    }

    def __init__(self, num_players=4, max_steps=20000, render_mode=None):
        super().__init__()
        if type(num_players) is not int or num_players not in (3, 4):
            raise ValueError("num_players must be 3 or 4")
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be a positive integer or None")
        if render_mode not in (None, "ansi", "human"):
            raise ValueError("render_mode must be None, 'ansi', or 'human'")
        self.num_players, self.max_steps, self.render_mode = (
            num_players,
            max_steps,
            render_mode,
        )
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self._indices = {a: i for i, a in enumerate(self.possible_agents)}
        self.action_spaces = {
            a: spaces.Discrete(ACTION_COUNT) for a in self.possible_agents
        }
        # Header 24; map 38+54+108+72; bank 5; own cards 15;
        # public player rows 10 each; discards N; offer 10; last theft 3.
        self.observation_size = 329 + 11 * num_players
        self.observation_spaces = {
            a: spaces.Dict(
                {
                    "observation": spaces.Box(
                        0, np.inf, (self.observation_size,), dtype=np.float64
                    ),
                    "action_mask": spaces.MultiBinary(ACTION_COUNT),
                }
            )
            for a in self.possible_agents
        }
        self.np_random, _ = seeding.np_random()

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
            for i, a in enumerate(self.possible_agents):
                self.action_space(a).seed(seed + i)
        self.agents = self.possible_agents[:]
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {a: {} for a in self.agents}
        self._skip_agent_selection = None
        self.terrain = np.array(
            [0] * 3 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 3 + [DESERT]
        )
        self.np_random.shuffle(self.terrain)
        tokens = np.array([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12])
        self.numbers = np.zeros(19, dtype=int)
        while True:
            self.np_random.shuffle(tokens)
            self.numbers[self.terrain != DESERT] = tokens
            red = set(np.flatnonzero(np.isin(self.numbers, [6, 8])))
            if not any(j in red for h in red for j in HEX_NEIGHBORS[h]):
                break
        self.ports = np.full(54, -1, dtype=int)
        port_types = np.array([0, 1, 2, 3, 4, 5, 5, 5, 5])
        self.np_random.shuffle(port_types)
        for edge, kind in zip(PORT_EDGES, port_types):
            self.ports[list(EDGES[edge])] = kind
        self.roads = np.full(72, -1, dtype=int)
        self.owners = np.full(54, -1, dtype=int)
        self.buildings = np.zeros(54, dtype=int)
        self.resources = np.zeros((self.num_players, 5), dtype=int)
        self.bank = np.full(5, 19, dtype=int)
        self.development = np.zeros((self.num_players, 5), dtype=int)
        self.new_development = np.zeros((self.num_players, 5), dtype=int)
        self.played = np.zeros((self.num_players, 4), dtype=int)
        self.development_deck = make_development_deck()
        self.np_random.shuffle(self.development_deck)
        self.robber = int(np.flatnonzero(self.terrain == DESERT)[0])
        self.longest_road = self.largest_army = -1
        self.road_lengths = [0] * self.num_players
        # Random first seat is equivalent to the symmetric pregame dice contest.
        self.first_player = int(self.np_random.integers(self.num_players))
        seats = [
            (self.first_player + i) % self.num_players for i in range(self.num_players)
        ]
        self.setup_order = seats + seats[::-1]
        self.setup_index = 0
        self.setup_vertex = -1
        self.turn = self.first_player
        self.turn_number = self.steps = self.dice = 0
        self.phase = "setup_settlement"
        self.resume_phase = "roll"
        self.dev_played = False
        self.remaining = 0
        self.discards = np.zeros(self.num_players, dtype=int)
        self.winners = []
        self.last_theft = (-1, -1, -1)
        self._clear_trade()
        self._select(self.turn)

    def _clear_trade(self):
        self.proposer = self.partner = -1
        self.offer_give = np.zeros(5, dtype=int)
        self.offer_want = np.zeros(5, dtype=int)

    def _select(self, p):
        self.agent_selection = self.possible_agents[p]

    def _counts(self, p):
        return (
            int(np.count_nonzero((self.owners == p) & (self.buildings == 1))),
            int(np.count_nonzero((self.owners == p) & (self.buildings == 2))),
            int(np.count_nonzero(self.roads == p)),
        )

    def _score(self, p, private=True):
        settlements, cities, _ = self._counts(p)
        score = (
            settlements
            + 2 * cities
            + 2 * (self.longest_road == p)
            + 2 * (self.largest_army == p)
        )
        return int(score + (self.development[p, 4] if private else 0))

    def _check_win(self):
        if self.phase.startswith("setup") or self._score(self.turn) < 10:
            return False
        winner = self.possible_agents[self.turn]
        self.winners = [winner]
        for a in self.agents:
            self.terminations[a] = True
            self.rewards[a] = 1.0 if a == winner else -1.0
            self.infos[a] = {
                "winners": [winner],
                "scores": [self._score(p) for p in range(self.num_players)],
            }
        return True

    def _start_turn(self):
        self.phase = "roll"
        self.resume_phase = "roll"
        self.turn_number += 1
        self.dice = 0
        self.dev_played = False
        self.new_development[self.turn] = 0
        self.last_theft = (-1, -1, -1)
        self._select(self.turn)
        self._check_win()

    def _settlement_sites(self, p, setup=False):
        if self._counts(p)[0] >= 5:
            return []
        return [
            v
            for v in range(54)
            if self.owners[v] == -1
            and all(self.owners[n] == -1 for n in NEIGHBORS[v])
            and (setup or any(self.roads[e] == p for e in VERTEX_EDGES[v]))
        ]

    def _road_sites(self, p):
        if self._counts(p)[2] >= 15:
            return []
        return [
            e
            for e, vs in enumerate(EDGES)
            if self.roads[e] == -1
            and any(
                self.owners[v] == p
                or (
                    self.owners[v] == -1
                    and any(self.roads[n] == p for n in VERTEX_EDGES[v])
                )
                for v in vs
            )
        ]

    def _longest(self, p):
        owned = [e for e in range(72) if self.roads[e] == p]
        bits = {e: 1 << i for i, e in enumerate(owned)}

        @lru_cache(None)
        def walk(v, used):
            if used and self.owners[v] not in (-1, p):
                return 0
            best = 0
            for e in VERTEX_EDGES[v]:
                if e in bits and not used & bits[e]:
                    a, b = EDGES[e]
                    best = max(best, 1 + walk(b if a == v else a, used | bits[e]))
            return best

        return max((walk(v, 0) for e in owned for v in EDGES[e]), default=0)

    @staticmethod
    def _award(values, holder, minimum):
        best = max(values)
        if best < minimum:
            return -1
        leaders = [p for p, value in enumerate(values) if value == best]
        return holder if holder in leaders else leaders[0] if len(leaders) == 1 else -1

    def _update_awards(self):
        self.road_lengths = [self._longest(p) for p in range(self.num_players)]
        self.longest_road = self._award(self.road_lengths, self.longest_road, 5)
        self.largest_army = self._award(
            self.played[:, 0].tolist(), self.largest_army, 3
        )

    def _pay(self, p, cost):
        self.resources[p] -= cost
        self.bank += cost

    def _gain(self, p, resource, count=1):
        count = min(count, int(self.bank[resource]))
        self.bank[resource] -= count
        self.resources[p, resource] += count

    def _produce(self, roll):
        demand = np.zeros_like(self.resources)
        for h in range(19):
            if self.numbers[h] != roll or h == self.robber or self.terrain[h] == DESERT:
                continue
            for v in HEX_VERTICES[h]:
                if self.owners[v] >= 0:
                    demand[self.owners[v], self.terrain[h]] += self.buildings[v]
        for r in range(5):
            claimants = np.flatnonzero(demand[:, r])
            if self.bank[r] >= demand[:, r].sum() or len(claimants) == 1:
                for p in claimants:
                    self._gain(p, r, int(demand[p, r]))

    def _resolve_roll(self, roll):
        self.dice = roll
        self.resume_phase = "main"
        if roll == 7:
            totals = self.resources.sum(axis=1)
            self.discards = np.where(totals > 7, totals // 2, 0)
            self._next_discard()
        else:
            self._produce(roll)
            self.phase = "main"

    def _next_discard(self):
        for offset in range(self.num_players):
            p = (self.turn + offset) % self.num_players
            if self.discards[p]:
                self.phase = "discard"
                self._select(p)
                return
        self.phase = "robber"
        self._select(self.turn)

    def _victims(self):
        return sorted(
            {
                int(self.owners[v])
                for v in HEX_VERTICES[self.robber]
                if self.owners[v] not in (-1, self.turn)
            }
        )

    def _finish_effect(self):
        self.remaining = 0
        self.phase = self.resume_phase
        self._select(self.turn)

    def _trade_rates(self, p):
        ports = set(self.ports[self.owners == p])
        return [
            2 if r in ports else 3 if GENERIC_PORT in ports else 4 for r in range(5)
        ]

    def _can_trade(self, p):
        return bool(self.resources[p].sum()) and any(
            self.resources[q].sum() for q in range(self.num_players) if q != p
        )

    def _mask(self, agent):
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        if (
            agent not in self.agents
            or agent != self.agent_selection
            or self.terminations[agent]
            or self.truncations[agent]
        ):
            return mask
        p = self._indices[agent]
        if self.phase == "setup_settlement":
            mask[[BUILD_SETTLEMENT + v for v in self._settlement_sites(p, True)]] = 1
        elif self.phase == "setup_road":
            mask[
                [
                    BUILD_ROAD + e
                    for e in VERTEX_EDGES[self.setup_vertex]
                    if self.roads[e] < 0
                ]
            ] = 1
        elif self.phase in ("roll", "main"):
            if not self.dev_played:
                for d in range(4):
                    legal = self.development[p, d] > self.new_development[p, d]
                    if d == Development.ROAD_BUILDING:
                        legal = legal and bool(self._road_sites(p))
                    if d == Development.YEAR_OF_PLENTY:
                        legal = legal and bool(self.bank.sum())
                    mask[PLAY_DEVELOPMENT + d] = legal
            if self.phase == "roll":
                mask[ROLL] = 1
            else:
                mask[END_TURN] = 1
                if np.all(self.resources[p] >= ROAD_COST):
                    mask[[BUILD_ROAD + e for e in self._road_sites(p)]] = 1
                if np.all(self.resources[p] >= SETTLEMENT_COST):
                    mask[[BUILD_SETTLEMENT + v for v in self._settlement_sites(p)]] = 1
                if self._counts(p)[1] < 4 and np.all(self.resources[p] >= CITY_COST):
                    mask[
                        [
                            BUILD_CITY + int(v)
                            for v in np.flatnonzero(
                                (self.owners == p) & (self.buildings == 1)
                            )
                        ]
                    ] = 1
                mask[BUY_DEVELOPMENT] = bool(self.development_deck) and np.all(
                    self.resources[p] >= DEVELOPMENT_COST
                )
                for give, rate in enumerate(self._trade_rates(p)):
                    for want in range(5):
                        mask[MARITIME + give * 5 + want] = (
                            give != want
                            and self.resources[p, give] >= rate
                            and self.bank[want] > 0
                        )
                        for offset, alternate, available in (
                            (
                                MARITIME_3,
                                3,
                                GENERIC_PORT in self.ports[self.owners == p],
                            ),
                            (MARITIME_4, 4, True),
                        ):
                            mask[offset + give * 5 + want] = (
                                available
                                and alternate > rate
                                and give != want
                                and self.resources[p, give] >= alternate
                                and self.bank[want] > 0
                            )
                mask[TRADE] = self._can_trade(p)
        elif self.phase == "discard":
            mask[SELECT_RESOURCE : SELECT_RESOURCE + 5] = self.resources[p] > 0
        elif self.phase == "robber":
            mask[MOVE_ROBBER : MOVE_ROBBER + 19] = 1
            mask[MOVE_ROBBER + self.robber] = 0
        elif self.phase == "victim":
            mask[[SELECT_PLAYER + q for q in self._victims()]] = 1
        elif self.phase == "free_roads":
            mask[[BUILD_ROAD + e for e in self._road_sites(p)]] = 1
            mask[CANCEL] = 1
        elif self.phase == "plenty":
            mask[SELECT_RESOURCE : SELECT_RESOURCE + 5] = self.bank > 0
        elif self.phase == "monopoly":
            mask[SELECT_RESOURCE : SELECT_RESOURCE + 5] = 1
        elif self.phase == "trade_partner":
            mask[CANCEL] = 1
            for q in range(self.num_players):
                mask[SELECT_PLAYER + q] = q != p and self.resources[q].sum() > 0
        elif self.phase == "trade_draft":
            mask[CANCEL] = 1
            mask[INVITE_OFFER] = (
                p == self.turn
                and not self.offer_give.any()
                and not self.offer_want.any()
            )
            for r in range(5):
                mask[OFFER_GIVE + r] = (
                    self.offer_want[r] == 0
                    and self.offer_give[r] < self.resources[p, r]
                )
                # Requests use only public card totals, never the other hand's composition.
                mask[OFFER_WANT + r] = (
                    self.offer_give[r] == 0
                    and self.offer_want[r] < 19
                    and self.offer_want.sum() < self.resources[self.partner].sum()
                )
                mask[REMOVE_GIVE + r] = self.offer_give[r] > 0
                mask[REMOVE_WANT + r] = self.offer_want[r] > 0
            mask[SUBMIT] = self.offer_give.any() and self.offer_want.any()
        elif self.phase == "trade_response":
            mask[REJECT] = 1
            mask[ACCEPT] = np.all(self.resources[p] >= self.offer_want)
            mask[COUNTER] = bool(self.resources[p].sum())
        return mask

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        if (
            not self.action_space(agent).contains(action)
            or not self._mask(agent)[action]
        ):
            raise ValueError(f"Illegal action {action!r} for {agent} in {self.phase}")
        self._cumulative_rewards[agent] = 0
        self._clear_rewards()
        self.steps += 1
        self._step(int(action), self._indices[agent])
        if not self.winners:
            self._check_win()
        if (
            not self.winners
            and self.max_steps is not None
            and self.steps >= self.max_steps
        ):
            for a in self.agents:
                self.truncations[a] = True
                self.infos[a] = {"winners": []}
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def _step(self, action, p):
        if self.phase == "setup_settlement":
            v = action - BUILD_SETTLEMENT
            self.owners[v], self.buildings[v] = p, 1
            self.setup_vertex = v
            if self.setup_index >= self.num_players:
                for h in VERTEX_HEXES[v]:
                    if self.terrain[h] != DESERT:
                        self._gain(p, self.terrain[h])
            self.phase = "setup_road"
        elif self.phase == "setup_road":
            self.roads[action - BUILD_ROAD] = p
            self.setup_index += 1
            self._update_awards()
            if self.setup_index == len(self.setup_order):
                self.turn = self.first_player
                self._start_turn()
            else:
                self.turn = self.setup_order[self.setup_index]
                self._select(self.turn)
                self.phase = "setup_settlement"
        elif self.phase in ("roll", "main"):
            self._normal_action(action, p)
        elif self.phase == "discard":
            r = action - SELECT_RESOURCE
            self.resources[p, r] -= 1
            self.bank[r] += 1
            self.discards[p] -= 1
            self._next_discard()
        elif self.phase == "robber":
            self.robber = action - MOVE_ROBBER
            if self._victims():
                self.phase = "victim"
            else:
                self._finish_effect()
        elif self.phase == "victim":
            victim = action - SELECT_PLAYER
            total = int(self.resources[victim].sum())
            resource = -1
            if total:
                draw = int(self.np_random.integers(total))
                resource = int(
                    np.searchsorted(
                        np.cumsum(self.resources[victim]), draw, side="right"
                    )
                )
                self.resources[victim, resource] -= 1
                self.resources[p, resource] += 1
            self.last_theft = (p, victim, resource)
            self._finish_effect()
        elif self.phase == "free_roads":
            if action == CANCEL:
                self._finish_effect()
            else:
                self.roads[action - BUILD_ROAD] = p
                self.remaining -= 1
                self._update_awards()
                if not self.remaining or not self._road_sites(p):
                    self._finish_effect()
        elif self.phase == "plenty":
            self._gain(p, action - SELECT_RESOURCE)
            self.remaining -= 1
            if not self.remaining or not self.bank.sum():
                self._finish_effect()
        elif self.phase == "monopoly":
            r = action - SELECT_RESOURCE
            total = self.resources[:, r].sum()
            self.resources[:, r] = 0
            self.resources[p, r] = total
            self._finish_effect()
        elif self.phase == "trade_partner":
            if action == CANCEL:
                self._finish_trade()
            else:
                self.proposer, self.partner = p, action - SELECT_PLAYER
                self.phase = "trade_draft"
        elif self.phase == "trade_draft":
            if action == CANCEL:
                self._finish_trade()
            elif action == INVITE_OFFER:
                self.proposer, self.partner = self.partner, self.proposer
                self._select(self.proposer)
            elif action == SUBMIT:
                self.phase = "trade_response"
                self._select(self.partner)
            elif OFFER_GIVE <= action < OFFER_WANT:
                self.offer_give[action - OFFER_GIVE] += 1
            elif OFFER_WANT <= action < REMOVE_GIVE:
                self.offer_want[action - OFFER_WANT] += 1
            elif REMOVE_GIVE <= action < REMOVE_WANT:
                self.offer_give[action - REMOVE_GIVE] -= 1
            else:
                self.offer_want[action - REMOVE_WANT] -= 1
        elif self.phase == "trade_response":
            if action == COUNTER:
                self.proposer, self.partner = self.partner, self.proposer
                self.offer_give[:] = 0
                self.offer_want[:] = 0
                self.phase = "trade_draft"
            else:
                if action == ACCEPT:
                    delta = self.offer_want - self.offer_give
                    self.resources[self.proposer] += delta
                    self.resources[self.partner] -= delta
                self._finish_trade()

    def _finish_trade(self):
        self._clear_trade()
        self.phase = "main"
        self._select(self.turn)

    def _normal_action(self, action, p):
        if action == ROLL:
            self._resolve_roll(int(self.np_random.integers(1, 7, size=2).sum()))
        elif action == END_TURN:
            self.turn = (self.turn + 1) % self.num_players
            self._start_turn()
        elif action == BUY_DEVELOPMENT:
            self._pay(p, DEVELOPMENT_COST)
            d = self.development_deck.pop()
            self.development[p, d] += 1
            self.new_development[p, d] += 1
        elif action == TRADE:
            self.phase = "trade_partner"
        elif PLAY_DEVELOPMENT <= action < BUILD_ROAD:
            d = action - PLAY_DEVELOPMENT
            self.development[p, d] -= 1
            self.played[p, d] += 1
            self.dev_played = True
            self.resume_phase = self.phase
            self.phase = ("robber", "free_roads", "plenty", "monopoly")[d]
            self.remaining = 2 if d in (1, 2) else 0
            if d == 0:
                self._update_awards()
        elif BUILD_ROAD <= action < BUILD_SETTLEMENT:
            self._pay(p, ROAD_COST)
            self.roads[action - BUILD_ROAD] = p
            self._update_awards()
        elif BUILD_SETTLEMENT <= action < BUILD_CITY:
            self._pay(p, SETTLEMENT_COST)
            v = action - BUILD_SETTLEMENT
            self.owners[v], self.buildings[v] = p, 1
            self._update_awards()
        elif BUILD_CITY <= action < MOVE_ROBBER:
            self._pay(p, CITY_COST)
            self.buildings[action - BUILD_CITY] = 2
        else:
            if action >= MARITIME_4:
                give, want = divmod(action - MARITIME_4, 5)
                count = 4
            elif action >= MARITIME_3:
                give, want = divmod(action - MARITIME_3, 5)
                count = 3
            else:
                give, want = divmod(action - MARITIME, 5)
                count = self._trade_rates(p)[give]
            self.resources[p, give] -= count
            self.bank[give] += count
            self._gain(p, want)

    def observe(self, agent):
        p = self._indices[agent]
        header = [
            p,
            self._indices[self.agent_selection],
            self.turn,
            self.first_player,
            self.turn_number,
            self.steps,
            self.max_steps or 0,
            PHASES.index(self.phase),
            self.setup_index,
            self.setup_vertex + 1,
            self.dice,
            self.robber,
            self.longest_road + 1,
            self.largest_army + 1,
            int(self.dev_played),
            PHASES.index(self.resume_phase),
            self.remaining,
            len(self.development_deck),
            self.proposer + 1,
            self.partner + 1,
            int(bool(self.winners)),
            self.turn + 1 if self.winners else 0,
            self._score(p),
            10,
        ]
        public = []
        for q in range(self.num_players):
            public.extend(
                [
                    self.resources[q].sum(),
                    self.development[q].sum(),
                    self._score(q, private=bool(self.winners)),
                    *self._counts(q),
                    *self.played[q],
                ]
            )
        offer_visible = self.phase == "trade_response" or (
            self.phase == "trade_draft" and p == self.proposer
        )
        thief, victim, resource = self.last_theft
        theft = [thief + 1, victim + 1, resource + 1 if p in (thief, victim) else 0]
        vector = np.concatenate(
            [
                header,
                self.terrain,
                self.numbers,
                self.ports + 1,
                self.owners + 1,
                self.buildings,
                self.roads + 1,
                self.bank,
                self.resources[p],
                self.development[p],
                self.new_development[p],
                public,
                self.discards,
                self.offer_give if offer_visible else np.zeros(5),
                self.offer_want if offer_visible else np.zeros(5),
                theft,
            ]
        )
        return {
            "observation": vector.astype(np.float64),
            "action_mask": self._mask(agent),
        }

    def action_label(self, action):
        controls = {
            ROLL: "Roll dice",
            END_TURN: "End turn",
            BUY_DEVELOPMENT: "Buy development",
            TRADE: "Open trade",
            ACCEPT: "Accept trade",
            REJECT: "Reject trade",
            COUNTER: "Counteroffer",
            SUBMIT: "Submit offer",
            CANCEL: "Cancel / finish",
            INVITE_OFFER: "Invite partner's offer",
        }
        if action in controls:
            return controls[action]
        for offset, rate in ((MARITIME_3, 3), (MARITIME_4, 4)):
            if offset <= action < offset + 25:
                give, want = divmod(action - offset, 5)
                return (
                    f"Maritime {rate}:1 {Resource(give).name} for {Resource(want).name}"
                )
        if MARITIME <= action < OFFER_GIVE:
            give, want = divmod(action - MARITIME, 5)
            return f"Maritime {Resource(give).name} for {Resource(want).name}"
        for offset, size, label in [
            (PLAY_DEVELOPMENT, 4, "Play development"),
            (BUILD_ROAD, 72, "Road"),
            (BUILD_SETTLEMENT, 54, "Settlement"),
            (BUILD_CITY, 54, "City"),
            (MOVE_ROBBER, 19, "Robber hex"),
            (SELECT_PLAYER, 4, "Player"),
            (SELECT_RESOURCE, 5, "Resource"),
            (OFFER_GIVE, 5, "Offer +"),
            (OFFER_WANT, 5, "Request +"),
            (REMOVE_GIVE, 5, "Offer -"),
            (REMOVE_WANT, 5, "Request -"),
        ]:
            if offset <= action < offset + size:
                return f"{label} {action - offset}"
        raise ValueError("Unknown action")

    def render(self):
        if self.render_mode is None:
            return None
        lines = [
            f"Catan | turn {self.turn_number} | {self.agent_selection}: {self.phase} | dice {self.dice}"
        ]
        lines.append(
            "Hexes: "
            + "; ".join(
                f"{h}: {'DESERT' if r == DESERT else Resource(r).name}/{self.numbers[h]}"
                + (" ROBBER" if h == self.robber else "")
                for h, r in enumerate(self.terrain)
            )
        )
        lines.append(
            "Harbors (edge:type): "
            + ", ".join(f"{e}:{self.ports[EDGES[e][0]]}" for e in PORT_EDGES)
        )
        for p, a in enumerate(self.possible_agents):
            sites = [
                (int(v), int(self.buildings[v]))
                for v in np.flatnonzero(self.owners == p)
            ]
            lines.append(
                f"{a}: {self._score(p, private=bool(self.winners))} public points | "
                f"resources={self.resources[p].sum()} development={self.development[p].sum()} | "
                f"buildings={sites} roads={np.flatnonzero(self.roads == p).tolist()} "
                f"road length={self.road_lengths[p]} knights={self.played[p, 0]}"
            )
        lines.append(
            f"Longest road: {self.longest_road}; largest army: {self.largest_army}"
        )
        if self.winners:
            lines.append(f"Winner: {self.winners[0]}")
        result = "\n".join(lines)
        if self.render_mode == "human":
            print(result)
            return None
        return result

    def close(self):
        pass
