"""Harmonies with both board sides, solo play, and optional Nature's Spirits."""

from collections import Counter, deque
from typing import ClassVar

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from game_room._harmonies_data import ANIMALS

BLUE, GRAY, BROWN, GREEN, YELLOW, RED = range(1, 7)
TOKEN_COUNTS = (23, 23, 21, 19, 19, 15)
# Five flat-top columns of lengths 5,4,5,4,5; odd columns are shifted down.
CELLS = tuple((q, row - q // 2) for q in range(5) for row in range(5 - q % 2))
CELL_INDEX = {cell: i for i, cell in enumerate(CELLS)}
DIRECTIONS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
NEIGHBORS = tuple(
    tuple(
        CELL_INDEX[(q + dq, r + dr)]
        for dq, dr in DIRECTIONS
        if (q + dq, r + dr) in CELL_INDEX
    )
    for q, r in CELLS
)
END_TURN = 0
GROUP_OFFSET = 1
DRAFT_OFFSET = 6
PLACE_OFFSET = 11
SETTLE_OFFSET = PLACE_OFFSET + 6 * 23
SPIRIT_OFFSET = SETTLE_OFFSET + 42 * 23
SOLO_DISCARD_OFFSET = SPIRIT_OFFSET + 2
ACTION_COUNT = SOLO_DISCARD_OFFSET + 5


def place_action(color, cell):
    if not 1 <= color <= 6 or not 0 <= cell < 23:
        raise ValueError("invalid token placement")
    return PLACE_OFFSET + (color - 1) * 23 + cell


def settle_action(card, cell):
    if not 0 <= card < 42 or not 0 <= cell < 23:
        raise ValueError("invalid animal placement")
    return SETTLE_OFFSET + card * 23 + cell


def _rotations(pattern):
    rotated = []
    for _ in range(6):
        rotated.append(pattern)
        pattern = tuple((-r, q + r, stack) for q, r, stack in pattern)
    return tuple(rotated)


# Precompute in-bounds patterns for each animal and cube position.
PATTERNS = tuple(
    tuple(
        tuple(
            tuple((CELL_INDEX[(q + dq, r + dr)], stack) for dq, dr, stack in pattern)
            for pattern in _rotations(card.pattern)
            if all((q + dq, r + dr) in CELL_INDEX for dq, dr, _ in pattern)
        )
        for q, r in CELLS
    )
    for card in ANIMALS
)


def env(**kwargs):
    return wrappers.OrderEnforcingWrapper(
        wrappers.AssertOutOfBoundsWrapper(raw_env(**kwargs))
    )


def groups(cells):
    """Connected components of a set of cell indices."""
    unseen = set(cells)
    result = []
    while unseen:
        todo = [min(unseen)]
        unseen.remove(todo[0])
        group = []
        while todo:
            c = todo.pop()
            group.append(c)
            for n in NEIGHBORS[c]:
                if n in unseen:
                    unseen.remove(n)
                    todo.append(n)
        result.append(group)
    return result


class raw_env(AECEnv):
    metadata: ClassVar[dict] = {
        "name": "harmonies_v0",
        "render_modes": ["ansi", "human"],
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(
        self,
        num_players=2,
        board_side="A",
        nature_spirits=False,
        max_steps=10000,
        render_mode=None,
    ):
        super().__init__()
        if type(num_players) is not int or not 1 <= num_players <= 4:
            raise ValueError("num_players must be an integer from 1 to 4")
        if board_side not in ("A", "B"):
            raise ValueError("board_side must be 'A' or 'B'")
        if type(nature_spirits) is not bool:
            raise ValueError("nature_spirits must be a bool")
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be a positive integer or None")
        if render_mode not in (None, "ansi", "human"):
            raise ValueError("render_mode must be None, 'ansi', or 'human'")
        self.num_players, self.board_side, self.nature_spirits = (
            num_players,
            board_side,
            nature_spirits,
        )
        self.max_steps, self.render_mode = max_steps, render_mode
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self._indices = {a: i for i, a in enumerate(self.possible_agents)}
        self.action_spaces = {
            a: spaces.Discrete(ACTION_COUNT) for a in self.possible_agents
        }
        self.observation_spaces = {
            a: spaces.Dict(
                {
                    "observation": spaces.Box(
                        0, np.inf, (181 + 92 * num_players,), dtype=np.float64
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
        self.boards = [[() for _ in CELLS] for _ in self.agents]
        self.cubes = [[0] * 23 for _ in self.agents]
        self.owners = [-1] * 42
        self.progress = [0] * 42
        self.bag = [c for c, count in enumerate(TOKEN_COUNTS, 1) for _ in range(count)]
        self.np_random.shuffle(self.bag)
        self.market_size = 3 if self.num_players == 1 else 5
        self.market = [
            [self.bag.pop() for _ in range(3)] for _ in range(self.market_size)
        ]
        self.deck = list(range(32))
        self.np_random.shuffle(self.deck)
        self.cards = [self.deck.pop() for _ in range(self.market_size)]
        self.card_discards = []
        self.token_discards = []
        spirits = list(range(32, 42))
        self.np_random.shuffle(spirits)
        self.spirit_offers = [
            [spirits.pop(), spirits.pop()] if self.nature_spirits else []
            for _ in self.agents
        ]
        self.unused_spirits = spirits
        self.spirit_choices = [-1] * self.num_players
        self.first_player = int(self.np_random.integers(self.num_players))
        self.turn_player = self.first_player
        self.agent_selection = self.possible_agents[self.turn_player]
        self.steps = self.turns = 0
        self.round_number = 1
        self.final_round = False
        self.winners = []
        self.scores = None
        self._start_turn()

    def _start_turn(self):
        self.taken = self.drafted = False
        self.remaining = Counter()

    def _needs_spirit(self):
        return self.nature_spirits and self.spirit_choices[self.turn_player] == -1

    def _capacity(self, player):
        return (
            sum(
                owner == player and self.progress[i] < len(ANIMALS[i].points)
                for i, owner in enumerate(self.owners)
            )
            < 4
        )

    def can_place(self, player, color, cell):
        if self.cubes[player][cell]:
            return False
        stack = self.boards[player][cell]
        if not stack:
            return True
        if color == GRAY:
            return stack in ((GRAY,), (GRAY, GRAY))
        if color == BROWN:
            return stack == (BROWN,)
        if color == GREEN:
            return stack in ((BROWN,), (BROWN, BROWN))
        if color == RED:
            return stack in ((RED,), (BROWN,), (GRAY,))
        return False

    def can_settle(self, player, card, cell):
        if (
            self.owners[card] != player
            or self.progress[card] >= len(ANIMALS[card].points)
            or self.cubes[player][cell]
        ):
            return False
        board = self.boards[player]
        for pattern in PATTERNS[card][cell]:
            matches = True
            for target, expected in pattern:
                actual = board[target]
                if expected == (0, RED):
                    valid = (
                        len(actual) == 2
                        and actual[-1] == RED
                        and actual[0] in (RED, BROWN, GRAY)
                    )
                else:
                    valid = actual == expected
                if not valid:
                    matches = False
                    break
            if matches:
                return True
        return False

    def _mask(self, agent):
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        if (
            agent not in self.agents
            or agent != self.agent_selection
            or self.terminations[agent]
            or self.truncations[agent]
        ):
            return mask
        if self._needs_spirit():
            mask[SPIRIT_OFFSET : SPIRIT_OFFSET + 2] = 1
            return mask
        p = self.turn_player
        if not self.taken:
            for i, offer in enumerate(self.market):
                mask[GROUP_OFFSET + i] = bool(offer)
        for color in range(1, 7):
            if self.remaining[color]:
                for cell in range(23):
                    mask[place_action(color, cell)] = self.can_place(p, color, cell)
        if not self.drafted and self._capacity(p):
            for i, card in enumerate(self.cards):
                mask[DRAFT_OFFSET + i] = card != -1
        for card in range(42):
            if self.owners[card] == p and self.progress[card] < len(
                ANIMALS[card].points
            ):
                for cell in range(23):
                    mask[settle_action(card, cell)] = self.can_settle(p, card, cell)
        if self.taken and not sum(self.remaining.values()):
            mask[END_TURN] = 1
            if self.num_players == 1 and not self.drafted:
                for i, card in enumerate(self.cards):
                    mask[SOLO_DISCARD_OFFSET + i] = card != -1
        return mask

    def observe(self, agent):
        p = self._indices[agent]
        values = [
            p,
            self.turn_player,
            self.first_player,
            self.steps,
            self.max_steps or 0,
            self.round_number,
            int(self.board_side == "B"),
            int(self.nature_spirits),
            int(self.taken),
            int(self.drafted),
            int(self.final_round),
            len(self.bag),
            len(self.deck),
            self.num_players,
            self.turns,
            int(self._needs_spirit()),
        ]
        for i in range(5):
            counts = Counter(self.market[i]) if i < self.market_size else Counter()
            values += [counts[c] for c in range(1, 7)]
        values += [self.cards[i] + 1 if i < self.market_size else 0 for i in range(5)]
        values += [self.remaining[c] for c in range(1, 7)]
        discarded = Counter(self.token_discards)
        values += [discarded[c] for c in range(1, 7)]
        values += [o + 1 for o in self.owners]
        values += self.progress
        for player in range(self.num_players):
            for cell, stack in enumerate(self.boards[player]):
                values += (
                    list(stack) + [0] * (3 - len(stack)) + [self.cubes[player][cell]]
                )
        values += [
            self.spirit_offers[p][i] + 1 if i < len(self.spirit_offers[p]) else 0
            for i in range(2)
        ]
        values += [int(i in self.card_discards) for i in range(32)]
        return {
            "observation": np.array(values, dtype=np.float64),
            "action_mask": self._mask(agent),
        }

    def score_details(self, player):
        board = self.boards[player]
        top = [stack[-1] if stack else 0 for stack in board]
        trees = [i for i, t in enumerate(top) if t == GREEN]
        mountains = [i for i, t in enumerate(top) if t == GRAY]
        fields = groups(i for i, t in enumerate(top) if t == YELLOW)
        water = {i for i, t in enumerate(top) if t == BLUE}
        buildings = [
            i for i, stack in enumerate(board) if len(stack) == 2 and stack[-1] == RED
        ]
        tree_points = sum((0, 1, 3, 7)[len(board[i])] for i in trees)
        mountain_points = sum(
            (0, 1, 3, 7)[len(board[i])]
            for i in mountains
            if any(top[j] == GRAY for j in NEIGHBORS[i])
        )
        field_points = 5 * sum(len(g) >= 2 for g in fields)
        building_points = 5 * sum(
            len({top[n] for n in NEIGHBORS[i]} - {0}) >= 3 for i in buildings
        )
        if self.board_side == "B":
            water_points = 5 * len(groups(set(range(23)) - water))
        else:
            longest = 0
            # River length is the graph diameter in vertices, NOT the longest
            # simple path: shortcuts and loops use shortest paths (official FAQ).
            for start in sorted(water):
                distances = {start: 1}
                queue = deque([start])
                while queue:
                    cell = queue.popleft()
                    for n in NEIGHBORS[cell]:
                        if n in water and n not in distances:
                            distances[n] = distances[cell] + 1
                            queue.append(n)
                longest = max(longest, max(distances.values()))
            water_points = (0, 0, 2, 5, 8, 11, 15)[min(longest, 6)] + max(
                0, longest - 6
            ) * 4
        animal_points = sum(
            ANIMALS[i].points[self.progress[i] - 1]
            for i in range(32)
            if self.owners[i] == player and self.progress[i]
        )
        spirit_points = 0
        spirit = self.spirit_choices[player]
        if spirit >= 0 and self.progress[spirit]:
            if spirit == 32:
                spirit_points = sum(10 if len(g) >= 3 else 2 for g in fields)
            elif spirit == 33:
                spirit_points = 5 * len(fields)
            elif spirit == 34:
                spirit_points = 4 * sum(len(board[i]) >= 2 for i in trees)
            elif spirit == 35:
                spirit_points = sum(1 if len(board[i]) == 3 else 3 for i in trees)
            elif spirit == 36:
                spirit_points = 4 * len(groups(buildings))
            elif spirit == 37:
                spirit_points = 6 * sum(len(g) >= 2 for g in groups(buildings))
            elif spirit == 38:
                spirit_points = 4 * sum(len(board[i]) >= 2 for i in mountains)
            elif spirit == 39:
                spirit_points = sum(1 if len(board[i]) == 3 else 3 for i in mountains)
            elif spirit == 40:
                spirit_points = 7 * sum(len(g) >= 2 for g in groups(water))
            elif spirit == 41:
                spirit_points = 2 * len(water)
        return {
            "trees": tree_points,
            "mountains": mountain_points,
            "fields": field_points,
            "buildings": building_points,
            "water": water_points,
            "animals": animal_points,
            "spirits": spirit_points,
        }

    def _finish(self):
        details = [self.score_details(p) for p in range(self.num_players)]
        self.scores = [sum(d.values()) for d in details]
        counts = [sum(0 < c <= 32 for c in cubes) for cubes in self.cubes]
        ranks = list(zip(self.scores, counts))
        self.winners = [
            a for a, rank in zip(self.possible_agents, ranks) if rank == max(ranks)
        ]
        for a in self.agents:
            self.terminations[a] = True
            self.rewards[a] = (
                float(self.scores[0])
                if self.num_players == 1
                else 1.0
                if a in self.winners
                else -1.0
            )
            self.infos[a] = {
                "scores": dict(zip(self.possible_agents, self.scores)),
                "score_details": dict(zip(self.possible_agents, details)),
                "animal_counts": dict(zip(self.possible_agents, counts)),
                "winners": self.winners[:],
            }
            if self.num_players == 1:
                spirit = self.spirit_choices[0]
                group_spirit = spirit in (32, 33, 36, 37, 40)
                suns = sum(
                    self.scores[0] >= s for s in (40, 70, 90, 110, 130, 140, 150, 160)
                )
                self.infos[a]["suns"] = (
                    suns
                    + (self.board_side == "A")
                    + (2 if spirit == -1 else int(group_spirit))
                )

    def _end_turn(self):
        p = self.turn_player
        if sum(not stack for stack in self.boards[p]) <= 2:
            self.final_round = True
        if self.num_players == 1:
            self.token_discards.extend(c for offer in self.market for c in offer)
            self.market = [[] for _ in range(self.market_size)]
        for i, offer in enumerate(self.market):
            if not offer:
                if len(self.bag) < 3:
                    self.final_round = True
                self.market[i] = [self.bag.pop() for _ in range(min(3, len(self.bag)))]
        for i, card in enumerate(self.cards):
            if card == -1 and self.deck:
                self.cards[i] = self.deck.pop()
        self.turns += 1
        next_player = (p + 1) % self.num_players
        if next_player == self.first_player:
            if self.final_round:
                self._finish()
                return
            self.round_number += 1
        self.turn_player = next_player
        self.agent_selection = self.possible_agents[next_player]
        self._start_turn()

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        if (
            not self.action_space(agent).contains(action)
            or not self._mask(agent)[action]
        ):
            raise ValueError(f"Illegal action {action!r} for {agent}")
        action = int(action)
        self._cumulative_rewards[agent] = 0
        self._clear_rewards()
        self.steps += 1
        p = self.turn_player
        if SPIRIT_OFFSET <= action < SOLO_DISCARD_OFFSET:
            card = self.spirit_offers[p][action - SPIRIT_OFFSET]
            self.unused_spirits.extend(c for c in self.spirit_offers[p] if c != card)
            self.spirit_offers[p] = []
            self.spirit_choices[p] = card
            self.owners[card] = p
        elif action >= SOLO_DISCARD_OFFSET:
            i = action - SOLO_DISCARD_OFFSET
            self.card_discards.append(self.cards[i])
            self.cards[i] = -1
            self._end_turn()
        elif action >= SETTLE_OFFSET:
            card, cell = divmod(action - SETTLE_OFFSET, 23)
            self.cubes[p][cell] = card + 1
            self.progress[card] += 1
        elif action >= PLACE_OFFSET:
            color, cell = divmod(action - PLACE_OFFSET, 23)
            color += 1
            self.boards[p][cell] += (color,)
            self.remaining[color] -= 1
        elif action >= DRAFT_OFFSET:
            slot = action - DRAFT_OFFSET
            self.owners[self.cards[slot]] = p
            self.cards[slot] = -1
            self.drafted = True
        elif action >= GROUP_OFFSET:
            slot = action - GROUP_OFFSET
            self.remaining = Counter(self.market[slot])
            self.market[slot] = []
            self.taken = True
        else:
            self._end_turn()
        if (
            not any(self.terminations.values())
            and self.max_steps is not None
            and self.steps >= self.max_steps
        ):
            self.truncations = dict.fromkeys(self.agents, True)
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def render(self):
        if self.render_mode is None:
            return None
        lines = [
            f"Harmonies | side {self.board_side} | round {self.round_number}",
            f"Tokens: {self.market}",
            "Animals: "
            + ", ".join(ANIMALS[c].name if c >= 0 else "-" for c in self.cards),
        ]
        for p, a in enumerate(self.possible_agents):
            lines.append(f"{a}: {sum(self.score_details(p).values())} points")
            lines.append(
                " ".join(
                    f"{i}:{''.join(str(t) for t in stack) or '-'}"
                    f"{'*' if self.cubes[p][i] else ''}"
                    for i, stack in enumerate(self.boards[p])
                )
            )
            lines.append(
                ", ".join(
                    f"{ANIMALS[c].name} ({self.progress[c]}/{len(ANIMALS[c].points)})"
                    for c in range(42)
                    if self.owners[c] == p
                )
            )
        if self.winners:
            lines.append(f"Final scores: {self.scores}; winners: {self.winners}")
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self):
        pass
