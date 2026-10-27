"""Kingdomino, 2017 base rules. See docs/rules-kingdomino.md."""

from typing import ClassVar

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

EMPTY, WHEAT, FOREST, WATER, GRASS, SWAMP, MINE, CASTLE = range(8)
# Domino number is its index + 1. Each half is (terrain, crowns).
# Inventory checked against the photograph linked in the rules reference.
DOMINOES = (
    *[((WHEAT, 0), (WHEAT, 0))] * 2,
    *[((FOREST, 0), (FOREST, 0))] * 4,
    *[((WATER, 0), (WATER, 0))] * 3,
    *[((GRASS, 0), (GRASS, 0))] * 2,
    ((SWAMP, 0), (SWAMP, 0)),
    ((WHEAT, 0), (FOREST, 0)),
    ((WHEAT, 0), (WATER, 0)),
    ((WHEAT, 0), (GRASS, 0)),
    ((WHEAT, 0), (SWAMP, 0)),
    ((FOREST, 0), (WATER, 0)),
    ((FOREST, 0), (GRASS, 0)),
    ((WHEAT, 1), (FOREST, 0)),
    ((WHEAT, 1), (WATER, 0)),
    ((WHEAT, 1), (GRASS, 0)),
    ((WHEAT, 1), (SWAMP, 0)),
    ((WHEAT, 1), (MINE, 0)),
    *[((FOREST, 1), (WHEAT, 0))] * 4,
    ((FOREST, 1), (WATER, 0)),
    ((FOREST, 1), (GRASS, 0)),
    *[((WATER, 1), (WHEAT, 0))] * 2,
    *[((WATER, 1), (FOREST, 0))] * 4,
    ((WHEAT, 0), (GRASS, 1)),
    ((WATER, 0), (GRASS, 1)),
    ((WHEAT, 0), (SWAMP, 1)),
    ((GRASS, 0), (SWAMP, 1)),
    ((MINE, 1), (WHEAT, 0)),
    ((WHEAT, 0), (GRASS, 2)),
    ((WATER, 0), (GRASS, 2)),
    ((WHEAT, 0), (SWAMP, 2)),
    ((GRASS, 0), (SWAMP, 2)),
    ((MINE, 2), (WHEAT, 0)),
    ((SWAMP, 0), (MINE, 2)),
    ((SWAMP, 0), (MINE, 2)),
    ((WHEAT, 0), (MINE, 3)),
)
INITIAL_DRAFT, PLACE, DRAFT = range(3)
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
PLACE_OFFSET = 4
DISCARD = PLACE_OFFSET + 9 * 9 * 4
ACTION_COUNT = DISCARD + 1


def placement_action(row, column, direction):
    """First half at (row, column), second in direction right/down/left/up."""
    if not (0 <= row < 9 and 0 <= column < 9 and 0 <= direction < 4):
        raise ValueError("placement coordinates or direction out of range")
    return PLACE_OFFSET + (row * 9 + column) * 4 + direction


def decode_placement(action):
    if not PLACE_OFFSET <= action < DISCARD:
        raise ValueError("not a placement action")
    cell, direction = divmod(int(action) - PLACE_OFFSET, 4)
    row, column = divmod(cell, 9)
    return row, column, direction


def env(**kwargs):
    return wrappers.OrderEnforcingWrapper(
        wrappers.AssertOutOfBoundsWrapper(raw_env(**kwargs))
    )


class raw_env(AECEnv):
    metadata: ClassVar[dict] = {
        "name": "kingdomino_v0",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self, num_players=4, max_steps=10000, render_mode=None):
        super().__init__()
        if type(num_players) is not int or not 2 <= num_players <= 4:
            raise ValueError("num_players must be an integer from 2 to 4")
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be a positive integer or None")
        if render_mode not in (None, "ansi", "human"):
            raise ValueError("render_mode must be None, 'ansi', or 'human'")
        self.num_players = num_players
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self._indices = {a: i for i, a in enumerate(self.possible_agents)}
        self.action_spaces = {
            a: spaces.Discrete(ACTION_COUNT) for a in self.possible_agents
        }
        self.observation_spaces = {
            a: spaces.Dict(
                {
                    "observation": spaces.Box(
                        0, np.inf, (78 + 162 * num_players,), dtype=np.float64
                    ),
                    "action_mask": spaces.MultiBinary(ACTION_COUNT),
                }
            )
            for a in self.possible_agents
        }
        self.np_random, _ = seeding.np_random()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

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
        self.boards = np.zeros((self.num_players, 9, 9, 2), dtype=np.int8)
        self.boards[:, 4, 4, 0] = CASTLE
        self.deck = list(range(1, 49))
        self.np_random.shuffle(self.deck)
        count = 12 * self.num_players
        self.removed, self.deck = self.deck[count:], self.deck[:count]
        self.row_size = 3 if self.num_players == 3 else 4
        self.current = self._draw_row()
        self.current_owners = [-1] * self.row_size
        self.next_row = []
        self.next_owners = []
        self.initial_order = list(range(self.num_players)) * (
            2 if self.num_players == 2 else 1
        )
        self.np_random.shuffle(self.initial_order)
        self.cursor = 0
        self.round_number = 1
        self.phase = INITIAL_DRAFT
        self.steps = 0
        self.discarded = []
        self.placed = [[] for _ in self.possible_agents]
        self.winners = []
        self.scores = None
        self.agent_selection = self.possible_agents[self.initial_order[0]]

    def _draw_row(self):
        return (
            sorted(self.deck.pop() for _ in range(self.row_size)) if self.deck else []
        )

    @property
    def pending_domino(self):
        return (
            self.current[self.cursor]
            if self.phase == PLACE and self.cursor < len(self.current)
            else 0
        )

    def _legal_placements(self, player):
        board = self.boards[player]
        occupied = np.argwhere(board[:, :, 0] != EMPTY)
        low = occupied.min(axis=0)
        high = occupied.max(axis=0)
        halves = DOMINOES[self.pending_domino - 1]
        legal = []
        for r in range(9):
            for c in range(9):
                if board[r, c, 0] != EMPTY:
                    continue
                for direction, (dr, dc) in enumerate(DIRECTIONS):
                    r2, c2 = r + dr, c + dc
                    if not (0 <= r2 < 9 and 0 <= c2 < 9) or board[r2, c2, 0] != EMPTY:
                        continue
                    if (
                        max(high[0], r, r2) - min(low[0], r, r2) >= 5
                        or max(high[1], c, c2) - min(low[1], c, c2) >= 5
                    ):
                        continue
                    connected = False
                    for (rr, cc), (terrain, _) in zip(((r, c), (r2, c2)), halves):
                        for nr, nc in DIRECTIONS:
                            y, x = rr + nr, cc + nc
                            if (
                                0 <= y < 9
                                and 0 <= x < 9
                                and board[y, x, 0] in (terrain, CASTLE)
                            ):
                                connected = True
                    if connected:
                        legal.append(placement_action(r, c, direction))
        return legal

    def _mask(self, agent):
        mask = np.zeros(ACTION_COUNT, dtype=np.int8)
        if (
            agent not in self.agents
            or agent != self.agent_selection
            or self.terminations[agent]
            or self.truncations[agent]
        ):
            return mask
        if self.phase == PLACE:
            legal = self._legal_placements(self._indices[agent])
            mask[legal if legal else [DISCARD]] = 1
        else:
            owners = (
                self.current_owners if self.phase == INITIAL_DRAFT else self.next_owners
            )
            for i, owner in enumerate(owners):
                mask[i] = owner == -1
        return mask

    def observe(self, agent):
        values = [
            self._indices[agent],
            self._indices[self.agent_selection],
            self.phase,
            self.cursor,
            self.round_number,
            len(self.deck),
            self.steps,
            self.max_steps or 0,
            self.pending_domino,
            self.num_players,
        ]
        values.extend(
            [p + 1 for p in self.initial_order] + [0] * (4 - len(self.initial_order))
        )
        for row, owners in (
            (self.current, self.current_owners),
            (self.next_row, self.next_owners),
        ):
            for i in range(4):
                values.extend([row[i], owners[i] + 1] if i < len(row) else [0, 0])
        values.extend(self.boards.ravel())
        values.extend(int(d in self.discarded) for d in range(1, 49))
        return {
            "observation": np.array(values, dtype=np.float64),
            "action_mask": self._mask(agent),
        }

    def score_details(self, player):
        """Return (points, largest territory, total crowns) for tie-breaking."""
        board = self.boards[player]
        seen = set()
        score = largest = 0
        for r in range(9):
            for c in range(9):
                terrain = int(board[r, c, 0])
                if terrain in (EMPTY, CASTLE) or (r, c) in seen:
                    continue
                stack = [(r, c)]
                seen.add((r, c))
                size = crowns = 0
                while stack:
                    rr, cc = stack.pop()
                    size += 1
                    crowns += int(board[rr, cc, 1])
                    for dr, dc in DIRECTIONS:
                        y, x = rr + dr, cc + dc
                        if (
                            0 <= y < 9
                            and 0 <= x < 9
                            and (y, x) not in seen
                            and board[y, x, 0] == terrain
                        ):
                            seen.add((y, x))
                            stack.append((y, x))
                score += size * crowns
                largest = max(largest, size)
        return score, largest, int(board[:, :, 1].sum())

    def _finish(self):
        details = [self.score_details(p) for p in range(self.num_players)]
        self.scores = [d[0] for d in details]
        self.winners = [
            a for a, d in zip(self.possible_agents, details) if d == max(details)
        ]
        for a in self.agents:
            self.terminations[a] = True
            self.rewards[a] = 1.0 if a in self.winners else -1.0
            self.infos[a] = {
                "scores": dict(zip(self.possible_agents, self.scores)),
                "score_details": dict(zip(self.possible_agents, details)),
                "winners": self.winners[:],
            }

    def _advance(self):
        self.cursor += 1
        if self.cursor == self.row_size:
            if not self.next_row:
                # Keep the final public row, with all its tiles marked consumed.
                self._finish()
                return
            self.current, self.current_owners = self.next_row, self.next_owners
            self.next_row = self._draw_row()
            self.next_owners = [-1] * len(self.next_row)
            self.cursor = 0
            self.round_number += 1
        self.phase = PLACE
        self.agent_selection = self.possible_agents[self.current_owners[self.cursor]]

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
        self._cumulative_rewards[agent] = 0
        self._clear_rewards()
        player = self._indices[agent]
        self.steps += 1
        action = int(action)
        if self.phase == INITIAL_DRAFT:
            self.current_owners[action] = player
            self.cursor += 1
            if self.cursor == self.row_size:
                self.next_row = self._draw_row()
                self.next_owners = [-1] * len(self.next_row)
                self.cursor = 0
                self.phase = PLACE
                self.agent_selection = self.possible_agents[self.current_owners[0]]
            else:
                self.agent_selection = self.possible_agents[
                    self.initial_order[self.cursor]
                ]
        elif self.phase == PLACE:
            domino = self.pending_domino
            if action == DISCARD:
                self.discarded.append(domino)
            else:
                r, c, direction = decode_placement(action)
                dr, dc = DIRECTIONS[direction]
                self.boards[player, r, c] = DOMINOES[domino - 1][0]
                self.boards[player, r + dr, c + dc] = DOMINOES[domino - 1][1]
                self.placed[player].append(domino)
            # A zero marks a consumed tile, so rows never duplicate held tiles.
            self.current[self.cursor] = 0
            if self.next_row:
                self.phase = DRAFT
            else:
                self._advance()
        else:
            self.next_owners[action] = player
            self._advance()
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
        symbols = ".WFLGSMC"
        lines = [
            f"Kingdomino | round {self.round_number} | phase {self.phase}",
            f"Current: {list(zip(self.current, self.current_owners))}",
            f"Next: {list(zip(self.next_row, self.next_owners))}",
        ]
        for p, a in enumerate(self.possible_agents):
            lines.append(f"{a}: {self.score_details(p)[0]} points")
            for row in self.boards[p]:
                lines.append(
                    " ".join(f"{symbols[t]}{c}" if t else ".." for t, c in row)
                )
        if self.winners:
            lines.append(f"Winners: {', '.join(self.winners)}")
        elif self.agents:
            lines.append(f"To act: {self.agent_selection}")
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self):
        pass
