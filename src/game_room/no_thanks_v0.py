"""No Thanks! base game (AMIGO 2024). See docs/rules-no-thanks.md."""

from typing import ClassVar

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

TAKE, PASS = range(2)


def env(**kwargs):
    return wrappers.OrderEnforcingWrapper(
        wrappers.AssertOutOfBoundsWrapper(raw_env(**kwargs))
    )


class raw_env(AECEnv):
    metadata: ClassVar[dict] = {
        "name": "no_thanks_v0",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self, num_players=3, max_steps=10000, render_mode=None):
        super().__init__()
        if type(num_players) is not int or not 3 <= num_players <= 7:
            raise ValueError("num_players must be an integer from 3 to 7")
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be a positive integer or None")
        if render_mode not in (None, "human", "ansi"):
            raise ValueError("render_mode must be None, 'human', or 'ansi'")
        self.num_players = num_players
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self._indices = {a: i for i, a in enumerate(self.possible_agents)}
        self.action_spaces = {a: spaces.Discrete(2) for a in self.possible_agents}
        self.observation_spaces = {
            a: spaces.Dict(
                {
                    "observation": spaces.Box(
                        0, np.inf, (8 + 33 * num_players,), dtype=np.float64
                    ),
                    "action_mask": spaces.MultiBinary(2),
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
        self.deck = list(range(3, 36))
        self.np_random.shuffle(self.deck)
        self.removed, self.deck = self.deck[:9], self.deck[9:]
        self.card = self.deck.pop()
        self.pot = 0
        self.chips = [
            11 if self.num_players <= 5 else 9 if self.num_players == 6 else 7
        ] * self.num_players
        self.cards = [set() for _ in self.possible_agents]
        self.steps = 0
        self.winners = []
        self.scores = None
        self.agent_selection = self.possible_agents[
            int(self.np_random.integers(self.num_players))
        ]

    def score(self, player):
        """Penalty points: lower is better. Includes private counters."""
        cards = self.cards[player]
        return sum(c for c in cards if c - 1 not in cards) - self.chips[player]

    def _mask(self, agent):
        mask = np.zeros(2, dtype=np.int8)
        if (
            agent in self.agents
            and agent == self.agent_selection
            and not self.terminations[agent]
            and not self.truncations[agent]
        ):
            mask[TAKE] = 1
            mask[PASS] = self.chips[self._indices[agent]] > 0
        return mask

    def observe(self, agent):
        player = self._indices[agent]
        values = [
            player,
            self._indices[self.agent_selection],
            self.card or 0,
            self.pot,
            len(self.deck),
            self.chips[player],
            self.steps,
            self.max_steps or 0,
        ]
        values.extend(int(c in cards) for cards in self.cards for c in range(3, 36))
        return {
            "observation": np.array(values, dtype=np.float64),
            "action_mask": self._mask(agent),
        }

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
        if action == PASS:
            self.chips[player] -= 1
            self.pot += 1
            self.agent_selection = self.possible_agents[(player + 1) % self.num_players]
        else:
            self.cards[player].add(self.card)
            self.chips[player] += self.pot
            self.pot = 0
            self.card = self.deck.pop() if self.deck else None
            if self.card is None:
                self.scores = [self.score(p) for p in range(self.num_players)]
                self.winners = [
                    a
                    for a, s in zip(self.possible_agents, self.scores)
                    if s == min(self.scores)
                ]
                for a in self.agents:
                    self.terminations[a] = True
                    self.rewards[a] = 1.0 if a in self.winners else -1.0
                    self.infos[a] = {
                        "scores": dict(zip(self.possible_agents, self.scores)),
                        "winners": self.winners[:],
                    }
        if (
            self.card is not None
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
            f"No Thanks! | card {self.card} | pot {self.pot} | deck {len(self.deck)}"
        ]
        for i, a in enumerate(self.possible_agents):
            lines.append(f"{a}: cards {sorted(self.cards[i])}")
        if self.scores is not None:
            lines.append(
                f"Scores (lower is better): {self.scores}; winners: {self.winners}"
            )
        elif self.agents:
            lines.append(f"{self.agent_selection}: take / pass")
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self):
        pass
