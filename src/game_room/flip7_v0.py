"""Flip 7 (rules edition 3.1), a sequential, imperfect-information game.

See docs/rules-flip7.md for the observation layout and rule conventions.
"""

from collections import Counter, deque
from enum import IntEnum
from typing import ClassVar

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers


class Card(IntEnum):
    """Numbers use IDs 0..12; the remaining IDs describe special cards."""

    PLUS_2 = 13
    PLUS_4 = 14
    PLUS_6 = 15
    PLUS_8 = 16
    PLUS_10 = 17
    TIMES_2 = 18
    FREEZE = 19
    FLIP_THREE = 20
    SECOND_CHANCE = 21


HIT = 0
STAY = 1
TARGET_OFFSET = 2
CARD_COUNT = 22
# Scheduler instructions are public; no future card identities are stored here.
DEAL, FORCED_DRAW, END_THREE, RESOLVE, TURN = range(1, 6)
ACTIVE, STAYED, BUSTED, FROZEN = range(4)


def make_deck() -> list[int]:
    """Return one unshuffled 94-card deck."""
    return (
        [0]
        + [number for number in range(1, 13) for _ in range(number)]
        + list(range(Card.PLUS_2, Card.TIMES_2 + 1))
        + [
            card
            for card in range(Card.FREEZE, Card.SECOND_CHANCE + 1)
            for _ in range(3)
        ]
    )


def env(**kwargs):
    """Create a wrapped AEC environment; illegal actions raise an exception."""
    return wrappers.OrderEnforcingWrapper(
        wrappers.AssertOutOfBoundsWrapper(raw_env(**kwargs))
    )


class raw_env(AECEnv):
    metadata: ClassVar[dict] = {
        "name": "flip7_v0",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(
        self,
        num_players: int = 3,
        target_score: int = 200,
        max_rounds: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        if (
            isinstance(num_players, bool)
            or not isinstance(num_players, int)
            or not 3 <= num_players <= 18
        ):
            raise ValueError("num_players must be an integer from 3 to 18")
        if (
            isinstance(target_score, bool)
            or not isinstance(target_score, int)
            or target_score < 1
        ):
            raise ValueError("target_score must be a positive integer")
        if max_rounds is not None and (
            isinstance(max_rounds, bool)
            or not isinstance(max_rounds, int)
            or max_rounds < 1
        ):
            raise ValueError("max_rounds must be a positive integer or None")
        if render_mode not in (None, "ansi", "human"):
            raise ValueError("render_mode must be None, 'ansi', or 'human'")
        self.num_players = num_players
        self.target_score = target_score
        self.max_rounds = max_rounds
        self.render_mode = render_mode
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self._indices = {agent: i for i, agent in enumerate(self.possible_agents)}
        self.action_spaces = {
            agent: spaces.Discrete(TARGET_OFFSET + num_players)
            for agent in self.possible_agents
        }
        # Header + per-seat scores/status/cards + discard + scheduler + deferred.
        self._observation_size = 10 + num_players * 24 + CARD_COUNT + 128 * 3 + 3
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        0, np.inf, (self._observation_size,), dtype=np.float64
                    ),
                    "action_mask": spaces.MultiBinary(TARGET_OFFSET + num_players),
                }
            )
            for agent in self.possible_agents
        }
        self.np_random, _ = seeding.np_random()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        # Accept unused options for compatibility with PettingZoo's reset API.
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
            for i, agent in enumerate(self.possible_agents):
                self.action_space(agent).seed(seed + i)
        self.agents = self.possible_agents[:]
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {agent: {} for agent in self.agents}
        self._skip_agent_selection = None
        self.scores = [0] * self.num_players
        self.deck = make_deck()
        self.np_random.shuffle(self.deck)
        self.discard = []
        self.round_number = 0
        self.dealer = 0
        self.winners = []
        self._start_round()
        self._advance()
        self._accumulate_rewards()

    def _start_round(self):
        self.round_number += 1
        self.hands = [[] for _ in self.possible_agents]
        self.status = [ACTIVE] * self.num_players
        self.flip7_player = None
        self._exhausted = False
        self.pending_card = None
        self.deferred = []
        # The player to the dealer's left acts first; dealer acts last.
        first = (self.dealer + 1) % self.num_players
        self.tasks = deque(
            (DEAL, (first + i) % self.num_players, 0) for i in range(self.num_players)
        )
        self.tasks.append((TURN, first, 0))

    def _round_score(self, player):
        if self.status[player] == BUSTED:
            return 0
        hand = self.hands[player]
        total = sum(card for card in hand if card <= 12)
        total *= 2 if Card.TIMES_2 in hand else 1
        total += sum(2 * (card - 12) for card in hand if 13 <= card <= 17)
        return total + (15 if self.flip7_player == player else 0)

    def _draw(self, player, forced=False):
        if not self.deck:
            self.deck, self.discard = self.discard, []
            self.np_random.shuffle(self.deck)
        if not self.deck:
            # Large tables can hold the entire deck without a Flip 7. The
            # rulebook has no adjudication for this; bank the round (see docs).
            self._exhausted = True
            return
        card = self.deck.pop()
        hand = self.hands[player]
        if card <= 12:
            if card in hand:
                if Card.SECOND_CHANCE in hand:
                    hand.remove(Card.SECOND_CHANCE)
                    self.discard.extend([int(Card.SECOND_CHANCE), card])
                    return
                self.status[player] = BUSTED
            hand.append(card)
            if self.status[player] != BUSTED and sum(c <= 12 for c in hand) == 7:
                self.flip7_player = player
        elif card <= Card.TIMES_2:
            hand.append(card)
        elif card == Card.SECOND_CHANCE:
            if card not in hand:
                hand.append(card)
            elif self._targets(player, card):
                self._ask_target(player, card)
            else:
                self.discard.append(card)
        elif forced:
            self.deferred.append(card)
        else:
            self._ask_target(player, card)

    def _targets(self, player, card):
        return [
            i
            for i in range(self.num_players)
            if self.status[i] == ACTIVE
            and (
                card != Card.SECOND_CHANCE
                or (i != player and Card.SECOND_CHANCE not in self.hands[i])
            )
        ]

    def _ask_target(self, player, card):
        self.pending_card = int(card)
        self.agent_selection = self.possible_agents[player]

    def _advance(self):
        """Run chance events until the next player decision or game end."""
        while True:
            if (
                self.flip7_player is not None
                or ACTIVE not in self.status
                or self._exhausted
            ) and self._finish_round():
                return
            if self.pending_card is not None:
                return
            kind, player, card = self.tasks.popleft()
            if kind == TURN:
                self.tasks.append((TURN, (player + 1) % self.num_players, 0))
                if self.status[player] == ACTIVE:
                    self.agent_selection = self.possible_agents[player]
                    return
            elif kind in (DEAL, FORCED_DRAW):
                if self.status[player] == ACTIVE:
                    self._draw(player, forced=kind == FORCED_DRAW)
            elif kind == END_THREE:
                if self.status[player] == BUSTED:
                    self.hands[player].extend(self.deferred)
                else:
                    for action_card in reversed(self.deferred):
                        self.tasks.appendleft((RESOLVE, player, action_card))
                self.deferred = []
            elif kind == RESOLVE:
                # Busting in a chained Flip Three cancels outstanding actions.
                if self.status[player] == BUSTED:
                    self.hands[player].append(card)
                else:
                    self._ask_target(player, card)

    def _finish_round(self):
        round_scores = [self._round_score(i) for i in range(self.num_players)]
        self.scores = [a + b for a, b in zip(self.scores, round_scores)]
        for hand in self.hands:
            self.discard.extend(hand)
        self.discard.extend(self.deferred)
        self.discard.extend(card for kind, _, card in self.tasks if kind == RESOLVE)
        if self.pending_card is not None:
            self.discard.append(self.pending_card)
        self.tasks.clear()
        self.deferred = []
        self.pending_card = None
        finished = max(self.scores) >= self.target_score
        truncated = (
            not finished
            and self.max_rounds is not None
            and self.round_number >= self.max_rounds
        )
        if finished:
            best = max(self.scores)
            self.winners = [
                a for a, s in zip(self.possible_agents, self.scores) if s == best
            ]
        for i, agent in enumerate(self.possible_agents):
            self.rewards[agent] += float(round_scores[i])
            self.infos[agent] = {
                "scores": dict(zip(self.possible_agents, self.scores)),
                "round_scores": dict(zip(self.possible_agents, round_scores)),
                "completed_round": self.round_number,
                "winners": self.winners[:],
            }
            self.terminations[agent] = finished
            self.truncations[agent] = truncated
        if finished or truncated:
            # Clear hands now that their cards have moved to the discard pile.
            self.hands = [[] for _ in self.possible_agents]
            self.agent_selection = self.possible_agents[0]
            return True
        self.dealer = (self.dealer + 1) % self.num_players
        self._start_round()
        return False

    def _mask(self, agent):
        mask = np.zeros(TARGET_OFFSET + self.num_players, dtype=np.int8)
        if (
            agent not in self.agents
            or self.terminations[agent]
            or self.truncations[agent]
            or agent != self.agent_selection
        ):
            return mask
        player = self._indices[agent]
        if self.pending_card is None:
            mask[HIT] = 1
            mask[STAY] = bool(self.hands[player])
        else:
            for target in self._targets(player, self.pending_card):
                mask[TARGET_OFFSET + target] = 1
        return mask

    def observe(self, agent):
        header = [
            self._indices[agent],
            self._indices[self.agent_selection],
            self.dealer,
            self.round_number,
            len(self.deck),
            self.target_score,
            self.max_rounds or 0,
            int(self.pending_card is not None),
            0 if self.pending_card is None else self.pending_card,
            0 if self.flip7_player is None else self.flip7_player + 1,
        ]
        rows = []
        for i, hand in enumerate(self.hands):
            counts = Counter(hand)
            rows.extend([self.scores[i], self.status[i]])
            rows.extend(counts[c] for c in range(CARD_COUNT))
        discard = Counter(self.discard)
        deferred = [card + 1 for card in self.deferred]
        deferred += [0] * (3 - len(deferred))
        queue = np.zeros((128, 3), dtype=np.float64)
        for i, task in enumerate(self.tasks):
            queue[i] = task
        observation = np.concatenate(
            [
                header,
                rows,
                [discard[c] for c in range(CARD_COUNT)],
                queue.ravel(),
                deferred,
            ]
        ).astype(np.float64)
        return {"observation": observation, "action_mask": self._mask(agent)}

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
        if self.pending_card is not None:
            target = int(action) - TARGET_OFFSET
            card, self.pending_card = self.pending_card, None
            self.hands[target].append(card)
            if card == Card.FREEZE:
                self.status[target] = FROZEN
            elif card == Card.FLIP_THREE:
                self.tasks.appendleft((END_THREE, target, 0))
                for _ in range(3):
                    self.tasks.appendleft((FORCED_DRAW, target, 0))
        elif action == STAY:
            self.status[player] = STAYED
        else:
            self._draw(player)
        self._advance()
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def render(self):
        if self.render_mode is None:
            return None
        lines = [f"Flip 7 | round {self.round_number} | deck {len(self.deck)}"]
        names = ["active", "stayed", "busted", "frozen"]
        for i, agent in enumerate(self.possible_agents):
            cards = [str(c) if c <= 12 else Card(c).name for c in self.hands[i]]
            lines.append(
                f"{agent}: {self.scores[i]} points | {names[self.status[i]]} | {', '.join(cards)}"
            )
        if self.winners:
            lines.append(f"Winners: {', '.join(self.winners)}")
        elif self.agents:
            decision = (
                "Hit / Stay"
                if self.pending_card is None
                else f"target for {Card(self.pending_card).name}"
            )
            lines.append(f"{self.agent_selection}: {decision}")
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self):
        pass
