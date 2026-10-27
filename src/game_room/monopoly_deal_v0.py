"""Monopoly Deal, using the linked 2008 rules. See docs/rules-monopoly-deal.md."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers


class Color(IntEnum):
    BROWN = 0
    LIGHT_BLUE = 1
    PINK = 2
    ORANGE = 3
    RED = 4
    YELLOW = 5
    GREEN = 6
    DARK_BLUE = 7
    RAILROAD = 8
    UTILITY = 9


RENTS = (
    (1, 2),
    (1, 2, 3),
    (1, 2, 4),
    (1, 3, 5),
    (2, 3, 6),
    (2, 4, 6),
    (2, 4, 7),
    (3, 8),
    (1, 2, 3, 4),
    (1, 2),
)
SET_SIZES = tuple(map(len, RENTS))
PROPERTY_VALUES = (1, 1, 2, 2, 3, 3, 4, 4, 2, 2)


@dataclass(frozen=True)
class Card:
    kind: str
    value: int
    colors: tuple[int, ...] = ()


def _cards():
    cards = []
    for color, count in enumerate(SET_SIZES):
        cards.extend([Card("property", PROPERTY_VALUES[color], (color,))] * count)
    for colors, count, value in [
        ((0, 1), 1, 1),
        ((1, 8), 1, 4),
        ((2, 3), 2, 2),
        ((4, 5), 2, 3),
        ((6, 7), 1, 4),
        ((6, 8), 1, 4),
        ((8, 9), 1, 2),
        (tuple(range(10)), 2, 0),
    ]:
        cards.extend([Card("property", value, colors)] * count)
    for kind, count, value in [
        ("deal_breaker", 2, 5),
        ("forced_deal", 3, 3),
        ("sly_deal", 3, 3),
        ("just_say_no", 3, 4),
        ("debt_collector", 3, 3),
        ("birthday", 3, 2),
        ("double_rent", 2, 1),
        ("house", 3, 3),
        ("hotel", 2, 4),
        ("pass_go", 10, 1),
    ]:
        cards.extend([Card(kind, value)] * count)
    for colors in [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]:
        cards.extend([Card("rent", 1, colors)] * 2)
    cards.extend([Card("rent_any", 3, tuple(range(10)))] * 3)
    for value, count in [(1, 6), (2, 5), (3, 3), (4, 3), (5, 2), (10, 1)]:
        cards.extend([Card("money", value)] * count)
    return tuple(cards)


CARDS = _cards()
CARD_COUNT = len(CARDS)
# Enough slots even when all 39 properties and five buildings stand alone.
GROUP_COUNT = 44
# Card selection is reused for payment, discard, responses and deal targets.
SELECT_CARD = 0
PLAY = CARD_COUNT
BANK = PLAY + CARD_COUNT
MOVE = BANK + CARD_COUNT
SELECT_GROUP = MOVE + CARD_COUNT
SELECT_COLOR = SELECT_GROUP + GROUP_COUNT
SELECT_PLAYER = SELECT_COLOR + 10
END_TURN = SELECT_PLAYER + 5
ACCEPT = END_TURN + 1
ACTION_COUNT = ACCEPT + 1
PHASES = (
    "play",
    "color",
    "group",
    "target",
    "take",
    "give",
    "rent_group",
    "double",
    "response",
    "pay",
    "discard",
)


@dataclass
class PropertyGroup:
    color: int
    cards: list[int] = field(default_factory=list)
    buildings: list[int] = field(default_factory=list)

    @property
    def complete(self):
        return len(self.cards) == SET_SIZES[self.color]


def make_deck():
    """Physical card IDs, excluding the four reference cards."""
    return list(range(CARD_COUNT))


def env(**kwargs):
    return wrappers.OrderEnforcingWrapper(
        wrappers.AssertOutOfBoundsWrapper(raw_env(**kwargs))
    )


class raw_env(AECEnv):
    metadata: ClassVar[dict] = {
        "name": "monopoly_deal_v0",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self, num_players=3, max_steps=10000, render_mode=None):
        super().__init__()
        if type(num_players) is not int or not 2 <= num_players <= 5:
            raise ValueError("num_players must be an integer from 2 to 5")
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
                        0,
                        np.inf,
                        (22 + num_players * 2 + CARD_COUNT * 4,),
                        dtype=np.float64,
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
        self.deck = make_deck()
        self.np_random.shuffle(self.deck)
        self.discard = []
        self.hands = [[] for _ in self.agents]
        self.banks = [[] for _ in self.agents]
        self.groups = [[None] * GROUP_COUNT for _ in self.agents]
        self.turn = 0
        self.turn_number = 1
        self.steps = 0
        self.winners = []
        self.plays = 0
        self.phase = "play"
        self._clear_pending()
        for p in range(self.num_players):
            self._draw(p, 5)
        self._start_turn()

    def _clear_pending(self):
        self.card = self.color = self.group = self.target = -1
        self.taken = self.offered = -1
        self.moving = False
        self.amount = 0
        self.targets = []
        self.cancelled = False

    def _select(self, p):
        self.agent_selection = self.possible_agents[p]

    def _draw(self, p, count):
        for _ in range(count):
            if not self.deck:
                self.deck, self.discard = self.discard, []
                self.np_random.shuffle(self.deck)
            if not self.deck:
                break
            self.hands[p].append(self.deck.pop())

    def _start_turn(self):
        self.plays = 0
        self.phase = "play"
        self._clear_pending()
        self._select(self.turn)
        if not self._check_win():
            self._draw(self.turn, 5 if not self.hands[self.turn] else 2)

    def _complete_colors(self, p):
        return {g.color for g in self.groups[p] if g is not None and g.complete}

    def _check_win(self):
        if len(self._complete_colors(self.turn)) < 3:
            return False
        winner = self.possible_agents[self.turn]
        self.winners = [winner]
        for a in self.agents:
            self.terminations[a] = True
            self.rewards[a] = 1.0 if a == winner else -1.0
            self.infos[a] = {
                "winners": self.winners[:],
                "turn_number": self.turn_number,
            }
        return True

    def _owned_groups(self, p):
        return [(i, g) for i, g in enumerate(self.groups[p]) if g is not None]

    def _property_location(self, p, card):
        for i, g in self._owned_groups(p):
            if card in g.cards or card in g.buildings:
                return i
        raise ValueError("Card is not on this player's table")

    def _loose_properties(self, p):
        return [c for _, g in self._owned_groups(p) if not g.complete for c in g.cards]

    def _building_destinations(self, p, card):
        kind = CARDS[card].kind
        return [
            i
            for i, g in self._owned_groups(p)
            if g.complete
            and g.color < 8
            and not any(CARDS[c].kind == kind for c in g.buildings)
            and (kind == "house" or any(CARDS[c].kind == "house" for c in g.buildings))
        ]

    def _destinations(self, p, color, moving=-1):
        result = [
            i
            for i, g in self._owned_groups(p)
            if g.color == color
            and len(g.cards) < SET_SIZES[color]
            and moving not in g.cards
        ]
        empty = next((i for i, g in enumerate(self.groups[p]) if g is None), None)
        if empty is not None:
            result.append(empty)
        return result

    def _payable(self, p):
        return self.banks[p] + [
            c
            for _, g in self._owned_groups(p)
            for c in g.cards + g.buildings
            if CARDS[c].value > 0
        ]

    def _playable(self, p, c):
        kind = CARDS[c].kind
        if kind in ("money", "just_say_no", "double_rent"):
            return False
        if kind in ("house", "hotel"):
            return bool(self._building_destinations(p, c))
        if kind in ("rent", "rent_any"):
            return any(
                g.cards and g.color in CARDS[c].colors for _, g in self._owned_groups(p)
            )
        if kind in ("sly_deal", "forced_deal"):
            return (kind != "forced_deal" or bool(self._loose_properties(p))) and any(
                self._loose_properties(q) for q in range(self.num_players) if q != p
            )
        if kind == "deal_breaker":
            return any(
                self._complete_colors(q) for q in range(self.num_players) if q != p
            )
        return True

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
        if self.phase == "play":
            mask[END_TURN] = 1
            if self.plays < 3:
                for c in self.hands[p]:
                    mask[BANK + c] = CARDS[c].kind != "property"
                    mask[PLAY + c] = self._playable(p, c)
            for _, g in self._owned_groups(p):
                for c in g.cards:
                    mask[MOVE + c] = (
                        bool(self._destinations(p, g.color, c))
                        or len(CARDS[c].colors) > 1
                    )
                for c in g.buildings:
                    mask[MOVE + c] = bool(self._building_destinations(p, c))
        elif self.phase == "color":
            for color in CARDS[self.card].colors:
                if self._destinations(p, color, self.card if self.moving else -1):
                    mask[SELECT_COLOR + color] = 1
        elif self.phase == "group":
            dest = (
                self._building_destinations(p, self.card)
                if CARDS[self.card].kind in ("house", "hotel")
                else self._destinations(p, self.color, self.card if self.moving else -1)
            )
            mask[[SELECT_GROUP + i for i in dest]] = 1
        elif self.phase == "target":
            for q in range(self.num_players):
                if q == p:
                    continue
                kind = CARDS[self.card].kind
                if kind in ("sly_deal", "forced_deal") and not self._loose_properties(
                    q
                ):
                    continue
                if kind == "deal_breaker" and not self._complete_colors(q):
                    continue
                mask[SELECT_PLAYER + q] = 1
        elif self.phase == "take":
            if CARDS[self.card].kind == "deal_breaker":
                for i, g in self._owned_groups(self.target):
                    mask[SELECT_GROUP + i] = g.complete
            else:
                mask[self._loose_properties(self.target)] = 1
        elif self.phase == "give":
            mask[self._loose_properties(p)] = 1
        elif self.phase == "rent_group":
            for i, g in self._owned_groups(p):
                mask[SELECT_GROUP + i] = bool(
                    g.cards and g.color in CARDS[self.card].colors
                )
        elif self.phase == "double":
            mask[ACCEPT] = 1
            if self.plays < 3 and CARDS[self.card].kind == "rent":
                for c in self.hands[p]:
                    mask[c] = CARDS[c].kind == "double_rent"
        elif self.phase == "response":
            mask[ACCEPT] = 1
            for c in self.hands[p]:
                mask[c] = CARDS[c].kind == "just_say_no"
        elif self.phase == "pay":
            mask[self._payable(p)] = 1
        elif self.phase == "discard":
            mask[self.hands[p]] = 1
        return mask

    def _consume(self, card):
        self.hands[self.turn].remove(card)
        self.plays += 1
        self.discard.append(card)

    def _remove_property(self, p, card):
        i = self._property_location(p, card)
        g = self.groups[p][i]
        (g.cards if card in g.cards else g.buildings).remove(card)
        if not g.cards and not g.buildings:
            self.groups[p][i] = None
        return g.color

    def _receive_property(self, p, card, color):
        # Transfers keep the current color; reorganization is a turn action.
        dest = self._destinations(p, color)[0]
        if self.groups[p][dest] is None:
            self.groups[p][dest] = PropertyGroup(color)
        self.groups[p][dest].cards.append(card)

    def _resume(self):
        self.phase = "play"
        self._select(self.turn)
        self._clear_pending()
        self._check_win()

    def _begin_effect(self, targets):
        self.targets = list(targets)
        self._next_target()

    def _next_target(self):
        if not self.targets:
            self._resume()
            return
        self.target = self.targets.pop(0)
        self.cancelled = False
        self.phase = "response"
        self._select(self.target)

    def _resolve_effect(self):
        if self.cancelled:
            self._next_target()
            return
        kind = CARDS[self.card].kind
        if kind in ("sly_deal", "forced_deal"):
            color = self._remove_property(self.target, self.taken)
            if kind == "forced_deal":
                offered_color = self._remove_property(self.turn, self.offered)
                self._receive_property(self.target, self.offered, offered_color)
            self._receive_property(self.turn, self.taken, color)
            self._next_target()
        elif kind == "deal_breaker":
            g = self.groups[self.target][self.group]
            self.groups[self.target][self.group] = None
            dest = self.groups[self.turn].index(None)
            self.groups[self.turn][dest] = g
            self._next_target()
        else:
            self.phase = "pay"
            self._select(self.target)
            self.debt = self.amount
            if not self._payable(self.target):
                self._next_target()

    def _end_turn(self):
        if len(self.hands[self.turn]) > 7:
            self.phase = "discard"
        else:
            self.turn = (self.turn + 1) % self.num_players
            self.turn_number += 1
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
            raise ValueError(f"Illegal action {action!r} for {agent} in {self.phase}")
        action = int(action)
        self._cumulative_rewards[agent] = 0
        self._clear_rewards()
        p = self._indices[agent]
        self.steps += 1
        self._step(action, p)
        if (
            self.max_steps is not None
            and self.steps >= self.max_steps
            and not self.winners
        ):
            for a in self.agents:
                self.truncations[a] = True
                self.infos[a] = {"winners": [], "turn_number": self.turn_number}
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def _step(self, action, p):
        if self.phase == "play":
            if action == END_TURN:
                self._end_turn()
            elif BANK <= action < MOVE:
                c = action - BANK
                self.hands[p].remove(c)
                self.banks[p].append(c)
                self.plays += 1
            elif MOVE <= action < SELECT_GROUP:
                self.card = action - MOVE
                self.moving = True
                self.phase = "color" if CARDS[self.card].kind == "property" else "group"
            else:
                self.card = action - PLAY
                kind = CARDS[self.card].kind
                if kind == "property":
                    self.phase = "color"
                elif kind in ("house", "hotel"):
                    self.phase = "group"
                else:
                    self._consume(self.card)
                    if kind == "pass_go":
                        self._draw(p, 2)
                        self._resume()
                    elif kind == "birthday":
                        self.amount = 2
                        self._begin_effect(
                            [
                                (p + i) % self.num_players
                                for i in range(1, self.num_players)
                            ]
                        )
                    elif kind in ("rent", "rent_any"):
                        self.phase = "rent_group"
                    else:
                        self.phase = "target"
        elif self.phase == "color":
            self.color = action - SELECT_COLOR
            self.phase = "group"
        elif self.phase == "group":
            dest = action - SELECT_GROUP
            if self.moving:
                self._remove_property(p, self.card)
            else:
                self.hands[p].remove(self.card)
                self.plays += 1
            if self.groups[p][dest] is None:
                self.groups[p][dest] = PropertyGroup(self.color)
            g = self.groups[p][dest]
            (g.cards if CARDS[self.card].kind == "property" else g.buildings).append(
                self.card
            )
            self._resume()
        elif self.phase == "target":
            self.target = action - SELECT_PLAYER
            kind = CARDS[self.card].kind
            if kind in ("sly_deal", "forced_deal", "deal_breaker"):
                self.phase = "take"
            else:
                if kind == "debt_collector":
                    self.amount = 5
                self._begin_effect([self.target])
        elif self.phase == "take":
            if CARDS[self.card].kind == "deal_breaker":
                self.group = action - SELECT_GROUP
            else:
                self.taken = action
            if CARDS[self.card].kind == "forced_deal":
                self.phase = "give"
            else:
                self._begin_effect([self.target])
        elif self.phase == "give":
            self.offered = action
            self._begin_effect([self.target])
        elif self.phase == "rent_group":
            self.group = action - SELECT_GROUP
            g = self.groups[p][self.group]
            self.amount = RENTS[g.color][len(g.cards) - 1]
            if g.complete:
                kinds = {CARDS[c].kind for c in g.buildings}
                if "house" in kinds:
                    self.amount += 3 + (4 if "hotel" in kinds else 0)
            self.phase = "double"
        elif self.phase == "double":
            if action == ACCEPT:
                if CARDS[self.card].kind == "rent_any":
                    self.phase = "target"
                else:
                    self._begin_effect(
                        [(p + i) % self.num_players for i in range(1, self.num_players)]
                    )
            else:
                self._consume(action)
                self.amount *= 2
        elif self.phase == "response":
            if action == ACCEPT:
                self._resolve_effect()
            else:
                self.hands[p].remove(action)
                self.discard.append(action)
                self.cancelled = not self.cancelled
                self._select(self.turn if p == self.target else self.target)
        elif self.phase == "pay":
            if action in self.banks[p]:
                self.banks[p].remove(action)
                self.banks[self.turn].append(action)
            else:
                color = self._remove_property(p, action)
                if CARDS[action].kind == "property":
                    self._receive_property(self.turn, action, color)
                else:
                    self.banks[self.turn].append(action)
            self.debt -= CARDS[action].value
            if self._check_win():
                return
            if self.debt <= 0 or not self._payable(p):
                self._next_target()
        elif self.phase == "discard":
            self.hands[p].remove(action)
            self.deck.insert(0, action)
            self._end_turn()

    def observe(self, agent):
        p = self._indices[agent]
        header = [
            p,
            self._indices[self.agent_selection],
            self.turn,
            self.turn_number,
            self.steps,
            self.max_steps or 0,
            self.plays,
            PHASES.index(self.phase),
            len(self.deck),
            self.card + 1,
            self.color + 1,
            self.group + 1,
            self.target + 1,
            self.taken + 1,
            self.offered + 1,
            int(self.moving),
            self.amount,
            getattr(self, "debt", 0) if self.phase == "pay" else 0,
            int(self.cancelled),
            sum(1 << q for q in self.targets),
            int(bool(self.winners)),
            self.turn + 1 if self.winners else 0,
        ]
        rows = np.zeros((CARD_COUNT, 4), dtype=np.float64)
        for c in self.hands[p]:
            rows[c] = [1, p + 1, 0, 0]
        for q in range(self.num_players):
            header.extend([len(self.hands[q]), len(self._complete_colors(q))])
            for c in self.banks[q]:
                rows[c] = [2, q + 1, 0, 0]
            for i, g in self._owned_groups(q):
                for c in g.cards:
                    rows[c] = [3, q + 1, i + 1, g.color + 1]
                for c in g.buildings:
                    rows[c] = [4, q + 1, i + 1, g.color + 1]
        for c in self.discard:
            rows[c] = [5, 0, 0, 0]
        return {
            "observation": np.concatenate([header, rows.ravel()]).astype(np.float64),
            "action_mask": self._mask(agent),
        }

    def action_label(self, action):
        """Human-readable label for an action ID in the current phase."""
        if action == END_TURN:
            return "End turn"
        if action == ACCEPT:
            return "Accept / continue"
        for offset, end, label in [
            (SELECT_PLAYER, END_TURN, "Player"),
            (SELECT_COLOR, SELECT_PLAYER, "Color"),
            (SELECT_GROUP, SELECT_COLOR, "Group"),
        ]:
            if offset <= action < end:
                return f"{label} {action - offset}"
        for offset, label in [
            (MOVE, "Move"),
            (BANK, "Bank"),
            (PLAY, "Play"),
            (0, "Select"),
        ]:
            if offset <= action < offset + CARD_COUNT:
                c = action - offset
                return f"{label} #{c} {CARDS[c].kind} ({CARDS[c].value}M)"
        raise ValueError("Unknown action")

    def render(self):
        if self.render_mode is None:
            return None
        lines = [
            f"Monopoly Deal | turn {self.turn_number} | {self.agent_selection}: {self.phase}"
        ]
        for p, agent in enumerate(self.possible_agents):
            groups = "; ".join(
                f"{i}: {Color(g.color).name} {g.cards} buildings={g.buildings}"
                for i, g in self._owned_groups(p)
            )
            lines.append(
                f"{agent}: hand={len(self.hands[p])} bank={self.banks[p]} | {groups}"
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
