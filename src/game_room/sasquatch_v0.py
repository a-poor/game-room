"""Sell Me a Sasquatch. Rules: docs/rules-sasquatch.md; API: docs/sasquatch-v0.md."""

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers


class Card(IntEnum):
    TINY = 0
    MEDIUM = 1
    BIG = 2
    GIANT = 3
    POISON_PILL_BUG = 4
    LOAN_SHARK = 5
    TROJAN_HORSE = 6
    PLATONIC_ISOLATOR = 7
    SUPER_DETRITAL_REPOSITIONER = 8
    DETRITAL_REPOSITIONER = 9
    SPECTROELECTRIC_OPTIMETER = 10
    CRYPTOZOOPTIC_EXPANDER = 11


CARD_COUNTS = (20, 25, 16, 4, 10, 10, 12, 3, 3, 6, 5, 6)
SET_SIZES = (4, 3, 2, 1, 2, 2, 3)
CARD_COUNT = 12
DEAL_SLOTS = 120
SELECT_CARD = 0  # + card type, used for offers, expansion, sets and theft
PLAY = 12  # + card type, only Thingamabobs are playable
SELECT_SLOT = 24  # + deal slot 0..119, independent of hidden card identity
SELECT_PLAYER = 144  # + seat 0..5
SELECT_DEAL = 150  # + seat 0..5
PASS = 156
ACCEPT = 157
REVERSE = 158
STOP = 159
ACTION_COUNT = 160
PHASES = (
    "offer_card",
    "offer_place",
    "offer_reveal",
    "buyer_reveal",
    "things",
    "remove",
    "reveal",
    "expand_card",
    "expand_place",
    "point_target",
    "choose",
    "nasty",
    "steal",
)


@dataclass
class DealCard:
    kind: int
    owner: int
    contributor: int
    face_up: bool = False


def make_deck():
    """Return the 120-card inventory as interchangeable card-type IDs."""
    return [kind for kind, count in enumerate(CARD_COUNTS) for _ in range(count)]


def env(**kwargs):
    return wrappers.OrderEnforcingWrapper(
        wrappers.AssertOutOfBoundsWrapper(raw_env(**kwargs))
    )


class raw_env(AECEnv):
    metadata: ClassVar[dict] = {
        "name": "sasquatch_v0",
        "render_modes": ["human", "ansi"],
        "render_fps": 1,
        "is_parallelizable": False,
    }

    def __init__(self, num_players=3, max_steps=10000, render_mode=None):
        super().__init__()
        if type(num_players) is not int or not 2 <= num_players <= 6:
            raise ValueError("num_players must be an integer from 2 to 6")
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be a positive integer or None")
        if render_mode not in (None, "ansi", "human"):
            raise ValueError("render_mode must be None, 'ansi', or 'human'")
        self.num_players = num_players
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.target_score = 4 if num_players in (3, 4) else 5
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self._indices = {agent: i for i, agent in enumerate(self.possible_agents)}
        self.action_spaces = {
            a: spaces.Discrete(ACTION_COUNT) for a in self.possible_agents
        }
        self.observation_spaces = {
            a: spaces.Dict(
                {
                    "observation": spaces.Box(
                        0, np.inf, (644 + 14 * num_players,), dtype=np.float64
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
        self.collections = [[] for _ in self.agents]
        self.points = [0] * self.num_players
        self.deals = [None] * DEAL_SLOTS
        self.turn = 0
        self.buyer = 0 if self.num_players > 2 else -1
        self.turn_number = 1
        self.steps = 0
        self.winners = []
        for p in range(self.num_players):
            self._refill(p)
        self._start_turn()

    def _select(self, p):
        self.agent_selection = self.possible_agents[p]

    def _refill(self, p):
        while len(self.hands[p]) < 5:
            if not self.deck:
                self.deck, self.discard = self.discard, []
                self.np_random.shuffle(self.deck)
            if not self.deck:
                break
            self.hands[p].append(self.deck.pop())

    def _start_turn(self):
        self.offerers = (
            [self.turn]
            if self.num_players == 2
            else [
                (self.turn + i) % self.num_players for i in range(1, self.num_players)
            ]
        )
        self.active_deals = (
            list(range(2)) if self.num_players == 2 else self.offerers[:]
        )
        self.offer_index = 0
        self.offered_count = 0
        self.initial_reveals = []
        self.pending_card = -1
        self.passes = 0
        self.tool = self.tool_actor = -1
        self.remaining = 0
        self.victim = self.beneficiary = -1
        self.phase = "offer_card"
        self._select(self.offerers[0])

    def _slots(self, owner=None, hidden=False):
        return [
            i
            for i, c in enumerate(self.deals)
            if c is not None
            and (owner is None or c.owner == owner)
            and (not hidden or not c.face_up)
        ]

    def _add_to_deal(self, p, owner, card):
        slot = self.deals.index(None)
        self.hands[p].remove(card)
        self.deals[slot] = DealCard(card, owner, p)
        return slot

    def _offer_added(self):
        self.pending_card = -1
        self.offered_count += 1
        self.phase = "offer_reveal" if self.offered_count == 3 else "offer_card"

    def _things_start(self):
        self.phase = "things"
        self.passes = 0
        self._select(self.turn)

    def _tool_legal(self, p, card):
        if card == Card.PLATONIC_ISOLATOR:
            return any(score > self.points[p] for score in self.points)
        if card == Card.SUPER_DETRITAL_REPOSITIONER:
            return len(self._slots()) >= 2
        if card == Card.DETRITAL_REPOSITIONER:
            return bool(self._slots())
        if card == Card.SPECTROELECTRIC_OPTIMETER:
            return bool(self._slots(hidden=True))
        if card == Card.CRYPTOZOOPTIC_EXPANDER:
            return bool(self.hands[p])
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
        p = self._indices[agent]
        if self.phase in ("offer_card", "expand_card"):
            mask[self.hands[p]] = 1
        elif self.phase in ("offer_place", "expand_place"):
            mask[[SELECT_DEAL + owner for owner in self.active_deals]] = 1
        elif self.phase == "offer_reveal":
            slots = self._slots() if self.num_players == 2 else self._slots(p)
            mask[[SELECT_SLOT + slot for slot in slots]] = 1
        elif self.phase in ("buyer_reveal", "reveal"):
            mask[[SELECT_SLOT + slot for slot in self._slots(hidden=True)]] = 1
        elif self.phase == "things":
            mask[PASS] = 1
            for card in set(self.collections[p]):
                mask[PLAY + card] = self._tool_legal(p, card)
        elif self.phase == "remove":
            mask[[SELECT_SLOT + slot for slot in self._slots()]] = 1
        elif self.phase == "point_target":
            for q in range(self.num_players):
                mask[SELECT_PLAYER + q] = self.points[q] > self.points[p]
        elif self.phase == "choose":
            if self.num_players == 2:
                mask[ACCEPT] = mask[REVERSE] = 1
            else:
                mask[[SELECT_DEAL + owner for owner in self.active_deals]] = 1
        elif self.phase == "nasty":
            counts = Counter(self.collections[p])
            for card in range(4, 7):
                mask[card] = counts[card] >= SET_SIZES[card]
        elif self.phase == "steal":
            mask[STOP] = 1
            mask[self.collections[self.victim]] = 1
        return mask

    def _finish_tool(self):
        self.phase = "things"
        self._select(self.tool_actor)
        self.tool = self.tool_actor = -1
        self.pending_card = -1
        self.remaining = 0

    def _distribute(self, choice):
        if self.num_players > 2:
            self.buyer = choice
        for card in self.deals:
            if card is None:
                continue
            if self.num_players == 2:
                recipient = 1 - card.owner if choice == REVERSE else card.owner
            else:
                recipient = self.turn if card.owner == choice else card.owner
            self.collections[recipient].append(card.kind)
        self.deals = [None] * DEAL_SLOTS
        # Fixed priority resolves simultaneous sets; owners choose which Nasty
        # type to resolve first when they have more than one complete type.
        first = self.turn if self.num_players == 2 else self.buyer
        self.set_order = [
            (first + i) % self.num_players for i in range(self.num_players)
        ]
        self._advance_sets()

    def _advance_sets(self):
        self.remaining = 0
        self.victim = self.beneficiary = -1
        for p in self.set_order:
            counts = Counter(self.collections[p])
            if any(counts[card] >= SET_SIZES[card] for card in range(4, 7)):
                self.phase = "nasty"
                self._select(p)
                return
        # All Nasty cascades are exhausted before any Creature points are paid.
        for p in self.set_order:
            counts = Counter(self.collections[p])
            for card in range(4):
                sets = counts[card] // SET_SIZES[card]
                if sets:
                    self._discard_set(p, card, sets * SET_SIZES[card])
                    self.points[p] += sets
        self._end_turn()

    def _discard_set(self, p, card, count):
        for _ in range(count):
            self.collections[p].remove(card)
            self.discard.append(card)

    def _nasty(self, p, card):
        self._discard_set(p, card, SET_SIZES[card])
        beneficiary = 1 - p if self.num_players == 2 else self.buyer
        if beneficiary == p:
            # The new Buyer cannot gain anything by stealing from themself.
            self._advance_sets()
        elif card == Card.TROJAN_HORSE:
            if self.points[p]:
                self.points[p] -= 1
                self.points[beneficiary] += 1
            self._advance_sets()
        elif self.collections[p]:
            self.victim, self.beneficiary = p, beneficiary
            self.remaining = 1 if card == Card.POISON_PILL_BUG else 2
            self.phase = "steal"
            self._select(beneficiary)
        else:
            self._advance_sets()

    def _end_turn(self):
        for offset in range(self.num_players):
            self._refill((self.turn + offset) % self.num_players)
        high = max(self.points)
        leaders = [p for p, score in enumerate(self.points) if score == high]
        if high >= self.target_score and len(leaders) == 1:
            winner = self.possible_agents[leaders[0]]
            self.winners = [winner]
            for a in self.agents:
                self.terminations[a] = True
                self.rewards[a] = 1.0 if a == winner else -1.0
                self.infos[a] = {
                    "winners": [winner],
                    "scores": self.points[:],
                    "turn_number": self.turn_number,
                }
            return
        self.turn = 1 - self.turn if self.num_players == 2 else self.buyer
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
        self._cumulative_rewards[agent] = 0
        self._clear_rewards()
        self.steps += 1
        self._step(int(action), self._indices[agent])
        if (
            not self.winners
            and self.max_steps is not None
            and self.steps >= self.max_steps
        ):
            for a in self.agents:
                self.truncations[a] = True
                self.infos[a] = {
                    "winners": [],
                    "scores": self.points[:],
                    "turn_number": self.turn_number,
                }
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def _step(self, action, p):
        if self.phase == "offer_card":
            if self.num_players == 2:
                self.pending_card = action
                self.phase = "offer_place"
            else:
                self._add_to_deal(p, p, action)
                self._offer_added()
        elif self.phase == "offer_place":
            self._add_to_deal(p, action - SELECT_DEAL, self.pending_card)
            self._offer_added()
        elif self.phase == "offer_reveal":
            self.initial_reveals.append(action - SELECT_SLOT)
            self.offer_index += 1
            if self.offer_index < len(self.offerers):
                self.offered_count = 0
                self.phase = "offer_card"
                self._select(self.offerers[self.offer_index])
            else:
                for slot in self.initial_reveals:
                    self.deals[slot].face_up = True
                if self.num_players == 2:
                    self._things_start()
                else:
                    self.phase = "buyer_reveal"
                    self._select(self.turn)
        elif self.phase == "buyer_reveal":
            self.deals[action - SELECT_SLOT].face_up = True
            self._things_start()
        elif self.phase == "things":
            if action == PASS:
                self.passes += 1
                if self.passes == self.num_players:
                    self.phase = "choose"
                    self._select(1 - self.turn if self.num_players == 2 else self.turn)
                else:
                    self._select((p + 1) % self.num_players)
            else:
                self.tool = action - PLAY
                self.tool_actor = p
                self.collections[p].remove(self.tool)
                self.discard.append(self.tool)
                self.passes = 0
                if self.tool == Card.PLATONIC_ISOLATOR:
                    self.phase = "point_target"
                elif self.tool in (
                    Card.SUPER_DETRITAL_REPOSITIONER,
                    Card.DETRITAL_REPOSITIONER,
                ):
                    self.remaining = (
                        2 if self.tool == Card.SUPER_DETRITAL_REPOSITIONER else 1
                    )
                    self.phase = "remove"
                elif self.tool == Card.SPECTROELECTRIC_OPTIMETER:
                    self.phase = "reveal"
                else:
                    self.phase = "expand_card"
        elif self.phase == "remove":
            slot = action - SELECT_SLOT
            self.discard.append(self.deals[slot].kind)
            self.deals[slot] = None
            self.remaining -= 1
            if not self.remaining:
                self._finish_tool()
        elif self.phase == "reveal":
            self.deals[action - SELECT_SLOT].face_up = True
            self._finish_tool()
        elif self.phase == "point_target":
            target = action - SELECT_PLAYER
            self.points[target] -= 1
            self.points[p] += 1
            self._finish_tool()
        elif self.phase == "expand_card":
            self.pending_card = action
            self.phase = "expand_place"
        elif self.phase == "expand_place":
            self._add_to_deal(p, action - SELECT_DEAL, self.pending_card)
            self._finish_tool()
        elif self.phase == "choose":
            self._distribute(action if self.num_players == 2 else action - SELECT_DEAL)
        elif self.phase == "nasty":
            self._nasty(p, action)
        elif self.phase == "steal":
            if action == STOP:
                self._advance_sets()
            else:
                self.collections[self.victim].remove(action)
                self.collections[p].append(action)
                self.remaining -= 1
                if not self.remaining or not self.collections[self.victim]:
                    self._advance_sets()

    def observe(self, agent):
        p = self._indices[agent]
        pending_owner = self.tool_actor if self.tool_actor >= 0 else self.turn
        header = [
            p,
            self._indices[self.agent_selection],
            self.turn,
            self.buyer + 1,
            self.turn_number,
            self.steps,
            self.max_steps or 0,
            self.target_score,
            PHASES.index(self.phase),
            self.offer_index,
            self.offered_count,
            self.passes,
            self.tool + 1,
            self.tool_actor + 1,
            self.remaining,
            self.victim + 1,
            self.beneficiary + 1,
            self.pending_card + 1 if p == pending_owner else 0,
            self._indices[self.winners[0]] + 1 if self.winners else 0,
            len(self.deck),
        ]
        for q in range(self.num_players):
            counts = Counter(self.collections[q])
            header.extend(
                [
                    self.points[q],
                    len(self.hands[q]),
                    *[counts[c] for c in range(CARD_COUNT)],
                ]
            )
        hand, discard = Counter(self.hands[p]), Counter(self.discard)
        rows = np.zeros((DEAL_SLOTS, 5), dtype=np.float64)
        for i, card in enumerate(self.deals):
            if card is not None:
                known = card.face_up or card.contributor == p
                rows[i] = [
                    1,
                    card.owner + 1,
                    card.contributor + 1,
                    int(card.face_up),
                    card.kind + 1 if known else 0,
                ]
        vector = np.concatenate(
            [
                header,
                [hand[c] for c in range(CARD_COUNT)],
                [discard[c] for c in range(CARD_COUNT)],
                rows.ravel(),
            ]
        )
        return {
            "observation": vector.astype(np.float64),
            "action_mask": self._mask(agent),
        }

    def action_label(self, action):
        controls = {
            PASS: "Pass Thingamabob opportunity",
            ACCEPT: "Accept deals",
            REVERSE: "Reverse deals",
            STOP: "Stop stealing",
        }
        if action in controls:
            return controls[action]
        if SELECT_CARD <= action < PLAY:
            return f"Select {Card(action).name}"
        if PLAY <= action < SELECT_SLOT:
            return f"Play {Card(action - PLAY).name}"
        if SELECT_SLOT <= action < SELECT_PLAYER:
            return f"Deal slot {action - SELECT_SLOT}"
        if SELECT_PLAYER <= action < SELECT_DEAL:
            return f"Player {action - SELECT_PLAYER}"
        if SELECT_DEAL <= action < PASS:
            return f"Deal in front of player {action - SELECT_DEAL}"
        raise ValueError("Unknown action")

    def render(self):
        if self.render_mode is None:
            return None
        lines = [
            f"Sell Me a Sasquatch | turn {self.turn_number} | {self.agent_selection}: {self.phase}"
        ]
        if self.buyer >= 0:
            lines.append(f"Buyer Squatch: player_{self.buyer}")
        for p, a in enumerate(self.possible_agents):
            cards = Counter(self.collections[p])
            collection = (
                ", ".join(
                    f"{Card(c).name} x{count}" for c, count in sorted(cards.items())
                )
                or "empty"
            )
            lines.append(
                f"{a}: {self.points[p]} points | hand={len(self.hands[p])} | collection: {collection}"
            )
        for owner in self.active_deals:
            cards = [
                f"#{slot} {Card(self.deals[slot].kind).name if self.deals[slot].face_up else '?'}"
                for slot in self._slots(owner)
            ]
            lines.append(f"Deal {owner}: {', '.join(cards) or 'empty'}")
        if self.winners:
            lines.append(f"Winner: {self.winners[0]}")
        result = "\n".join(lines)
        if self.render_mode == "human":
            print(result)
            return None
        return result

    def close(self):
        pass
