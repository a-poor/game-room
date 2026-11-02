"""Cards and tile identities for American Mahjong.

A pattern is one concrete suit/rank realization of a card line. Keep singles
and pairs separate, even when the same tile occurs in multiple groups.
"""

from collections import Counter
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from game_room._mahjong_2026 import pattern_data

BAMS, CRAKS, DOTS = range(3)
EAST, SOUTH, WEST, NORTH, GREEN, RED, WHITE, FLOWER, JOKER = range(27, 36)
TILE_COUNT = 36
TILE_NAMES = tuple(
    f"{rank} {suit}" for suit in ("bam", "crak", "dot") for rank in range(1, 10)
) + ("east", "south", "west", "north", "green", "red", "white", "flower", "joker")
TILE_COUNTS = (4,) * 34 + (8, 8)
MATCHING_DRAGONS = (GREEN, RED, WHITE)


def suited(suit, rank):
    """Return the tile ID for a suit (BAMS/CRAKS/DOTS) and rank 1–9."""
    if type(suit) is not int or not 0 <= suit < 3:
        raise ValueError("suit must be BAMS, CRAKS, or DOTS")
    if type(rank) is not int or not 1 <= rank <= 9:
        raise ValueError("rank must be an integer from 1 to 9")
    return suit * 9 + rank - 1


@dataclass(frozen=True)
class Group:
    tile: int
    size: int

    def __post_init__(self):
        if type(self.tile) is not int or not 0 <= self.tile < JOKER:
            raise ValueError("group tile must be a natural tile ID")
        if type(self.size) is not int or not 1 <= self.size <= 6:
            raise ValueError("group size must be an integer from 1 to 6")


@dataclass(frozen=True)
class HandPattern:
    name: str
    groups: tuple[Group, ...]
    value: int = 25
    concealed: bool = False
    jokerless_bonus: bool = True

    def __post_init__(self):
        object.__setattr__(self, "groups", tuple(self.groups))
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("pattern name must be nonempty")
        if not all(isinstance(g, Group) for g in self.groups):
            raise ValueError("groups must contain Group objects")
        if sum(g.size for g in self.groups) != 14:
            raise ValueError("a pattern must contain exactly 14 tiles")
        if type(self.value) is not int or self.value < 1:
            raise ValueError("hand value must be a positive integer")
        if type(self.concealed) is not bool or type(self.jokerless_bonus) is not bool:
            raise ValueError("concealed and jokerless_bonus must be bools")
        required, total = Counter(), Counter()
        for group in self.groups:
            total[group.tile] += group.size
            if group.size < 3:
                required[group.tile] += group.size
        if (
            any(n > TILE_COUNTS[t] for t, n in required.items())
            or sum(max(0, n - TILE_COUNTS[t]) for t, n in total.items())
            > TILE_COUNTS[JOKER]
        ):
            raise ValueError("pattern cannot be made with the 152-tile set")


@dataclass(frozen=True)
class Exposure:
    tile: int
    size: int
    jokers: int = 0

    def __post_init__(self):
        Group(self.tile, self.size)
        if self.size < 3:
            raise ValueError("an exposure requires at least three tiles")
        if type(self.jokers) is not int or not 0 <= self.jokers < self.size:
            raise ValueError("an exposure must contain its claimed natural tile")


@dataclass(frozen=True)
class Card:
    name: str
    patterns: tuple[HandPattern, ...]

    def __post_init__(self):
        object.__setattr__(self, "patterns", tuple(self.patterns))
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("card name must be nonempty")
        if not self.patterns or not all(
            isinstance(p, HandPattern) for p in self.patterns
        ):
            raise ValueError("card requires at least one HandPattern")
        if len({p.name for p in self.patterns}) != len(self.patterns):
            raise ValueError("pattern names must be unique within a card")

    @cached_property
    def _matcher(self):
        return _Matcher(self)

    def matching_hands(self, hand, exposures=()):
        """Return pattern indices matching concealed tile counts and exposures."""
        counts = np.array([hand.get(t, 0) for t in range(TILE_COUNT)])
        if (
            np.any(counts < 0)
            or np.any(counts != np.floor(counts))
            or counts.sum() + sum(e.size for e in exposures) != 14
        ):
            return ()
        physical = counts.copy()
        for e in exposures:
            physical[e.tile] += e.size - e.jokers
            physical[JOKER] += e.jokers
        if np.any(physical > TILE_COUNTS):
            return ()
        indices, needed = self._matcher.remaining(exposures)
        natural = counts[:JOKER]
        matches = np.all(natural <= needed, axis=1) & np.all(
            natural >= self._matcher.required[indices], axis=1
        )
        return tuple(int(i) for i in indices[matches])


class _Matcher:
    def __init__(self, card):
        self.card = card
        self.total = np.zeros((len(card.patterns), JOKER), dtype=np.int16)
        self.required = np.zeros_like(self.total)
        self.groups = []
        for i, pattern in enumerate(card.patterns):
            self.groups.append(Counter(pattern.groups))
            for g in pattern.groups:
                self.total[i, g.tile] += g.size
                if g.size < 3:
                    self.required[i, g.tile] += g.size
        self.cache = {}

    def remaining(self, exposures):
        key = tuple(sorted((e.tile, e.size) for e in exposures))
        if key not in self.cache:
            used = Counter(Group(t, n) for t, n in key)
            indices = np.array(
                [
                    i
                    for i, p in enumerate(self.card.patterns)
                    if (not key or not p.concealed)
                    and all(self.groups[i][g] >= n for g, n in used.items())
                ],
                dtype=np.int64,
            )
            needed = self.total[indices].copy()
            for tile, size in key:
                needed[:, tile] -= size
            if len(self.cache) >= 512:
                self.cache.clear()
            self.cache[key] = indices, needed
        return self.cache[key]


NMJL_2026 = Card(
    "NMJL 2026",
    tuple(
        HandPattern(
            name, tuple(Group(t, n) for t, n in groups), value, concealed, bonus
        )
        for name, groups, value, concealed, bonus in pattern_data()
    ),
)
