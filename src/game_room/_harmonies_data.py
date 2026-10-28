"""Harmonies component facts; see docs/rules-harmonies.md for sources."""

from typing import NamedTuple


class Animal(NamedTuple):
    name: str
    points: tuple[int, ...]
    # Offsets from the cube cell, with bottom-to-top stacks; 0 is a building base wildcard.
    pattern: tuple[tuple[int, int, tuple[int, ...]], ...]
    spirit: bool = False


ANIMALS = (
    Animal(
        "Crocodile", (4, 9, 15), ((2, 0, (3, 3, 4)), (1, 0, (1,)), (0, 0, (1,))), False
    ),
    Animal("Stingray", (4, 10, 16), ((1, 0, (2,)), (0, 1, (2,)), (0, 0, (1,))), False),
    Animal("Hippopotamus", (3, 6, 10, 16), ((1, 0, (2, 2, 2)), (0, 0, (1,))), False),
    Animal("Otter", (5, 10, 16), ((2, 0, (4,)), (1, 0, (4,)), (0, 0, (1,))), False),
    Animal("Frog", (2, 4, 6, 10, 15), ((1, 0, (4,)), (0, 0, (1,))), False),
    Animal("Duck", (2, 4, 8, 13), ((1, 0, (0, 6)), (0, 0, (1,))), False),
    Animal("Flamingo", (4, 10, 16), ((1, 0, (5,)), (0, 0, (1,)), (0, 1, (5,))), False),
    Animal("Lizard", (5, 10, 16), ((2, 0, (5,)), (1, 0, (5,)), (0, 0, (0, 6))), False),
    Animal("Mouse", (5, 10, 17), ((0, 1, (5,)), (0, 0, (0, 6)), (-1, 0, (5,))), False),
    Animal(
        "Peacock", (5, 10, 17), ((0, 1, (1,)), (0, 0, (0, 6)), (-1, 0, (1,))), False
    ),
    Animal("Squirrel", (4, 9, 15), ((1, 0, (3, 3, 4)), (0, 0, (0, 6))), False),
    Animal(
        "Hedgehog", (5, 12), ((1, 0, (3, 4)), (0, 0, (0, 6)), (0, 1, (3, 4))), False
    ),
    Animal(
        "Bee",
        (8, 18),
        ((0, 1, (5,)), (0, 0, (3, 4)), (-1, 1, (5,)), (-1, 0, (5,))),
        False,
    ),
    Animal("Bear", (5, 11), ((1, 0, (2, 2)), (0, 0, (4,)), (0, 1, (2, 2))), False),
    Animal("Rabbit", (5, 10, 17), ((2, 0, (0, 6)), (1, 0, (4,)), (0, 0, (4,))), False),
    Animal("Toucan", (4, 9, 14), ((1, 0, (1,)), (0, 0, (3, 4)), (0, 1, (1,))), False),
    Animal("Dog", (4, 8, 13), ((1, 0, (0, 6)), (0, 0, (3, 4))), False),
    Animal("Koala", (3, 6, 10, 15), ((1, 0, (4,)), (0, 0, (3, 4))), False),
    Animal("Wolf", (4, 10, 16), ((1, 0, (5,)), (0, 0, (3, 3, 4)), (0, 1, (5,))), False),
    Animal(
        "Kingfisher",
        (5, 11, 18),
        ((0, 1, (1,)), (0, 0, (3, 3, 4)), (-1, 0, (1,))),
        False,
    ),
    Animal("Penguin", (4, 10, 16), ((0, 1, (1,)), (0, 0, (2,)), (-1, 0, (1,))), False),
    Animal("Bat", (3, 6, 10, 15), ((1, 0, (3, 3, 4)), (0, 0, (2,))), False),
    Animal("Fennec Fox", (4, 9, 16), ((2, 0, (5,)), (1, 0, (2,)), (0, 0, (2,))), False),
    Animal("Polar Bear", (5, 11), ((1, 0, (1,)), (0, 0, (2, 2)), (0, 1, (1,))), False),
    Animal("Eagle", (5, 11), ((1, 0, (5,)), (0, 0, (2, 2, 2))), False),
    Animal("Meerkat", (2, 5, 9, 14), ((1, 0, (5,)), (0, 0, (2,))), False),
    Animal("Crow", (4, 9), ((0, 1, (0, 6)), (0, 0, (5,)), (-1, 0, (0, 6))), False),
    Animal("Llama", (5, 12), ((2, 0, (2, 2)), (1, 0, (5,)), (0, 0, (5,))), False),
    Animal(
        "Heron", (5, 10, 17), ((0, 1, (3, 4)), (0, 0, (5,)), (-1, 0, (3, 4))), False
    ),
    Animal(
        "Raccoon",
        (6, 12),
        ((0, 1, (1,)), (0, 0, (5,)), (-1, 1, (1,)), (-1, 0, (1,))),
        False,
    ),
    Animal("Ladybug", (2, 5, 8, 12, 17), ((1, 0, (4,)), (0, 0, (5,))), False),
    Animal(
        "Black Panther", (5, 11), ((2, 0, (3, 4)), (1, 0, (3, 4)), (0, 0, (5,))), False
    ),
    Animal("Lion Spirit", (0,), ((1, 0, (3, 4)), (0, 0, (5,)), (-1, 0, (5,))), True),
    Animal(
        "Butterfly Spirit",
        (0,),
        ((1, 1, (5,)), (1, 0, (1,)), (0, 1, (1,)), (0, 0, (5,))),
        True,
    ),
    Animal(
        "Deer Spirit", (0,), ((1, 0, (3, 3, 4)), (0, 0, (3, 4)), (-1, 0, (4,))), True
    ),
    Animal("Owl Spirit", (0,), ((1, 0, (4,)), (0, 0, (3, 3, 4)), (0, 1, (4,))), True),
    Animal("Raven Spirit", (0,), ((2, 0, (0, 6)), (1, 0, (4,)), (0, 0, (0, 6))), True),
    Animal("Stork Spirit", (0,), ((2, 0, (5,)), (1, 0, (0, 6)), (0, 0, (0, 6))), True),
    Animal("Ram Spirit", (0,), ((1, 0, (2, 2)), (0, 0, (2, 2, 2))), True),
    Animal("Marmot Spirit", (0,), ((0, 1, (2,)), (0, 0, (2, 2)), (-1, 0, (2,))), True),
    Animal(
        "Dragonfly Spirit", (0,), ((1, 0, (3, 4)), (0, 0, (1,)), (-1, 0, (3, 4))), True
    ),
    Animal("Turtle Spirit", (0,), ((1, 0, (2, 2)), (0, 0, (1,)), (-1, 0, (1,))), True),
)
