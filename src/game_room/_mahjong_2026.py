"""2026 NMJL card facts expanded into concrete suit/rank combinations.

Sources and transcription checks are documented in docs/rules-mahjong.md.
This is an independent data expansion, not an implementation of another
engine's card language. No card artwork or explanatory prose is included.
"""

from itertools import combinations, permutations


def pattern_data():
    # IDs agree with mahjong_cards: winds E/S/W/N, dragons green/red/white.
    east, south, west, north, flower, white = 27, 28, 29, 30, 34, 33
    rows = []
    seen = set()
    counts = {}

    def tile(suit, rank):
        return 9 * suit + rank - 1

    def dragon(suit):
        return 31 + suit

    def add(section, line, value, groups, concealed=False):
        groups = tuple(sorted(groups))
        name = f"{section} #{line}"
        key = name, groups
        if key in seen:
            return
        seen.add(key)
        counts[name] = counts.get(name, 0) + 1
        rows.append(
            (
                f"{name} / {counts[name]:03d}",
                groups,
                value,
                concealed,
                section != "Singles and Pairs",
            )
        )

    def year(s):
        return [(tile(s, 2), 1), (white, 1), (tile(s, 2), 1), (tile(s, 6), 1)]

    news = [(t, 1) for t in (north, east, west, south)]
    for a, b, c in permutations(range(3)):

        def run(s, ranks, sizes):
            return [(tile(s, r), n) for r, n in zip(ranks, sizes)]

        add("2026", 1, 25, run(a, [2], [3]) + [(white, 3)] + run(b, [2, 6], [4, 4]))
        for r in (2, 6):
            add(
                "2026",
                2,
                25,
                year(a) + [(dragon(a), 3), (tile(b, r), 4), (dragon(b), 3)],
            )
        add("2026", 3, 25, [(flower, 3)] + year(a) + [(tile(b, 2), 3), (tile(c, 6), 4)])
        add(
            "2026", 4, 30, [(tile(a, 2), 2), (white, 2)] + run(b, [2, 6], [3, 3]) + news
        )

        for s in (a, b):
            add("2468", 1, 25, run(a, [2, 4], [3, 3]) + run(s, [6, 8], [4, 4]))
        add(
            "2468",
            2,
            30,
            [(flower, 2)] + run(a, [2, 8], [4, 4]) + run(b, [4, 6], [2, 2]),
        )
        add("2468", 3, 30, [(east, 2), (west, 2)] + run(a, [2, 4, 6, 8], [2, 3, 3, 2]))
        add(
            "2468",
            4,
            25,
            [(tile(a, 2), 4), (dragon(a), 3), (tile(b, 8), 4), (dragon(b), 3)],
        )
        add("2468", 5, 25, [(flower, 3)] + run(a, [2, 4, 6, 8], [2, 2, 3, 4]))
        for r in (2, 4, 6, 8):
            add(
                "2468",
                6,
                25,
                run(a, [2, 4, 6, 8], [1] * 4)
                + [(tile(b, r), 4), (dragon(b), 1), (tile(c, r), 4), (dragon(c), 1)],
            )
            add(
                "2468",
                7,
                30,
                [(flower, 3), (flower, 3)]
                + run(a, [2, 4, 6, 8], [1] * 4)
                + [(tile(b, r), 4)],
            )
        add(
            "2468",
            8,
            30,
            [(flower, 2)]
            + run(a, [2, 4, 6, 8], [1, 1, 1, 3])
            + run(b, [2, 4, 6, 8], [1, 1, 1, 3]),
            True,
        )

        for r in range(1, 10):
            add(
                "Any Like Numbers",
                1,
                30,
                [(tile(a, r), 4), (flower, 6), (tile(b, r), 4)],
            )
            add(
                "Any Like Numbers",
                2,
                25,
                [
                    (tile(a, r), 4),
                    (dragon(a), 1),
                    (tile(b, r), 3),
                    (dragon(b), 1),
                    (tile(c, r), 4),
                    (dragon(c), 1),
                ],
            )
            for d in range(3):
                add(
                    "Any Like Numbers",
                    3,
                    25,
                    [
                        (flower, 2),
                        (tile(a, r), 4),
                        (tile(b, r), 2),
                        (tile(c, r), 4),
                        (dragon(d), 2),
                    ],
                )
            add("Quints", 1, 40, [(tile(a, r), 5), (tile(b, r), 4), (tile(c, r), 5)])
        for r in range(1, 8):
            add("Quints", 2, 45, [(flower, 2)] + run(a, range(r, r + 3), [5, 2, 5]))
        for r, s in combinations(range(1, 10), 2):
            add("Quints", 3, 40, [(tile(a, r), 5), (tile(a, s), 5), (dragon(b), 4)])

        for r in (1, 5):
            add("Consecutive Run", 1, 25, run(a, range(r, r + 5), [2, 3, 2, 3, 4]))
        for r in range(1, 6):
            for s in (a, b):
                add(
                    "Consecutive Run",
                    2,
                    25,
                    [(flower, 3), (tile(a, r), 4)]
                    + run(s, range(r + 1, r + 4), [1] * 3)
                    + [(tile(a, r + 4), 4)],
                )
        for r in range(1, 8):
            add(
                "Consecutive Run",
                3,
                25,
                run(a, [r, r + 1], [2, 2])
                + run(b, [r, r + 1], [3, 3])
                + [(tile(c, r + 2), 4)],
            )
            for s in (a, b):
                add(
                    "Consecutive Run",
                    5,
                    25,
                    [
                        (flower, 3),
                        (tile(a, r), 2),
                        (tile(s, r + 1), 2),
                        (tile(a, r + 2), 3),
                        (dragon(s), 4),
                    ],
                )
            add(
                "Consecutive Run",
                7,
                25,
                [(flower, 2)] + run(a, range(r, r + 3), [4] * 3),
            )
            add(
                "Consecutive Run",
                7,
                25,
                [
                    (flower, 2),
                    (tile(a, r), 4),
                    (tile(b, r + 1), 4),
                    (tile(c, r + 2), 4),
                ],
            )
        for r in range(1, 7):
            for s in (a, b):
                add(
                    "Consecutive Run",
                    4,
                    25,
                    run(a, [r, r + 1], [3, 3]) + run(s, [r + 2, r + 3], [4, 4]),
                )
            add(
                "Consecutive Run",
                8,
                30,
                run(a, range(r, r + 3), [1, 2, 3])
                + run(b, range(r, r + 3), [1, 2, 3])
                + [(tile(c, r + 3), 2)],
                True,
            )
        for r in range(1, 9):
            add("Consecutive Run", 6, 30, [(flower, 6)] + run(a, [r, r + 1], [4, 4]))

        add("13579", 1, 25, run(a, [1, 3, 5, 7, 9], [2, 3, 2, 3, 4]))
        add(
            "13579",
            1,
            25,
            run(a, [1, 3], [2, 3]) + [(tile(b, 5), 2)] + run(c, [7, 9], [3, 4]),
        )
        for r in (1, 5):
            add(
                "13579",
                2,
                25,
                run(a, [r, r + 2], [3, 3]) + run(b, [r + 2, r + 4], [4, 4]),
            )
            add(
                "13579",
                3,
                30,
                [(north, 2), (south, 2)] + run(a, [r, r + 2, r + 4], [4, 2, 4]),
            )
            add(
                "13579",
                5,
                25,
                [(flower, 3)] + run(a, [r, r + 2, r + 4], [2, 2, 3]) + [(dragon(a), 4)],
            )
            add(
                "13579",
                6,
                25,
                run(a, [r, r + 2], [2, 2])
                + run(b, [r, r + 2], [3, 3])
                + [(tile(c, r + 4), 4)],
            )
            add(
                "13579",
                8,
                35,
                [(flower, 2)]
                + run(a, [r, r + 2, r + 4], [2, 2, 2])
                + [(tile(b, r), 3), (tile(c, r), 3)],
                True,
            )
        for r in (1, 3, 5, 7, 9):
            add(
                "13579",
                4,
                25,
                [(tile(a, n), 2 if n == r else 1) for n in (1, 3, 5, 7, 9)]
                + [(tile(b, r), 4), (tile(c, r), 4)],
            )
        for s in (a, b):
            add("13579", 7, 30, run(a, [1, 9], [4, 4]) + run(s, [3, 5, 7], [2, 2, 2]))
        add(
            "13579",
            9,
            30,
            [(flower, 2)] + run(a, [1, 3, 5, 7, 9], [1, 1, 1, 3, 3]) + [(dragon(b), 3)],
            True,
        )

        for sizes in ([4, 3, 3, 4], [3, 4, 4, 3]):
            add(
                "Winds and Dragons", 1, 25, list(zip((north, east, west, south), sizes))
            )
        for r in range(1, 7):
            for d in range(3):
                add(
                    "Winds and Dragons",
                    2,
                    25,
                    run(a, range(r, r + 4), [1] * 4)
                    + [(dragon(s), 4 if s == d else 3) for s in range(3)],
                )
        for r in range(1, 10):
            winds = (north, south) if r % 2 else (east, west)
            add(
                "Winds and Dragons",
                3 if r % 2 else 4,
                25,
                [(w, 3) for w in winds] + [(tile(a, r), 4), (tile(b, r), 4)],
            )
        for w in (east, south, west, north):
            add(
                "Winds and Dragons",
                5,
                25,
                [(flower, 3), (flower, 3), (w, 4), (dragon(a), 4)],
            )
        add(
            "Winds and Dragons",
            6,
            25,
            run(a, [1, 2, 3, 4], [1] * 4)
            + list(zip((north, east, west, south), (1, 2, 3, 4))),
        )
        for winds in ((north, south), (east, west)):
            add(
                "Winds and Dragons",
                7,
                25,
                [(flower, 2)]
                + [(w, 4) for w in winds]
                + [(dragon(a), 2), (dragon(b), 2)],
            )
        add(
            "Winds and Dragons",
            8,
            30,
            list(zip((north, east, west, south), (2, 3, 3, 2))) + year(a),
            True,
        )

        for s in (b, c):
            add(
                "369",
                1,
                25,
                run(a, [3, 6], [3, 3]) + [(tile(b, 6), 4), (tile(s, 9), 4)],
            )
        add(
            "369",
            2,
            25,
            run(a, [3, 6], [2, 2]) + run(b, [3, 6], [3, 3]) + [(tile(c, 9), 4)],
        )
        for s in (a, b):
            add(
                "369",
                3,
                25,
                [(flower, 3)] + run(a, [3, 6, 9], [2, 3, 2]) + [(dragon(s), 4)],
            )
        add("369", 4, 30, run(a, [3, 6], [2, 2]) + run(b, [6, 9], [3, 3]) + news)
        for r in (3, 6, 9):
            add(
                "369",
                5,
                25,
                [(flower, 2)]
                + [(tile(a, n), 2 if n == r else 1) for n in (3, 6, 9)]
                + [(tile(b, r), 4), (tile(c, r), 4)],
            )
        add(
            "369",
            6,
            30,
            [(flower, 2)] + run(a, [3, 6, 9], [3] * 3) + run(b, [3, 6, 9], [1] * 3),
            True,
        )

        for r in range(1, 10):
            add(
                "Singles and Pairs",
                1,
                50,
                [(w, 2) for w in (north, east, west, south)]
                + [(tile(s, r), 1) for s in range(3)]
                + [(dragon(s), 1) for s in range(3)],
                True,
            )
        add(
            "Singles and Pairs",
            2,
            50,
            run(a, [2, 4, 6, 8], [1, 1, 2, 2])
            + run(b, [2, 4, 6, 8], [1, 1, 2, 2])
            + [(tile(c, 8), 2)],
            True,
        )
        add(
            "Singles and Pairs",
            3,
            50,
            [(flower, 2)]
            + run(a, [3, 6, 9], [2, 1, 1])
            + run(b, [3, 6, 9], [1, 2, 1])
            + run(c, [3, 6, 9], [1, 1, 2]),
            True,
        )
        for r in range(1, 4):
            add("Singles and Pairs", 4, 50, run(a, range(r, r + 7), [2] * 7), True)
        add(
            "Singles and Pairs",
            5,
            50,
            run(a, [1, 3, 5, 7, 9], [2, 1, 1, 1, 2])
            + run(b, [1, 3, 5, 7, 9], [2, 1, 1, 1, 2]),
            True,
        )
        add(
            "Singles and Pairs",
            6,
            75,
            [(flower, 2)] + year(a) + year(b) + year(c),
            True,
        )
    return tuple(rows)
