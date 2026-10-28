# Harmonies v0

## Sources and scope

Based on the [Libellud English rulebook](https://cdn.svc.asmodee.net/production-libelludv2/uploads/2026/07/HARMONIES_Rules_EN_compressed-1.pdf)
and the publisher's [FAQ](https://www.libellud.com/game/harmonies/), particularly
its clarification of circular river scoring and unlimited animal cubes.
Supports 1–4 players, both personal-board sides, solo play, and all ten optional
Nature's Spirit cards. Pulse and Crescendo are excluded.

The 32 animal cards and ten spirits were checked against the
[card sheet and factual inventory](https://github.com/miles2542/harmonies-bot/tree/main/docs).
The engine is implemented independently; no external engine code or artwork is
bundled. Card metadata is in
[`_harmonies_data.py`](../src/game_room/_harmonies_data.py). The rules illustration
supplies the board geometry, stacking rules, and scoring tables.

## Usage

```python
from game_room import harmonies_v0

env = harmonies_v0.env(
    num_players=2, board_side="A", nature_spirits=False,
    max_steps=10000, render_mode="ansi",
)
env.reset(seed=42)
for agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    action = None if terminated or truncated else env.action_space(agent).sample(
        observation["action_mask"]
    )
    env.step(action)
print(env.render())
env.close()
```

`raw_env` exposes the unwrapped AEC environment. Rendering supports `ansi` and
`human`. `max_steps=None` disables the safety limit. Reset seeds both game
randomness and action spaces; `options` is accepted but unused. Illegal actions
raise `ValueError` without mutation. Use `None` for finished agents.

## Rules and scoring

A turn takes one token group, places its three tokens, optionally drafts one
animal card, and may settle animals. These actions can interleave. No more than
four unfinished cards may be held; completed cards free a slot. Habitat patterns
allow rotations, not reflections, and require exact stack heights. Cube placement
is permanent; its cell cannot receive further tokens, but other habitat cells
may change or be reused. Buildings require two tokens with a red top.

Trees score 1/3/7 by height. Mountains score 1/3/7 only with an adjacent mountain.
Each field group of at least two cells scores five. Buildings score five with
three distinct neighboring top colors. Side A scores the best river using the
longest **shortest** path between water cells. Side B scores five per island,
including empty land. Animal progress scores the highest uncovered value;
activated spirits add their end-game landscape bonuses.

An empty pouch at refill or at most two empty cells triggers completion of the
current round. Everyone receives the same number of turns. Highest score wins,
then most animal cubes, then shared victory.

## Solo play and spirits

Solo uses three token groups and three displayed animal cards. At turn end,
unused groups go to the box and all groups refill. If no animal was drafted,
the player may discard one displayed card at the end of the turn. The goal is
score maximization; the terminal reward is the final score, not a win/loss value.
Final info also reports `suns`, including the board-side and spirit modifiers.

With `nature_spirits=True`, each player privately receives two spirit cards.
Choosing one is mandatory at the start of their first turn. It occupies one of
the four card slots until its cube is placed. Scoring depends on the final board
and only activates if the cube was placed. The rejected spirit leaves play.
Both ordinary and spirit cubes prevent further token placement in their cell.

## Environment conventions

- The seeded RNG chooses the first player; subsequent turns follow ascending
  seat indices, wrapping around. This replaces the real-world first-player rule.
- All placements, groups, and drafted cards are public. Bag order, draw-pile
  order, unchosen spirit identities, and opponents' spirit offers are hidden.
- Cubes are unlimited as clarified by the publisher FAQ. Ordinary animal cubes
  count for the tie-break; the distinct Nature's Spirit cube does not.
- An explicit `END_TURN` preserves the opportunity to take optional actions
  after placing the third token. It is legal only after a group was taken and
  all its tokens were placed. There is no voluntary token discard.
- Pouch exhaustion is checked when attempting a refill. Drawing the last three
  tokens successfully does not itself trigger the ending; an empty refill does.
- A building habitat requires a two-high building, with a red, brown, or gray
  base. A single red token is an unfinished building and does not qualify.
- Intermediate rewards are zero. Multiplayer winners receive +1 and opponents
  −1; tied rewards are not necessarily zero-sum. Solo receives its final score.
- Truncation yields zero rewards and no winner. Natural completion takes
  precedence over the decision limit.

Final info contains `scores`, `score_details`, and `animal_counts` mappings by
agent name and a `winners` list. Solo also includes `suns`. Infos are empty during
play. `score_details(player)` computes the public board's breakdown into
`trees`, `mountains`, `fields`, `buildings`, `water`, `animals`, and `spirits`.

## Board and component representation

The board contains 23 flat-top hexes in columns of lengths 5, 4, 5, 4, 5.
Odd columns are shifted down by half a cell. `CELLS` lists axial `(q, r)`
coordinates in column order; `CELL_INDEX` maps coordinates to cell IDs.
For column `q`, row `y` maps to `(q, y - q//2)`. `NEIGHBORS` contains adjacent
cell IDs. Cell IDs are 0–22.

Token IDs are blue 1, gray 2, brown 3, green 4, yellow 5, red 6. Initial counts
are 23, 23, 21, 19, 19, and 15 respectively. Stacks are bottom-to-top. Legal
nonempty stacks are single tokens, two/three gray tokens, one/two brown tokens,
green over one/two brown tokens, and red over a single red/brown/gray token.

`ANIMALS` contains 42 immutable records. Each has `.name`, `.points`, `.pattern`,
and `.spirit`. The pattern is a tuple of `(dq, dr, required_stack)` entries
relative to the cube's cell. `(0, RED)` denotes a valid building with any legal
base; zero otherwise denotes absence in observations. Patterns have all six
rotations. Habitat cells other than the target may already hold cubes.

Animal card IDs are 0–31, spirits 32–41. The following reference uses cumulative
scores after placing 1, 2, ... cubes; no cubes means zero points.

| ID | Animal | Cumulative scores |
| --- | --- | --- |
| 0 | Crocodile | 4, 9, 15 |
| 1 | Stingray | 4, 10, 16 |
| 2 | Hippopotamus | 3, 6, 10, 16 |
| 3 | Otter | 5, 10, 16 |
| 4 | Frog | 2, 4, 6, 10, 15 |
| 5 | Duck | 2, 4, 8, 13 |
| 6 | Flamingo | 4, 10, 16 |
| 7 | Lizard | 5, 10, 16 |
| 8 | Mouse | 5, 10, 17 |
| 9 | Peacock | 5, 10, 17 |
| 10 | Squirrel | 4, 9, 15 |
| 11 | Hedgehog | 5, 12 |
| 12 | Bee | 8, 18 |
| 13 | Bear | 5, 11 |
| 14 | Rabbit | 5, 10, 17 |
| 15 | Toucan | 4, 9, 14 |
| 16 | Dog | 4, 8, 13 |
| 17 | Koala | 3, 6, 10, 15 |
| 18 | Wolf | 4, 10, 16 |
| 19 | Kingfisher | 5, 11, 18 |
| 20 | Penguin | 4, 10, 16 |
| 21 | Bat | 3, 6, 10, 15 |
| 22 | Fennec Fox | 4, 9, 16 |
| 23 | Polar Bear | 5, 11 |
| 24 | Eagle | 5, 11 |
| 25 | Meerkat | 2, 5, 9, 14 |
| 26 | Crow | 4, 9 |
| 27 | Llama | 5, 12 |
| 28 | Heron | 5, 10, 17 |
| 29 | Raccoon | 6, 12 |
| 30 | Ladybug | 2, 5, 8, 12, 17 |
| 31 | Black Panther | 5, 11 |

| ID | Spirit | Activated end-game bonus |
| --- | --- | --- |
| 32 | Lion Spirit | Each yellow group: 2 points at size 1–2, 10 at size 3+. |
| 33 | Butterfly Spirit | 5 per yellow group, including singletons. |
| 34 | Deer Spirit | 4 per tree of height 2 or 3. |
| 35 | Owl Spirit | 3 per tree of height 1 or 2; 1 per height 3. |
| 36 | Raven Spirit | 4 per connected building group, including singletons. |
| 37 | Stork Spirit | 6 per connected group of at least two buildings. |
| 38 | Ram Spirit | 4 per mountain of height 2 or 3, even isolated. |
| 39 | Marmot Spirit | 3 per mountain of height 1 or 2; 1 per height 3, even isolated. |
| 40 | Dragonfly Spirit | 7 per connected group of at least two water cells. |
| 41 | Turtle Spirit | 2 per water cell. |

## Actions

The action space is `Discrete(1122)`. Inactive or finished agents have zero masks.
The active player can mix legal optional actions with token choices/placements.

| IDs | Meaning |
| --- | --- |
| 0 | `END_TURN`. |
| 1–5 | `GROUP_OFFSET + slot`: take a token group. |
| 6–10 | `DRAFT_OFFSET + slot`: draft a displayed animal. |
| 11–148 | `place_action(color, cell)`: place one selected token. |
| 149–1114 | `settle_action(card, cell)`: settle a cube from that card. |
| 1115–1116 | `SPIRIT_OFFSET + slot`: choose one of two initial spirits. |
| 1117–1121 | `SOLO_DISCARD_OFFSET + slot`: discard an animal and end the turn. |

Placement encoding is `11 + 23*(color-1) + cell`; settlement encoding is
`149 + 23*card_id + cell`. A settlement is allowed if **any** rotation matches;
the rotation need not be chosen, since only the cube position changes state.
Solo uses only slots 0–2. Markets refill at turn end, never immediately after
drafting or taking tokens. `can_place` and `can_settle` inspect placement legality
for raw-environment debugging; policies should use the mask.

## Observation layout

The dictionary contains a float64 `observation` of length `181 + 92*N` and an
int8 `action_mask` of length 1122. All seat indices are absolute.

The first 16 values are:

1. Observer seat.
2. Turn owner seat.
3. First player seat.
4. Decisions so far.
5. Decision limit, or zero when disabled.
6. Round number, starting at one.
7. Board side (A = 0, B = 1).
8. Nature's Spirits enabled flag.
9. Token group taken this turn flag.
10. Animal drafted this turn flag.
11. Final round triggered flag.
12. Tokens remaining in pouch.
13. Animal cards remaining in draw pile.
14. Player count.
15. Completed turns.
16. Current player still needs to choose their spirit flag.

Then follow, in order:

- Five groups of six token counts, in color order. Absent solo slots are zero.
- Five animal-market IDs + 1, with zero for unavailable/absent slots.
- Six counts of selected tokens still to place.
- Six public discarded-token counts (solo leftovers).
- 42 card owners, seat + 1; zero means unowned.
- 42 progress counts (number of placed cubes).
- `N` boards, each with 23 cells containing four values: three bottom-to-top
  token IDs padded with zero, then cube card ID + 1 or zero when absent.
- Two private spirit-offer IDs + 1 for the observer, padded with zero.
- 32 public animal-discard flags (solo).

Board state and progress include completed cards. Returning arrays are copies.
Rendering exposes public state only. Raw object attributes expose the hidden
state for tests and debugging and should not be read by policies.
