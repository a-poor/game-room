# Kingdomino v0

## Source and scope

Uses the [Blue Orange 2017 multilingual rulebook](https://blueorangegames.eu/wp-content/uploads/2017/04/KINGDOMINO_rules_ML.pdf),
English pages 2–4, with the French setup text on page 5 clarifying that the owner
of each randomly drawn king chooses its domino. This edition uses 36 randomly
selected dominoes and rows of three for three players. It does not use the
alternative setup that reveals four tiles and discards the unclaimed one.

The numbered terrain/crown inventory was checked against the
[individual tile records and photograph](https://github.com/RuPaulsDataRace/Kingdomino-For-Queens).
The dataset's summary table understates mine crowns: its individual records and
photograph show ten (one single, three doubles, one triple). The implementation
uses those individual records. No artwork is bundled.

`game_room.kingdomino_v0.env(num_players=4, max_steps=10000, render_mode=None)`
returns a PettingZoo AEC environment for 2–4 players. `raw_env` is unwrapped.
Only the base 5×5 game is implemented: no Dynasty, Middle Kingdom, Harmony,
Mighty Duel, or expansions. Render modes are `ansi` and `human`.

## Rules and conventions

- Two players use 24 random dominoes and two kings each. Three use 36 and one
  king each. Four use all 48 and one king each.
- Rows are sorted by printed domino number. Initial king order is a seeded
  random permutation of all kings, including both kings per player at two seats.
  Each king's owner chooses one available domino in that order.
- Once the opening row is claimed, reveal the next row. Starting at the lowest
  domino number in the current row, each king's owner places their current
  domino, then chooses an unclaimed domino in the next row. That choice determines
  the next round's order. At two players, a player acts twice per row.
- The last row has placements only. Every player receives 12 dominoes; games
  take six placement rounds at two players and twelve at three or four players.
- Place both halves on empty squares, with at least one half orthogonally
  adjacent to an existing matching terrain or the wild starting castle.
  The two new halves cannot supply their own required connection.
- The bounding rectangle of all occupied squares, including the castle, must
  be at most five rows by five columns. The castle need not be centered.
- Discard a domino only if no legal placement exists. No voluntary discard.
- Each orthogonally connected terrain region scores its area times its total
  crowns. Crownless regions score zero. The castle neither scores nor joins
  regions. Highest score wins; ties break by largest region, then total crowns,
  then shared victory.

The exact domino data is exported as immutable `DOMINOES`: entry `number - 1`
contains two `(terrain, crowns)` pairs. Terrain IDs are `EMPTY=0`, `WHEAT=1`,
`FOREST=2`, `WATER=3`, `GRASS=4`, `SWAMP=5`, `MINE=6`, and `CASTLE=7`.
The 48 dominoes contain 96 terrain squares and 39 crowns.

## Coordinates, actions, and phases

Each kingdom is stored on a 9×9 canvas with its castle at `(4, 4)`. This represents
all valid 5×5 kingdoms around the castle without forcing an early choice of
where its edges will lie. Rows increase downward; columns increase rightward.
Placement legality still limits the occupied bounding rectangle to 5×5.

The action space is `Discrete(329)`:

| IDs | Meaning |
| --- | --- |
| 0–3 | Claim the corresponding slot in the current opening row or next row. |
| 4–327 | Place the current domino at an anchor and orientation. |
| 328 | `DISCARD`: discard an unplaceable domino. |

For placement, use `placement_action(row, column, direction)` or
`4 + (row * 9 + column) * 4 + direction`. The first half of `DOMINOES[number-1]`
is at the anchor. The second half is right (0), down (1), left (2), or up (3).
`decode_placement(action)` reverses this encoding.

Phases are `INITIAL_DRAFT=0`, `PLACE=1`, and `DRAFT=2`. A turn usually requires
one placement action followed by one claim action from the same player. Slot
indices refer to ascending domino order, not seat indices. Only slots 0–2 are
used at three players. Consumed current-row dominoes become zero without
shifting the remaining slots.

The action mask lists all legal actions for the selected player and is zero
for other players or finished agents. `DISCARD` is available only when no
placement is possible. Invalid actions raise `ValueError` without changing
state. Finished agents receive `None`. There is no parallel API.

## Observations

`observe(agent)` returns a float64 `observation` vector of length `78 + 162*N`
and an int8 `action_mask` of length 329. Seat indices are absolute.

The first ten values are:

1. Observer seat (zero-based).
2. Selected seat (zero-based).
3. Phase ID.
4. Cursor: opening draft position or current-row slot. Equals row size at end.
5. Placement round number, starting at one.
6. Unrevealed draw-pile size.
7. Decisions taken so far.
8. Decision limit, or zero when disabled.
9. Pending domino number during placement, otherwise zero.
10. Number of players.

The remaining fields, in order:

- Four opening-order entries: seat + 1, padded with zero for the absent fourth
  king at three players. This shuffled order is public and stays fixed.
- Four current-row `(domino number, owner seat + 1)` pairs, then four next-row
  pairs. Zero owner means unclaimed; absent slots are `(0, 0)`. A consumed
  current tile has number zero but retains its owner until the row is replaced.
- Every player's 9×9 board, in absolute seat order, then row-major cell order:
  two values `(terrain ID, crowns)` per cell, totaling `162*N` values.
- 48 public discard flags, indexed by domino number minus one.

All boards, revealed dominoes, and king assignments are public. Unrevealed
order and unused dominoes are hidden. Removed tiles are not marked as discards.
Arrays are copies. Infos are empty during play. The raw object exposes game
state for testing; policies should consume observations and infos only.

## End of game and rewards

Intermediate rewards are zero. Each winner receives +1; other players receive
−1 (not necessarily zero-sum). Terminal info contains `scores`, `score_details`
(each seat's `(points, largest region, crowns)`), and `winners`.
`score_details(player)` computes these public board statistics at any time.

A complete game takes exactly `24*N` decisions, counting opening claims and
final placements. The optional `max_steps` limit defaults to 10,000; `None`
disables it. Truncation awards no rewards or winner. Natural completion wins
if it coincides with the limit.

`reset(seed=...)` seeds game randomness and action spaces; `options` is accepted
but unused. Text rendering displays public boards, row assignments, and scores.
