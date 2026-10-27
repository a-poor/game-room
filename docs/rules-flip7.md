# Flip 7 environment reference

Implementation: `game_room.flip7_v0`, based on The Op's
[official rules, edition 3.1](https://cdn.shopify.com/s/files/1/0611/3958/3198/files/25_FLIP_7_TB_RULES_C_Rev_9_2_25_ND.pdf?v=1756935535).
The game name belongs to its respective owner; this repository supplies an
independent implementation and does not include card artwork.

## Implemented rules

The deck contains 94 cards: one zero, n copies of each number n from 1 through
12, one each of +2/+4/+6/+8/+10/x2, and three each of Freeze, Flip Three, and
Second Chance. Each round starts with one deal per active seat. Special cards
interrupt that deal. Players then choose Hit or Stay in seat order; staying
requires at least one card in front of the player.

A duplicate number busts its recipient unless they have Second Chance, in which
case the duplicate and Second Chance are discarded immediately. A second Second
Chance must go to another active player without one, or be discarded if there
is no eligible recipient. Freeze banks a target's current hand. Flip Three
forces up to three draws, stopping on a bust or seven distinct numbers. Every
card counts toward those three draws. Second Chance works immediately; Freeze
and further Flip Three cards wait until the three draws finish and are canceled
if their recipient busts.

Seven number cards end the round for everyone. Otherwise the round ends when
everyone stays, freezes, or busts. Non-busted players score their numbers,
doubled if they hold x2, then add modifiers and the 15-point Flip 7 bonus.
The highest total wins once a completed round has brought someone to the target
score. The unused deck carries forward between rounds. Only discards are
reshuffled when it runs out; cards on the table remain out of the draw pile.

## Explicit conventions

- Seats are fixed as `player_0` through `player_N-1`. The first dealer is seat 0;
  dealing and turns start on the dealer's left, and the dealer rotates each round.
- Seats already frozen or busted during the opening deal are skipped.
- Multiple deferred actions resolve in draw order. Each Flip Three chain resolves
  before the next deferred action. The player who drew a special card chooses its
  target, including when forced to draw it by someone else. A player frozen while
  resolving deferred actions still chooses targets for their remaining actions;
  a bust cancels their remaining actions. The printed rules do not specify all
  of these ordering cases.
- When the only eligible target is oneself, target selection remains an explicit
  decision with a single legal action.
- Tied leaders share victory; the rules do not prescribe a tiebreaker.
- If every card is on the table and both draw and discard piles are empty when
  a draw is needed, bank the round and proceed normally. This rare large-table
  case has no printed ruling; this convention avoids an impossible draw.
- Observations retain publicly revealed cards even after players turn them face
  down. This gives policies perfect recall of public card counts, never deck
  order. Deferred action order and scheduled events are also public.
- This version uses a single deck and the standard 3-18 player competitive game.
  The one/two-player round-count challenge and multiple decks are not implemented.

## API and rewards

`env(num_players=3, target_score=200, max_rounds=None, render_mode=None)` returns a
wrapped PettingZoo AEC environment. `raw_env(...)` exposes the underlying class.
`reset(seed=...)` seeds shuffling and action-space sampling; reset options are
accepted but unused. `render_mode` accepts `None`, `"ansi"`, or `"human"`.

Each round awards its scored points as rewards to all players together. Stayed,
frozen, and busted players remain episode agents but are skipped for ordinary
turns. On episode termination/truncation, drain remaining agents with `step(None)`
as required by the AEC API. Natural termination takes precedence over a round
limit. `info` includes `scores`, `round_scores`, `completed_round`, and `winners`
after a round ends, and retains that information until the next completed round.
No future draw information is exposed through observations, rendering, or infos.

## Observation layout

`observe(agent)` returns a dictionary:

- `action_mask`: an `int8` vector of length `2 + N`. Legal decisions have value 1;
  all other entries are zero. Non-selected and finished agents have all-zero masks.
- `observation`: a flat `float64` vector of length `419 + 24*N`, with the following
  consecutive blocks. All values represent nonnegative integers. Seats are in
  absolute order, not rotated relative to the observer.

| Block | Size | Contents |
| --- | --- | --- |
| Header | 10 | Observer seat, acting seat, dealer seat, round number (1-based), draw-pile size, target score, max rounds (0 means unlimited), target-decision flag, pending card ID (0 when absent), Flip 7 seat plus one (0 when absent) |
| Player rows | N * 24 | For each seat: banked total, round status, counts for each of 22 card IDs |
| Discard counts | 22 | Counts in the current discard pile by card ID |
| Scheduled events | 128 * 3 | Ordered triples `(kind, seat, card)`, padded with zeros |
| Deferred actions | 3 | Card ID plus one, in draw order, padded with zeros |

Round status codes: `0` active, `1` stayed, `2` busted, `3` frozen. Banked totals
update together at round end. Final observations clear hands after moving cards
to discard; `round_scores` in infos preserves the final scoring result.

Scheduled event kinds: `1` opening deal, `2` forced draw, `3` end of a Flip Three,
`4` resolve a deferred action, `5` next ordinary turn. Only kind 4 uses the card
field. Draw events contain no card identities. The fixed padding exceeds the
maximum scheduler size for the supported player counts and single deck.

| Card IDs | Meaning |
| --- | --- |
| 0-12 | Number cards |
| 13-17 | +2, +4, +6, +8, +10 |
| 18 | x2 |
| 19 | Freeze |
| 20 | Flip Three |
| 21 | Second Chance |

Actions: `HIT=0`, `STAY=1`, and `TARGET_OFFSET + seat` with `TARGET_OFFSET=2`.
The `Card` enum exports symbolic names for special cards. The vector layout and
action IDs are part of the versioned `flip7_v0` interface.
