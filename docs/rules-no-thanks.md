# No Thanks! v0

## Source and scope

Based on the [AMIGO English rulebook](https://blog.amigo-spiele.de/content/ap/rule/02455-GB-AmigoRule.pdf),
base-game rules on pages 1–10. The special-card expansion is not included.

`game_room.no_thanks_v0.env(num_players=3, max_steps=10000, render_mode=None)`
returns a PettingZoo AEC environment for 3–7 players. `raw_env` exposes the
unwrapped implementation. Render modes are `ansi` and `human`.

## Rules and conventions

- Shuffle the 33 cards numbered 3–35; remove nine unseen. Play the remaining 24.
- Each player starts with 11 chips at 3–5 players, nine at six, or seven at seven.
  Unused chips stay out of play.
- Taking the face-up card also takes all chips on it. That same player then
  decides whether to take or pass on the next card.
- Passing costs one chip, adds it to the card, and advances to the next seat.
  A player with no chips must take the card.
- Collected cards are public. Chip counts are private. Removed cards and the
  future draw order are hidden; chips on the current card are public.
- Once the last card is taken, sum each player's isolated cards and the lowest
  number in each consecutive run, then subtract remaining chips. Lowest wins;
  tied leaders share victory. Scores can be negative.
- The first seat is selected by the seeded RNG, replacing the rulebook's
  real-world starting-player criterion. Play proceeds through ascending seat
  indices, wrapping at the last seat.

There are no intermediate rewards. Each winner receives +1 and each other
player −1 at termination (not necessarily zero-sum). Final `info` contains
`scores` keyed by agent name and `winners` as a list of names. Scores that would
reveal opponents' chips are not supplied during play. Text rendering shows
public cards and the pot, with final scores only after a completed game.

The default safety limit is 10,000 decisions; `max_steps=None` disables it.
Truncation awards no reward or winner and does not reveal private scores.
Natural completion takes precedence if it coincides with the limit.

## Actions

| ID | Constant | Meaning |
| --- | --- | --- |
| 0 | `TAKE` | Take the card and its chips; draw the next card if any. |
| 1 | `PASS` | Pay one chip and advance to the next player. |

Always use `observation["action_mask"]`. It is zero for inactive or finished
agents; passing is masked when the active player has no chips. Illegal actions
raise `ValueError` before modifying state. Finished agents must receive `None`.
There is no parallel API.

## Observation

The dictionary contains a float64 `observation` vector of length `8 + 33*N`
and an int8 `action_mask` of length 2, where `N` is the number of players.
All seat indices are absolute and zero-based.

The first eight values are:

1. Observer seat.
2. Selected seat.
3. Current card value, or zero after completion.
4. Chips on the current card.
5. Number of cards remaining in the draw pile, excluding the face-up card.
6. Observer's own chip count.
7. Decisions taken so far.
8. Decision limit, or zero when disabled.

Then follow `N` rows of 33 binary values in seat order, indicating ownership
of each card from 3 through 35. No removed-card identities, future draw order,
or opponent chip counts are exposed. Returned arrays are independent copies.

`reset(seed=...)` seeds both game randomness and per-agent action spaces.
`options` is accepted but unused. The raw Python object is intended for debugging
and tests; policies must use observations and infos rather than reading its
private game state.
