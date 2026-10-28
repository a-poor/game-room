# Ticket to Ride v0

## Sources and edition

This environment implements the **2025 North America base game**, using the
[Days of Wonder English rulebook](https://cdn.svc.asmodee.net/production-daysofwonder/uploads/2025/07/7201N_TICKET2RIDEV2_RULES_EN_20250425_WEB.pdf).
It uses 33 destination tickets and an opening offer of four tickets, of which
at least two must be kept. It does not implement Europe, other maps, or the
USA 1910 expansion modes.

The 100 routes and original 30 tickets were transcribed from the
[Rob217 component dataset](https://github.com/Rob217/TicketToRideAnalysis/tree/master/data/USA)
and checked against the official board illustration. The three additional
2025 tickets are Atlanta–Washington (4), Omaha–New Orleans (8), and
San Francisco–Washington (21), documented in the
[2025 edition review](https://opinionatedgamers.com/2025/08/29/ticket-to-ride-2025-refresh-edition/).
The network contains 36 cities, 22 double-route pairs, and 309 train spaces.
Only component facts are included; no artwork is bundled.

## Usage

```python
from game_room import ticket_to_ride_v0

env = ticket_to_ride_v0.env(num_players=4, max_steps=10000, render_mode="ansi")
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

Supports 2–5 players with the AEC API, `ansi`/`human` text rendering, and
`max_steps=None` to disable the decision limit. `raw_env` is the unwrapped
implementation. Reset seeds both game randomness and action spaces; `options`
is accepted but unused. Illegal actions raise `ValueError` before state changes.
Inactive and finished players have zero masks. Finished players receive `None`.

## Rule behavior

Each player has 45 trains and four train cards. Each turn draws train cards,
claims one complete route, or draws tickets. A face-up locomotive consumes the
whole draw; a blind locomotive does not. Train discards recycle, and three
face-up locomotives trigger replacement. Gray routes require a single color
with optional locomotives. Players choose exactly how many locomotives to spend.
Double routes cannot share an owner; with 2–3 players only one side may be used.

Opening ticket rejects are pooled, shuffled, and placed beneath the remaining
deck. Later draws offer up to three tickets, require keeping at least one, and
let the player order rejects at the bottom. Held tickets cannot be discarded.

Finishing a turn with at most two trains triggers one final turn for everyone,
including that player. Final scores combine route points, successful/failed
tickets, and the ten-point longest-trail bonus (shared on ties). Trails can
revisit cities but cannot reuse edges. Winner ties use completed-ticket count,
then possession of the longest-trail bonus, then shared victory.

## Environment conventions

- The first player is chosen by the seeded RNG. Seats proceed in ascending
  order, wrapping around. Opening ticket selection starts at that seat.
- Drawing train cards takes two choices unless a face-up locomotive is taken
  or no legal second card remains. There is no voluntary pass or single-card
  draw option. Claim selection commits the player to a legal payment.
- If too few non-locomotive cards exist outside hands to form a valid market,
  the three-locomotive replacement is suspended. The visible cards remain
  available; replacement resumes as soon as the available pool permits it.
  This avoids an endless reshuffle in a case the rulebook does not adjudicate.
- If all three turn options are unavailable, a forced `PASS` is allowed. A
  full table of consecutive forced passes truncates with
  `info["truncation_reason"] == "no_legal_moves"`, without inventing a winner.
- Intermediate rewards are zero. At natural completion each winner receives
  +1 and other players −1; rewards are not necessarily zero-sum. Truncation
  gives zero rewards. Natural completion takes precedence over `max_steps`.

Final infos contain `scores`, `completed_tickets`, and `longest_paths` mappings
by agent name, all players' ticket IDs in `tickets`, and a `winners` list.
Infos are empty during ordinary play. Final scores may be negative.

## Component IDs and actions

`CITIES` is a tuple of names; `CITY_INDEX` maps names to integer city IDs.
`ROUTES` and `TICKETS` are immutable tuples in the order defined in
[`_ticket_to_ride_data.py`](../src/game_room/_ticket_to_ride_data.py).
Routes expose `.cities`, `.length`, and `.color`; tickets expose `.cities` and
`.points`. City and card IDs are zero-based. Route lengths score 1, 2, 4, 7, 10,
and 15 points for lengths 1–6.

Train color IDs: purple 0, red 1, orange 2, yellow 3, green 4, blue 5, white 6,
black 7, locomotive 8. Gray routes use −1. The deck contains twelve of each
ordinary color and fourteen locomotives.

The action space is `Discrete(184)`:

| IDs | Meaning |
| --- | --- |
| 0 | `DRAW_BLIND`: draw the top train card. |
| 1–5 | `MARKET_OFFSET + slot`: take one visible train card. |
| 6 | `DRAW_TICKETS`: request up to three tickets. |
| 7–106 | `CLAIM_OFFSET + route_id`: select a route. |
| 107–162 | Pay for the selected route. |
| 163–178 | Choose a subset of offered tickets to keep. |
| 179–182 | `RETURN_OFFSET + slot`: return a rejected ticket. |
| 183 | `PASS`: only when no ordinary turn action is available. |

Payment encoding is `payment_action(color, locomotives)`, or
`107 + 7*color + locomotives`. The remaining cost is paid in that one color.
All-locomotive gray payments use color 0 as their canonical encoding. Colored
routes always use their own color, even for an all-locomotive payment.

Ticket keeping uses `KEEP_OFFSET + bits`, where bit `i` keeps offer slot `i`.
The zero subset is never legal. After a normal draw, return unkept slots in the
order they should eventually be drawn: the first returned card is closer to
the top than subsequent returns. No return-order choice is needed during setup.

Phases: `SETUP=0`, `TURN=1`, `SECOND_DRAW=2`, `PAYMENT=3`, `CHOOSE_TICKETS=4`,
`RETURN_TICKETS=5`. Use the mask rather than assuming one decision per turn.
`connected(player, cities)` and `longest_path(player)` inspect public routes.

## Observation layout

The observation dictionary contains a float64 vector of length `180 + 4*N`
and an int8 action mask of length 184. All seat indices are absolute.

The first 15 values are:

1. Observer seat.
2. Turn owner seat.
3. Phase ID.
4. First player seat.
5. Decisions so far.
6. Decision limit, or zero for no limit.
7. Completed turns, excluding setup choices.
8. Final-turn countdown + 1; zero means not triggered.
9. Train draw-pile size.
10. Ticket draw-pile size.
11. Selected route ID + 1, or zero.
12. Number of players who have completed opening ticket selection.
13. Consecutive forced passes.
14. Size of the observer's visible ticket offer.
15. Number of offer slots still to return, visible only to their owner.

Then follow, in order:

- Five market card IDs + 1 (zero marks an empty slot).
- Nine public train-discard counts by color.
- 100 route owners, seat + 1; zero means unclaimed.
- `N` rows of four public values: trains remaining, route points, train-hand
  size, held-ticket count.
- Nine own train-card counts.
- 33 own held-ticket flags.
- Four private offered-ticket IDs + 1, padded with zeros. During setup every
  observer can see their own initial offer, even before their selection turn.
- Four private flags for rejected offer slots awaiting return.
- One setup flag.

Hand composition, other players' tickets/offers, future draw order, and returned
ticket identities/order are hidden from opponents. Rendering shows public
information only. Observation arrays are copies. Raw-object attributes expose
internal state for debugging and must not be used as policy inputs.
