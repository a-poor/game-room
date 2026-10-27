# Catan v0

`game_room.catan_v0` implements the Catan base game for 3–4 players using
PettingZoo's sequential AEC API. The source is the
[official 2020 base rules and almanac](https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf).
It uses the variable-board setup with nonadjacent red numbers and the almanac's
combined trade/build phase. Expansions, two-player variants and the preplaced
beginner layout are not included.

## Running a game

```python
from game_room import catan_v0

env = catan_v0.env(num_players=4, max_steps=20_000, render_mode="ansi")
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

Defaults: four players, a 20,000-decision episode limit, and no rendering.
`max_steps=None` disables the limit. Decisions include individual selections,
discards and trade-drafting steps, so the limit is not a turn count. Repeated
negotiation can otherwise continue indefinitely. Reaching the limit truncates
all agents without a winner or terminal reward. `reset(seed=...)` seeds board
generation, the starting seat, development deck, dice, robber draws and action
spaces. `options` is accepted but unused.

The environment supports `ansi` and `human` text rendering, with hex, vertex and
edge IDs to identify the public position. Rendering does not expose resource
compositions or unplayed development-card types. There is no parallel API:
discards, trades and robber choices temporarily change the decision maker.

Intermediate rewards are zero. The winner receives +1 and every opponent −1.
The first player with at least 10 points **on their own turn** wins immediately,
including before rolling, after taking an award or after buying a victory-point
card. An off-turn award recipient must wait for their turn. Terminal
`info["winners"]` lists the winning agent, and `info["scores"]` gives final scores
in seat order. Truncation reports an empty winners list and does not reveal
hidden scores.

## Board and setup

The board contains 19 hexes, 54 intersections and 72 edges. `HEX_COORDS` contains
axial `(q, r)` coordinates in increasing `(r, q)` order, with radius two.
`VERTEX_COORDS` contains exact integer drawing coordinates, sorted by `(y, x)`.
A hex center is `(2*q+r, 3*r)`; its six corner offsets are
`(1,1), (0,2), (-1,1), (-1,-1), (0,-2), (1,-1)`.
Multiplying x by `sqrt(3)` gives regular hexagon proportions for a custom renderer.

The exported immutable tuples define the complete public geometry:

- `HEX_VERTICES[hex]`: its six corner IDs.
- `EDGES[edge]`: its two vertex IDs, sorted; edges are lexicographically ordered.
- `VERTEX_EDGES[vertex]`, `NEIGHBORS[vertex]`, `VERTEX_HEXES[vertex]` and
  `HEX_NEIGHBORS[hex]`: adjacency lists.
- `COAST`: the 30 coastal edges in boundary order.
- `PORT_EDGES`: nine nonoverlapping harbor edges, at coast indices
  `0, 3, 6, 10, 13, 16, 20, 23, 26`.

Terrain is shuffled with three hills, four forests, four pastures, four fields,
three mountains and one desert. The eighteen number tokens are shuffled until
no 6 or 8 borders another 6 or 8. The desert has number zero and starts with the
robber. The five specialized harbors and four generic harbors are shuffled among
the fixed harbor locations; either endpoint controls the harbor when settled.

A uniformly random seat starts (equivalent to a symmetric pregame dice contest).
Setup proceeds clockwise once, then in reverse order, including two consecutive
placements for the last player. Each decision places a settlement, followed by
an adjoining road. All settlements obey the distance rule. Resources from the
second settlement are granted immediately. The first setup player then starts
the first normal turn.

## Implemented rules

Each turn starts in the roll phase. One eligible development card may be played
before rolling or during the combined trade/build phase, including after buying
other cards. Bought cards cannot be played that turn; victory-point cards are
an exception and contribute to the automatic win check immediately. Victory
points remain hidden until game end. The deck contains fourteen knights, two
each of Road Building, Year of Plenty and Monopoly, and five victory points.

Production uses two independent six-sided dice and pays one resource per
settlement or two per city on a matching, unblocked hex. There are nineteen cards
of each resource. If a shortage affects multiple claimants, nobody gets that
resource; if only one player claims it, they receive what remains. Other
resources still produce normally.

Rolling seven triggers mandatory discards for hands larger than seven. Players
choose half their cards, rounded down, one card at a time. Once all discards are
finished, the turn owner moves the robber to a different hex and chooses an
adjacent opponent to rob. Theft samples uniformly over that opponent's cards,
not uniformly over resource types. An empty-handed adjacent opponent is a legal
victim and yields nothing. If no opponent is adjacent, the effect ends without
a theft. Knights move and rob without causing discards.

Building costs, in `Resource` order:

| Action | Brick | Lumber | Wool | Grain | Ore |
| --- | --- | --- | --- | --- | --- |
| Road | 1 | 1 | 0 | 0 | 0 |
| Settlement | 1 | 1 | 1 | 1 | 0 |
| City upgrade | 0 | 0 | 0 | 2 | 3 |
| Development card | 0 | 0 | 1 | 1 | 1 |

Piece limits are fifteen roads, five settlements and four cities per player.
A city replaces an owned settlement and returns that settlement to supply.
Normal settlements require an owned adjoining road and no occupied neighboring
intersection. New roads connect to an owned building or road; an opponent's
building blocks extension through its intersection.

Longest Road is the longest continuous trail of distinct owned edges: branches
cannot be added together, cycles count, and opponents' buildings stop traversal.
A road may start or end at a blocked intersection. The award requires at least
five edges. The incumbent retains it on a tie; if the incumbent loses it and
multiple players tie, nobody holds it until a unique leader emerges. Largest
Army requires three played knights and changes hands only when surpassed.
Both awards are worth two points and are recomputed when relevant pieces or
knights change. Settlements can interrupt an opponent's road.

Road Building places up to two legal roads without paying resources, subject to
remaining pieces. Year of Plenty selects two available bank resources, allowing
the same type twice. Monopoly collects every opponent's cards of one chosen
type. These effects can finish before rolling and return to the roll phase.

## Trading and interface conventions

Bank trading uses the best available rate for the offered resource: 4:1 normally,
3:1 with a generic harbor, or 2:1 with that resource's specialized harbor.
The best-rate action is a convenience; 4:1 is always also allowed, and 3:1
remains available with a generic harbor even if a specialized harbor offers 2:1.
Each maritime action receives one card. Multiple actions express larger trades.
The newly built harbor is usable immediately. The requested bank card must be
available, and the two resource types must differ.

Domestic trades support arbitrary multi-resource bundles, acceptance, rejection
and counteroffers. Both sides must give something, and the same resource cannot
appear on both sides. Development cards cannot be traded. All negotiations
include the turn owner; opponents cannot trade with each other.

A tabletop conversation is serialized as follows:

1. The turn owner opens a trade and chooses a partner.
2. They draft an offer one card at a time, or invite the partner to make the
   first offer. Drafts can be edited or cancelled without moving resources.
3. Submission reveals the offer. The recipient can accept if they can pay,
   reject, or compose a counteroffer. Counteroffers reverse proposer/recipient
   roles and start with empty bundles.
4. Acceptance exchanges both bundles atomically. Rejection or cancellation ends
   that negotiation. Control returns to the turn owner, who can open another
   trade with anyone or continue building.

This protocol does not impose a numerical limit on offers or counteroffers.
Requests may include resources the recipient does not own; the proposing
player's action mask does not inspect the recipient's private resource mix.
Requests are bounded by nineteen cards per resource and the recipient's public
hand size. A draft is only visible to its author until submission. Past public
offers are not retained in the current observation, so policies wanting a trade
history must remember it.

Other explicit interface conventions:

- Discards resolve in clockwise order starting with the turn owner. Their
  required counts are captured when seven is rolled, before any discards.
- A Road Building effect can be ended early with `CANCEL`; legal placements
  already made remain. It cannot be started without a legal road site and piece.
- Year of Plenty can start with one bank card remaining and finishes when the
  bank is empty. It cannot be started with an empty bank.
- Resource and development hands are arrays of counts. Discarded resources
  return to the bank; played development cards never return to the deck.
- The last stolen resource type is visible only to the thief and victim. Other
  players see who was involved. This record resets at the next normal turn.
- There is no “claim victory” action: wins are checked after every decision and
  at the beginning of a normal turn. Winning takes priority over a step limit
  reached by that same decision.

## Action reference

The action space is `Discrete(317)`. Use the exported constants and the current
mask rather than assuming a single decision completes a turn. Illegal actions
raise `ValueError` without modifying state or advancing randomness. Only the
selected live agent has a nonzero mask. Dead agents must step with `None`.
`env.unwrapped.action_label(action)` describes an action ID for debugging.

`Resource` IDs are brick 0, lumber 1, wool 2, grain 3, ore 4.
`Development` IDs are knight 0, Road Building 1, Year of Plenty 2, Monopoly 3,
victory point 4. Victory-point cards have no play action.

| Constant / expression | IDs | Purpose |
| --- | --- | --- |
| `ROLL` | 0 | Roll for production |
| `END_TURN` | 1 | Finish the combined trade/build phase |
| `BUY_DEVELOPMENT` | 2 | Buy the top development card |
| `TRADE` | 3 | Open partner selection |
| `ACCEPT`, `REJECT`, `COUNTER` | 4–6 | Respond to an offer |
| `SUBMIT`, `CANCEL` | 7–8 | Submit/cancel a draft; cancel partner selection or finish free roads |
| `PLAY_DEVELOPMENT + kind` | 9–12 | Play an eligible knight/progress card |
| `BUILD_ROAD + edge` | 13–84 | Place a road in setup, normal building or Road Building |
| `BUILD_SETTLEMENT + vertex` | 85–138 | Place a settlement in setup or normal building |
| `BUILD_CITY + vertex` | 139–192 | Upgrade a settlement |
| `MOVE_ROBBER + hex` | 193–211 | Choose a new robber hex |
| `SELECT_PLAYER + seat` | 212–215 | Choose a victim or trade partner |
| `SELECT_RESOURCE + resource` | 216–220 | Discard, take a Plenty card or select the Monopoly resource |
| `MARITIME + give * 5 + receive` | 221–245 | Trade with the bank at the best rate |
| `OFFER_GIVE + resource` | 246–250 | Add one card to the offered bundle |
| `OFFER_WANT + resource` | 251–255 | Add one card to the requested bundle |
| `REMOVE_GIVE + resource` | 256–260 | Remove one offered card |
| `REMOVE_WANT + resource` | 261–265 | Remove one requested card |
| `INVITE_OFFER` | 266 | Ask the selected partner to draft the first proposal |
| `MARITIME_3 + give * 5 + receive` | 267–291 | Choose 3:1 with a generic harbor when 2:1 is also available |
| `MARITIME_4 + give * 5 + receive` | 292–316 | Choose 4:1 when a better rate is also available |

## Observations

`observe(agent)` returns `observation`, a nonnegative `float64` vector of length
`329 + 11 * num_players`, and `action_mask`, 317 `int8` entries. Seat indices are
absolute. Topology is static and available from the module; generated terrain,
numbers, ports and pieces are all included in the vector.

The first 24 entries are:

| Index | Value |
| --- | --- |
| 0–3 | Observer, selected decision maker, turn owner, first player's seat |
| 4 | Normal turn number, or zero during setup |
| 5–6 | Decisions taken; decision limit, or zero if unlimited |
| 7 | Phase index in `PHASES` |
| 8–9 | Setup placement index; last setup vertex + 1, or zero |
| 10 | Dice sum, or zero before rolling |
| 11 | Robber hex |
| 12–13 | Longest Road and Largest Army holders + 1, or zero |
| 14 | Whether a development card has been played this turn |
| 15–16 | Return phase after a development/robber effect; remaining free roads/Plenty selections |
| 17 | Development cards remaining in the deck |
| 18–19 | Current trade proposer and partner + 1, or zero |
| 20–21 | Whether a winner exists; winning seat + 1, or zero |
| 22–23 | Observer's actual score including hidden points; winning threshold (10) |

`PHASES`, in order: setup_settlement, setup_road, roll, main, discard, robber,
victim, free_roads, plenty, monopoly, trade_partner, trade_draft, trade_response.

The remaining sections follow consecutively:

| Length | Contents |
| --- | --- |
| 19 | Terrain resource IDs, or 5 for desert |
| 19 | Hex production numbers, zero for desert |
| 54 | Harbor type + 1 per vertex: zero means none, 1–5 specialized, 6 generic |
| 54 | Building owner + 1 per vertex, zero if vacant |
| 54 | Building size per vertex: 0 vacant, 1 settlement, 2 city |
| 72 | Road owner + 1 per edge, zero if vacant |
| 5 | Public bank counts in resource order |
| 5 | Observer's resource counts |
| 5 | Observer's unplayed development counts (including cards bought this turn) |
| 5 | Observer's newly bought, not-yet-eligible development counts |
| `10 * N` | Public player rows, described below |
| `N` | Outstanding discard counts per player |
| 5 + 5 | Offered and requested bundles, or zero when hidden/no offer |
| 3 | Last thief + 1, victim + 1, resource + 1; zero for absent or hidden values |

Each public player row contains resource hand size, unplayed development-card
count, public score, settlement count, city count, road count, and four played
development counts in knight/progress order. Scores reveal victory-point cards
only once someone wins. Before then, opponents' resource composition,
development types and deck order are never included. The observer's own score
and cards remain visible. Award lengths can be derived from the public road
network, while award ownership is explicit in the header.
