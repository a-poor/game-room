# American Mahjong v0

`game_room.mahjong_v0` is a four-player PettingZoo AEC environment. One episode
is one deal, ending with Mah Jongg or a wall game. Its default card is
`game_room.mahjong_cards.NMJL_2026`; pass a different `Card` to configure the
winning hands without changing the engine.

## Usage

```python
from game_room import mahjong_v0
from game_room.mahjong_cards import NMJL_2026

env = mahjong_v0.env(card=NMJL_2026, render_mode="ansi")
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

`num_players` must be 4. The default decision limit is 10,000; `max_steps=None`
disables it. `reset(seed=...)` seeds game randomness and action spaces.
`options` is accepted but unused. `raw_env` provides the unwrapped environment.
Illegal actions raise before changing game state. Inactive and finished agents
have zero action masks; finished agents accept `None`.

## Sources and card edition

Core play follows the American Mah Jongg Association's
[rules overview](https://www.americanmahjonggassociation.com/american-mah-jongg-rules),
[Charleston reference](https://guide.americanmahjonggassociation.com/how-to-play-american-mah-jongg/the-charleston/),
[discard claims](https://guide.americanmahjonggassociation.com/how-to-play-american-mah-jongg/claiming-discarded-tile-exposure/),
[joker exchanges](https://guide.americanmahjonggassociation.com/how-to-play-american-mah-jongg/jokers-and-joker-exchange/),
[payments](https://guide.americanmahjonggassociation.com/how-to-play-american-mah-jongg/hand-value-and-payment/),
and [dead hands](https://guide.americanmahjonggassociation.com/how-to-play-american-mah-jongg/dead-hand/).
The conventions below specify how physical timing and adjudication become
discrete, legal actions.

The default card data is independently expanded from the
[2026 card guide](https://guide.americanmahjonggassociation.com/2026-card/).
Values and concealed flags were cross-checked against the
[Riichi Advanced 2026 card data, pinned revision](https://github.com/EpicOrange/riichi_advanced/blob/0f0b358ab92f266cb414738f1da9195973909faf/priv/static/mods/american/am_card_nmjl_2026.majs).
No implementation code, artwork, or card commentary is copied into this package.
This environment is not affiliated with or endorsed by the NMJL.

The card has 55 numbered lines, including printed alternatives, and expands
to 1,077 distinct concrete patterns. The expansion preserves suit relationships,
fixed versus variable ranks, matching/opposite dragons, and printed group
boundaries. The public guide resolves three transcription differences in the
cross-check source: Any Like Numbers #3 permits any dragon; Consecutive Run #5
has three flowers; Singles and Pairs #2 puts its second set of singles in the
second suit. Tests cover all realizations and these specific cases.

The default is pinned to 2026 and never changes according to the system date.
Record the card alongside seeds when comparing experiments.

## Rule behavior

The 152 tiles include four of each suited tile, wind and dragon, eight
interchangeable flowers, and eight jokers. East starts with 14 tiles; others
start with 13. Seats advance to the right in ascending index order modulo four.
East is chosen by the seeded RNG. The shuffled wall represents the distribution
produced by physical walls and a dice break.

The first Charleston passes three tiles right, across, then left. All players
must agree to a second Charleston, which passes left, across, then right.
One stop vote proceeds to courtesy negotiation. Opposite seats offer 0–3 tiles;
each pair exchanges its smaller offer. Jokers cannot be passed.

Blind passing is available on the first left and last right only. Each `BLIND`
action substitutes one unseen incoming tile for a tile from the player's hand.
No incoming tiles are shown until all outgoing selections are committed.
The engine shuffles each outgoing packet and circulates tiles through blind
commitments before delivering retained tiles. If every player blind-passes all
three on the first left, East must select one natural to start circulation.
If this happens on the last right, passing ends and courtesy is skipped.

After passing, East discards without drawing. Subsequent turns draw one tile
automatically after responses to the previous discard. Claiming takes the
discard instead of a draw, exposes the chosen group, and gives the claimant
the turn. Pungs, kongs, quints and sextets are supported. There are no chows,
concealed-kong actions, flower replacement draws, or kong replacement draws.
Exactly 14 tiles, including exposures, are required to win.

A natural discard can complete an exposed group of at least three identical
tiles. It can complete a single, pair, or concealed pattern only for Mah Jongg.
Discarded jokers cannot be claimed. Responses are sealed and resolved together:
Mah Jongg beats an exposure, then proximity in turn order breaks ties. Even
after the last wall draw, the final discard can be claimed before a wall game.

Jokers fill groups of three or more, including an entirely joker-filled group
in a winning concealed rack. Singles and pairs must be natural. An exposed
group always contains its claimed natural tile. During their turn a player
can redeem matching natural tiles for jokers in any exposure, including their
own and valid exposures of a dead player. Exchanges can repeat. Calling a
discard solely to redeem a joker is unavailable.

## Scoring and rewards

Rewards are **net point payments**, not binary win/loss rewards. Intermediate
rewards are zero. A discard win charges its discarder twice the hand value and
the other opponents once; self-pick charges every opponent twice. Winning
after a joker exchange is self-picked, including after beginning that turn
with an exposure claim. East has no additional multiplier.

A jokerless win doubles payments again when the pattern's `jokerless_bonus`
is true. The 2026 Singles and Pairs patterns disable that bonus. Exposed jokers
count when determining whether a hand is jokerless. If multiple patterns
match, the engine selects the largest payment, breaking equal-value ties by
card order. All opponents pay, including dead players. Payments sum to zero.

Wall games and termination with three dead hands have no winners and zero
payments. A decision-limit truncation also has zero rewards. Natural completion
takes precedence over the limit. Terminal info contains `card`, `end_reason`,
`scores` by agent name, and `winners`. A win adds `pattern`, `pattern_index`,
`hand_value`, `jokerless`, `self_picked`, and `discarder` (or `None`).

## Digital play conventions

- There is a `START` decision for East's initial winning hand. A heavenly hand
  must be declared; otherwise East continues to the Charleston. Later East can
  declare Mah Jongg after all passing is complete.
- Passes and discard responses are serialized for AEC, but commitments remain
  private until resolution. Every live opponent responds, including players
  whose only legal action is `PASS`. Hidden eligibility never determines which
  opponents are asked to respond.
- A claim chooses its final group size and joker composition in one action.
  Exposures cannot be edited afterward except through joker redemption. The
  mask prevents exposures incompatible with every card pattern. Selecting an
  exposure does not commit the player to a particular card line.
- A player may declare an opponent dead during their own `TURN` when public
  discards and exposures prove that no card pattern remains possible. The
  proof accounts for redeemable jokers and does not inspect concealed hands
  or wall order. Only correct challenges are legal; they are adjudicated
  immediately. Dead seats stop drawing, discarding and responding, but their
  agents remain until episode completion so they can receive final payments.
- Physical mistakes, disputed challenges, misnaming, timing races, and their
  penalties are excluded. There is no out-of-turn picking, deliberate false
  Mah Jongg, or manual racking. Invalid actions are rejected atomically.
- Episodes do not rotate dealers across a multi-hand session. Blanks, betting,
  three-player adaptations, tournament bonuses and house variants are excluded.

## Custom cards

```python
from game_room import mahjong_v0
from game_room.mahjong_cards import BAMS, FLOWER, Card, Group, HandPattern, suited

custom = Card("My practice card", (
    HandPattern("Three kongs and flowers", (
        Group(FLOWER, 2),
        Group(suited(BAMS, 2), 4),
        Group(suited(BAMS, 5), 4),
        Group(suited(BAMS, 8), 4),
    ), value=25),
))
env = mahjong_v0.env(card=custom)
```

`Card`, `HandPattern`, `Group`, and `Exposure` are frozen dataclasses. Lists of
patterns/groups are copied to tuples. A pattern represents **one concrete
suit/rank realization**: enumerate all allowed alternatives when creating a
card. `Group(tile, size)` accepts a natural tile ID and size 1–6. A pattern must
total 14 tiles and be physically realizable with the supplied tile set.
Pattern names must be unique within a card.

`HandPattern` has `value=25`, `concealed=False`, and `jokerless_bonus=True`
defaults. Set `concealed=True` for hands marked C, and `jokerless_bonus=False`
for Singles and Pairs. Runs, years and NEWS are separate single/pair groups,
not melds. Keep repeated groups separate: two `Group(FLOWER, 3)` entries are
not `Group(FLOWER, 6)`.

`card.matching_hands(hand, exposures=())` accepts a mapping of concealed tile
IDs to counts and returns matching pattern indices. An `Exposure(tile, size,
jokers=0)` states its represented natural tile, total size, and joker count.
Exposures match whole groups in any order. `NMJL_2026` is exported from
`mahjong_cards` and accessible as `mahjong_v0.NMJL_2026`.

The card is public, immutable episode configuration, available as
`env.unwrapped.card`; it is not repeated in each observation. Policies trained
across cards must receive that configuration separately. This keeps the action
and observation spaces identical for different cards.

## Tiles and action encoding

Tile IDs 0–8 are bams 1–9, 9–17 craks, and 18–26 dots. IDs 27–30 are east,
south, west and north; 31–33 green, red and white dragons; 34 flower; 35 joker.
`suited(BAMS|CRAKS|DOTS, rank)` returns an ID. Matching dragons are green/bam,
red/crak and white/dot. White also supplies zero in year patterns.

The action space is `Discrete(89)`:

| IDs | Meaning |
| --- | --- |
| 0–35 | Select this tile for a pass or discard, according to phase. |
| 36 | `PASS`: decline the current discard. |
| 37 | `MAHJONG`: declare a legal winning hand. |
| 38 | `CONTINUE`: start Charleston or agree to the second one. |
| 39 | `STOP`: veto the second Charleston. |
| 40 | `BLIND`: commit one unseen incoming tile to the current pass. |
| 41–44 | `COURTESY_OFFSET + n`: offer to exchange n tiles, 0–3. |
| 45–68 | `claim_action(size, jokers)`: claim and expose a group. |
| 69–84 | `exchange_action(player, slot)`: redeem one exposed joker. |
| 85–88 | `DEAD_OFFSET + player`: declare a publicly impossible hand dead. |

Claim encoding is `45 + 6*(size-3) + jokers`, where `3 <= size <= 6` and
`0 <= jokers < size`; unused combinations stay masked. Exchange encoding is
`69 + 4*player + slot`, with absolute seat and exposure slot in 0–3. Always use
the mask; a player can take several consecutive actions.

Phase IDs are `START=0`, `CHARLESTON=1`, `SECOND_VOTE=2`, `COURTESY_VOTE=3`,
`TURN=4`, `RESPOND=5`, and `FINISHED=6`. Pass indices 0–5 represent the two
Charlestons; index 6 is courtesy.

## Observation layout

Observations contain a float64 vector of length 644 and an int8 action mask of
length 89. Seat indices are absolute. The first 20 entries are:

1. Observer seat.
2. Current decision-maker seat.
3. Turn owner seat (East during setup; discarder during responses).
4. Phase ID.
5. East/dealer seat.
6. Decisions so far.
7. Decision limit; zero means unlimited.
8. Number of discards made.
9. Tiles remaining in the wall.
10. Pass index.
11. Number of seats finished selecting for the pass.
12. Pending discard tile ID + 1, or zero.
13. Pending discarder seat + 1, or zero.
14. Whether East must supply a natural to resolve an all-blind cycle.
15. Discarder liable for a current-turn win, seat + 1; zero means self-pick.
16. Discard history length.
17. Observer's required pass count.
18. Observer's number of selected pass slots.
19. Observer's number of blind pass slots.
20. Observer's sealed discard response action + 1, or zero.

Then follow:

| Indices | Contents |
| --- | --- |
| 20–23 | Votes + 1; zero unsubmitted. Second vote: 1 stop, 2 continue. Courtesy: 1–4 for 0–3 tiles. |
| 24–27 | Public concealed hand sizes, including reserved outgoing tiles. |
| 28–31 | Dead-hand flags. |
| 32–79 | Four seats × four exposure slots × (tile ID + 1, size, jokers). Empty slots are zero. |
| 80–115 | Observer's available concealed tile counts. |
| 116–151 | Observer's reserved outgoing natural tile counts. |
| 152–187 | Counts of unclaimed tiles in the discard pool. |
| 188–643 | 152 ordered history slots × (discarder + 1, tile ID + 1, claimed flag), padded with zeros. |

The pending discard is already in the pool/history; do not count it twice.
Claimed tiles remain in history but are removed from pool counts. History is
bounded by the tile inventory and finite number of exposures. Passing fields
can retain their last public values outside setup; interpret them by phase.

Opponent hands, wall order, sealed response choices, and reserved/forwarded
pass identities are hidden. Rendering exposes public information only.
Observation arrays are independent copies. Raw object attributes are debugging
state and must not be used as policy inputs.
