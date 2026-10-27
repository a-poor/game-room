# Sell Me a Sasquatch v0 environment

`game_room.sasquatch_v0` implements the supplied
[Sell Me a Sasquatch rules](rules-sasquatch.md) for 2–6 players using PettingZoo's
sequential AEC API. The 120-card inventory and all effects come from that
repository document. Creature names within a size category do not affect play,
so interchangeable cards share a type ID.

## Usage

```python
from game_room import sasquatch_v0

env = sasquatch_v0.env(num_players=3, render_mode="ansi")
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

`num_players` defaults to 3. Seat 0 starts as Buyer, or as the active player in
the two-player game. `reset(seed=...)` seeds the deck, recycling shuffles and
action spaces. `options` is accepted but unused.

`max_steps` defaults to 10,000 individual decisions, including each card
selection and response. Reaching the limit truncates all agents with no winner
or terminal reward. Set it to `None` for unlimited play. `render_mode` accepts
`None`, `"ansi"`, or `"human"`. Text rendering shows collections, scores, hand
sizes and public deal cards, but never private hands or facedown card types.
There is no parallel API.

Rewards are zero until the game ends, then +1 for the winner and −1 for every
opponent. Terminal `info` contains `winners` (agent names), `scores` (seat order)
and `turn_number`. On truncation, `winners` is empty. A win takes priority over a
step limit reached by the same decision.

## Turn flow

### Three to six players

All players except the Buyer contribute three hand cards to their own deal and
select one to reveal. The Buyer then reveals one additional card from any deal.
Everyone receives a Thingamabob opportunity before the Buyer chooses a deal.
The chosen deal enters the outgoing Buyer's collection, its offerer becomes the
new Buyer, and every unchosen deal enters its offerer's collection.

All Nasty sets resolve before any Creature scoring. The **new Buyer** receives
Nasty penalties, including those suffered by the outgoing Buyer. After all sets
are resolved, everyone refills to five and the game checks for a winner. The new
Buyer begins the next turn.

### Two players

The active player selects three hand cards individually, assigning each to
either player's deal. All four splits (3/0, 2/1, 1/2 and 0/3) are supported. The
active player reveals one of the three cards. There is no additional Buyer
reveal. After the Thingamabob window, the other player accepts the deals or
reverses them. Both collections receive their allocated cards. Nasty penalties
benefit the victim's opponent; there is no Buyer Squatch. After set resolution
and refill, the active player alternates.

### Thingamabobs

Only Thingamabobs in collections can be played. Their card is discarded before
its effect resolves. Hands can only contribute to deals, including through
Cryptozooptic Expander; deal cards cannot be played directly as Thingamabobs.

| Card | Resolution decisions |
| --- | --- |
| Platonic Isolator | Choose someone with strictly more points and steal one point |
| Super Detrital Repositioner | Select two distinct deal slots, in the same deal or different deals, to discard |
| Detrital Repositioner | Select one deal slot to discard |
| Spectroelectric Optimeter | Select one facedown deal slot to turn face up |
| Cryptozooptic Expander | Select one hand card, then choose the deal to receive it facedown |

A player can play multiple Thingamabobs in a window. The effect must have legal
targets before the card can be played. In particular, the Super Repositioner
requires at least two cards on offer, and the Expander requires a hand card.
Removal can target face-up or facedown cards. Removed cards become public in
the discard pile. A tool cannot be cancelled after starting its effect.

### Sets, scoring and wins

Owners must trade in every completed Nasty set. When an owner has more than one
complete Nasty type, they choose which type to resolve next. Poison Pill Bug
permits stealing zero or one collection card; Loan Shark permits zero, one or
two. Trojan Horse transfers one existing point, or nothing if the victim has
zero. Stealing can create new completed sets, which are resolved before the
turn finishes. Cards being stolen remain collection cards; they do not enter
hands and cannot be played as tools during this resolution phase.

After all Nasty effects are exhausted, Creature sets score automatically: four
Tiny, three Medium, two Big or one Giant earns a point. Multiple sets of a size
all score, and incomplete remainders stay in the collection. Set cards are
discarded and can return through deck recycling.

All hands refill to five before the end-of-turn victory check. Three or four
players need four points; two, five or six players need five. The winner must
be the unique highest scorer at or above the threshold. A tie for highest score
continues play, even when multiple players have reached the threshold. Reaching
the target during a Thingamabob or Nasty effect does not end the game.

## Sequential conventions

The source rules describe some simultaneous decisions and open conversation.
These explicit conventions make them reproducible in an AEC environment:

- **Sealed initial offers:** In multiplayer, offerers commit clockwise from the
  Buyer. Their selected face-up cards are revealed together after every offerer
  has committed. Later offerers therefore cannot inspect earlier offerers'
  initial reveals while building their own offers. Card counts and contributors
  remain public, but facedown identities are private.
- **Thingamabob window:** The outgoing Buyer (or active player in two-player
  mode) receives the first opportunity. A player can play repeatedly or pass.
  Passing moves to the next seat clockwise. Any played Thingamabob resets the
  consecutive-pass count to zero. Only a full circuit of passes closes the
  window and enables the deal decision. Thus, everyone gets a final opportunity
  after the last effect, including players who passed earlier.
- **Simultaneous Nasties:** Scan players clockwise, starting with the new Buyer
  in multiplayer or the active player in two-player mode. The first player with
  a completed Nasty set chooses a type to resolve. Resolve its entire penalty,
  including up to two Loan Shark thefts, then rescan from the start. Newly
  completed Nasty sets take part in this scan. Creature scoring waits until
  none remain anywhere.
- **Buyer's own Nasties:** In multiplayer, a new Buyer's own completed Nasty set
  is discarded without a self-transfer or theft choice; stealing from oneself
  cannot change the collection or score. In two-player mode the opponent
  always benefits instead.
- **Empty deals:** A deal remains selectable after all its cards are removed.
  Choosing that empty deal in multiplayer still transfers the Buyer Squatch.
  Expansion can add to any offered deal, including an empty one. The multiplayer
  Buyer's seat has no deal; two-player mode always has two deals.
- **Known facedown cards:** A contributor remembers their own offered or added
  cards. Another player's hidden addition to their deal is not shown to them.
  Only a face-up reveal makes that card's type visible to everyone.
- **Refill order:** Refill clockwise starting with the outgoing Buyer/active
  player. Shuffle the discard pile whenever the draw pile empties. With the
  supplied inventory, resolving all sets leaves enough cards to refill legal
  positions. The draw routine defensively stops if both piles are empty.
- **Point supply:** Scores are integers without a physical token-supply cap.
  The source lists 28 tokens but gives no exhausted-supply rule. This avoids an
  undefined scoring state during extended ties; it does not introduce another
  victory condition.

These conventions are part of the `v0` rules and interface. A change that makes
saved policies or experiment results incompatible should introduce a new
environment version.

## Card IDs

`make_deck()` returns 120 type IDs with the counts below. `Card` names and
`CARD_COUNTS` expose the same mapping. Copies of a type are interchangeable.

| ID | `Card` member | Copies | Set size |
| --- | --- | --- | --- |
| 0 | `TINY` | 20 | 4 |
| 1 | `MEDIUM` | 25 | 3 |
| 2 | `BIG` | 16 | 2 |
| 3 | `GIANT` | 4 | 1 |
| 4 | `POISON_PILL_BUG` | 10 | 2 |
| 5 | `LOAN_SHARK` | 10 | 2 |
| 6 | `TROJAN_HORSE` | 12 | 3 |
| 7 | `PLATONIC_ISOLATOR` | 3 | — |
| 8 | `SUPER_DETRITAL_REPOSITIONER` | 3 | — |
| 9 | `DETRITAL_REPOSITIONER` | 6 | — |
| 10 | `SPECTROELECTRIC_OPTIMETER` | 5 | — |
| 11 | `CRYPTOZOOPTIC_EXPANDER` | 6 | — |

## Actions

The action space is `Discrete(160)`. Use the exported offsets and the mask for
each decision. Illegal actions raise `ValueError` without changing state or
advancing randomness. Only the selected live agent has a nonzero mask.
Terminated or truncated agents must step with `None`.
`env.unwrapped.action_label(action)` provides readable action labels.

| Expression | IDs | Meaning |
| --- | --- | --- |
| `SELECT_CARD + card_type` | 0–11 | Choose a hand card, completed Nasty type, or collection card to steal, depending on phase |
| `PLAY + card_type` | 12–23 | Play a collection Thingamabob; only types 7–11 can be legal |
| `SELECT_SLOT + slot` | 24–143 | Select an individual deal card to reveal or remove |
| `SELECT_PLAYER + seat` | 144–149 | Select the target of Platonic Isolator |
| `SELECT_DEAL + seat` | 150–155 | Select a deal for placement/expansion or the Buyer's chosen deal |
| `PASS` | 156 | Pass the current Thingamabob opportunity |
| `ACCEPT` | 157 | Accept two-player deal allocations |
| `REVERSE` | 158 | Reverse two-player deal allocations |
| `STOP` | 159 | Take no further cards from a Nasty victim |

Deal slots identify **positions, not hidden card types**. They are assigned from
the first empty slot in a 120-slot array. Removing a card empties its slot;
a subsequent addition may reuse it. Existing cards never move to another slot
while deals are being considered. All slots are cleared when deals enter
collections. A slot's owner identifies whose deal it belongs to; its contributor
identifies who supplied the card.

A two-player hand selection is followed by a deal-placement decision. Expander
uses the same two-step process. The pending card stays in its hand until placed,
so merely selecting it never creates or destroys a card. Other effect targets
are likewise separate AEC decisions.

## Observations

`observe(agent)` returns `observation`, a nonnegative `float64` vector of length
`644 + 14 * num_players`, and `action_mask`, 160 `int8` entries. Seats are absolute,
not observer-relative. Observations contain private information only for the
requested agent. Agents should consume their own observations, as with the
other imperfect-information environments in this package.

The first 20 entries are:

| Index | Meaning |
| --- | --- |
| 0–2 | Observer's seat, selected decision maker, outgoing Buyer/active player's seat |
| 3 | Current Buyer Squatch holder + 1; zero in two-player mode |
| 4–6 | Turn number starting at 1, decisions taken, decision limit (zero if unlimited) |
| 7 | Required winning score |
| 8 | Phase index in `PHASES` |
| 9–10 | Current/completed offerer index; number of initial cards placed by that offerer |
| 11 | Consecutive Thingamabob passes |
| 12–13 | Pending Thingamabob type + 1 and its player's seat + 1; zero if absent |
| 14 | Remaining removals or Nasty thefts |
| 15–16 | Nasty victim and beneficiary seat + 1; zero if absent |
| 17 | Pending hand-card type + 1, visible only to its owner; zero if absent/hidden |
| 18 | Winning seat + 1, or zero |
| 19 | Draw-pile size |

`PHASES`, in order: offer_card, offer_place, offer_reveal, buyer_reveal, things,
remove, reveal, expand_card, expand_place, point_target, choose, nasty, steal.
Setup counters retain their final values until the next turn. The turn's Buyer
and the current Buyer Squatch holder differ after multiplayer distribution;
Nasty benefits follow the current holder.

Next come 14 entries per player, in seat order: points, hand size, and twelve
public collection counts in card-type order. These are followed by twelve
observer hand counts and twelve public discard counts.

The final 600 entries are 120 deal-slot rows with five entries each:

1. Occupied flag (1 if occupied, otherwise the entire row is zero).
2. Deal owner's seat + 1.
3. Contributor's seat + 1.
4. Face-up flag, public to everyone.
5. Card type + 1 if face up or contributed by the observer; otherwise zero.

The identities of opponents' hand cards, unknown facedown deal cards and draw
order are never exposed. Masks for revealing or removing a hidden card depend
on the slot and visibility, not its type. Initial reveal choices remain sealed
until all initial offers are committed. Discard identities are public, but
shuffle order is hidden. Past offers, removed slots and shuffled discards are
not retained as history; recurrent policies can remember earlier observations.
