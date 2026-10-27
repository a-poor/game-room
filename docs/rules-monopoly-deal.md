# Monopoly Deal v0

`game_room.monopoly_deal_v0` implements the single-pack game for 2–5 players
through PettingZoo's sequential AEC API. It follows the
[2008 Hasbro instruction sheet supplied for this implementation](https://www.buffalolib.org/sites/default/files/gaming-unplugged/inst/Monopoly%20Deal%20Card%20Game%20Instructions.pdf).
The four reference cards are excluded, leaving 106 playable cards. Six or more
players and the two-pack variant are not supported.

## Usage

```python
from game_room import monopoly_deal_v0

env = monopoly_deal_v0.env(num_players=3, render_mode="ansi")
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

`num_players` defaults to 3. `max_steps` defaults to 10,000 individual decisions,
including selections, payments, responses and rearrangements. Reaching the limit
truncates all agents with zero terminal reward and no winner. Use `None` for no
limit; unrestricted free rearrangements can otherwise keep a policy in a turn
forever. `render_mode` accepts `None`, `"ansi"`, or `"human"`. Rendering shows only
public cards and hand sizes. `reset(seed=...)` seeds the shuffle and action spaces.
Player 0 starts, followed by increasing seat indices. `options` is accepted but
unused. There is no parallel environment.

Rewards are zero until a winner is declared: the winner receives +1 and every
other player receives −1. Terminal `info["winners"]` contains the winning agent
name, or an empty list on truncation; `info["turn_number"]` records the turn.
The winner needs full sets in three **different** colors. Winning is checked on
the turn owner's turn, including after receiving payment, and at the beginning
of their turn before drawing. An off-turn recipient must wait for their turn.

## Rules and explicit conventions

Players begin with five cards. A turn starts by drawing two, or five if the hand
is empty. Bank, property, building and action plays each consume one of three
plays. Pass Go draws two immediately. A player may end after zero to three
plays; the environment leaves the turn open after the third play for free
rearrangement. More than seven hand cards must then be selected individually
and placed at the **bottom of the draw pile**, as specified by this edition.
The first selected excess card will be drawn before later selected excess cards.

Money and action cards can be banked. Property cards, including wildcards,
cannot be banked. Banked action cards remain money even when transferred.
Payments come from bank or table cards, never the hand, with no change. The
payer chooses one card at a time until the bill is covered or no payable assets
remain. Properties may come from full sets. Zero-value rainbow wildcards are
never payable. Houses and hotels used for payment become bank money.

All card effects are implemented: Sly Deal, Forced Deal, Deal Breaker, Just Say
No, Debt Collector, Birthday, both rent types, Double the Rent, House, Hotel and
Pass Go. Two-color rent and birthdays resolve clockwise for each opponent;
any-color rent and debts target one opponent. Each affected opponent gets a
response decision, even without a Just Say No, so the decision itself does not
reveal their hand. Just Say No alternates between the victim and the initiating
player, toggling cancellation; it only cancels that victim's obligation in a
multi-player effect. Cancelled actions still consume their plays.

The sheet does not specify every interaction. These conventions are part of
this environment version:

- Used action cards form a discard pile. When the draw pile is empty, shuffle
  the discard pile to replenish it. If both are empty, draw only what is
  available; turns continue.
- Just Say No responses do not consume the three normal plays, including a
  counter-response by the turn owner.
- Double the Rent is bundled after choosing a rent group and before responses.
  Each doubler consumes a play. Two doublers produce four times the rent when
  three plays remain. Following the sheet's wording “standard Rent card,” only
  two-color rent can be doubled in this version.
- Neither property in a Forced Deal may belong to a complete set, following
  the restriction printed on that card. Deal Breaker moves an entire full group
  including its buildings; it does not merge it into another group.
- Properties are arranged in explicit groups, each capped at its full-set size.
  Multiple groups of the same color are permitted but count as one winning
  color. Wildcards may form a set without a single-color property; the supplied
  sheet does not impose an anchor-card requirement.
- On your turn, rearranging table properties and buildings is free, including
  after the third play. Wildcards can change to any of their printed colors.
  A new group uses the first empty slot. A property cannot be moved into the
  group it already occupies; a singleton may move to a new slot.
- Individual properties received through payment, Sly Deal or Forced Deal keep
  their current color and join the first incomplete matching group, or start a
  new group. The recipient can rearrange them on their turn. This deterministic
  placement also applies to an off-turn Forced Deal recipient.
- A house requires a complete non-railroad, non-utility set without a house.
  A hotel additionally requires a house and no hotel. These conditions apply
  to free building moves as well. If a set is broken or its house is removed,
  remaining buildings stay in that group, may be paid or rearranged later,
  and add no rent until the set is complete and has a house. A building-only
  group remains on the table but cannot charge rent or count as a set.
- A win ends the game immediately; any unresolved opponents' obligations are
  abandoned. Only the current turn owner can claim it.

## Card IDs and rent

`CARDS[id]` is an immutable `Card(kind, value, colors)` description. Values are
in millions. IDs represent **physical copies**, so identical cards have distinct
IDs; no card ID is reused within a game. `make_deck()` returns IDs 0–105.

| IDs | Kind |
| --- | --- |
| 0–27 | Single-color properties in the color order below |
| 28 | Brown/light blue wildcard |
| 29 | Light blue/railroad wildcard |
| 30–31 | Pink/orange wildcards |
| 32–33 | Red/yellow wildcards |
| 34 | Green/dark blue wildcard |
| 35 | Green/railroad wildcard |
| 36 | Railroad/utility wildcard |
| 37–38 | Zero-value all-color wildcards |
| 39–40 | Deal Breaker |
| 41–43 | Forced Deal |
| 44–46 | Sly Deal |
| 47–49 | Just Say No |
| 50–52 | Debt Collector |
| 53–55 | Birthday |
| 56–57 | Double the Rent |
| 58–60 | House |
| 61–62 | Hotel |
| 63–72 | Pass Go |
| 73–82 | Two-color rents: two each for brown/light blue, pink/orange, red/yellow, green/dark blue, railroad/utility |
| 83–85 | Any-color rent |
| 86–105 | Money: six 1M, five 2M, three 3M, three 4M, two 5M, one 10M |

| Color ID (`Color`) | Properties in full set | Property value | Rent by number of properties |
| --- | --- | --- | --- |
| 0 Brown | 2 | 1M | 1, 2 |
| 1 Light blue | 3 | 1M | 1, 2, 3 |
| 2 Pink | 3 | 2M | 1, 2, 4 |
| 3 Orange | 3 | 2M | 1, 3, 5 |
| 4 Red | 3 | 3M | 2, 3, 6 |
| 5 Yellow | 3 | 3M | 2, 4, 6 |
| 6 Green | 3 | 4M | 2, 4, 7 |
| 7 Dark blue | 2 | 4M | 3, 8 |
| 8 Railroad | 4 | 2M | 1, 2, 3, 4 |
| 9 Utility | 2 | 2M | 1, 2 |

A house adds 3M; a hotel adds another 4M. Rent amounts are also tabulated in this
[supplemental card reference](https://playmonopolydeal.com/how-to-play); that
site's house rules are not the rules of this environment.

## Actions

The action space is `Discrete(485)`. Use the exported offsets rather than
hard-coded numbers. Only the selected agent has a nonzero mask. Illegal actions
raise `ValueError` without changing the environment. Terminated or truncated
agents must step with `None`, following PettingZoo's dead-step convention.

| Expression | IDs | Meaning |
| --- | --- | --- |
| `SELECT_CARD + card_id` | 0–105 | Select a payment, excess hand card, deal property, doubler or Just Say No |
| `PLAY + card_id` | 106–211 | Start playing a hand card |
| `BANK + card_id` | 212–317 | Bank a hand card |
| `MOVE + card_id` | 318–423 | Start rearranging a table card |
| `SELECT_GROUP + group_index` | 424–467 | Select a group slot |
| `SELECT_COLOR + color_id` | 468–477 | Select a property color |
| `SELECT_PLAYER + seat_index` | 478–482 | Select an opponent |
| `END_TURN` | 483 | Finish normal plays and rearrangements |
| `ACCEPT` | 484 | Resolve a response chain or finish adding rent doublers |

Properties use `PLAY` → color → group. Buildings use `PLAY` → group. Rearranging
uses `MOVE` with the same selection steps. Rent uses `PLAY` → rent group → zero
or more doublers → `ACCEPT`, then an opponent selection for any-color rent.
Deals use `PLAY` → opponent → their card/group; Forced Deal additionally selects
your offered card. Responses and payments switch `agent_selection` temporarily.
Always consult the mask at each decision. `raw_env.action_label(action)` describes
an action for debugging; `env.unwrapped` exposes it on the wrapped environment.

## Observation layout

`observe(agent)` returns a dictionary containing `observation` (a nonnegative
`float64` vector of length `446 + 2 * num_players`) and `action_mask` (485 `int8`
entries). All seat indices are absolute, not relative to the observer.

The first 22 entries are:

| Index | Meaning |
| --- | --- |
| 0 | Observing seat |
| 1 | Selected decision-maker's seat |
| 2 | Turn owner's seat |
| 3 | Turn number, starting at 1 |
| 4 | Decisions taken |
| 5 | Step limit, or 0 for unlimited |
| 6 | Normal plays used this turn |
| 7 | Phase index in `PHASES` |
| 8 | Draw-pile size |
| 9–14 | Pending card, chosen color, chosen group, target seat, taken card, offered card; each encoded as ID + 1, or 0 if absent |
| 15 | Whether the pending placement is a free move |
| 16 | Original payment amount for the current effect |
| 17 | Remaining debt in the payment phase, otherwise 0 |
| 18 | Whether the effect is currently cancelled |
| 19 | Bitset of opponents still queued after the current target; clockwise order |
| 20 | Whether a winner exists |
| 21 | Winning seat + 1, or 0 |

`PHASES`, in order: play, color, group, target, take, give, rent_group, double,
response, pay, discard. Pending selections remain visible until the action is
finished; a hand card selected for placement is publicly committed at that point.

Next come two entries per seat: hand size and number of distinct complete colors.
Finally come 106 rows of four entries each, in physical-card-ID order:

1. Zone: 0 unknown, 1 observer's hand, 2 bank, 3 property, 4 building, 5 action discard.
2. Owner's seat + 1, or 0 for unknown/discard.
3. Group index + 1, or 0 when not in a group.
4. Assigned color + 1, or 0 when not in a group.

Opponents' hand identities and draw-pile identities/order are not exposed.
Card types/values/colors are static and available through `CARDS`. Action discard
identities are public; excess cards moved to the draw pile become unknown in the
current observation (a recurrent policy may remember earlier decisions). Group
slots are local to each player's table. Empty groups have no rows. The 44 slots
allow all properties and buildings to be represented, even when separated.
