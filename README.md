# game-room

[![Tests](https://github.com/a-poor/game-room/actions/workflows/test.yml/badge.svg)](https://github.com/a-poor/game-room/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/game-room)](https://pypi.org/project/game-room/)
[![Python](https://img.shields.io/pypi/pyversions/game-room)](https://pypi.org/project/game-room/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A collection of PettingZoo environments for a few different games.

## Installation

Requires Python 3.14+. Install from [PyPI](https://pypi.org/project/game-room/)
with pip:

```sh
python -m pip install game-room
```

Or add it to a [uv](https://docs.astral.sh/uv/) project with `uv add game-room`.
To install directly from a checkout, use `python -m pip install .`.

## Getting Started

Save the following as `play.py`, then run `python play.py` (or
`uv run python play.py` in a uv project) to simulate a
three-player Flip 7 game with random legal actions and print the final scores:

```python
from game_room import flip7_v0

env = flip7_v0.env(num_players=3, render_mode="ansi")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        action = None
    else:
        action = env.action_space(agent).sample(observation["action_mask"])
    env.step(action)

print(env.render())
env.close()
```

## Flip 7

`game_room.flip7_v0` implements the 94-card game for 3-18 players using
PettingZoo's turn-based AEC API. It includes the opening deal, all special cards,
chained Flip Threes, deck recycling, and complete games across multiple rounds.

Actions are `0` (Hit), `1` (Stay), and `2 + seat_index` (choose the recipient
of an action card). Always use the mask: target selection is a separate decision,
and the same player can act multiple times in succession. Invalid actions raise
`ValueError` without changing game state.

Rewards are points earned at each round's end, so each player's episode return
equals their final score. Winners are reported in `info["winners"]` at game end;
the reward is not a win/loss reward. By default, a game ends after a completed
round brings someone to 200 points. Set `target_score` to change that threshold,
or `max_rounds` to truncate training episodes at a round boundary (default: no
limit). Tied leaders share the win. Scores and the last completed round's scores
are available in `info` after the first round.

The environment supports `ansi` and `human` text rendering. It has no parallel
API because each decision can immediately affect other players.

See [rules and observation reference](docs/rules-flip7.md) for rule conventions,
card IDs, and the public observation layout.

The `_v0` suffix identifies version 0 of the environment's rules and interface.
It is independent of the package version (`0.1.1`). Changes that invalidate
existing policies or experiment comparisons should introduce a new environment
version, such as `flip7_v1`, rather than silently replacing `flip7_v0`.

## Monopoly Deal

`game_room.monopoly_deal_v0` implements the 106-card game for 2–5 players with
private hands, banking, property sets and wildcards, all action cards, player-chosen
payments, and Just Say No counter-response chains. It uses the same AEC loop
shown above; replace the import and environment construction with:

```python
from game_room import monopoly_deal_v0

env = monopoly_deal_v0.env(num_players=3, render_mode="ansi")
```

Always sample from the action mask. Playing a card may require several decisions
to select a color, group, opponent, payment, or response. Rewards are +1 for the
winner and −1 for the other players; intermediate rewards are zero. The first
player with three complete sets of different colors on their own turn wins.
Episodes truncate after 10,000 decisions by default; set `max_steps=None` to
disable this limit.

See the [rules, action and observation reference](docs/rules-monopoly-deal.md)
for the supplied 2008 rules, explicit conventions, physical card IDs, and the
hidden-information observation layout.

## Catan

`game_room.catan_v0` implements the base game for 3–4 players on a randomized
19-hex board. It includes snake-order setup, resource production and shortages,
building, harbor trades, negotiated player trades with counteroffers, the robber,
all development cards, Longest Road and Largest Army.

```python
from game_room import catan_v0

env = catan_v0.env(num_players=4, render_mode="ansi")
```

Use the same masked AEC loop shown above. Resource hands and unplayed development
card types are private; public board state and hand sizes are observable. A win
requires at least 10 points on the player's own turn, awarding +1 to the winner
and −1 to opponents. Episodes truncate after 20,000 decisions by default;
`max_steps=None` removes this limit.

The environment uses the rulebook's variable board and combined trade/build
phase. See the [rules, board geometry and API reference](docs/rules-catan.md)
for action IDs, the observation layout, and the sequential trading protocol.

## Sell Me a Sasquatch

`game_room.sasquatch_v0` implements the supplied 120-card game for 2–6 players.
It includes hidden offers, all five Thingamabobs, Nasty thefts and cascading
sets, Creature scoring, changing Buyers, and the two-player accept/reverse mode.

```python
from game_room import sasquatch_v0

env = sasquatch_v0.env(num_players=3, render_mode="ansi")
```

Use the same masked AEC loop shown above. Hands and unknown facedown deal cards
stay private; contributors can see their own cards. Wins are checked after all
sets resolve and hands refill. The unique leader must have four points with
3–4 players, or five points with 2 or 5–6 players; ties continue. Rewards are +1
for the winner and −1 for opponents. The default limit is 10,000 decisions;
`max_steps=None` disables it.

See the [supplied rules](docs/rules-sasquatch.md) and
[environment reference](docs/sasquatch-v0.md) for the action and observation
layout, sealed offers, Thingamabob opportunities, and set-resolution conventions.

## Development

From a checkout, install the package and development tools:

```sh
uv sync --group dev
```

`dev` is the dependency group defined in `pyproject.toml` for pytest and Ruff.
uv includes this group by default, so `uv sync` also works; `--group dev` makes
the intent explicit. These tools are not runtime dependencies and are not
installed when users install the published package.

```sh
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

Tests include PettingZoo API and seed checks for every environment, controlled
rule scenarios, hidden-information checks, and complete games checking resource
and card conservation and cumulative rewards.

See [publishing instructions](docs/publishing.md) for building and releasing
the package. The implementation is distributed under the [MIT license](LICENSE).
