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

Tests include PettingZoo API and seed checks, controlled special-card scenarios,
and full randomized games checking card conservation and cumulative rewards.

See [publishing instructions](docs/publishing.md) for building and releasing
the package. The implementation is distributed under the [MIT license](LICENSE).
