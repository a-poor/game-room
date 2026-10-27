# Publishing game-room

The initial distribution name is `game-room`, the Python import is `game_room`,
and the first planned version is `0.1.0`. Python 3.14+ is currently required.
The package uses the MIT license and uv's build backend.

## Validate and build

Run from the repository root:

```sh
uv sync --locked --group dev
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv build --no-sources
uvx --from twine twine check --strict dist/game_room-0.1.0*
```

The build produces a source archive (`.tar.gz`) and a wheel (`.whl`) in `dist/`.
uv builds the wheel from the source archive by default, checking that the source
distribution is sufficient to build the package. The source archive includes
tests and documentation; the wheel includes the importable package, typing
marker, metadata, and license. Development dependencies are not wheel runtime
requirements.

Before releasing, install the wheel into a fresh environment and run the README
example from outside the checkout. Confirm the imported package comes from that
environment's site-packages.

## Registry setup

1. Create or select your PyPI and TestPyPI accounts. They are separate services.
2. Confirm the distribution name is available or owned by your account on each
   service. A missing project page does not guarantee the name can be registered.
3. Configure authentication locally. For manual uploads, use a registry API token
   through `UV_PUBLISH_TOKEN` or uv's credential prompt. Never commit tokens.
4. When a repository is hosted, add its actual URL to `[project.urls]` and configure
   a Trusted Publisher if you want releases through CI. No remote repository or
   publisher is configured by this checkout's initial packaging setup.

## Test release

With TestPyPI credentials configured:

```sh
uv publish --publish-url https://test.pypi.org/legacy/ dist/game_room-0.1.0.tar.gz dist/game_room-0.1.0-py3-none-any.whl
```

To test that artifact, create a fresh environment, install the runtime
dependencies from regular PyPI, then install this package from TestPyPI without
dependency resolution:

```sh
uv venv /tmp/game-room-test-release
uv pip install --python /tmp/game-room-test-release/bin/python 'pettingzoo>=1.24.3,<2' 'gymnasium>=1.0,<2' 'numpy>=2.1'
uv pip install --python /tmp/game-room-test-release/bin/python --index-url https://test.pypi.org/simple/ --no-deps game-room==0.1.0
```

## Production release

After verifying the test release, use production PyPI credentials:

```sh
uv publish dist/game_room-0.1.0.tar.gz dist/game_room-0.1.0-py3-none-any.whl
```

PyPI release files cannot be overwritten. For subsequent releases, update
`project.version`, refresh `uv.lock`, rebuild, and publish only the intended
version's artifacts. Update these versioned commands accordingly. After the
first upload succeeds, remove the README's pre-release installation notice.

References: [uv packaging guide](https://docs.astral.sh/uv/guides/package/),
[Python packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
