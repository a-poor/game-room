# Publishing game-room

The distribution name is `game-room`, the Python import is `game_room`,
and version `0.1.0` is published on [PyPI](https://pypi.org/project/game-room/0.1.0/).
Python 3.14+ is currently required. The commands below show the initial release;
substitute the new version when preparing subsequent releases.
The package uses the MIT license and uv's build backend.

## GitHub Actions releases

The `Tests` workflow runs lint, formatting, tests, and package builds on Python
3.14 across Linux, macOS, and Windows for pushes to `main` and pull requests.
It can also be run manually, and is reused by the publishing workflow.

The `Publish` workflow runs when a GitHub release is published. It waits for
the tests, verifies the release tag matches `v` plus `project.version`, builds
and checks the distributions, tests an installed wheel, and uploads to PyPI
using Trusted Publishing. No long-lived API token is needed.

Before the first automated release:

1. Push `.github/workflows/test.yml` and `.github/workflows/publish.yml` to GitHub.
2. Create the `pypi` environment in the repository's Settings > Environments.
3. In [PyPI publishing settings](https://pypi.org/manage/project/game-room/settings/publishing/),
   add a GitHub Trusted Publisher with these exact values:

   | Field | Value |
   | --- | --- |
   | Owner | `a-poor` |
   | Repository | `game-room` |
   | Workflow filename | `publish.yml` |
   | Environment | `pypi` |

For the next release, bump `project.version` (for example, to `0.1.1`), run
`uv lock`, commit and push the changes, then publish a GitHub release tagged
`v0.1.1` at that commit. Draft releases do not upload packages. Published
prereleases also upload to PyPI; use an appropriate PEP 440 prerelease version
in the package metadata. Do not release `v0.1.0` through this workflow: that
version is already on PyPI and its files cannot be replaced.

The manual commands below remain available for local verification and releases.

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
4. For CI publishing, configure the Trusted Publisher described above. Repository
   URLs are already recorded in `[project.urls]`.

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
version's artifacts. Update these versioned commands accordingly.

References: [uv packaging guide](https://docs.astral.sh/uv/guides/package/),
[Python packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
