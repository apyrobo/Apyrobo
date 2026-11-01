# Releasing apyrobo

A release is: bump the version, move the changelog entries, check, tag, upload.
`scripts/release_check.py` does the checking; it never bumps, tags or uploads.

## 1. Bump the version

```bash
scripts/bump_version.sh X.Y.Z
```

This updates `pyproject.toml`, `apyrobo/__version__.py` and adds a
`## [X.Y.Z]` stub to `CHANGELOG.md`. It does **not** touch `server.json`:
set both `"version"` fields there to `X.Y.Z` by hand (the MCP Registry reads
them). Move the `[Unreleased]` entries under the new heading.

## 2. Check

```bash
pip install build twine
python scripts/release_check.py
```

Every line must say `PASS`; the script exits 1 otherwise. It checks that:

| Check | What a FAIL usually means |
| --- | --- |
| pyproject.toml and server.json versions match | `server.json` was not bumped |
| CHANGELOG.md has a section for this version | `[Unreleased]` was not renamed |
| mcp-name marker in README.md | the `<!-- mcp-name: … -->` line was removed |
| sdist and wheel build | packaging error (`python -m build`) |
| twine check | the README does not render on PyPI |
| mcp-name marker in wheel METADATA | the README is not the package description |
| wheel installs in a fresh venv | a dependency or packaging problem |
| apyrobo mcp answers initialize | the installed server does not start, or `apyrobo/__version__.py` disagrees with `pyproject.toml` |
| apyrobo mcp lists its tools | the server starts but serves no tools |
| an over-limit move is clamped | the safety clamp is not applied in the installed package |

The last four install the wheel in a temporary venv (dependencies come from
PyPI, so this needs network) and talk to `apyrobo mcp` over stdio on
`mock://`, with no robot. `--static` runs only the first three checks;
`--keep` leaves the built files and venv in the temp folder.

## 3. Tag

```bash
git tag vX.Y.Z && git push origin vX.Y.Z
```

Pushing a `v*` tag runs `.github/workflows/python-package.yml`, which builds,
publishes to PyPI through a Trusted Publisher (the `pypi` environment must be
set up on GitHub and pypi.org, see the comment at the top of that file) and
creates the GitHub Release. Without the Trusted Publisher, upload by hand
with `python -m twine upload dist/*`.

After the PyPI release, publish `server.json` to the MCP Registry with
`mcp-publisher` (the registry checks the `mcp-name` marker on PyPI).
