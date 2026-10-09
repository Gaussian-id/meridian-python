# meridian (Python)

The Meridian SDK for Python. Private until the owner decides names and the public date (O47).

## Contract run

The suite normally runs against an in-memory fake engine. `uv run python scripts/contract.py` runs it against the
real local engine instead, and refuses (naming variables only, never values) unless all of these hold:

- no paid provider key (`GAUSSMERIDIAN_GEMINI_API_KEY`, `GAUSSMERIDIAN_OPENROUTER_API_KEY`) is set in the process
  environment or in any engine env file: the canonical `meridian-engine.env`, the one in `MERIDIAN_RUNTIME_DIR`, and
  the `GAUSSMERIDIAN_ENV_FILE` target are all read. The files are not parsed: any line that names a paid key sets it
  unless the line is exactly an empty assignment (`NAME=`, `NAME=""`, `NAME=''`, optionally after `export ` or
  `set `), so a whitespace-only value counts as set. A file with a character some reader hides (a non-ASCII space or
  format character, a default-ignorable code point, a control character, a lone CR) cannot be checked and is refused.
  When Docker is available, a test compares this with what `docker compose config` itself resolves;
- every path it reads stays inside the local runtime folder;
- every engine URL is loopback. `localhost` is accepted, but it is resolved by the system resolver (hosts file, DNS),
  which the guard does not control; prefer `http://127.0.0.1:8180` or `http://[::1]:8180`;
- no `http`, `https` or `all` proxy is set in the environment (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, any letter
  case) or, on Windows, in the registry (read even when `NO_PROXY` or another proxy variable is set). A PAC file,
  WPAD and the WinHTTP proxy are not detected; the suite runs with `NO_PROXY=*` so its own traffic ignores them.

The suite runs as `uv run --frozen --offline --no-env-file pytest -m "not fake_only"`: nothing is resolved or
downloaded and no `.env` file adds variables.

What the guard does not cover: it reads files and its own environment, not the running engine. The environment of the
process that runs `docker compose` (a bare `NAME` line inherits from it), a runtime folder made elsewhere through
`LOCALAPPDATA`, and other variables that reach the network (`GAUSSMERIDIAN_DECISION_*`,
`GAUSSMERIDIAN_USAGE_RECEIVER_URL`) are outside the two-key rule. Whether a contract run must require simulator mode
or inspect the running engine container's environment (by name only) is an open owner decision; it is settled before
any contract run.

`pytest --noconftest` drops the suite's conftest and with it the `engine` fixture, so no test can reach an engine at
all (the engine tests error with `fixture 'engine' not found`).
