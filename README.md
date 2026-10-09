# meridian (Python)

The Meridian SDK for Python. Private until the owner decides names and the public date (O47).

## Contract run

The suite normally runs against an in-memory fake engine. `uv run python scripts/contract.py` runs it against the
real local engine instead, and refuses (naming variables only, never values) unless all of these hold:

- no paid provider key (`GAUSSMERIDIAN_GEMINI_API_KEY`, `GAUSSMERIDIAN_OPENROUTER_API_KEY`) is set in the process
  environment or in any engine env file: the canonical `meridian-engine.env`, the one in `MERIDIAN_RUNTIME_DIR`, and
  the `GAUSSMERIDIAN_ENV_FILE` target are all read;
- every path it reads stays inside the local runtime folder;
- every engine URL is loopback. `localhost` is accepted, but it is resolved by the system resolver (hosts file, DNS),
  which the guard does not control; prefer `http://127.0.0.1:8180` or `http://[::1]:8180`;
- no proxy is configured (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, or the system proxy on Windows).

The guard reads files and its own environment. It does not inspect the running engine.
