<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Gaussian-id/meridian-python/main/assets/brand/meridian-lockup-light.svg">
    <img alt="Gauss Meridian" src="https://raw.githubusercontent.com/Gaussian-id/meridian-python/main/assets/brand/meridian-lockup-dark.svg" width="360">
  </picture>
</p>

# gauss-meridian (Python)

The official Python SDK for Gauss Meridian: one API for many AI models, routed by Meridian.

**Status:** Public preview, v0.1.0; Python 3.10 or later. The suite runs against an in-memory fake engine; the run
against a real local engine is simulator-only (see [Contract run](#contract-run-simulator-only)).

The code examples below are illustrative: they need an engine at `MERIDIAN_URL` and keys in your environment. The test
suite checks the same calls against its fake engine.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Three ways to connect](#three-ways-to-connect)
- [Response contract](#response-contract)
- [Streaming](#streaming)
- [Forwarding a browser's chat through your server](#forwarding-a-browsers-chat-through-your-server)
- [Admin client](#admin-client)
- [Retries, timeouts and idempotency](#retries-timeouts-and-idempotency)
- [Configuration](#configuration)
- [Contract run (simulator only)](#contract-run-simulator-only)
- [Development](#development)

## Install

| Package on PyPI | Import | Python |
|---|---|---|
| `gauss-meridian` | `gauss_meridian` | 3.10 or later |

```sh
pip install gauss-meridian                  # the core: httpx is its only dependency
pip install "gauss-meridian[fastapi]"       # adds FastAPI (>=0.110) for the forwarding route
pip install "gauss-meridian[django]"        # adds Django (>=4.2)
pip install "gauss-meridian[flask]"         # adds Flask (>=3.0)
```

With other tools:

```sh
uv add gauss-meridian --extra django
poetry add gauss-meridian --extras flask
```

The distribution installs the import package `gauss_meridian`, so it does not clash with Google's `google-meridian`,
which installs `meridian`.

`import gauss_meridian` (and `gauss_meridian.forward`, `gauss_meridian.admin`, `gauss_meridian.types`) never imports a
framework. `import gauss_meridian.flask` without Flask installed raises an `ImportError` that names the extra to
install (the same for `gauss_meridian.fastapi` and `gauss_meridian.django`).

## Quick start

Read the URL and the project key (`gmr_...`) from the environment, never from source code:

```sh
export MERIDIAN_URL="https://your-engine.example"   # the engine's public listener
export MERIDIAN_KEY="..."                           # a project key from your secret store
```

A chat call:

```python
from gauss_meridian import Meridian

client = Meridian()  # MERIDIAN_URL and MERIDIAN_KEY from the environment; or Meridian(url, key)

result = client.chat.create({'messages': [{'role': 'user', 'content': 'Hello'}]})
if result.error is None:
    print(result.data['choices'][0]['message']['content'])
    print(result.data['usage'])  # only prompt_tokens, completion_tokens and total_tokens
else:
    print(result.error.code, result.error.message, result.error.request_id)
```

A streaming call:

```python
opened = client.chat.stream({'messages': [{'role': 'user', 'content': 'Tell me a story'}]})
if opened.error is None:
    stream = opened.data
    for chunk in stream:
        print(chunk.delta, end='')
    if stream.error is not None:  # read after the loop
        print(stream.error.code)
```

The async client is the same API, awaited:

```python
from gauss_meridian import AsyncMeridian

async with AsyncMeridian() as client:
    result = await client.chat.create({'messages': [{'role': 'user', 'content': 'Hello'}]})
```

`client.models.list()` lists the models. The SDK never fills in a model: without one the engine chooses
(`meridian/auto`). Request and reply types are in `gauss_meridian.types` (`ChatParams`, `ChatCompletion`, ...).

To request a reasoning effort, set `model` to `meridian/auto:low` or append `:<effort>` to a named model. Efforts are case-sensitive: `low`, `medium`, `high`, `xhigh`, `max`. Browser forwarding drops `model` by default; allow it explicitly with `AllowRule(model=True)` or include the exact suffixed string in its model allowlist.

## Three ways to connect

1. **The server client** (`Meridian` / `AsyncMeridian`): your server calls the engine with a project key (`gmr_...`).
2. **A forwarding route** (FastAPI, Django or Flask): your browser or mobile code calls a route on your own server,
   which checks the user with your `authorize` function and adds the key. The key never reaches the browser. See
   [Forwarding](#forwarding-a-browsers-chat-through-your-server).
3. **The engine's API directly**, from your server, with the key from an environment variable:

   ```sh
   curl "$MERIDIAN_URL/v1/chat/completions" \
     -H "Authorization: Bearer $MERIDIAN_KEY" \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Hello"}]}'
   ```

Never put a key in source code, in a browser bundle or in a repository: read it from the environment.

## Response contract

### Results, not exceptions

Every call, sync or async, returns a `Result`: an `Ok` or an `Err`. API, network and configuration failures never
raise.

| | `Ok[T]` | `Err` |
|---|---|---|
| `.data` | the reply | `None` |
| `.error` | `None` | a `MeridianError` |
| `.request_id` | the engine's `x-request-id` for the call, when it had one | not present: read `.error.request_id` |

`Result`, `Ok`, `Err`, `MeridianError`, `ErrorCode` and `ErrorType` are all exported from `gauss_meridian`. Test with
`if result.error is None:`; a type checker then narrows `result.data` to the reply.

The async client returns the same types from awaited calls (`await client.chat.create(...)`).

### The chat completion

`chat.create` gives an `Ok` whose `.data` is the engine's chat completion (type `ChatCompletion`):

| Field | Meaning |
|---|---|
| `id`, `object`, `created`, `model` | the completion's id, its kind, the creation time and the model that answered |
| `choices[]` | each has `index`, `finish_reason` and a `message` (`role`, `content`, and `reasoning`, `refusal` or `tool_calls` when present) |
| `usage` | the three token counts: `prompt_tokens`, `completion_tokens`, `total_tokens` |

`usage` holds those three counts and nothing else: no cost or price field leaves the SDK. A stream chunk is a
`ChatChunk` with `delta` (the text of the piece, `''` when it has none) and `raw` (the engine's chunk as a `dict`).

### The error object

`result.error` is a `MeridianError` (a frozen dataclass):

| Field | Meaning |
|---|---|
| `code` | an `ErrorCode`: one of the engine's 28 codes, or the SDK's own `network_error` and `aborted` (table below) |
| `type` | an `ErrorType`: one of the engine's five (`invalid_request_error`, `authentication_error`, `permission_error`, `rate_limit_error`, `api_error`), or `client_error` for the SDK's own errors, configuration errors included |
| `message` | a readable message |
| `status` | the HTTP status (0 when nothing was received) |
| `retryable` | whether trying again later may help |
| `request_id` | the engine's id for the request; give it to support with any question about a call |
| `engine_code` | set only when the engine sent a code this SDK does not know: `code` is then `internal`, and the raw code is also in the message |
| `retry_after` | the response's `Retry-After` header as sent (delta-seconds or an HTTP date) when it was valid, else `None`. Wait at least that long before trying a `rate_limited` call again yourself |

`code` and `type` are `Literal` unions, so a type checker catches a misspelt code in
`if result.error.code == 'rate_limited':`. `engine_code` and `retry_after` are not part of an error's `repr` or `==`.

### Error codes

| Code | HTTP status | Retryable |
|---|---|---|
| `invalid_request` | 400 | no |
| `capability_unsupported` | 400 | no |
| `invalid_api_key` | 401 | no |
| `key_revoked` | 401 | no |
| `key_expired` | 401 | no |
| `invalid_token` | 401 | no |
| `insufficient_credit` | 402 | no |
| `project_unavailable` | 403 | no |
| `organisation_unavailable` | 403 | no |
| `model_not_allowed` | 403 | no |
| `byok_required` | 403 | no |
| `project_host_mismatch` | 403 | no |
| `permission_denied` | 403 | no |
| `entitlement_missing` | 403 | no |
| `model_not_found` | 404 | no |
| `route_not_found` | 404 | no |
| `not_found` | 404 | no |
| `method_not_allowed` | 405 | no |
| `idempotency_conflict` | 409 | no |
| `conflict` | 409 | no |
| `body_too_large` | 413 | no |
| `rate_limited` | 429 | yes |
| `hard_limit_reached` | 429 | no |
| `internal` | 500 | no |
| `upstream_unavailable` | 502 | yes |
| `upstream_error` | 502 | no |
| `service_unavailable` | 503 | yes |
| `deadline_exceeded` | 504 | yes |
| `network_error` (SDK) | the real status of an unreadable reply, else 0 | yes |
| `aborted` (SDK) | 0 | no |

The engine's 28 codes come first. `network_error` means the request failed, a reply could not be read, or a stream
broke. `aborted` means a stream was cancelled or left early. An engine code marked retryable is still not retryable
when the engine said `retryable: false`. An unreadable error reply (a proxy page, an empty body) is a `network_error`
with the real status, retryable for 429 and 5xx. The `type` of an error is `client_error` for the two SDK codes and for
configuration errors (`invalid_request` with status 0).

### When the client cannot send

A client built with a missing or malformed URL or key (or with both `http_client` and `transport`) is still built; it
answers every call with an `invalid_request` error of type `client_error` and sends nothing. So does a closed client
(`client.is_closed`, after `close()` / `aclose()` or its `with` block), and a client whose own `http_client` you
closed: the message is "The client is closed.". The shared clients the forwarding routes use are rebuilt after a close.
Don't close a shared client while it is serving requests: closing it ends the calls and streams in flight.

### What may raise

Building a forwarding helper without a callable `authorize` (or with an unusable `response_type`, or an async function
as an async view's `client`) raises `TypeError`; so does Python itself (wrong argument types); and
`asyncio.CancelledError` always propagates after cleanup. Nothing else raises.

## Streaming

```python
opened = client.chat.stream({'messages': [{'role': 'user', 'content': 'Tell me a story'}]})
if opened.error is None:
    stream = opened.data
    for chunk in stream:
        print(chunk.delta, end='')
    if stream.error is not None:  # read after the loop: a stream that broke or was cut short says so here
        print(stream.error.code)
    print(stream.usage)  # the three token counts, once the stream finished
```

- **Open:** `chat.stream` returns a `Result`. An `Err` means the call failed before any byte (a bad key, a refusal);
  an `Ok` holds the stream.
- **Iterate:** `for chunk in stream` gives `ChatChunk`s. `stream.text()` reads the whole stream and returns a
  `Result[str]`. `stream.iter_bytes()` gives the engine's events byte for byte, for forwarding.
- **Cancel:** `stream.cancel()` stops it and closes the connection; the stream's error is then `aborted`.
- **Usage:** `stream.usage` holds the three token counts once the stream finished (`None` before that). The SDK asks
  the engine for the usage chunk itself (`stream_options.include_usage`) and owns `stream` and `stream_options`.
- **A mid-stream error:** `chat.stream` has already returned `Ok`, so the failure shows in `stream.error` after the
  loop: the engine's error frame as its own code, a broken connection or a frame that is not JSON as `network_error`,
  a stream that ended without its end marker as `network_error` ("The stream ended before it finished.").
  `stream.request_id` is the response's request id.
- **Second-read rule:** a stream is read once (with `for`, `text()` or `iter_bytes()`). A second read records `aborted`
  ("The stream was already read."). Iterating gives no chunks, `text()` answers an `Err` (never an empty success), and
  `iter_bytes()` / `aiter_bytes()` give one error frame.

Async:

```python
from contextlib import aclosing

opened = await client.chat.stream({'messages': [{'role': 'user', 'content': 'Hi'}]})
if opened.error is None:
    stream = opened.data
    async with aclosing(stream.__aiter__()) as chunks:
        async for chunk in chunks:
            print(chunk.delta, end='')
            if chunk.delta.endswith('.'):
                break
    print(stream.error)  # final: the iteration is closed
    # or: text = await stream.text();  await stream.cancel()
```

An async stream's `.error` is final only once its iteration is closed. A loop that runs to the end closes it; one left
with `break` (or an exception) is closed later by the event loop, so close it yourself first: iterate inside
`aclosing(...)` as above, or `await stream.cancel()`. Cancelling the task that iterates an async stream closes the
connection, records `aborted` and lets the `CancelledError` propagate.

## Forwarding a browser's chat through your server

The route is public and spends your credit, so it is the SDK's security boundary:

- **`authorize(request)` is required.** Only the literal `True` lets the request through. A response of your framework
  (a 401, a redirect) is returned to the browser as it is. Anything else (`False`, `None`, `1`, a user object) or a
  raise is a 403 `permission_denied`, and the engine is never called.
- **POST and JSON only.** Another method is a 405 (`Allow: POST`); a content type other than `application/json` is a
  415, answered before `authorize` runs (so a cross-site form cannot reach it without a CORS preflight).
- **The body:** at most 8 MiB (413 `body_too_large`, read in chunks and never past the cap), UTF-8 JSON, nested at
  most 64 deep, with a non-empty `messages` list (400 otherwise).
- **What reaches the engine:** `messages` as sent (add a fixed system prompt server side if you need one);
  `temperature` when it is a number; `max_tokens` clamped to `[1, cap]`, and the cap when the browser sends none
  (`AllowRule(max_tokens_cap=...)`, default 4096); `model` only when you allow it; other options only when you name
  them. `n`, `stream`, `stream_options`, `max_completion_tokens` and `model` are never copied raw: the route sets
  streaming itself from `"stream": true`.
- **Idempotency keys:** a browser's `idempotency-key` is passed on only when it is 1 to 255 visible ASCII characters; otherwise the SDK makes its own.
- **What the browser learns:** errors about the end user's own request keep their status and code, and a passed-on
  `rate_limited` carries the engine's `Retry-After`. Errors about your own key, account or credit (`invalid_api_key`,
  `key_revoked`, `key_expired`, `project_unavailable`, `organisation_unavailable`, `project_host_mismatch`,
  `byok_required`, `insufficient_credit`, `hard_limit_reached`, `entitlement_missing`), codes this SDK does not know,
  malformed error replies and SDK configuration errors all become one generic answer, with the request id kept:

  ```json
  {"error": {"message": "The service is temporarily unavailable.", "type": "api_error",
             "code": "service_unavailable", "retryable": false, "request_id": "..."}}
  ```

  with status 503, or as the stream's last frame when streaming had already started.
- **The stream guard fails closed:** each event is judged before it is sent. An event it cannot judge (another event
  name, an unknown field, data that is not a JSON object, an unreadable `error`) ends the stream with the generic
  frame. It assumes the SDK talks to the engine directly: a proxy in between that adds SSE fields makes streams fail
  closed (safely).
- **Clients:** by default each route uses one long-lived shared client per settings, never one per request. A
  `client=` you pass must be a client, or a plain function returning the same long-lived client every time.

```python
from gauss_meridian.forward import AllowRule

allow = AllowRule(model=['your-model-id'], max_tokens_cap=1024, options=['top_p'])
```

### FastAPI

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from gauss_meridian.fastapi import forward_route, meridian_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await (await meridian_client()).aclose()  # close the shared client on the loop that used it


app = FastAPI(lifespan=lifespan)


async def authorize(request: Request) -> object:
    if request.headers.get('x-user-token') == 'valid':  # your own session check
        return True
    return JSONResponse({'error': 'log in first'}, status_code=401)


app.post('/api/meridian/chat')(forward_route(authorize=authorize))
```

`meridian_client` is also a dependency (`Depends(meridian_client)`) for your own endpoints. It is an async function,
so it is not a `client=` for `forward_route` (that raises `TypeError`). Under trio there is no asyncio loop to share
a client on: pass `forward_route(client=...)` there.

### Django

```python
# settings.py: optional, MERIDIAN_URL / MERIDIAN_KEY from the environment are used otherwise
import os

MERIDIAN = {'URL': os.environ['MERIDIAN_URL'], 'KEY': os.environ['MERIDIAN_KEY']}

# urls.py
from django.http import JsonResponse
from django.urls import path
from gauss_meridian.django import async_forward_view, forward_view


def authorize(request):
    return True if request.user.is_authenticated else JsonResponse({'error': 'log in first'}, status=401)


urlpatterns = [path('api/meridian/chat', forward_view(authorize=authorize))]
```

- `forward_view` is for WSGI. `async_forward_view` is for ASGI only (under WSGI it answers the generic 503); its
  `authorize` may be async (use `await request.auser()` there). Noticing a browser that leaves mid-stream needs
  Django 5.0 or later; the sync view works from Django 4.2.
- The body is read in chunks, never through `request.body`, so `DATA_UPLOAD_MAX_MEMORY_SIZE` does not apply. Under
  ASGI Django spools the whole body to a temporary file before the view runs: set a body limit in your server or
  proxy.
- Both views are CSRF-exempt by default (JSON only, 415 before `authorize`, and `authorize` guards access); pass
  `csrf_exempt=False` to keep Django's CSRF check.
- Under WSGI, a middleware whose `process_response` raises on a streamed response leaves the engine's stream open.

### Flask

```python
import os

from flask import Flask, jsonify, make_response, session
from gauss_meridian.flask import forward_blueprint

app = Flask(__name__)
app.config['MERIDIAN_URL'] = os.environ['MERIDIAN_URL']  # optional: the environment is used otherwise
app.config['MERIDIAN_KEY'] = os.environ['MERIDIAN_KEY']


def authorize(request):
    if session.get('user_id'):
        return True
    return make_response(jsonify(error='log in first'), 401)  # a response, not a tuple


app.register_blueprint(forward_blueprint(authorize=authorize))  # POST /api/meridian/chat
```

- `authorize` returns `True` or a werkzeug/Flask `Response`. A tuple such as `(jsonify(...), 401)` is not a response,
  so it is a 403: build it with `make_response(...)`.
- An app `MAX_CONTENT_LENGTH` below 8 MiB applies first: Flask answers its own 413.
- Flask has no CSRF check of its own. With Flask-WTF's `CSRFProtect` set up before the blueprint is registered, the
  blueprint is exempt by default (same reason as Django); pass `csrf_exempt=False` to keep the check.
- `meridian_client()` (and `MeridianExtension(app).client()`) is the shared client for the app's config.

Another framework can use `gauss_meridian.forward.Forwarder` / `AsyncForwarder` directly: give it the request's method,
content type, idempotency key and a chunked body reader, and close the streamed reply on every exit path.

## Admin client

`MeridianAdmin` / `AsyncMeridianAdmin` use an integrator key (`gmi_...`) on the engine's admin listener
(`MERIDIAN_ADMIN_URL`, `MERIDIAN_INTEGRATOR_KEY`). Server side only.

```python
from gauss_meridian.admin import MeridianAdmin

admin = MeridianAdmin()
org = admin.orgs.create({'name': 'Acme', 'external_reference': 'acme-1'})
if org.error is None:
    project = admin.projects.create(org.data['id'], {'name': 'web'})
    if project.error is None:
        issued = admin.keys.create(org.data['id'], project.data['id'], {'name': 'server'})
        if issued.error is None:
            store_securely(issued.data['api_key'])  # shown once; prints as [redacted]
```

It wraps 21 operations:

| Group | Calls |
|---|---|
| `orgs` | `create`, `get`, `get_by_reference`, `list` |
| `projects` | `create`, `get`, `list`, `update`, `delete` |
| `keys` | `create`, `list`, `revoke` |
| `provider_credentials` | `list`, `put`, `delete` |
| `project_models` | `get`, `put` |
| `models` | `list`, `get` |
| `analytics` | `usage` |
| `logs` | `list` |

The supplier catalogue is not wrapped (it holds only supplier prices), nor are renaming or deleting an organisation.

- **Typed results:** each call returns a `Result` of the engine's reply shape (`OrganisationBody`, `ProjectBody`,
  `KeyBody`, `IssuedKey` with `api_key` as a `Secret`, `CredentialMetadata`, `ModelAllowlist`, `CatalogueModelBody`,
  `UsageReport`, `LogPage`, or a list of one); a delete returns `Result[None]`. A type checker flags a misspelt key
  such as `org.data['idd']`.
- **Retries:** reads and deletes retry normally; writes (POST, PUT, PATCH) retry only after `rate_limited`, since a
  repeat could do the work twice. `orgs.create` with a non-blank `external_reference` retries normally (the engine
  replays it). Admin calls send no idempotency key. A delete whose 204 was lost and that is retried may answer
  `not_found`: the resource is gone either way.
- **Path ids** are each one encoded segment. An id that is blank, `.` or `..` (also after one percent-decode), or an
  organisation id of `by-reference`, is refused with `invalid_request` and nothing is sent.
- **No price or cost anywhere:** every key naming a cost, a price, a per-million rate, a recovery or write-off rate
  (or bleed, revenue, uncollected, written off) is removed from every admin result, at any depth.
- **Secrets:** `keys.create` returns the new key once, as a `Secret`: a `str` whose `repr()` is `[redacted]`, so
  `repr`/`str` of a result or of a dict holding it, f-strings of those, `%s`/`%r` logging of them and pytest's
  assertion output hide it. The value itself is the key: it compares, concatenates and encodes as the key, and
  `json.dumps(result.data)` carries it. `provider_credentials.put` holds a text `credential` as a `Secret` in its own
  copy of your body and sends it unchanged.

  What a `Secret` cannot hide: anything that makes a plain `str` from it or reads its characters shows the key:
  `str(key)`, `f'{key}'`, `format(key)` / `'{}'.format(key)`, `'%s' % key`, `+` and `''.join(...)`, slicing, string
  methods that return text, `%(api_key)s` / `'{api_key}'.format(**data)` formatting of a dict holding it, logging the
  bare value (`log.info('%s', key)`, `log.info(key)`), an exception built from it, JSON and structured log
  formatters, `pickle.dumps`, and pytest's line diff of two strings.

## Retries, timeouts and idempotency

- A call is retried at most twice (`max_retries=`, default 2), and only for the retryable codes above.
- Retries happen only before the first byte of the reply body is read: never mid-stream.
- A `Retry-After` of up to 30 seconds is waited for; a longer one is not, and the error is returned at once. A
  `Retry-After` of 0, in the past or unreadable falls back to backoff with jitter.
- Each chat call (`create` or `stream`) sends one `idempotency-key`, reused across its retries, so the engine runs it
  once. Pass your own with `idempotency_key=`: 1 to 255 characters, each visible ASCII (0x21 to 0x7E); `None` or `''`
  makes one; any other value is refused (`invalid_request`, nothing sent).
- When a retry finds its key still in use (`idempotency_conflict`, 409) after an earlier attempt failed with anything
  but `rate_limited`, the earlier attempt's error is returned: that attempt reached the engine.
- A buffered `chat.create` waits up to 300 seconds for its reply; streams and admin calls wait up to 60 seconds. These
  are limits per connect, read, write and pool wait, not on the whole call: a stream that keeps sending runs as long as
  it sends, and a call that is retried waits again on each attempt. `timeout=` (seconds) on the client sets one such
  limit for every call. Invalid values of `timeout` and `max_retries` (negative, NaN, infinite, `bool`) fall back to
  the defaults.

## Configuration

| Setting | Where | Default | Meaning |
|---|---|---|---|
| `MERIDIAN_URL` | environment, or `Meridian(url, ...)` | none (required) | the engine's public URL (absolute `http` or `https`) |
| `MERIDIAN_KEY` | environment, or `Meridian(url, key)` | none (required) | a project key (`gmr_...`) |
| `MERIDIAN_ADMIN_URL` | environment, or `MeridianAdmin(url, ...)` | none (required) | the engine's admin URL |
| `MERIDIAN_INTEGRATOR_KEY` | environment, or `MeridianAdmin(url, key)` | none (required) | an integrator key (`gmi_...`) |
| `timeout=` | client option | 60 s (300 s to read a buffered chat reply) | seconds, per connect, read, write and pool wait |
| `max_retries=` | client option | 2 | retries for the retryable codes |
| `http_client=` / `transport=` | client option | none | your own `httpx` client or transport (not both) |
| `idempotency_key=` | `chat.create` / `chat.stream` | generated | 1 to 255 visible ASCII characters |
| `MERIDIAN` (`URL`, `KEY`) | Django `settings.py` | the environment | settings for the Django forwarding view |
| `MERIDIAN_URL`, `MERIDIAN_KEY` | Flask `app.config` | the environment | settings for the Flask blueprint |

An argument given as `None` falls back to its environment variable; an explicit `''` does not. A missing or malformed
setting does not raise: the client answers every call with `invalid_request` (see
[When the client cannot send](#when-the-client-cannot-send)).

## Contract run (simulator only)

The suite normally runs against an in-memory fake engine. `uv run python scripts/contract.py` runs it against the
real local engine instead, and only an engine in simulator mode (a fake provider; no paid call is ever made). Start
the engine first (the engine's `scripts/local-up` with all provider keys empty); the run changes nothing there.
Run it with the same `LOCALAPPDATA` (or `XDG_STATE_HOME` / `HOME`) the engine was started with, and, if the engine
project is not `meridian-engine`, `GAUSSMERIDIAN_LOCAL_PROJECT`. The script refuses (naming variables only, never
values) unless all of these hold:

- docker proves a live simulator, with fixed `docker ps`, `docker port` and formatted `docker inspect` calls (never
  `logs`, never a raw inspect or environment): (S1) exactly one `provider-sim` container runs in the project; (S2)
  exactly one `engine` container runs, owning `127.0.0.1:8180` and `:8181`, which match `MERIDIAN_PUBLIC_URL` and
  `MERIDIAN_ADMIN_URL`; (S3) its provider base URLs are `provider-sim:8184`, `GAUSSMERIDIAN_PROVIDER_SIMULATOR` is
  `true` and each of the eight paid keys equals the public dummy `sim-local-key` (compared inside the docker template, so only match
  tokens come back); (S4) its compose `environment_file` label is the env file of the runtime folder checked below (or
  `GAUSSMERIDIAN_ENV_FILE`), or, with no label, the runbook's marker `.sim-started-<container id>` in that folder holds
  the engine's id. A throwaway runtime folder (a redirected `LOCALAPPDATA`) is accepted only because S1 to S4 prove the
  running engine was started from it. Docker missing, or any error, is a refusal;
- no paid provider key (`GAUSSMERIDIAN_GEMINI_API_KEY`, `GAUSSMERIDIAN_OPENROUTER_API_KEY`,
  `GAUSSMERIDIAN_OPENAI_API_KEY`, `GAUSSMERIDIAN_ANTHROPIC_API_KEY`, `GAUSSMERIDIAN_XAI_API_KEY`,
  `GAUSSMERIDIAN_DEEPSEEK_API_KEY`, `GAUSSMERIDIAN_MISTRAL_API_KEY`, `GAUSSMERIDIAN_QWEN_API_KEY`) is set in the process
  environment or in any engine env file: the canonical `meridian-engine.env`, the one in `MERIDIAN_RUNTIME_DIR`, and
  the `GAUSSMERIDIAN_ENV_FILE` target are all read. Any character outside printable ASCII and tab anywhere in the
  file (after one leading byte order mark), a lone CR included, is refused: the file cannot be checked. The files
  are not parsed: any line that names a paid key, in any letter case, sets it unless the line is exactly an empty
  assignment (`NAME=` or `NAME=""`, optionally after `export ` or `set `), so a whitespace-only value counts as set
  and so does `NAME=''` (both local-up readers keep the two quotes, which turns simulator mode off). A test compares
  this with the three readers that matter, each when its tool is installed: `docker compose config`, a copy of
  local-up.ps1's `Get-EnvValue` under `pwsh` and a copy of local-up.sh's `env_value` under `bash`;
- every path it reads stays inside the local runtime folder;
- every engine URL is loopback. `localhost` is accepted, but it is resolved by the system resolver (hosts file, DNS),
  which the guard does not control; prefer `http://127.0.0.1:8180` or `http://[::1]:8180`;
- no `http`, `https` or `all` proxy is set in the environment (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, any letter
  case) and, on Windows, no proxy server entry of any kind (SOCKS and FTP included) is in the registry (read even
  when `NO_PROXY` or another proxy variable is set). A PAC file, WPAD and the WinHTTP proxy are not detected; the
  suite runs with `NO_PROXY=*` so its own traffic ignores them.

The suite runs as `uv run --frozen --offline --no-env-file pytest -m "not fake_only"`: nothing is resolved or
downloaded and no `.env` file adds variables.

The simulator matrix (`tests/test_contract_sim.py`, marked `contract`, skipped in the normal suite) runs in sync and
async mode: a canary first (the fixed reply `sim token 1` ... `sim token 10`, 10 completion tokens, no cost; a mismatch
stops the session), chat create, an invalid key (`invalid_api_key`), a stream of exactly ten text pieces with usage,
admin provisioning (organisation, project, key issued, listed, revoked), a revoked key used directly (refused),
forwarding through the core forwarder and the FastAPI adapter, forwarding with the revoked key (the generic 503), and the
`sim-fail` fallback (reply shape only). A mid-stream error cannot be produced by the simulator, so it stays fake-only.
Each admin case creates its own organisation; the seeded keys are never revoked.

What the guard does not cover: the environment of the process that runs `docker compose` (a bare `NAME` line inherits
from it) and other variables that reach the network (`GAUSSMERIDIAN_DECISION_*`, `GAUSSMERIDIAN_USAGE_RECEIVER_URL`)
are outside the two-key rule; in simulator mode the engine's own compose override fixes the provider settings.

Engine-side carry (not changed here): `Get-EnvValue` in the engine's `scripts/local-common.ps1` uses a
culture-sensitive `StartsWith`, which skips control characters, marks and some letters inside a name; it should use
`StartsWith("$Name=", [StringComparison]::Ordinal)`. Both local-up readers (`Get-EnvValue` and `env_value` in
`scripts/local-common.sh`) strip double quotes only, so `NAME=''` turns simulator mode off; both should strip single
quotes too. Until then the guard refuses what those readers would read differently.

`pytest --noconftest` drops the suite's conftest and with it the `engine` fixture, so no test can reach an engine at
all (the engine tests error with `fixture 'engine' not found`).

## Development

```sh
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run python scripts/generate_types.py --check
```

The suite runs on Python 3.10 and on the latest Python; `tests/test_package.py` builds the wheel and sdist with
`uv build` into a temporary directory and checks what they ship: the package source, `README.md` and `LICENSE` ship,
and `assets/` does not (the README's logo links point at the repository's raw files, so they render on PyPI too).

The Meridian logo is a Gaussian trademark; the Apache-2.0 licence covers the code, not the logo. The approved lockups
are in `assets/brand/`.
