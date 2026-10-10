import asyncio
import json
import logging
import pprint
import re
import reprlib
import traceback
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest

from gauss_meridian import Ok
from gauss_meridian.admin import AsyncMeridianAdmin, MeridianAdmin, Secret
from tests.support.fake_engine import Scripted, engine_error
from tests.support.modes import resolve
from tests.support.spec import UNWRAPPED_OPERATIONS, integrator_operations, wrapped_operations

if TYPE_CHECKING:
    from gauss_meridian._generated.admin import (
        CreateOrganisation,
        CreateProject,
        IssueKey,
        ModelAllowlist,
        SetCredential,
        UpdateProject,
    )

pytestmark = [pytest.mark.anyio, pytest.mark.fake_only]

ORG, PROJECT, KEY, PROVIDER, REFERENCE, MODEL = 'org-1', 'proj-1', 'key-1', 'gemini', 'ref-1', 'model-1'
# Every wrapped operation: its dotted name and the arguments of one valid call. Ids are sentinels that never
# collide with a literal word of any path, so each concrete segment maps back to its template parameter.
OPS: dict[str, tuple[Any, ...]] = {
    'orgs.create': ({'name': 'A', 'external_reference': 'ext-1'},),
    'orgs.get': (ORG,),
    'orgs.get_by_reference': (REFERENCE,),
    'orgs.list': (),
    'projects.create': (ORG, {'name': 'P'}),
    'projects.get': (ORG, PROJECT),
    'projects.list': (ORG,),
    'projects.update': (ORG, PROJECT, {'name': 'Q'}),
    'projects.delete': (ORG, PROJECT),
    'keys.create': (ORG, PROJECT),
    'keys.list': (ORG, PROJECT),
    'keys.revoke': (ORG, PROJECT, KEY),
    'provider_credentials.list': (ORG,),
    'provider_credentials.put': (ORG, PROVIDER, {'credential': 'c'}),
    'provider_credentials.delete': (ORG, PROVIDER),
    'project_models.get': (ORG, PROJECT),
    'project_models.put': (ORG, PROJECT, {'models': ['m']}),
    'models.list': (),
    'models.get': (MODEL,),
    'analytics.usage': (PROJECT,),
    'logs.list': (PROJECT,),
}
# How many leading arguments of each call are path ids
ID_COUNT = {
    'orgs.get': 1,
    'orgs.get_by_reference': 1,
    'projects.create': 1,
    'projects.get': 2,
    'projects.list': 1,
    'projects.update': 2,
    'projects.delete': 2,
    'keys.create': 2,
    'keys.list': 2,
    'keys.revoke': 3,
    'provider_credentials.list': 1,
    'provider_credentials.put': 2,
    'provider_credentials.delete': 2,
    'project_models.get': 2,
    'project_models.put': 2,
    'models.get': 1,
}
DELETES = ['projects.delete', 'keys.revoke', 'provider_credentials.delete']
# The ones whose first id is an organisation id (the engine routes /v1/orgs/by-reference/... elsewhere)
ORG_FIRST = [d for d in ID_COUNT if d not in ('orgs.get_by_reference', 'models.get')]
TEMPLATE = {
    ORG: '{org}',
    PROJECT: '{project}',
    KEY: '{key}',
    PROVIDER: '{provider}',
    REFERENCE: '{reference}',
    MODEL: '{model}',
}
PRICED = re.compile(
    'cost|price|per_million|recovery_rate|write_off_rate|bleed|revenue|uncollected|written_off', re.IGNORECASE
)
SECRET = 'gmr_' + 'a' * 64  # a key shaped like one, built at run time


def admin(mode: str, engine: Any, **options: Any) -> Any:
    kwargs: dict[str, Any] = {'url': engine.admin_url, 'key': engine.integrator_key, 'transport': engine.transport}
    kwargs.update(options)
    return (MeridianAdmin if mode == 'sync' else AsyncMeridianAdmin)(**kwargs)


async def call(client: Any, dotted: str, *args: Any, **kwargs: Any) -> Any:
    group, method = dotted.split('.')
    return await resolve(getattr(getattr(client, group), method)(*args, **kwargs))


def with_id(dotted: str, position: int, value: str) -> tuple[Any, ...]:
    args = list(OPS[dotted])
    args[position] = value
    return tuple(args)


def as_template(path: str) -> str:
    return '/'.join(TEMPLATE.get(segment, segment) for segment in path.split('/'))


def script_for(dotted: str) -> Scripted:
    return Scripted(status=204) if dotted in DELETES else Scripted(json={'ok': True})


def raw_paths(engine: Any) -> list[str]:
    return [r.url.raw_path.decode() for r in engine.requests]


# ---------------------------------------------------------------------------------------------------------------------
# What is wrapped
# ---------------------------------------------------------------------------------------------------------------------


async def test_wraps_exactly_the_21_integrator_operations_and_not_the_supplier_catalogue(
    mode: str, engine: Any
) -> None:
    client = admin(mode, engine)
    for dotted in OPS:
        engine.next(script_for(dotted))
        await call(client, dotted, *OPS[dotted])
    sent = {(r.method, as_template(r.url.path)) for r in engine.requests}
    # Both directions: nothing is sent that the spec lacks, and nothing in the spec (but the named pair) is unsent
    assert sent == set(wrapped_operations())
    assert len(OPS) == len(wrapped_operations()) == 21
    unwrapped = {op['operationId'] for key, op in integrator_operations().items() if key not in wrapped_operations()}
    assert unwrapped == UNWRAPPED_OPERATIONS == {'supplier_catalogue_active', 'supplier_catalogue_version'}
    assert not any(path.startswith('/v1/catalogue') for _, path in sent)
    assert not hasattr(client, 'catalogue')
    assert not hasattr(client.orgs, 'update')
    assert not hasattr(client.orgs, 'delete')
    assert not hasattr(client.orgs, 'rename')


async def test_bearer_gmi_and_no_idempotency_header_on_any_call(mode: str, engine: Any) -> None:
    client = admin(mode, engine)
    for dotted in OPS:
        engine.next(script_for(dotted))
        await call(client, dotted, *OPS[dotted])
    assert len(engine.requests) == 21
    for sent in engine.requests:
        assert sent.headers['authorization'] == f'Bearer {engine.integrator_key}'
        assert 'idempotency-key' not in sent.headers
        assert sent.url.host == 'admin.engine.test'


async def test_bodies_go_out_as_json_and_a_key_issue_without_a_body_sends_an_object(mode: str, engine: Any) -> None:
    client = admin(mode, engine)
    await call(client, 'projects.update', ORG, PROJECT, {'default_model': None})
    await call(client, 'keys.create', ORG, PROJECT)
    await call(client, 'keys.create', ORG, PROJECT, {'name': 'ci', 'expires_in_days': 30})
    assert engine.bodies() == [{'default_model': None}, {}, {'name': 'ci', 'expires_in_days': 30}]
    assert {r.headers['content-type'] for r in engine.requests} == {'application/json'}


async def test_query_parameters_are_encoded_and_none_is_left_out(mode: str, engine: Any) -> None:
    client = admin(mode, engine)
    await call(client, 'logs.list', 'p 1', page_size=5, cursor='a&b=c')
    await call(client, 'logs.list', 'p 1')
    await call(client, 'analytics.usage', 'p&1', from_='2026-01-01', to=None)
    logs, plain, usage = (r.url for r in engine.requests)
    assert logs.path == '/v1/logs'
    assert dict(logs.params) == {'project_id': 'p 1', 'page_size': '5', 'cursor': 'a&b=c'}
    assert dict(plain.params) == {'project_id': 'p 1'}
    assert usage.path == '/v1/analytics/usage'
    assert dict(usage.params) == {'project_id': 'p&1', 'from': '2026-01-01'}


@pytest.mark.parametrize('dotted', ['logs.list', 'analytics.usage'])
@pytest.mark.parametrize('project_id', ['', '   ', '\xa0'])
async def test_a_blank_project_id_for_a_query_is_refused_with_nothing_sent(
    mode: str, engine: Any, dotted: str, project_id: str
) -> None:
    result = await call(admin(mode, engine), dotted, project_id)
    assert (result.error.code, result.error.type, result.error.retryable) == ('invalid_request', 'client_error', False)
    assert engine.requests == []


async def test_the_public_signatures_take_the_generated_typed_dicts(engine: Any) -> None:
    # Typed clients, so pyright checks every body against the generated TypedDict; the replies stay Result[Any]
    org: CreateOrganisation = {'name': 'A', 'external_reference': 'ext'}
    project: CreateProject = {'name': 'P'}
    update: UpdateProject = {'status': 'active'}
    issue: IssueKey = {'name': 'ci'}
    models: ModelAllowlist = {'models': ['m']}
    credential: SetCredential = {'credential': 'c'}
    sync = MeridianAdmin(engine.admin_url, engine.integrator_key, transport=engine.transport)
    results = [
        sync.orgs.create(org),
        sync.projects.create(ORG, project),
        sync.projects.update(ORG, PROJECT, update),
        sync.keys.create(ORG, PROJECT, issue),
        sync.keys.create(ORG, PROJECT),
        sync.project_models.put(ORG, PROJECT, models),
        sync.provider_credentials.put(ORG, PROVIDER, credential),
    ]
    asynchronous = AsyncMeridianAdmin(engine.admin_url, engine.integrator_key, transport=engine.transport)
    results += [
        await asynchronous.orgs.create(org),
        await asynchronous.projects.create(ORG, project),
        await asynchronous.projects.update(ORG, PROJECT, update),
        await asynchronous.keys.create(ORG, PROJECT, issue),
        await asynchronous.project_models.put(ORG, PROJECT, models),
        await asynchronous.provider_credentials.put(ORG, PROVIDER, credential),
    ]
    assert [r.error for r in results] == [None] * 13
    assert len(engine.requests) == 13


# ---------------------------------------------------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------------------------------------------------

REFUSED_IDS = [
    '',
    '.',
    '..',
    '%2e%2E',
    '%2E.',
    '.%2E',
    '%2e',
    '%2E%2E',
    ' ',
    '   ',
    '\t',
    '\n',
    '\xa0',  # NBSP
    chr(0x2003),  # em space
    '%20',
    '%C2%A0',  # NBSP, percent-encoded
    '%09',
    '\ud800',  # a lone surrogate has no UTF-8 to encode
]


@pytest.mark.parametrize('bad', REFUSED_IDS, ids=repr)
async def test_an_unsafe_path_id_is_refused_in_every_position_with_nothing_sent(
    mode: str, engine: Any, bad: str
) -> None:
    client = admin(mode, engine)
    refusals: set[Any] = set()
    for dotted, count in ID_COUNT.items():
        for position in range(count):
            result = await call(client, dotted, *with_id(dotted, position, bad))
            assert (result.error.code, result.error.type, result.error.retryable) == (
                'invalid_request',
                'client_error',
                False,
            )
            assert result.data is None
            refusals.add(result.error)
    assert engine.requests == []
    # One refusal for every spelling and position: it cannot echo the id
    assert len(refusals) == 1


async def test_every_refusal_reads_the_same_whatever_the_spelling(mode: str, engine: Any) -> None:
    client = admin(mode, engine)
    errors = {(await call(client, 'projects.get', ORG, bad)).error for bad in REFUSED_IDS}
    errors.add((await call(client, 'projects.get', 'by-reference', PROJECT)).error)
    errors.add((await call(client, 'keys.list', 'by%2Dreference', PROJECT)).error)
    assert len(errors) == 1
    assert engine.requests == []


@pytest.mark.parametrize('org', ['by-reference', 'by%2Dreference', '%62y-reference', 'by-%72eference'])
async def test_an_organisation_id_of_by_reference_is_refused_with_nothing_sent(
    mode: str, engine: Any, org: str
) -> None:
    client = admin(mode, engine)
    baseline = (await call(client, 'orgs.get', '')).error
    for dotted in ORG_FIRST:
        result = await call(client, dotted, *with_id(dotted, 0, org))
        assert result.error == baseline, dotted
    assert engine.requests == []


async def test_by_reference_is_fine_anywhere_but_the_organisation_position(mode: str, engine: Any) -> None:
    client = admin(mode, engine)
    await call(client, 'orgs.get_by_reference', 'by-reference')
    await call(client, 'projects.get', ORG, 'by-reference')
    await call(client, 'keys.revoke', ORG, PROJECT, 'by-reference')
    await call(client, 'orgs.get', 'By-Reference')
    await call(client, 'orgs.get', 'by-reference2')
    assert raw_paths(engine) == [
        '/v1/orgs/by-reference/by-reference',
        '/v1/orgs/org-1/projects/by-reference',
        '/v1/orgs/org-1/projects/proj-1/keys/by-reference',
        '/v1/orgs/By-Reference',
        '/v1/orgs/by-reference2',
    ]


@pytest.mark.parametrize(
    ('value', 'segment'),
    [
        pytest.param('kopi-ü-日本-😀', 'kopi-%C3%BC-%E6%97%A5%E6%9C%AC-%F0%9F%98%80', id='unicode'),
        pytest.param('a/b', 'a%2Fb', id='slash'),
        pytest.param('%', '%25', id='percent'),
        pytest.param('%2F', '%252F', id='encoded-slash'),
        pytest.param('%2e%2e%2e', '%252e%252e%252e', id='encoded-dots-that-are-three'),
        pytest.param('%252e%252e', '%25252e%25252e', id='double-encoded-dots'),
        pytest.param('?', '%3F', id='question-mark'),
        pytest.param('a?b=c', 'a%3Fb%3Dc', id='query-lookalike'),
        pytest.param('#', '%23', id='hash'),
        pytest.param('a#b', 'a%23b', id='fragment-lookalike'),
        pytest.param('a\\b', 'a%5Cb', id='backslash'),
        pytest.param('a;b', 'a%3Bb', id='semicolon'),
        pytest.param('a b', 'a%20b', id='space'),
        pytest.param(' x ', '%20x%20', id='padded'),
        pytest.param(' . ', '%20.%20', id='padded-dot'),
        pytest.param('a\xa0b', 'a%C2%A0b', id='nbsp-inside'),
        pytest.param(chr(0x2024) * 2, '%E2%80%A4%E2%80%A4', id='one-dot-leaders'),
        pytest.param(chr(0xFF0E) * 2, '%EF%BC%8E%EF%BC%8E', id='fullwidth-dots'),
        pytest.param('..a', '..a', id='dots-with-a-letter'),
        pytest.param('...', '...', id='three-dots'),
        pytest.param('a\x00b', 'a%00b', id='nul'),
        pytest.param('%ff', '%25ff', id='undecodable-escape'),
    ],
)
async def test_every_other_id_is_one_encoded_segment(mode: str, engine: Any, value: str, segment: str) -> None:
    client = admin(mode, engine)
    await call(client, 'orgs.get_by_reference', value)
    await call(client, 'projects.get', value, value)
    await call(client, 'models.get', value)
    assert raw_paths(engine) == [
        f'/v1/orgs/by-reference/{segment}',
        f'/v1/orgs/{segment}/projects/{segment}',
        f'/v1/models/{segment}',
    ]


async def test_a_non_string_id_is_refused_never_raised(mode: str, engine: Any) -> None:
    client = admin(mode, engine)
    for bad in (None, 5, b'x', ['a']):
        result = await call(client, 'projects.get', ORG, bad)
        assert (result.error.code, result.error.type) == ('invalid_request', 'client_error')
    assert engine.requests == []


# ---------------------------------------------------------------------------------------------------------------------
# Retries: one rule, derived from the call
# ---------------------------------------------------------------------------------------------------------------------

FAILURES = {
    '503': lambda: engine_error('service_unavailable'),
    'dropped-connection': lambda: Scripted(network_error=True),
    '429': lambda: engine_error('rate_limited'),
}
GETS = ['orgs.get', 'keys.list', 'logs.list', 'models.list']
WRITES = ['projects.create', 'keys.create', 'provider_credentials.put', 'project_models.put', 'projects.update']


@pytest.mark.parametrize('failure', FAILURES)
@pytest.mark.parametrize('dotted', [*GETS, *DELETES])
async def test_a_read_and_a_delete_retry_normally(
    mode: str, engine: Any, clock: list[float], dotted: str, failure: str
) -> None:
    engine.next(*[FAILURES[failure]() for _ in range(3)])
    result = await call(admin(mode, engine), dotted, *OPS[dotted])
    assert result.error is not None
    assert len(engine.requests) == 3
    assert len(clock) == 2


@pytest.mark.parametrize('dotted', [*GETS, *DELETES])
async def test_a_read_and_a_delete_recover_after_a_failure(
    mode: str, engine: Any, clock: list[float], dotted: str
) -> None:
    engine.next(engine_error('upstream_unavailable'), script_for(dotted))
    result = await call(admin(mode, engine), dotted, *OPS[dotted])
    assert result.error is None
    assert len(engine.requests) == 2


@pytest.mark.parametrize('failure', ['503', 'dropped-connection'])
@pytest.mark.parametrize('dotted', WRITES)
async def test_a_write_is_not_repeated_after_a_failure_that_may_have_reached_the_engine(
    mode: str, engine: Any, clock: list[float], dotted: str, failure: str
) -> None:
    engine.next(*[FAILURES[failure]() for _ in range(3)])
    result = await call(admin(mode, engine), dotted, *OPS[dotted])
    assert result.error is not None
    assert len(engine.requests) == 1
    assert clock == []


@pytest.mark.parametrize('dotted', WRITES)
async def test_a_write_is_repeated_after_a_rate_limit_and_the_same_body_is_sent(
    mode: str, engine: Any, clock: list[float], dotted: str
) -> None:
    engine.next(*[FAILURES['429']() for _ in range(3)])
    result = await call(admin(mode, engine), dotted, *OPS[dotted])
    assert result.error.code == 'rate_limited'
    assert len(engine.requests) == 3
    assert len({json.dumps(b) for b in engine.bodies()}) == 1
    assert all('idempotency-key' not in r.headers for r in engine.requests)


async def test_a_key_issue_survives_a_rate_limit_and_returns_its_secret(
    mode: str, engine: Any, clock: list[float]
) -> None:
    reply = {'key_id': 'k', 'api_key': SECRET, 'key_prefix': 'gmr_aaaa', 'message': 'once'}
    engine.next(engine_error('rate_limited'), Scripted(status=201, json=reply))
    result = await call(admin(mode, engine), 'keys.create', ORG, PROJECT)
    assert result.error is None
    assert result.data['api_key'] == SECRET
    assert len(engine.requests) == 2


@pytest.mark.parametrize(
    ('body', 'repeats'),
    [
        pytest.param({'name': 'A', 'external_reference': 'ext'}, True, id='reference'),
        pytest.param({'name': 'A', 'external_reference': ' ext '}, True, id='padded-reference'),
        pytest.param({'name': 'A', 'external_reference': 'e'}, True, id='short-reference'),
        pytest.param({'name': 'A'}, False, id='none'),
        pytest.param({'name': 'A', 'external_reference': None}, False, id='null'),
        pytest.param({'name': 'A', 'external_reference': ''}, False, id='empty'),
        pytest.param({'name': 'A', 'external_reference': '   '}, False, id='spaces'),
        pytest.param({'name': 'A', 'external_reference': '\xa0'}, False, id='nbsp'),
        pytest.param({'name': 'A', 'external_reference': 7}, False, id='not-a-string'),
    ],
)
@pytest.mark.parametrize('failure', ['503', 'dropped-connection'])
async def test_org_create_retries_normally_only_with_a_non_blank_external_reference(
    mode: str, engine: Any, clock: list[float], body: dict[str, Any], repeats: bool, failure: str
) -> None:
    engine.next(*[FAILURES[failure]() for _ in range(3)])
    result = await call(admin(mode, engine), 'orgs.create', body)
    assert result.error is not None
    assert len(engine.requests) == (3 if repeats else 1)


@pytest.mark.parametrize('body', [{'name': 'A'}, {'name': 'A', 'external_reference': 'ext'}])
async def test_org_create_always_retries_a_rate_limit(
    mode: str, engine: Any, clock: list[float], body: dict[str, Any]
) -> None:
    engine.next(*[FAILURES['429']() for _ in range(3)])
    await call(admin(mode, engine), 'orgs.create', body)
    assert len(engine.requests) == 3


async def test_max_retries_zero_sends_once(mode: str, engine: Any, clock: list[float]) -> None:
    engine.next(*[FAILURES['503']() for _ in range(3)])
    await call(admin(mode, engine, max_retries=0), 'orgs.list')
    assert len(engine.requests) == 1


# ---------------------------------------------------------------------------------------------------------------------
# D-27: no price in any result
# ---------------------------------------------------------------------------------------------------------------------


def tainted() -> dict[str, Any]:
    return {
        'id': 'x',
        'cost': 1,
        'Cost_Charged': 2,
        'total_cost': 3,
        'supplier_price': 4,
        'provider_cost': 5,
        'input_per_million': 6,
        'recovery_rate': 7,
        'write_off_rate': 8,
        'summary': {'tokens': 3, 'cost': 9},
        'rows': [{'model': 'm', 'tokens': 1, 'price': 2}, [{'k': 1, 'unit_cost': 2}]],
    }


CLEAN = {
    'id': 'x',
    'summary': {'tokens': 3},
    'rows': [{'model': 'm', 'tokens': 1}, [{'k': 1}]],
}


@pytest.mark.parametrize('dotted', [d for d in OPS if d not in DELETES])
async def test_no_price_cost_or_rate_key_survives_in_any_result(mode: str, engine: Any, dotted: str) -> None:
    engine.next(Scripted(json=tainted()))
    result = await call(admin(mode, engine), dotted, *OPS[dotted])
    assert result.error is None
    assert not PRICED.search(json.dumps(result.data))
    assert dict(result.data) == CLEAN


async def test_a_list_reply_is_cleaned_too(mode: str, engine: Any) -> None:
    engine.next(Scripted(json=[tainted(), tainted()]), Scripted(json={'logs': [tainted()], 'has_more': False}))
    client = admin(mode, engine)
    assert (await call(client, 'orgs.list')).data == [CLEAN, CLEAN]
    assert (await call(client, 'logs.list', PROJECT)).data == {'logs': [CLEAN], 'has_more': False}


async def test_usage_and_log_replies_shaped_like_the_engines_lose_their_cost(mode: str, engine: Any) -> None:
    engine.next(
        Scripted(
            json={
                'summary': {'total_requests': 3, 'total_tokens': 9, 'total_cost': None},
                'model_performance': [{'model': 'm', 'requests': 3, 'tokens': 9, 'cost': None}],
                'time_range': {'start': 'a', 'end': 'b'},
            }
        ),
        Scripted(
            json={
                'logs': [
                    {'id': 'l', 'model': 'm', 'tokens_in': 1, 'tokens_out': 2, 'cost_charged': None, 'latency_ms': 9}
                ],
                'has_more': False,
                'next_cursor': None,
            }
        ),
    )
    client = admin(mode, engine)
    usage = await call(client, 'analytics.usage', PROJECT)
    logs = await call(client, 'logs.list', PROJECT)
    assert usage.data['summary'] == {'total_requests': 3, 'total_tokens': 9}
    assert usage.data['model_performance'] == [{'model': 'm', 'requests': 3, 'tokens': 9}]
    assert logs.data['logs'] == [{'id': 'l', 'model': 'm', 'tokens_in': 1, 'tokens_out': 2, 'latency_ms': 9}]


# ---------------------------------------------------------------------------------------------------------------------
# No content, and null
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('dotted', DELETES)
async def test_a_delete_answers_ok_none(mode: str, engine: Any, dotted: str) -> None:
    engine.next(Scripted(status=204, headers={'x-request-id': 'req-9'}))
    result = await call(admin(mode, engine), dotted, *OPS[dotted])
    assert result.__class__ is Ok
    assert result.error is None
    assert result.data is None
    assert result.request_id == 'req-9'


@pytest.mark.parametrize('dotted', DELETES)
async def test_a_delete_that_answers_200_with_an_object_is_still_done(mode: str, engine: Any, dotted: str) -> None:
    engine.next(Scripted(json={'deleted': True}))
    result = await call(admin(mode, engine), dotted, *OPS[dotted])
    assert (result.error, result.data) == (None, None)


@pytest.mark.parametrize('dotted', [d for d in OPS if d not in DELETES])
async def test_a_204_from_any_other_operation_is_a_network_error(mode: str, engine: Any, dotted: str) -> None:
    engine.next(Scripted(status=204))
    result = await call(admin(mode, engine), dotted, *OPS[dotted])
    assert result.data is None
    assert (result.error.code, result.error.type, result.error.status) == ('network_error', 'client_error', 204)
    assert len(engine.requests) == 1


@pytest.mark.parametrize('text', ['null', ' null\n'])
@pytest.mark.parametrize('status', [200, 201])
@pytest.mark.parametrize('dotted', list(OPS))
async def test_a_literal_null_reply_is_a_network_error_on_every_call(
    mode: str, engine: Any, dotted: str, status: int, text: str
) -> None:
    engine.next(Scripted(status=status, text=text, headers={'content-type': 'application/json'}))
    result = await call(admin(mode, engine), dotted, *OPS[dotted])
    assert result.data is None
    assert (result.error.code, result.error.type, result.error.status) == ('network_error', 'client_error', status)
    assert len(engine.requests) == 1


# ---------------------------------------------------------------------------------------------------------------------
# The issued key and the BYOK credential are not printed
# ---------------------------------------------------------------------------------------------------------------------

ISSUED = {'key_id': 'k1', 'api_key': SECRET, 'key_prefix': 'gmr_aaaa', 'message': 'Store this key now'}


async def issue(mode: str, engine: Any) -> Any:
    engine.next(Scripted(status=201, json=ISSUED, headers={'x-request-id': 'req-3'}))
    return await call(admin(mode, engine), 'keys.create', ORG, PROJECT)


async def test_the_issued_key_is_data_for_its_owner(mode: str, engine: Any) -> None:
    result = await issue(mode, engine)
    assert result.error is None
    assert result.request_id == 'req-3'
    data = result.data
    # It is a dict of strings: the owner reads it, stores it, uses it, and JSON carries it
    assert data.__class__ is dict
    assert isinstance(data['api_key'], str)
    assert data['api_key'] == SECRET
    assert dict(data) == ISSUED
    assert json.loads(json.dumps(data)) == ISSUED
    assert json.loads(json.dumps(data, indent=2, ensure_ascii=False)) == ISSUED
    assert SECRET in json.dumps(data)
    assert {**data}['api_key'] == SECRET
    assert 'Bearer ' + data['api_key'] == f'Bearer {SECRET}'
    assert f'Bearer {data["api_key"]}' == f'Bearer {SECRET}'
    assert data['api_key'].encode() == SECRET.encode()


async def test_the_issued_key_does_not_print_through_repr_str_format_or_pprint(mode: str, engine: Any) -> None:
    result = await issue(mode, engine)
    shown = [
        repr(result),
        str(result),
        f'{result}',
        '%s' % (result,),  # noqa: UP031
        '%r' % (result,),  # noqa: UP031
        '{}'.format(result),  # noqa: UP032
        repr(result.data),
        str(result.data),
        f'{result.data}',
        format(result.data),
        repr(result.data['api_key']),
        repr(list(result.data.items())),
        repr({'wrapped': result.data}),
        pprint.pformat(result),
        pprint.pformat(result.data),
        reprlib.repr(result.data),
        reprlib.repr(result),
    ]
    for text in shown:
        assert SECRET not in text, text
    # The rest of the reply stays readable
    assert 'k1' in repr(result.data)
    assert 'gmr_aaaa' in repr(result.data)


async def test_the_issued_key_does_not_print_through_logging(
    mode: str, engine: Any, caplog: pytest.LogCaptureFixture
) -> None:
    result = await issue(mode, engine)
    log = logging.getLogger('gauss_meridian.tests.issued')
    with caplog.at_level(logging.DEBUG):
        log.info('issued %s', result)
        log.info('issued %r', result)
        log.info('issued %s', result.data)
        log.info('issued %r', result.data)
        log.warning('issued %(r)s', {'r': result.data})
        log.info('issued %s', result.data['api_key'].__repr__())
        log.info(f'issued {result}')
        log.error('failed to store', exc_info=(AssertionError, AssertionError(str(result.data)), None))
    assert caplog.records
    assert SECRET not in caplog.text
    assert all(SECRET not in str(r.args) and SECRET not in r.getMessage() for r in caplog.records)


async def test_the_issued_key_does_not_print_in_pytest_assertion_output(mode: str, engine: Any) -> None:
    result = await issue(mode, engine)
    outputs: list[str] = []
    # pytest rewrites an assert in a test module, so the AssertionError's text is what a failing test would print
    with pytest.raises(AssertionError) as info:
        assert result.data == {'key_id': 'other'}
    outputs.append(str(info.value))
    with pytest.raises(AssertionError) as info:
        assert result == Ok({'key_id': 'other'})
    outputs.append(str(info.value))
    with pytest.raises(AssertionError) as info:
        assert result.data == {**ISSUED, 'message': 'changed'}
    outputs.append(str(info.value))
    with pytest.raises(AssertionError) as info:
        assert result.data['api_key'].startswith('zzz')
    outputs.append(str(info.value))
    with pytest.raises(AssertionError) as info:
        assert result.data['api_key'] in ['nothing']
    outputs.append(str(info.value))
    with pytest.raises(AssertionError) as info:
        assert result.data['api_key'] is None
    outputs.append(str(info.value))
    with pytest.raises(AssertionError) as info:
        assert result.data
        assert not result.data
    outputs.append(str(info.value))
    assert all('assert' in text for text in outputs)
    for text in outputs:
        assert SECRET not in text, text


async def test_the_issued_key_is_not_in_any_other_result_or_error(mode: str, engine: Any) -> None:
    engine.next(
        Scripted(json={'keys': [{'id': 'k1', 'key_prefix': 'gmr_aaaa'}]}),
        engine_error('invalid_request'),
    )
    client = admin(mode, engine)
    listed = await call(client, 'keys.list', ORG, PROJECT)
    failed = await call(client, 'keys.create', ORG, PROJECT, {'name': SECRET})
    assert listed.data == {'keys': [{'id': 'k1', 'key_prefix': 'gmr_aaaa'}]}
    assert SECRET not in repr(failed)
    assert SECRET not in repr(failed.error)


async def test_a_key_issue_that_is_not_the_documented_shape_passes_through(mode: str, engine: Any) -> None:
    engine.next(Scripted(status=201, json={'key_id': 'k1'}), Scripted(status=201, json={'api_key': 7}))
    client = admin(mode, engine)
    assert (await call(client, 'keys.create', ORG, PROJECT)).data == {'key_id': 'k1'}
    assert (await call(client, 'keys.create', ORG, PROJECT)).data == {'api_key': 7}


def test_a_secret_is_a_str_that_only_hides_in_repr() -> None:
    secret = Secret(SECRET)
    assert isinstance(secret, str)
    assert secret == SECRET
    assert hash(secret) == hash(SECRET)
    assert SECRET not in repr(secret)
    assert SECRET not in repr([secret])
    assert SECRET not in repr({'k': secret})
    assert str(secret) == SECRET
    assert f'{secret}' == SECRET
    assert json.dumps(secret) == json.dumps(SECRET)
    assert type(str(secret)) is str


async def test_a_byok_body_wrapped_in_secret_is_sent_in_full_and_does_not_print(mode: str, engine: Any) -> None:
    raw = 'AIza' + 'q' * 35
    body: SetCredential = {'credential': Secret(raw), 'policy': 'prefer'}
    engine.next(Scripted(json={'provider': 'gemini', 'hint': '…qqqq', 'policy': 'prefer'}))
    result = await call(admin(mode, engine), 'provider_credentials.put', ORG, PROVIDER, body)
    assert result.error is None
    assert engine.bodies() == [{'credential': raw, 'policy': 'prefer'}]
    assert raw not in repr(body)
    assert raw not in str(body)
    assert raw not in f'{body}'
    assert raw not in pprint.pformat(body)
    assert raw in json.dumps(body)
    assert raw not in repr(result)
    with pytest.raises(AssertionError) as info:
        assert body == {'credential': 'other'}
    assert raw not in str(info.value)


async def test_a_byok_credential_is_never_echoed_by_an_error(mode: str, engine: Any) -> None:
    raw = 'AIza' + 'q' * 35
    engine.next(engine_error('invalid_request'), Scripted(json={'unexpected': True}))
    client = admin(mode, engine)
    bad = await call(client, 'provider_credentials.put', ORG, PROVIDER, {'credential': raw})
    unusable = await call(client, 'provider_credentials.put', ORG, PROVIDER, {'credential': raw, 'extra': {1, 2}})
    assert bad.error is not None
    assert unusable.error is not None
    for result in (bad, unusable):
        assert raw not in repr(result)
        assert raw not in result.error.message


async def test_the_integrator_key_is_never_in_a_refusal_or_the_client_repr(mode: str, engine: Any) -> None:
    key = 'gmi_é' + '1' * 60
    client = admin(mode, engine, key=key)
    result = await call(client, 'orgs.list')
    assert result.error.code == 'invalid_request'
    assert result.error.type == 'client_error'
    # The codec's own message would quote a character of the key
    assert 'é' not in repr(result)
    assert not re.search(r'codec|xe9|position', result.error.message)
    assert '1' * 60 not in repr(client)
    assert engine.integrator_key not in repr(admin(mode, engine))
    assert engine.requests == []


# ---------------------------------------------------------------------------------------------------------------------
# Configuration, and nothing ever raises
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'override',
    [
        pytest.param({'key': ''}, id='no-key'),
        pytest.param({'url': ''}, id='no-url'),
        pytest.param({'url': '', 'key': ''}, id='neither'),
        pytest.param({'url': 'http://[::1'}, id='malformed-url'),
        pytest.param({'url': 'http://host:99999'}, id='bad-port'),
        pytest.param({'url': 'ftp://admin.engine.test'}, id='not-http'),
        pytest.param({'url': 'admin.engine.test'}, id='no-scheme'),
        pytest.param({'http_client': True}, id='client-and-transport-both'),
    ],
)
async def test_a_bad_setup_gives_a_client_whose_every_call_is_a_refusal_with_nothing_sent(
    mode: str, engine: Any, override: dict[str, Any]
) -> None:
    options = dict(override)
    if options.get('http_client'):
        options['http_client'] = httpx.Client() if mode == 'sync' else httpx.AsyncClient()
    client = admin(mode, engine, **options)
    for dotted in OPS:
        result = await call(client, dotted, *OPS[dotted])
        assert result.data is None, dotted
        assert (result.error.code, result.error.type, result.error.retryable) == (
            'invalid_request',
            'client_error',
            False,
        ), dotted
    assert engine.requests == []
    if 'http_client' in options:
        http_client: Any = options['http_client']
        await resolve(http_client.close() if mode == 'sync' else http_client.aclose())


async def test_the_environment_supplies_the_url_and_key(
    mode: str, engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('MERIDIAN_ADMIN_URL', engine.admin_url)
    monkeypatch.setenv('MERIDIAN_INTEGRATOR_KEY', engine.integrator_key)
    client = (MeridianAdmin if mode == 'sync' else AsyncMeridianAdmin)(transport=engine.transport)
    result = await call(client, 'orgs.list')
    assert result.error is None
    assert engine.requests[0].headers['authorization'] == f'Bearer {engine.integrator_key}'
    assert engine.requests[0].url.host == 'admin.engine.test'


async def test_the_public_settings_are_not_a_fallback(mode: str, engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('MERIDIAN_ADMIN_URL', raising=False)
    monkeypatch.delenv('MERIDIAN_INTEGRATOR_KEY', raising=False)
    monkeypatch.setenv('MERIDIAN_URL', engine.public_url)
    monkeypatch.setenv('MERIDIAN_KEY', engine.project_key)
    client = (MeridianAdmin if mode == 'sync' else AsyncMeridianAdmin)(transport=engine.transport)
    result = await call(client, 'orgs.list')
    assert result.error.code == 'invalid_request'
    assert engine.requests == []


async def test_an_explicit_empty_argument_beats_the_environment(
    mode: str, engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('MERIDIAN_ADMIN_URL', engine.admin_url)
    monkeypatch.setenv('MERIDIAN_INTEGRATOR_KEY', engine.integrator_key)
    result = await call(admin(mode, engine, key=''), 'orgs.list')
    assert result.error.code == 'invalid_request'
    assert engine.requests == []


async def test_network_decode_and_server_failures_are_error_values(mode: str, engine: Any, clock: list[float]) -> None:
    client = admin(mode, engine, max_retries=0)
    engine.next(
        Scripted(network_error=True),
        Scripted(timeout=True),
        Scripted(text='{not json', headers={'content-type': 'application/json'}),
        Scripted(text='[' * 100000, headers={'content-type': 'application/json'}),
        Scripted(status=502, text='<html>bad gateway</html>'),
        Scripted(status=200, chunks=[b'{"a":', b'1}'], drop_after=1),
    )
    codes = [(await call(client, 'orgs.list')).error.code for _ in range(6)]
    assert codes == ['network_error'] * 6


async def test_an_unknown_engine_code_is_internal_and_not_retried(mode: str, engine: Any, clock: list[float]) -> None:
    body = {'error': {'message': 'm', 'type': 'api_error', 'code': 'brand_new', 'request_id': 'r'}}
    engine.next(*[Scripted(status=503, json=body)] * 3)
    result = await call(admin(mode, engine), 'orgs.list')
    assert (result.error.code, result.error.engine_code, result.error.retryable) == ('internal', 'brand_new', False)
    assert len(engine.requests) == 1


async def test_a_known_code_keeps_its_type_status_and_request_id(mode: str, engine: Any) -> None:
    engine.next(engine_error('permission_denied'))
    result = await call(admin(mode, engine), 'orgs.list')
    assert (result.error.code, result.error.type, result.error.status) == ('permission_denied', 'permission_error', 403)
    assert result.error.request_id


async def test_a_body_that_cannot_be_encoded_is_a_refusal_never_a_raise(mode: str, engine: Any) -> None:
    client = admin(mode, engine)
    for body in ({'name': {1, 2}}, {'name': object()}, {'name': b'x'}):
        result = await call(client, 'projects.create', ORG, body)
        assert (result.error.code, result.error.type) == ('invalid_request', 'client_error')
    assert engine.requests == []


async def test_a_body_that_is_circular_is_a_refusal_never_a_raise(mode: str, engine: Any) -> None:
    loop: dict[str, Any] = {}
    loop['self'] = loop
    result = await call(admin(mode, engine), 'projects.create', ORG, loop)
    assert (result.error.code, result.error.type) == ('invalid_request', 'client_error')
    assert engine.requests == []


async def test_a_cancelled_wait_propagates_and_sends_nothing_more(engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import gauss_meridian._clock as clock_module

    async def cancelled(seconds: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(clock_module, 'asleep', cancelled)
    engine.next(*[engine_error('upstream_unavailable')] * 3)
    with pytest.raises(asyncio.CancelledError):
        await AsyncMeridianAdmin(engine.admin_url, engine.integrator_key, transport=engine.transport).orgs.get(ORG)
    assert len(engine.requests) == 1


def test_the_sync_client_is_a_context_manager(engine: Any) -> None:
    with MeridianAdmin(engine.admin_url, engine.integrator_key, transport=engine.transport) as client:
        assert client.orgs.list().error is None
    client.close()


async def test_the_async_client_is_an_async_context_manager(engine: Any) -> None:
    async with AsyncMeridianAdmin(engine.admin_url, engine.integrator_key, transport=engine.transport) as client:
        assert (await client.orgs.list()).error is None
    await client.aclose()


# ---------------------------------------------------------------------------------------------------------------------
# Review 1: deep replies, wrong-typed arguments, query values, error members, reply caps
# ---------------------------------------------------------------------------------------------------------------------


def depth_of(value: Any) -> int:
    """How deep lists nest, counted without recursion."""
    depth = 0
    while isinstance(value, list) and value:
        depth += 1
        value = cast('list[Any]', value)[0]
    return depth + (1 if isinstance(value, list) else 0)


@pytest.mark.parametrize('depth', [600, 5000])
@pytest.mark.parametrize('dotted', ['orgs.list', 'logs.list'])
async def test_a_deeply_nested_reply_is_a_result_or_a_network_error_never_a_raise(
    mode: str, engine: Any, dotted: str, depth: int
) -> None:
    text = '{"rows": ' + '[' * depth + ']' * depth + ', "cost": 1}'
    engine.next(Scripted(text=text, headers={'content-type': 'application/json'}))
    result = await call(admin(mode, engine, max_retries=0), dotted, *OPS[dotted])
    if depth == 600 or result.error is None:  # a parser that reads it hands it over whole, without its price
        assert result.error is None
        assert list(result.data) == ['rows']
        assert depth_of(result.data['rows']) == depth
    else:  # too deep for this interpreter's parser
        assert (result.error.code, result.error.status) == ('network_error', 200)


async def test_a_byok_credential_given_as_text_is_held_as_a_secret_and_sent_unchanged(mode: str) -> None:
    raw = 'AIza' + 'q' * 35
    body = {'credential': raw, 'policy': 'prefer'}
    held: list[str] = []
    sent: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # What a frame-locals error reporter (pytest --showlocals, Sentry) would capture from the SDK's transport
        sent.append(request.read())
        for frame, _ in traceback.walk_stack(None):
            if frame.f_globals.get('__name__') == 'gauss_meridian._transport':
                held.append(repr(frame.f_locals))
        return httpx.Response(200, json={'provider': 'gemini', 'hint': 'qqqq'})

    client = (MeridianAdmin if mode == 'sync' else AsyncMeridianAdmin)(
        'http://admin.engine.test', 'gmi_' + '1' * 64, transport=httpx.MockTransport(handler)
    )
    result = await call(client, 'provider_credentials.put', ORG, PROVIDER, body)
    assert result.error is None
    assert json.loads(sent[0]) == {'credential': raw, 'policy': 'prefer'}
    assert sent[0] == json.dumps({'credential': raw, 'policy': 'prefer'}).encode()
    assert held and all(raw not in text for text in held)
    # The caller's own dict is untouched
    assert body == {'credential': raw, 'policy': 'prefer'} and type(body['credential']) is str


WRITE_OPS = {
    'orgs.create': (),
    'projects.create': (ORG,),
    'projects.update': (ORG, PROJECT),
    'keys.create': (ORG, PROJECT),
    'provider_credentials.put': (ORG, PROVIDER),
    'project_models.put': (ORG, PROJECT),
}


@pytest.mark.parametrize('body', [None, 'x', 5, ['a'], b'x', {'a'}, ('a', 1)], ids=repr)
@pytest.mark.parametrize('dotted', WRITE_OPS)
async def test_a_body_that_is_not_a_mapping_is_refused_with_nothing_sent(
    mode: str, engine: Any, dotted: str, body: Any
) -> None:
    result = await call(admin(mode, engine), dotted, *WRITE_OPS[dotted], body)
    if dotted == 'keys.create' and body is None:  # no body is an empty one, and is sent
        assert result.error is None and engine.bodies() == [{}]
        return
    assert (result.error.code, result.error.type, result.error.retryable) == ('invalid_request', 'client_error', False)
    assert engine.requests == []


async def test_a_closed_admin_client_answers_every_call_with_an_error_value_and_sends_nothing(
    mode: str, engine: Any
) -> None:
    client = admin(mode, engine)
    await resolve(client.close() if mode == 'sync' else client.aclose())
    assert client.is_closed
    for dotted in OPS:
        result = await call(client, dotted, *OPS[dotted])
        error = result.error
        assert (error.code, error.type, error.retryable, error.message) == (
            'invalid_request',
            'client_error',
            False,
            'The client is closed.',
        )
    assert engine.requests == []


@pytest.mark.parametrize('dotted', WRITE_OPS)
async def test_a_body_holding_nan_or_infinity_is_refused_with_nothing_sent(mode: str, engine: Any, dotted: str) -> None:
    client = admin(mode, engine)
    for number in (float('nan'), float('inf'), float('-inf')):
        result = await call(client, dotted, *WRITE_OPS[dotted], {'name': 'a', 'n': number})
        assert (result.error.code, result.error.type) == ('invalid_request', 'client_error')
    assert engine.requests == []


async def test_an_empty_query_value_is_left_out(mode: str, engine: Any) -> None:
    client = admin(mode, engine)
    await call(client, 'logs.list', PROJECT, cursor='')
    await call(client, 'analytics.usage', PROJECT, from_='', to='')
    assert [dict(r.url.params) for r in engine.requests] == [{'project_id': PROJECT}] * 2


@pytest.mark.parametrize('page_size', [True, False, '5', 1.5, object(), [5]], ids=repr)
async def test_a_page_size_that_is_not_an_int_is_refused_with_nothing_sent(
    mode: str, engine: Any, page_size: Any
) -> None:
    result = await call(admin(mode, engine), 'logs.list', PROJECT, page_size=page_size)
    assert (result.error.code, result.error.type) == ('invalid_request', 'client_error')
    assert engine.requests == []


@pytest.mark.parametrize('status', [201, 200], ids=['created', 'replayed'])
async def test_org_create_is_ok_when_created_and_when_replayed(mode: str, engine: Any, status: int) -> None:
    org = {'id': 'o1', 'name': 'A', 'external_reference': 'ext-1'}
    engine.next(Scripted(status=status, json=org))
    result = await call(admin(mode, engine), 'orgs.create', {'name': 'A', 'external_reference': 'ext-1'})
    assert result.error is None
    assert result.data == org


ERROR_MEMBERS = [
    pytest.param('boom', id='string'),
    pytest.param({}, id='empty-object'),
    pytest.param(0, id='zero'),
    pytest.param(False, id='false'),
    pytest.param([], id='empty-list'),
    pytest.param({'message': 'm', 'type': 'api_error', 'code': 'internal', 'request_id': 'req_err'}, id='envelope'),
]


@pytest.mark.parametrize('error', ERROR_MEMBERS)
@pytest.mark.parametrize('dotted', list(OPS))
async def test_a_2xx_reply_with_an_error_member_is_a_network_error_on_every_call(
    mode: str, engine: Any, dotted: str, error: Any
) -> None:
    engine.next(Scripted(status=200, headers={'x-request-id': 'req_hdr'}, json={'ok': True, 'error': error}))
    result = await call(admin(mode, engine), dotted, *OPS[dotted])
    assert result.data is None
    assert (result.error.code, result.error.type, result.error.status) == ('network_error', 'client_error', 200)
    assert result.error.request_id == ('req_err' if isinstance(error, dict) and error else 'req_hdr')
    assert len(engine.requests) == 1


async def test_a_2xx_reply_with_a_null_error_member_is_a_result(mode: str, engine: Any) -> None:
    engine.next(Scripted(json={'id': 'o1', 'error': None}))
    result = await call(admin(mode, engine), 'orgs.get', ORG)
    assert result.error is None and result.data == {'id': 'o1', 'error': None}


async def test_a_reply_over_64_mib_is_refused_without_reading_it_all(mode: str, engine: Any) -> None:
    piece = b' ' * (1 << 20)
    engine.next(Scripted(status=200, chunks=[b'{"a": "', *[piece] * 80, b'"}']))
    result = await call(admin(mode, engine), 'orgs.list')
    assert (result.error.code, result.error.status) == ('network_error', 200)
    assert engine.parts_read() <= 66 and engine.upstream_closed()


async def test_an_error_body_over_1_mib_is_not_read_as_an_envelope(mode: str, engine: Any) -> None:
    big = {'error': {'message': 'm' * (1 << 20), 'type': 'invalid_request_error', 'code': 'invalid_request'}}
    engine.next(Scripted(status=400, headers={'x-request-id': 'req_big'}, json=big))
    result = await call(admin(mode, engine), 'orgs.list')
    assert (result.error.code, result.error.status, result.error.request_id) == ('network_error', 400, 'req_big')
    assert len(engine.requests) == 1
