import copy
import json
import re
from typing import Any, cast

import pytest

from gauss_meridian._prices import without_prices

# D-27: nothing priced, and none of the rates the operator earns on, may reach an integrator
PRICED = re.compile(
    'cost|price|per_million|recovery_rate|write_off_rate|bleed|revenue|uncollected|written_off', re.IGNORECASE
)


@pytest.mark.parametrize(
    'key',
    [
        'cost',
        'Cost',
        'cost_charged',
        'total_cost',
        'provider_cost',
        'supplier_price',
        'PRICE',
        'unit_price_usd',
        'input_per_million',
        'output_PER_MILLION_tokens',
        'recovery_rate',
        'Write_Off_Rate',
        'bleed',
        'revenue_share',
        'Uncollected_Total',
        'written_off',
    ],
)
def test_a_priced_key_is_dropped_at_the_top_and_at_depth(key: str) -> None:
    value: Any = {key: 1, 'tokens': 2, 'summary': {key: 3, 'n': 4}, 'rows': [{key: 5, 'id': 'a'}, [{key: 6, 'x': 7}]]}
    assert without_prices(value) == {'tokens': 2, 'summary': {'n': 4}, 'rows': [{'id': 'a'}, [{'x': 7}]]}


def test_a_priced_value_under_an_innocent_key_and_a_key_that_only_looks_close_are_kept() -> None:
    value: Any = {'note': 'the price is high', 'rate': 0.5, 'recovery': 'ok'}
    assert without_prices(value) == value


@pytest.mark.parametrize('value', [None, True, 3, 1.5, 'cost', [], {}, [None, 1, 'price']])
def test_plain_values_pass_through(value: Any) -> None:
    assert without_prices(value) == value


def test_any_depth_is_cleaned_without_recursion() -> None:
    deep: Any = {'cost': 1, 'id': 'leaf'}
    for _ in range(100_000):
        deep = [{'price': 2, 'next': deep}]
    out: Any = without_prices(deep)
    levels = 0
    while isinstance(out, list):
        levels += 1
        first: Any = cast('list[Any]', out)[0]
        assert list(first) == ['next']
        out = first['next']
    assert levels == 100_000 and out == {'id': 'leaf'}


def test_the_input_is_not_changed() -> None:
    value: Any = {'cost': 1, 'rows': [{'price': 2, 'id': 'a'}]}
    before = copy.deepcopy(value)
    without_prices(value)
    assert value == before


def test_a_real_shaped_reply_of_every_kind_comes_out_clean() -> None:
    log_page: Any = {
        'logs': [
            {'id': 'l', 'model': 'm', 'tokens_in': 1, 'tokens_out': 2, 'cost_charged': None, 'latency_ms': 9},
        ],
        'has_more': False,
        'next_cursor': None,
    }
    usage: Any = {
        'summary': {'total_requests': 3, 'total_tokens': 9, 'total_cost': None},
        'model_performance': [{'model': 'm', 'requests': 3, 'tokens': 9, 'cost': None}],
        'time_range': {'start': 'a', 'end': 'b'},
    }
    catalogue: Any = [
        {'id': 'm', 'provider': 'p', 'input_per_million': 1, 'output_per_million': 2, 'context_tokens': 8}
    ]
    credit: Any = {'provider': 'gemini', 'recovery_rate': 90, 'write_off_rate': 1, 'hint': '1234'}
    for reply in (log_page, usage, catalogue, credit):
        assert not PRICED.search(json.dumps(without_prices(reply)))
