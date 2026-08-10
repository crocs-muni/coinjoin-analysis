import json
import os

import pytest

from cj_process.parse_cj_logs import obtain_wallets_info
from cj_process.wallet_attribution import (
    COORDINATOR_WALLET_STRING,
    CoordinatorAttributionError,
    UnattributedCoinsError,
    assert_all_coins_attributed,
    assert_no_coordinator_for_joinmarket,
)

TXID = 'a' * 64


def make_coinjoins(inputs=None, outputs=None):
    return {TXID: {'inputs': inputs or {}, 'outputs': outputs or {}}}


def test_parse_module_reexports_the_guard_api():
    from cj_process import parse_cj_logs

    assert parse_cj_logs.assert_all_coins_attributed is assert_all_coins_attributed
    assert parse_cj_logs.assert_no_coordinator_for_joinmarket is assert_no_coordinator_for_joinmarket
    assert parse_cj_logs.CoordinatorAttributionError is CoordinatorAttributionError
    assert parse_cj_logs.UnattributedCoinsError is UnattributedCoinsError


def test_attributed_joinmarket_experiment_passes():
    coinjoins = make_coinjoins(
        inputs={"0": {"wallet_name": "wallet-000", "address": "bcrt1q0"}},
        outputs={"0": {"wallet_name": "wallet-001", "address": "bcrt1q1"}},
    )

    assert_no_coordinator_for_joinmarket(coinjoins)


def test_coordinator_label_aborts_joinmarket_experiment():
    coinjoins = make_coinjoins(
        outputs={"0": {"wallet_name": COORDINATOR_WALLET_STRING, "address": "bcrt1qfee"}})

    with pytest.raises(CoordinatorAttributionError) as excinfo:
        assert_no_coordinator_for_joinmarket(coinjoins)

    message = str(excinfo.value)
    assert 'JoinMarket has no coordinator' in message
    assert 'bcrt1qfee' in message


def test_coordinator_report_names_the_offending_records():
    coinjoins = make_coinjoins(
        inputs={"3": {"wallet_name": COORDINATOR_WALLET_STRING, "address": "bcrt1qin"}},
        outputs={"7": {"wallet_name": COORDINATOR_WALLET_STRING, "address": "bcrt1qout"}},
    )

    with pytest.raises(CoordinatorAttributionError) as excinfo:
        assert_no_coordinator_for_joinmarket(coinjoins)

    message = str(excinfo.value)
    assert '2 record(s)' in message
    assert 'inputs[3]' in message
    assert 'outputs[7]' in message


def test_coordinator_report_is_truncated():
    outputs = {
        str(i): {"wallet_name": COORDINATOR_WALLET_STRING, "address": f'bcrt1q{i}'}
        for i in range(9)
    }

    with pytest.raises(CoordinatorAttributionError) as excinfo:
        assert_no_coordinator_for_joinmarket(make_coinjoins(outputs=outputs))

    assert '... and 4 more' in str(excinfo.value)


def test_fully_attributed_experiment_needs_no_wallet_complaint():
    coinjoins = make_coinjoins(
        inputs={"0": {"wallet_name": "wallet-000", "address": "bcrt1q0"}},
        outputs={"0": {"wallet_name": COORDINATOR_WALLET_STRING, "address": "bcrt1qfee"}},
    )

    assert_all_coins_attributed(coinjoins)


def test_missing_wallet_name_aborts_the_experiment():
    taproot = 'bcrt1p' + 'y' * 56
    coinjoins = make_coinjoins(outputs={"0": {"address": taproot}})

    with pytest.raises(UnattributedCoinsError) as excinfo:
        assert_all_coins_attributed(coinjoins)

    message = str(excinfo.value)
    assert 'belong to no known wallet' in message
    assert taproot in message


def test_partial_recovery_is_caught_even_though_other_coins_are_fine():
    """The gap the coordinator fallback cannot cover: one address of a different length."""
    coinjoins = make_coinjoins(
        inputs={
            "0": {"wallet_name": "wallet-008", "address": "bcrt1q" + "a" * 38},
            "1": {"address": "bcrt1p" + "b" * 58},
        },
        outputs={"0": {"wallet_name": "wallet-008", "address": "bcrt1q" + "c" * 38}},
    )

    with pytest.raises(UnattributedCoinsError) as excinfo:
        assert_all_coins_attributed(coinjoins)

    assert '1 record(s)' in str(excinfo.value)
    assert 'inputs[1]' in str(excinfo.value)


def test_blank_wallet_name_counts_as_unattributed():
    coinjoins = make_coinjoins(outputs={"0": {"wallet_name": "", "address": "bcrt1qblank"}})

    with pytest.raises(UnattributedCoinsError):
        assert_all_coins_attributed(coinjoins)


def write_wallet(tmp_path, wallet_name, keys, coins='[]', unspent=None, client='wasabi-client'):
    """Lay out the docker-file wallet directory obtain_wallets_info() reads."""
    wallet_dir = tmp_path / 'data' / f'{client}-{wallet_name}'
    wallet_dir.mkdir(parents=True)
    (wallet_dir / 'keys.json').write_text(keys)
    (wallet_dir / 'coins.json').write_text(coins)
    if unspent is not None:
        (wallet_dir / 'unspent_coins.json').write_text(json.dumps(unspent))
    return wallet_dir


def test_unavailable_addresses_are_allowed_for_a_nonparticipating_wallet(tmp_path, caplog):
    write_wallet(tmp_path, '000', keys=json.dumps('timeout'))

    wallets_info, _ = obtain_wallets_info(str(tmp_path), False, True, {})

    assert wallets_info['wallet-000'] == {}
    assert 'wallet may not participate' in caplog.text


def test_recovered_addresses_keep_the_run_going(tmp_path):
    unspent = {'utxos': [{'address': 'bcrt1qrecovered', 'path': "m/84'/1'/0'/0/0"}]}
    write_wallet(
        tmp_path,
        '000',
        keys=json.dumps('This method is not available in joinmarket'),
        unspent=unspent,
    )

    wallets_info, _ = obtain_wallets_info(str(tmp_path), False, True, {})

    assert 'bcrt1qrecovered' in wallets_info['wallet-000']


def test_working_listkeys_wallet_is_untouched(tmp_path):
    keys = json.dumps([{'address': 'bcrt1qnormal', 'fullKeyPath': "84'/0'/0'/1/0"}])
    write_wallet(tmp_path, '000', keys=keys)

    wallets_info, _ = obtain_wallets_info(str(tmp_path), False, True, {})

    assert list(wallets_info['wallet-000']) == ['bcrt1qnormal']


@pytest.mark.parametrize('invalid_json', ['', '{invalid'])
def test_empty_or_invalid_coins_export_does_not_abort_collection(tmp_path, caplog, invalid_json):
    write_wallet(tmp_path, '000', keys='[]', coins=invalid_json)

    wallets_info, wallets_coins = obtain_wallets_info(str(tmp_path), False, True, {})

    assert wallets_info['wallet-000'] == {}
    assert wallets_coins['wallet-000'] == {}
    assert 'Empty or invalid coins.json' in caplog.text


@pytest.mark.parametrize('invalid_json', ['', '{invalid'])
def test_invalid_coins_export_keeps_valid_wallet_keys(tmp_path, invalid_json):
    keys = json.dumps([{'address': 'bcrt1qnormal', 'fullKeyPath': "84'/0'/0'/1/0"}])
    write_wallet(tmp_path, '000', keys=keys, coins=invalid_json)

    wallets_info, wallets_coins = obtain_wallets_info(str(tmp_path), False, True, {})

    assert list(wallets_info['wallet-000']) == ['bcrt1qnormal']
    assert wallets_coins['wallet-000'] == {}


@pytest.mark.parametrize('invalid_json', ['', '{invalid'])
def test_empty_or_invalid_keys_export_still_recovers_joinmarket_addresses(tmp_path, caplog, invalid_json):
    unspent = {'utxos': [{'address': 'bcrt1qrecovered', 'path': "m/84'/1'/0'/0/0"}]}
    write_wallet(
        tmp_path,
        '000',
        keys=invalid_json,
        coins=json.dumps('This method is not available in JoinMarket'),
        unspent=unspent,
        client='jcs',
    )

    wallets_info, wallets_coins = obtain_wallets_info(str(tmp_path), False, True, {})

    assert 'bcrt1qrecovered' in wallets_info['wallet-000']
    assert wallets_coins['wallet-000'] == {}
    assert 'Empty or invalid keys.json' in caplog.text
