"""Refuse to time coins when the run directory carries no fullnode block data."""
import json
import os

import pytest

from cj_process.parse_cj_logs import (
    MissingBlockDataError,
    load_tx_database_from_btccore,
    obtain_wallets_info,
)

TXID = 'b' * 64


def write_block(btc_node_dir, txid, block_time=1700000000):
    """Write one block export in the shape load_tx_database_from_btccore() reads."""
    os.makedirs(btc_node_dir, exist_ok=True)
    block = {'time': block_time, 'tx': [{'txid': txid, 'hash': 'block-hash'}]}
    with open(os.path.join(btc_node_dir, f'block_{txid[:8]}.json'), 'w') as file:
        json.dump(block, file)


def write_wallet(tmp_path, coins):
    """Lay out the docker-file wallet directory obtain_wallets_info() reads."""
    wallet_dir = tmp_path / 'data' / 'wasabi-client-000'
    wallet_dir.mkdir(parents=True)
    (wallet_dir / 'keys.json').write_text('[]')
    (wallet_dir / 'coins.json').write_text(json.dumps(coins))


def test_absent_btc_node_directory_names_the_path(tmp_path):
    missing = str(tmp_path / 'data' / 'btc-node')

    with pytest.raises(MissingBlockDataError) as excinfo:
        load_tx_database_from_btccore(missing)

    assert missing in str(excinfo.value)


def test_btc_node_directory_without_block_exports_is_rejected(tmp_path):
    btc_node = tmp_path / 'data' / 'btc-node'
    btc_node.mkdir(parents=True)
    (btc_node / 'debug.log').write_text('not a block export')

    with pytest.raises(MissingBlockDataError) as excinfo:
        load_tx_database_from_btccore(str(btc_node))

    assert 'block_*.json' in str(excinfo.value)


def test_present_block_exports_are_loaded(tmp_path):
    """The guard must not fire on a complete run."""
    btc_node = tmp_path / 'data' / 'btc-node'
    write_block(str(btc_node), TXID)

    tx_db = load_tx_database_from_btccore(str(btc_node))

    assert list(tx_db) == [TXID]
    assert tx_db[TXID]['mine_time'].startswith('2023-')


def test_coin_without_any_block_time_aborts_the_run(tmp_path):
    """A held coin with an empty transaction database is a broken run, not a quiet default."""
    write_wallet(tmp_path, [
        {'txid': TXID, 'index': 0, 'address': 'bcrt1qheld', 'amount': 1, 'anonymityScore': 1},
    ])

    with pytest.raises(MissingBlockDataError) as excinfo:
        obtain_wallets_info(str(tmp_path), False, True, {})

    message = str(excinfo.value)
    assert 'wallet-000' in message
    assert TXID in message
    assert os.path.join('data', 'btc-node') in message


@pytest.mark.parametrize('mine_time', [
    '2023-11-14 22:13:20.000',  # what the docker block export writes
    '2023-11-14T22:13:20Z',
    1700000000,
])
def test_synthetic_time_accepts_every_supported_mine_time(tmp_path, mine_time):
    """Synthetic times go through the shared mine_time contract, so any form the
    exporters produce is timed the same way instead of only the docker one."""
    unmatched_txid = 'c' * 64
    write_wallet(tmp_path, [
        {'txid': unmatched_txid, 'index': 0, 'address': 'bcrt1qheld', 'amount': 1, 'anonymityScore': 1},
    ])
    all_tx_db = {TXID: {'hash': 'block-hash', 'mine_time': mine_time}}

    _, wallets_coins = obtain_wallets_info(str(tmp_path), False, True, all_tx_db)

    coin = wallets_coins['wallet-000'][0]
    assert coin['block_hash'] == 'synthetic_block'
    assert coin['create_time'] == '2023-11-14 22:23:20.000'


@pytest.mark.parametrize('mine_time', [
    '2023-11-14 22:13:20.000',  # what the docker block export writes
    '2023-11-14T22:13:20Z',
    1700000000,
])
def test_coin_times_are_stored_in_the_canonical_form(tmp_path, mine_time):
    """Coin times are later compared as strings and re-parsed with a fixed format, so they
    are normalized on the way in rather than kept in the exporter's own representation."""
    spending_txid = 'd' * 64
    write_wallet(tmp_path, [
        {'txid': TXID, 'index': 0, 'address': 'bcrt1qspent', 'amount': 1, 'anonymityScore': 1,
         'spentBy': spending_txid},
    ])
    all_tx_db = {
        TXID: {'hash': 'block-hash', 'mine_time': mine_time},
        spending_txid: {'hash': 'spend-block-hash', 'mine_time': mine_time},
    }

    _, wallets_coins = obtain_wallets_info(str(tmp_path), False, True, all_tx_db)

    coin = wallets_coins['wallet-000'][0]
    assert coin['create_time'] == '2023-11-14 22:13:20.000'
    assert coin['destroy_time'] == '2023-11-14 22:13:20.000'


def test_synthetic_time_takes_the_latest_of_mixed_mine_times(tmp_path):
    """The latest block wins even when the database mixes the supported forms, which
    comparing the stored values instead of the parsed datetimes cannot do."""
    unmatched_txid = 'c' * 64
    write_wallet(tmp_path, [
        {'txid': unmatched_txid, 'index': 0, 'address': 'bcrt1qheld', 'amount': 1, 'anonymityScore': 1},
    ])
    all_tx_db = {
        TXID: {'hash': 'block-hash', 'mine_time': 1700000000},
        'd' * 64: {'hash': 'later-block-hash', 'mine_time': '2023-11-14 23:13:20.000'},
    }

    _, wallets_coins = obtain_wallets_info(str(tmp_path), False, True, all_tx_db)

    assert wallets_coins['wallet-000'][0]['create_time'] == '2023-11-14 23:23:20.000'


def test_wallet_holding_no_coins_needs_no_block_time(tmp_path):
    """A coinless wallet never needs a synthetic time, so an empty database is fine."""
    write_wallet(tmp_path, [])

    wallets_info, wallets_coins = obtain_wallets_info(str(tmp_path), False, True, {})

    assert wallets_info['wallet-000'] == {}
    assert wallets_coins['wallet-000'] == []
