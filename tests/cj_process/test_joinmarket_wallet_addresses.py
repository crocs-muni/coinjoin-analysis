import json
import logging
from unittest import mock

from cj_process import parse_cj_logs
from cj_process.joinmarket_client import joinmarket_wallet_addresses

SENTINEL = "This method is not available in JoinMarket"

OWN_UNSPENT_ADDRESS = "bcrt1qf9f97s96e9rx67usff4z5ks6q8ka6ank9d504j"
OWN_SPENT_ADDRESS = "bcrt1q7v6zc2jfarlcwv938pm2rza9c7evatfupz6clk"
PEER_ADDRESSES = (
    "bcrt1q752v6amgy6wtclw4e3d4twd46mmvpsrpa7xv9n",
    "bcrt1qaj48yrls96pz5hggnnf52j4s5u5p43ruvshqkd",
    "bcrt1q6zn5fd2rtfr8r3lrmsugn7ccg9s0whudsngumy",
)
MAKER_CJ_ADDRESS = "bcrt1qtsvf744algukqcsufvgl6muenc2vz40gx987dv"
MAKER_CHANGE_ADDRESS = "bcrt1qkal4qnsxty4l0epx6x78gu5f2a28jpgqzye38z"

WALLET_DISPLAY_LOG = f"""2026-08-04 14:27:01,001 [INFO]  Wallet display
6d4ede82bbeb0331363ad541994b2f8397065471a06518ee20b7a6682c7e266a:1 - path: m/84'/1'/0'/0/3, address: {OWN_SPENT_ADDRESS} , value: 100000
2026-08-04 14:29:34,889 [INFO]  Makers responded with: {{'J5RAD1JndALPagnO': [['2794b43d:0'], '03f79780', '{PEER_ADDRESSES[0]}', '{PEER_ADDRESSES[1]}', 'MEUCIQ==', 'ff605355']}}
            "address": "{PEER_ADDRESSES[2]}"
"""


def write_joinmarket_wallet(run_dir, client="jcs-000", unspent=None, log_text=WALLET_DISPLAY_LOG):
    wallet_dir = run_dir / "data" / client
    wallet_dir.mkdir(parents=True)
    (wallet_dir / "coins.json").write_text(json.dumps(SENTINEL), encoding="utf-8")
    (wallet_dir / "keys.json").write_text(json.dumps(SENTINEL), encoding="utf-8")
    if unspent is not None:
        (wallet_dir / "unspent_coins.json").write_text(json.dumps(unspent), encoding="utf-8")
    if log_text is not None:
        log_dir = wallet_dir / "joinmarket"
        log_dir.mkdir(parents=True)
        (log_dir / "jmwalletd.log").write_text(log_text, encoding="utf-8")
    return wallet_dir


UNSPENT_COINS = {
    "utxos": [
        {"address": OWN_UNSPENT_ADDRESS, "path": "m/84'/1'/0'/0/2", "value": 200000},
        {"path": "m/84'/1'/0'/0/9", "value": 1},
    ]
}


def test_addresses_are_recovered_from_the_unspent_export_and_the_logs(tmp_path):
    wallet_dir = write_joinmarket_wallet(tmp_path, unspent=UNSPENT_COINS)

    addresses = joinmarket_wallet_addresses(str(wallet_dir))

    assert addresses == {
        OWN_UNSPENT_ADDRESS: {"address": OWN_UNSPENT_ADDRESS, "path": "m/84'/1'/0'/0/2"},
        OWN_SPENT_ADDRESS: {"address": OWN_SPENT_ADDRESS, "path": "m/84'/1'/0'/0/3"},
    }


def test_counterparty_addresses_are_never_claimed_as_own(tmp_path):
    wallet_dir = write_joinmarket_wallet(tmp_path, unspent=UNSPENT_COINS)

    addresses = joinmarket_wallet_addresses(str(wallet_dir))

    for peer in PEER_ADDRESSES:
        assert peer not in addresses


def test_wallet_without_recoverable_artifacts_stays_empty(tmp_path):
    wallet_dir = write_joinmarket_wallet(tmp_path, unspent=None, log_text=None)

    assert joinmarket_wallet_addresses(str(wallet_dir)) == {}


def test_current_exported_maker_log_recovers_its_coinjoin_and_change_outputs(tmp_path):
    wallet_dir = write_joinmarket_wallet(tmp_path, log_text=None)
    logs_dir = wallet_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / "J58aUMAqe8RpvJxA.log").write_text(
        "[INFO ]  mycjaddr, mychange = "
        f"{MAKER_CJ_ADDRESS}, {MAKER_CHANGE_ADDRESS}\\n",
        encoding="utf-8",
    )

    addresses = joinmarket_wallet_addresses(str(wallet_dir))

    assert addresses == {
        MAKER_CJ_ADDRESS: {"address": MAKER_CJ_ADDRESS},
        MAKER_CHANGE_ADDRESS: {"address": MAKER_CHANGE_ADDRESS},
    }


def test_obtain_wallets_info_attributes_joinmarket_wallets(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    write_joinmarket_wallet(tmp_path, unspent=UNSPENT_COINS)

    options = parse_cj_logs.EmulParseOptions()
    options.READ_ONLY_COINJOIN_TX_INFO = False
    with (
        mock.patch.object(parse_cj_logs, "op", options, create=True),
        mock.patch.object(parse_cj_logs.anonymity_score, "parse_wallet_coins", return_value=[]),
    ):
        wallets_info, _ = parse_cj_logs.obtain_wallets_info(
            str(tmp_path),
            load_wallet_info_via_rpc=False,
            load_wallet_from_docker_files=True,
            all_tx_db={},
        )

    assert sorted(wallets_info["wallet-000"].keys()) == sorted([OWN_UNSPENT_ADDRESS, OWN_SPENT_ADDRESS])
    assert "Recovered 2 addresses" in caplog.text
