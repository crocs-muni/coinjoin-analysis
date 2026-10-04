from unittest import mock

from cj_process import parse_cj_logs


def test_extract_tx_info_does_not_query_rpc_for_missing_exported_input():
    raw_tx_db = {
        "coinjoin": {
            "txid": "coinjoin",
            "vin": [{"txid": "missing-funding", "vout": 0}],
            "vout": [],
        }
    }

    with mock.patch.object(parse_cj_logs.als, "run_command") as run_command:
        assert parse_cj_logs.extract_tx_info("coinjoin", raw_tx_db, allow_rpc=False) is None

    run_command.assert_not_called()


def test_empty_export_never_queries_rpc():
    """An empty export must stay offline instead of falling back to a live node."""
    with mock.patch.object(parse_cj_logs.als, "run_command") as run_command:
        assert parse_cj_logs.extract_tx_info("missing", {}, allow_rpc=False) is None

    run_command.assert_not_called()
