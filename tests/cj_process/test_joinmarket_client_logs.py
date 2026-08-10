"""A damaged jmwalletd.log must fail the parse, not look like a run without coinjoins."""

import pytest

from cj_process import cj_analysis as als

TXID = "d" * 64
TX_JSON_BLOCK = (
    "2025-02-10 05:33:48,936 [INFO]  obtained tx\n"
    "{\n"
    '    "hex": "0200000000",\n'
    f'    "txid": "{TXID}",\n'
    '    "nVersion": 2\n'
    "}\n"
)
TIMESTAMP_LINE = "2025-02-10 05:33:48,937 [INFO]  potentially earned = 0.00000006 BTC (6 sat)\n"


def write_log(tmp_path, content):
    log_path = tmp_path / "jmwalletd.log"
    log_path.write_text(content, encoding="utf-8")
    return str(log_path)


def test_joinmarket_find_coinjoins_parses_complete_log(tmp_path):
    log_file = write_log(tmp_path, TX_JSON_BLOCK + TIMESTAMP_LINE)

    hits = als.joinmarket_find_coinjoins(log_file)

    assert list(hits.keys()) == [TXID]
    assert hits[TXID]["timestamp"] == "2025-02-10 05:33:48.937"


def test_joinmarket_find_coinjoins_rejects_truncated_transaction_json(tmp_path):
    log_file = write_log(tmp_path, TX_JSON_BLOCK.rsplit("}\n", 1)[0])

    with pytest.raises(ValueError, match="unterminated transaction json"):
        als.joinmarket_find_coinjoins(log_file)


def test_joinmarket_find_coinjoins_rejects_log_ending_after_transaction_json(tmp_path):
    log_file = write_log(tmp_path, TX_JSON_BLOCK)

    with pytest.raises(ValueError, match="no log line follows the transaction json"):
        als.joinmarket_find_coinjoins(log_file)


def test_joinmarket_find_coinjoins_rejects_unparsable_transaction_json(tmp_path):
    log_file = write_log(
        tmp_path,
        TX_JSON_BLOCK.replace('"nVersion": 2', '"nVersion": ') + TIMESTAMP_LINE,
    )

    with pytest.raises(ValueError, match="Failed to parse JoinMarket client log"):
        als.joinmarket_find_coinjoins(log_file)


def test_joinmarket_find_coinjoins_does_not_hide_a_damaged_log_behind_earlier_hits(tmp_path):
    log_file = write_log(tmp_path, TX_JSON_BLOCK + TIMESTAMP_LINE + TX_JSON_BLOCK.rsplit("}\n", 1)[0])

    with pytest.raises(ValueError, match="unterminated transaction json"):
        als.joinmarket_find_coinjoins(log_file)


def test_joinmarket_find_coinjoins_reports_missing_log(tmp_path):
    with pytest.raises(FileNotFoundError):
        als.joinmarket_find_coinjoins(str(tmp_path / "absent.log"))
