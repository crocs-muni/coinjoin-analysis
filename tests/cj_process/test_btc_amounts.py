from cj_process.cj_analysis import btc_to_sats, extract_tx_info


def test_extract_tx_info_preserves_exact_satoshi_values() -> None:
    """JSON-RPC BTC floats must not lose a satoshi during normalization."""
    raw_txs = {
        "funding": {
            "vout": [
                {
                    "n": 0,
                    "value": 0.00012738,
                    "scriptPubKey": {"address": "funding-address"},
                }
            ]
        },
        "spending": {
            "blocktime": 0,
            "vin": [{"txid": "funding", "vout": 0}],
            "vout": [
                {
                    "n": 0,
                    "value": 0.00012738,
                    "scriptPubKey": {"address": "recipient-address"},
                }
            ],
        },
    }

    transaction = extract_tx_info("spending", raw_txs)

    assert transaction["inputs"]["0"]["value"] == 12_738
    assert transaction["outputs"]["0"]["value"] == 12_738


def test_btc_to_sats_preserves_exact_satoshi_value() -> None:
    assert btc_to_sats(0.00012738) == 12_738


def test_btc_to_sats_accepts_json_decimal_text() -> None:
    assert btc_to_sats("0.00012738") == 12_738


def test_btc_to_sats_accepts_integer_btc_amount() -> None:
    assert btc_to_sats(1) == 100_000_000
