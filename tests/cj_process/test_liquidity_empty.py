import json

from cj_process.cj_analysis import analyze_input_out_liquidity
from cj_process.cj_structs import MIX_PROTOCOL


def test_analyze_input_out_liquidity_allows_empty_coinjoins(tmp_path):
    result = analyze_input_out_liquidity(
        str(tmp_path),
        {},
        {},
        {},
        MIX_PROTOCOL.JOINMARKET,
    )

    assert result == {}
    with (tmp_path / "tx_reordering_stats.json").open() as file:
        assert json.load(file) == {}
