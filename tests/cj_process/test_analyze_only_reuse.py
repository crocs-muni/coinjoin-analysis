"""analyze_only reruns the analysis of an already collected experiment.

It must therefore be a pure function of coinjoin_tx_info.json: repeating it may not
change the numbers, and it may not depend on raw client artifacts that the action
does not require to be present.
"""

import json
from unittest import mock

import pytest

from cj_process import parse_cj_logs
from cj_process.cj_analysis import MIX_PROTOCOL
from utils import write_manifest

PRISON_ROWS = (
    f"1700000000,{'b' * 64}-0,RoundDisruption,100000,{'a' * 64},DoubleSpent",
    f"1700000600,{'c' * 64}-1,Cheating,{'d' * 64}",
)


def make_options(**overrides):
    options = parse_cj_logs.EmulParseOptions()
    values = {
        "LOAD_TXINFO_FROM_FILE": False,
        "LOAD_TXINFO_FROM_DOCKER_FILES": True,
        "READ_ONLY_COINJOIN_TX_INFO": False,
        "ASSUME_COORDINATOR_WALLET": False,
        "PARSE_ERRORS": True,
        "LOAD_COMPUTED_TRANSACTION_INFO": False,
        "SAVE_ANALYTICS_TO_FILE": False,
        "GENERATE_COINJOIN_GRAPH_BLIND": False,
        "GENERATE_COINJOIN_GRAPH": False,
    }
    values.update(overrides)
    for name, value in values.items():
        setattr(options, name, value)
    return options


def run_process_experiment(run_dir, options):
    """Run one experiment with everything mocked out except the parsing under test."""
    with (
        mock.patch.object(parse_cj_logs, "op", options, create=True),
        mock.patch.object(parse_cj_logs, "load_tx_database_from_btccore", return_value={}),
        mock.patch.object(parse_cj_logs, "obtain_wallets_info", return_value=({}, {})),
        mock.patch.object(parse_cj_logs.als, "remove_link_between_inputs_and_outputs"),
        mock.patch.object(parse_cj_logs.als, "compute_link_between_inputs_and_outputs"),
        mock.patch.object(parse_cj_logs.als, "analyze_input_out_liquidity") as analyze_liquidity,
    ):
        result = parse_cj_logs.process_experiment((str(run_dir), False))
    return result, analyze_liquidity


def count_prison_records(cjtx_stats):
    return sum(
        1
        for round_record in cjtx_stats["rounds"].values()
        if isinstance(round_record, dict)
        for log_record in round_record.get("logs", [])
        if log_record["type"] == parse_cj_logs.CJ_LOG_TYPES.UTXO_IN_PRISON.name
    )


def write_wasabi_run_with_prison(run_dir):
    log_path = run_dir / "data" / "wasabi-coordinator" / "coordinator" / "Logs.txt"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("round started but not completed\n", encoding="utf-8")
    write_manifest(run_dir, log_path)
    prison_path = log_path.parent / "WabiSabi" / "Prison.txt"
    prison_path.parent.mkdir()
    prison_path.write_text("".join(f"{row}\n" for row in PRISON_ROWS), encoding="utf-8")
    return prison_path


def write_collected_experiment(run_dir, **overrides):
    cjtx_stats = {
        "coinjoins": {},
        "wallets_info": {},
        "wallets_coins": {},
        "rounds": {"no_round": []},
        "address_wallet_mapping": {},
    }
    cjtx_stats.update(overrides)
    (run_dir / "coinjoin_tx_info.json").write_text(json.dumps(cjtx_stats), encoding="utf-8")
    return cjtx_stats


def test_analyze_only_does_not_duplicate_prison_records(tmp_path):
    write_wasabi_run_with_prison(tmp_path)

    collected, _ = run_process_experiment(tmp_path, make_options())
    assert count_prison_records(collected) == len(PRISON_ROWS)

    # Prison events are part of the collected file, so replaying the analysis on it
    # must keep reporting the same number of imprisoned UTXOs.
    for _ in range(2):
        saved = json.loads((tmp_path / "coinjoin_tx_info.json").read_text(encoding="utf-8"))
        assert count_prison_records(saved) == len(PRISON_ROWS)

        analyzed, _ = run_process_experiment(
            tmp_path,
            make_options(
                LOAD_TXINFO_FROM_FILE=True,
                LOAD_TXINFO_FROM_DOCKER_FILES=False,
                PARSE_ERRORS=False,
                LOAD_COMPUTED_TRANSACTION_INFO=True,
            ),
        )
        assert count_prison_records(analyzed) == len(PRISON_ROWS)


def test_collected_experiment_records_its_mix_protocol(tmp_path):
    write_wasabi_run_with_prison(tmp_path)

    collected, _ = run_process_experiment(tmp_path, make_options())

    assert collected["mix_protocol"] == MIX_PROTOCOL.WASABI2.value
    saved = json.loads((tmp_path / "coinjoin_tx_info.json").read_text(encoding="utf-8"))
    assert saved["mix_protocol"] == MIX_PROTOCOL.WASABI2.value


def test_detect_mix_protocol_prefers_wasabi_manifest_over_stale_joinmarket_artifacts(tmp_path):
    wasabi_log = tmp_path / "data" / "wasabi-coordinator" / "Logs.txt"
    wasabi_log.parent.mkdir(parents=True)
    wasabi_log.write_text("round started but not completed\n", encoding="utf-8")
    write_manifest(tmp_path, wasabi_log)

    stale_log = tmp_path / "data" / "jcs-000" / "joinmarket" / "jmwalletd.log"
    stale_log.parent.mkdir(parents=True)
    stale_log.write_text("stale JoinMarket log\n", encoding="utf-8")
    (tmp_path / "data" / "joinmarket_round_events.json").write_text("[]", encoding="utf-8")

    assert parse_cj_logs.detect_mix_protocol(str(tmp_path)) == MIX_PROTOCOL.WASABI2


def test_wasabi_manifest_keeps_collection_on_wasabi_with_stale_joinmarket_artifacts(tmp_path):
    write_wasabi_run_with_prison(tmp_path)
    stale_log = tmp_path / "data" / "jcs-000" / "joinmarket" / "jmwalletd.log"
    stale_log.parent.mkdir(parents=True)
    stale_log.write_text("stale JoinMarket log\n", encoding="utf-8")
    (tmp_path / "data" / "joinmarket_round_events.json").write_text("[]", encoding="utf-8")

    with (
        mock.patch.object(
            parse_cj_logs,
            "joinmarket_parse_coinjoin_logs",
            side_effect=AssertionError("stale JoinMarket client log was parsed"),
        ) as parse_joinmarket_logs,
        mock.patch.object(
            parse_cj_logs,
            "joinmarket_parse_round_events",
            side_effect=AssertionError("stale JoinMarket round events were parsed"),
        ) as parse_joinmarket_events,
    ):
        collected, analyze_liquidity = run_process_experiment(tmp_path, make_options())

    assert collected["mix_protocol"] == MIX_PROTOCOL.WASABI2.value
    assert count_prison_records(collected) == len(PRISON_ROWS)
    assert analyze_liquidity.call_args.args[-1] == MIX_PROTOCOL.WASABI2
    parse_joinmarket_logs.assert_not_called()
    parse_joinmarket_events.assert_not_called()


def test_analyze_only_keeps_joinmarket_protocol_without_raw_logs(tmp_path):
    # analyze_only requires coinjoin_tx_info.json only, so the JoinMarket client logs
    # the protocol was originally detected from are gone here.
    write_collected_experiment(tmp_path, mix_protocol=MIX_PROTOCOL.JOINMARKET.value)

    with mock.patch.object(parse_cj_logs, "assert_no_coordinator_for_joinmarket") as assert_no_coordinator:
        _, analyze_liquidity = run_process_experiment(
            tmp_path,
            make_options(
                LOAD_TXINFO_FROM_FILE=True,
                LOAD_TXINFO_FROM_DOCKER_FILES=False,
                PARSE_ERRORS=False,
                LOAD_COMPUTED_TRANSACTION_INFO=True,
            ),
        )

    assert analyze_liquidity.call_args.args[-1] == MIX_PROTOCOL.JOINMARKET
    assert_no_coordinator.assert_called_once()


def test_analyze_only_detects_protocol_of_experiment_collected_before_it_was_recorded(tmp_path):
    write_collected_experiment(tmp_path)
    events_path = tmp_path / "data" / "joinmarket_round_events.json"
    events_path.parent.mkdir()
    events_path.write_text("[]", encoding="utf-8")

    with mock.patch.object(parse_cj_logs, "assert_no_coordinator_for_joinmarket"):
        _, analyze_liquidity = run_process_experiment(
            tmp_path,
            make_options(
                LOAD_TXINFO_FROM_FILE=True,
                LOAD_TXINFO_FROM_DOCKER_FILES=False,
                PARSE_ERRORS=False,
                LOAD_COMPUTED_TRANSACTION_INFO=True,
            ),
        )

    assert analyze_liquidity.call_args.args[-1] == MIX_PROTOCOL.JOINMARKET
    saved = json.loads((tmp_path / "coinjoin_tx_info.json").read_text(encoding="utf-8"))
    assert saved["mix_protocol"] == MIX_PROTOCOL.JOINMARKET.value


def test_analyze_only_rejects_unknown_recorded_protocol(tmp_path):
    write_collected_experiment(tmp_path, mix_protocol="WASABI3")

    with pytest.raises(ValueError, match="unknown mix_protocol"):
        run_process_experiment(
            tmp_path,
            make_options(
                LOAD_TXINFO_FROM_FILE=True,
                LOAD_TXINFO_FROM_DOCKER_FILES=False,
                PARSE_ERRORS=False,
                LOAD_COMPUTED_TRANSACTION_INFO=True,
            ),
        )
