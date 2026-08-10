import json
from unittest import mock

import pytest

from cj_process import parse_cj_logs
from utils import write_manifest


def test_process_experiment_allows_wasabi_logs_without_complete_rounds(tmp_path):
    run_dir = tmp_path
    log_path = run_dir / "data" / "wasabi-coordinator" / "coordinator" / "Logs.txt"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("round started but not completed\n", encoding="utf-8")
    write_manifest(run_dir, log_path)

    options = parse_cj_logs.EmulParseOptions()
    for name, value in {
        "LOAD_TXINFO_FROM_FILE": False,
        "LOAD_TXINFO_FROM_DOCKER_FILES": True,
        "READ_ONLY_COINJOIN_TX_INFO": False,
        "ASSUME_COORDINATOR_WALLET": False,
        "PARSE_ERRORS": True,
        "LOAD_COMPUTED_TRANSACTION_INFO": False,
        "SAVE_ANALYTICS_TO_FILE": False,
        "GENERATE_COINJOIN_GRAPH_BLIND": False,
        "GENERATE_COINJOIN_GRAPH": False,
    }.items():
        setattr(options, name, value)

    with (
        mock.patch.object(parse_cj_logs, "op", options, create=True),
        mock.patch.object(parse_cj_logs, "load_tx_database_from_btccore", return_value={}),
        mock.patch.object(parse_cj_logs, "obtain_wallets_info", return_value=({}, {})),
        mock.patch.object(
            parse_cj_logs,
            "parse_backend_coinjoin_logs",
            return_value={},
        ) as parse_backend_coinjoin_logs,
        mock.patch.object(parse_cj_logs, "load_prison_data"),
        mock.patch.object(parse_cj_logs, "load_anonscore_data"),
        mock.patch.object(
            parse_cj_logs,
            "parse_coinjoin_errors",
            wraps=parse_cj_logs.parse_coinjoin_errors,
        ) as parse_coinjoin_errors,
        mock.patch.object(parse_cj_logs.als, "remove_link_between_inputs_and_outputs"),
        mock.patch.object(parse_cj_logs.als, "compute_link_between_inputs_and_outputs"),
        mock.patch.object(parse_cj_logs.als, "analyze_input_out_liquidity"),
    ):
        result = parse_cj_logs.process_experiment((str(run_dir), False))

    assert result["coinjoins"] == {}
    assert result["rounds"] == {"no_round": []}
    parse_backend_coinjoin_logs.assert_called_once_with(str(log_path), {}, allow_rpc=False)
    parse_coinjoin_errors.assert_called_once_with(result, str(log_path))

    coinjoin_info = json.loads((run_dir / "coinjoin_tx_info.json").read_text(encoding="utf-8"))
    coinjoin_stats = json.loads((run_dir / "coinjoin_tx_info_stats.json").read_text(encoding="utf-8"))
    assert coinjoin_info["coinjoins"] == {}
    assert coinjoin_info["rounds"] == {"no_round": []}
    assert coinjoin_stats["num_coinjoins"] == 0


def test_process_experiment_local_uses_rpc_with_empty_transaction_database(tmp_path):
    options = parse_cj_logs.EmulParseOptions()
    for name, value in {
        "LOAD_TXINFO_FROM_FILE": False,
        "LOAD_TXINFO_FROM_DOCKER_FILES": False,
        "READ_ONLY_COINJOIN_TX_INFO": False,
        "ASSUME_COORDINATOR_WALLET": False,
        "PARSE_ERRORS": True,
        "LOAD_COMPUTED_TRANSACTION_INFO": False,
        "SAVE_ANALYTICS_TO_FILE": False,
        "GENERATE_COINJOIN_GRAPH_BLIND": False,
        "GENERATE_COINJOIN_GRAPH": False,
    }.items():
        setattr(options, name, value)

    with (
        mock.patch.object(parse_cj_logs, "op", options, create=True),
        mock.patch.object(parse_cj_logs, "obtain_wallets_info", return_value=({}, {})),
        mock.patch.object(
            parse_cj_logs,
            "parse_wasabi_coordinator_coinjoins",
            return_value=({}, []),
        ) as parse_wasabi,
        mock.patch.object(parse_cj_logs, "load_prison_data"),
        mock.patch.object(parse_cj_logs, "load_anonscore_data"),
        mock.patch.object(parse_cj_logs.als, "remove_link_between_inputs_and_outputs"),
        mock.patch.object(parse_cj_logs.als, "compute_link_between_inputs_and_outputs"),
        mock.patch.object(parse_cj_logs.als, "analyze_input_out_liquidity"),
    ):
        result = parse_cj_logs.process_experiment((str(tmp_path), False))

    assert result["coinjoins"] == {}
    parse_wasabi.assert_called_once_with(str(tmp_path), {}, allow_rpc=True)


def test_empty_round_keeps_round_independent_error_events(tmp_path):
    log_path = tmp_path / "Logs.txt"
    log_path.write_text(
        "2026-01-01 00:00:00.000 [1] WARNING IdempotencyRequestCache.GetCachedResponseAsync "
        "WabiSabiProtocolException: Input banned\n",
        encoding="utf-8",
    )

    result = parse_cj_logs.parse_coinjoin_errors(
        {"coinjoins": {}, "rounds": {"no_round": []}},
        str(log_path),
    )

    event_groups = result["rounds"]["no_round"]
    events = [event for group in event_groups for matches in group.values() for event in matches]
    assert [event["type"] for event in events] == [parse_cj_logs.CJ_LOG_TYPES.INPUT_BANNED.name]


def test_manifest_hash_mismatch_is_rejected(tmp_path):
    log_path = tmp_path / "data" / "wasabi-coordinator" / "coordinator" / "Logs.txt"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("original coordinator log\n", encoding="utf-8")
    write_manifest(tmp_path, log_path)
    log_path.write_text("modified coordinator log\n", encoding="utf-8")

    with pytest.raises(ValueError, match="(size|hash) does not match manifest"):
        parse_cj_logs.find_wasabi_coordinator_log_files(str(tmp_path))


def test_incomplete_manifest_is_rejected_without_legacy_fallback(tmp_path):
    log_path = tmp_path / "data" / "wasabi-coordinator" / "coordinator" / "Logs.txt"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("coordinator log\n", encoding="utf-8")
    write_manifest(tmp_path, log_path, complete=False)

    with pytest.raises(ValueError, match="manifest is incomplete"):
        parse_cj_logs.find_wasabi_coordinator_log_files(str(tmp_path))


def write_legacy_prison(tmp_path, prison_rows):
    log_path = tmp_path / "data" / "wasabi-backend" / "backend" / "Logs.txt"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("coordinator log\n", encoding="utf-8")
    prison_path = log_path.parent / "WabiSabi" / "Prison.txt"
    prison_path.parent.mkdir()
    prison_path.write_text("".join(f"{row}\n" for row in prison_rows), encoding="utf-8")
    return prison_path


def test_load_prison_data_creates_records_for_rounds_without_coinjoin(tmp_path):
    round_id = "a" * 64
    utxo = f"{'b' * 64}-0"
    write_legacy_prison(tmp_path, [f"1700000000,{utxo},RoundDisruption,100000,{round_id},DoubleSpent"])

    # Prison entries exist precisely when rounds failed, so 'rounds' holds only the
    # placeholder written for experiments without a single completed CoinJoin.
    cjtx_stats = parse_cj_logs.load_prison_data({"rounds": {"no_round": []}}, str(tmp_path))

    assert cjtx_stats["rounds"]["no_round"] == []
    prison_logs = cjtx_stats["rounds"][round_id]["logs"]
    assert len(prison_logs) == 1
    assert prison_logs[0]["type"] == parse_cj_logs.CJ_LOG_TYPES.UTXO_IN_PRISON.name
    assert prison_logs[0]["utxo"] == utxo
    assert prison_logs[0]["adv_reason"] == "DoubleSpent"


def test_load_prison_data_keeps_logs_of_already_parsed_round(tmp_path):
    round_id = "a" * 64
    write_legacy_prison(tmp_path, [f"1700000000,{'b' * 64}-0,Cheating,{round_id}"])

    cjtx_stats = parse_cj_logs.load_prison_data(
        {"rounds": {round_id: {"logs": [{"type": "ROUND_STARTED"}]}}},
        str(tmp_path),
    )

    assert [entry["type"] for entry in cjtx_stats["rounds"][round_id]["logs"]] == [
        "ROUND_STARTED",
        parse_cj_logs.CJ_LOG_TYPES.UTXO_IN_PRISON.name,
    ]


def test_parse_backend_coinjoin_logs_returns_empty_for_incomplete_round(tmp_path):
    log_path = tmp_path / "Logs.txt"
    round_id = "a" * 64
    log_path.write_text(
        "2026-01-01 00:00:00.000 [1] INFO Arena.CreateRound (1) "
        f"Round ({round_id}): Created round with parameters: "
        "MaxSuggestedAmount:'1.0' BTC.\n",
        encoding="utf-8",
    )

    assert parse_cj_logs.parse_backend_coinjoin_logs(str(log_path), {}) == {}


def test_parse_backend_coinjoin_logs_parses_complete_round(tmp_path):
    log_path = tmp_path / "Logs.txt"
    round_id = "a" * 64
    txid = "b" * 64
    log_path.write_text(
        "2026-01-01 00:00:00.000 [1] INFO Arena.CreateRound (1) "
        f"Round ({round_id}): Created round with parameters: "
        "MaxSuggestedAmount:'1.0' BTC.\n"
        "2026-01-01 00:01:00.000 [1] INFO Arena.StepTransactionSigningPhaseAsync (1) "
        f"Round ({round_id}): Successfully broadcast the coinjoin: {txid}.\n",
        encoding="utf-8",
    )
    transaction = {"inputs": {}, "outputs": {}}

    with mock.patch.object(parse_cj_logs, "extract_tx_info", return_value=transaction) as extract_tx_info:
        result = parse_cj_logs.parse_backend_coinjoin_logs(str(log_path), {})

    extract_tx_info.assert_called_once_with(txid, {}, allow_rpc=True)
    assert result == {
        txid: {
            "inputs": {},
            "outputs": {},
            "round_id": round_id,
            "round_start_time": "2026-01-01 00:00:00.000",
            "broadcast_time": "2026-01-01 00:01:00.000",
            "is_blame_round": False,
            "is_cjtx": True,
        }
    }
