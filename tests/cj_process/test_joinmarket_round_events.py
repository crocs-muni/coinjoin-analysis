import hashlib
import json
import logging
from unittest import mock

import pytest

from cj_process import parse_cj_logs
from cj_process.parse_cj_logs import (
    find_joinmarket_client_log_files,
    find_joinmarket_round_events_file,
    joinmarket_parse_round_events,
)


def write_joinmarket_manifest(
    run_dir,
    events_file,
    *,
    positive_count,
    complete=True,
    engine='joinmarket',
    source_path='joinmarket_round_events.json',
):
    manifest = {
        'schema_version': '1.0',
        'engine': engine,
        'complete': complete,
        'reason': None if complete else 'round-event capture failed',
        'positive_count': positive_count,
        'sources': [{
            'path': source_path,
            'size_bytes': events_file.stat().st_size,
            'sha256': hashlib.sha256(events_file.read_bytes()).hexdigest(),
        }],
    }
    (run_dir / 'data' / 'coinjoin_label_manifest.json').write_text(
        json.dumps(manifest),
        encoding='utf-8',
    )


def collect_options():
    options = parse_cj_logs.EmulParseOptions()
    for name, value in {
        'LOAD_TXINFO_FROM_FILE': False,
        'LOAD_TXINFO_FROM_DOCKER_FILES': True,
        'READ_ONLY_COINJOIN_TX_INFO': False,
        'ASSUME_COORDINATOR_WALLET': False,
        'PARSE_ERRORS': True,
        'LOAD_COMPUTED_TRANSACTION_INFO': False,
        'SAVE_ANALYTICS_TO_FILE': False,
        'GENERATE_COINJOIN_GRAPH_BLIND': False,
        'GENERATE_COINJOIN_GRAPH': False,
    }.items():
        setattr(options, name, value)
    return options


def test_joinmarket_run_skips_wasabi_error_parsing(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "joinmarket_round_events.json").write_text("[]", encoding="utf-8")

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
        mock.patch.object(parse_cj_logs, "joinmarket_parse_round_events", return_value=({}, {})),
        mock.patch.object(parse_cj_logs, "parse_wasabi_coordinator_coinjoins") as parse_wasabi,
        mock.patch.object(parse_cj_logs, "parse_coinjoin_errors") as parse_wasabi_errors,
        mock.patch.object(parse_cj_logs, "load_prison_data") as load_prison_data,
        mock.patch.object(parse_cj_logs, "load_anonscore_data"),
        mock.patch.object(parse_cj_logs.als, "remove_link_between_inputs_and_outputs"),
        mock.patch.object(parse_cj_logs.als, "compute_link_between_inputs_and_outputs"),
        mock.patch.object(parse_cj_logs.als, "analyze_input_out_liquidity") as analyze_liquidity,
    ):
        result = parse_cj_logs.process_experiment((str(tmp_path), False))

    assert result["coinjoins"] == {}
    assert result["rounds"] == {}
    parse_wasabi.assert_not_called()
    parse_wasabi_errors.assert_not_called()
    load_prison_data.assert_not_called()
    assert analyze_liquidity.call_args.args[4] is parse_cj_logs.MIX_PROTOCOL.JOINMARKET
    assert "Wasabi coordinator" not in caplog.text


def test_joinmarket_manifest_round_events_take_precedence_over_client_logs(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    events_file = data_dir / 'joinmarket_round_events.json'
    events_file.write_text('[]', encoding='utf-8')
    write_joinmarket_manifest(tmp_path, events_file, positive_count=0)
    log_file = data_dir / 'jcs-000' / 'joinmarket' / 'jmwalletd.log'
    log_file.parent.mkdir(parents=True)
    log_file.write_text('legacy client log', encoding='utf-8')

    with (
        mock.patch.object(parse_cj_logs, 'op', collect_options(), create=True),
        mock.patch.object(parse_cj_logs, 'load_tx_database_from_btccore', return_value={}),
        mock.patch.object(parse_cj_logs, 'obtain_wallets_info', return_value=({}, {})),
        mock.patch.object(
            parse_cj_logs, 'joinmarket_parse_round_events', return_value=({}, {})
        ) as parse_events,
        mock.patch.object(parse_cj_logs, 'joinmarket_parse_coinjoin_logs') as parse_logs,
        mock.patch.object(parse_cj_logs, 'load_anonscore_data'),
        mock.patch.object(parse_cj_logs.als, 'remove_link_between_inputs_and_outputs'),
        mock.patch.object(parse_cj_logs.als, 'compute_link_between_inputs_and_outputs'),
        mock.patch.object(parse_cj_logs.als, 'analyze_input_out_liquidity'),
    ):
        parse_cj_logs.process_experiment((str(tmp_path), False))

    parse_events.assert_called_once_with(str(tmp_path), {})
    parse_logs.assert_not_called()


def test_legacy_joinmarket_client_logs_keep_precedence_without_manifest(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    (data_dir / 'joinmarket_round_events.json').write_text('[]', encoding='utf-8')
    log_file = data_dir / 'jcs-000' / 'joinmarket' / 'jmwalletd.log'
    log_file.parent.mkdir(parents=True)
    log_file.write_text('legacy client log', encoding='utf-8')

    with (
        mock.patch.object(parse_cj_logs, 'op', collect_options(), create=True),
        mock.patch.object(parse_cj_logs, 'load_tx_database_from_btccore', return_value={}),
        mock.patch.object(parse_cj_logs, 'obtain_wallets_info', return_value=({}, {})),
        mock.patch.object(parse_cj_logs, 'joinmarket_parse_coinjoin_logs', return_value={}) as parse_logs,
        mock.patch.object(
            parse_cj_logs, 'joinmarket_parse_round_events', return_value=({}, {})
        ) as parse_events,
        mock.patch.object(parse_cj_logs, 'load_anonscore_data'),
        mock.patch.object(parse_cj_logs.als, 'remove_link_between_inputs_and_outputs'),
        mock.patch.object(parse_cj_logs.als, 'compute_link_between_inputs_and_outputs'),
        mock.patch.object(parse_cj_logs.als, 'analyze_input_out_liquidity'),
    ):
        parse_cj_logs.process_experiment((str(tmp_path), False))

    parse_logs.assert_called_once_with(str(tmp_path), {}, allow_rpc=False)
    parse_events.assert_called_once_with(str(tmp_path), {})


def test_incomplete_joinmarket_manifest_falls_back_to_client_logs(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    events_file = data_dir / 'joinmarket_round_events.json'
    events_file.write_text('[]', encoding='utf-8')
    write_joinmarket_manifest(tmp_path, events_file, positive_count=0, complete=False)
    log_file = data_dir / 'jcs-000' / 'joinmarket' / 'jmwalletd.log'
    log_file.parent.mkdir(parents=True)
    log_file.write_text('tumbler client log', encoding='utf-8')
    parsed_coinjoins = {'txA': {'inputs': {}, 'outputs': {}}}

    with (
        mock.patch.object(parse_cj_logs, 'op', collect_options(), create=True),
        mock.patch.object(parse_cj_logs, 'load_tx_database_from_btccore', return_value={}),
        mock.patch.object(parse_cj_logs, 'obtain_wallets_info', return_value=({}, {})),
        mock.patch.object(
            parse_cj_logs, 'joinmarket_parse_coinjoin_logs', return_value=parsed_coinjoins
        ) as parse_logs,
        mock.patch.object(parse_cj_logs, 'joinmarket_parse_round_events') as parse_events,
        mock.patch.object(parse_cj_logs, 'analyze_coinjoin_stats', return_value={}),
        mock.patch.object(parse_cj_logs, 'load_anonscore_data'),
        mock.patch.object(parse_cj_logs.als, 'remove_link_between_inputs_and_outputs'),
        mock.patch.object(parse_cj_logs.als, 'compute_link_between_inputs_and_outputs'),
        mock.patch.object(parse_cj_logs.als, 'analyze_input_out_liquidity'),
    ):
        result = parse_cj_logs.process_experiment((str(tmp_path), False))

    assert result['coinjoins'] == parsed_coinjoins
    parse_logs.assert_called_once_with(str(tmp_path), {}, allow_rpc=False)
    parse_events.assert_not_called()
    assert 'round events are not authoritative' in caplog.text
    assert 'falling back to legacy client logs' in caplog.text


def test_incomplete_joinmarket_manifest_never_falls_back_to_round_events(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    events_file = data_dir / 'joinmarket_round_events.json'
    events_file.write_text('[]', encoding='utf-8')
    write_joinmarket_manifest(tmp_path, events_file, positive_count=0, complete=False)
    log_file = data_dir / 'jcs-000' / 'joinmarket' / 'jmwalletd.log'
    log_file.parent.mkdir(parents=True)
    log_file.write_text('empty but usable client log', encoding='utf-8')

    with (
        mock.patch.object(parse_cj_logs, 'op', collect_options(), create=True),
        mock.patch.object(parse_cj_logs, 'load_tx_database_from_btccore', return_value={}),
        mock.patch.object(parse_cj_logs, 'obtain_wallets_info', return_value=({}, {})),
        mock.patch.object(
            parse_cj_logs, 'joinmarket_parse_coinjoin_logs', return_value={}
        ) as parse_logs,
        mock.patch.object(parse_cj_logs, 'joinmarket_parse_round_events') as parse_events,
        mock.patch.object(parse_cj_logs, 'load_anonscore_data'),
        mock.patch.object(parse_cj_logs.als, 'remove_link_between_inputs_and_outputs'),
        mock.patch.object(parse_cj_logs.als, 'compute_link_between_inputs_and_outputs'),
        mock.patch.object(parse_cj_logs.als, 'analyze_input_out_liquidity'),
    ):
        result = parse_cj_logs.process_experiment((str(tmp_path), False))

    assert result['coinjoins'] == {}
    assert result['rounds'] == {'no_round': []}
    parse_logs.assert_called_once_with(str(tmp_path), {}, allow_rpc=False)
    parse_events.assert_not_called()


def test_incomplete_joinmarket_manifest_without_client_logs_fails_explicitly(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    events_file = data_dir / 'joinmarket_round_events.json'
    events_file.write_text('[]', encoding='utf-8')
    write_joinmarket_manifest(tmp_path, events_file, positive_count=0, complete=False)

    with (
        mock.patch.object(parse_cj_logs, 'op', collect_options(), create=True),
        mock.patch.object(parse_cj_logs, 'load_tx_database_from_btccore', return_value={}),
        mock.patch.object(parse_cj_logs, 'obtain_wallets_info', return_value=({}, {})),
        mock.patch.object(
            parse_cj_logs, 'joinmarket_parse_round_events'
        ) as parse_events,
        mock.patch.object(
            parse_cj_logs, 'parse_wasabi_coordinator_coinjoins'
        ) as parse_wasabi,
        pytest.raises(
            ValueError,
            match='manifest is incomplete and no usable JoinMarket client logs are available',
        ),
    ):
        parse_cj_logs.process_experiment((str(tmp_path), False))

    parse_wasabi.assert_not_called()
    parse_events.assert_not_called()


def test_unknown_engine_manifest_does_not_override_joinmarket_client_logs(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    events_file = data_dir / 'joinmarket_round_events.json'
    events_file.write_text('[]', encoding='utf-8')
    write_joinmarket_manifest(tmp_path, events_file, positive_count=0, engine='unknown')
    log_file = data_dir / 'jcs-000' / 'joinmarket' / 'jmwalletd.log'
    log_file.parent.mkdir(parents=True)
    log_file.write_text('joinmarket client log', encoding='utf-8')
    parsed_coinjoins = {'txA': {'inputs': {}, 'outputs': {}}}

    with (
        mock.patch.object(parse_cj_logs, 'op', collect_options(), create=True),
        mock.patch.object(parse_cj_logs, 'load_tx_database_from_btccore', return_value={}),
        mock.patch.object(parse_cj_logs, 'obtain_wallets_info', return_value=({}, {})),
        mock.patch.object(
            parse_cj_logs, 'joinmarket_parse_coinjoin_logs', return_value=parsed_coinjoins
        ) as parse_logs,
        mock.patch.object(parse_cj_logs, 'joinmarket_parse_round_events') as parse_events,
        mock.patch.object(parse_cj_logs, 'analyze_coinjoin_stats', return_value={}),
        mock.patch.object(parse_cj_logs, 'load_anonscore_data'),
        mock.patch.object(parse_cj_logs.als, 'remove_link_between_inputs_and_outputs'),
        mock.patch.object(parse_cj_logs.als, 'compute_link_between_inputs_and_outputs'),
        mock.patch.object(parse_cj_logs.als, 'analyze_input_out_liquidity'),
    ):
        result = parse_cj_logs.process_experiment((str(tmp_path), False))

    assert result['coinjoins'] == parsed_coinjoins
    parse_logs.assert_called_once_with(str(tmp_path), {}, allow_rpc=False)
    parse_events.assert_not_called()


def test_joinmarket_round_events_parse_reconciled_destination_match(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    events_file = data_dir / "joinmarket_round_events.json"
    events_file.write_text(
        json.dumps(
            [
                {
                    "round_id": 1,
                    "export_round_id": 1,
                    "engine": "joinmarket",
                    "status": "confirmed",
                    "taker": "jcs-000",
                    "candidate_makers": ["jcs-001", "jcs-002"],
                    "destination_address": "output-a0",
                    "destination_matches": [{"txid": "txA", "block_height": 7}],
                }
            ]
        ),
        encoding="utf-8",
    )
    raw_tx_db = {
        "funding": {
            "txid": "funding",
            "vout": [
                {
                    "n": 0,
                    "value": 0.0015,
                    "scriptPubKey": {"address": "input-a"},
                }
            ],
        },
        "txA": {
            "txid": "txA",
            "mine_time": "2026-06-13 09:10:00.000",
            "vin": [{"txid": "funding", "vout": 0}],
            "vout": [
                {
                    "n": 0,
                    "value": 0.001,
                    "scriptPubKey": {"address": "output-a0"},
                },
                {
                    "n": 1,
                    "value": 0.0004,
                    "scriptPubKey": {"address": "output-a1"},
                },
            ],
        },
    }

    assert find_joinmarket_round_events_file(str(tmp_path)) == str(events_file)

    coinjoins, rounds = joinmarket_parse_round_events(str(tmp_path), raw_tx_db)

    assert list(coinjoins) == ["txA"]
    assert coinjoins["txA"]["round_id"] == "1"
    assert coinjoins["txA"]["broadcast_time"] == "2026-06-13 09:10:00.000"
    assert coinjoins["txA"]["joinmarket_round_event"]["taker"] == "jcs-000"
    assert rounds["1"]["round_start_timestamp"] == "2026-06-13 09:10:00.000"


def test_joinmarket_round_events_allow_empty_label_file(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    events_file = data_dir / "joinmarket_round_events.json"
    events_file.write_text("[]", encoding="utf-8")

    coinjoins, rounds = joinmarket_parse_round_events(str(tmp_path), {})

    assert find_joinmarket_round_events_file(str(tmp_path)) == str(events_file)
    assert coinjoins == {}
    assert rounds == {}


def test_joinmarket_manifest_accepts_verified_confirmed_events(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    events_file = data_dir / 'joinmarket_round_events.json'
    events_file.write_text(
        json.dumps([
            {
                'round_id': 1,
                'export_round_id': 1,
                'status': 'confirmed',
                'destination_matches': [{'txid': 'txA', 'block_height': 7}],
                'timestamp': '2026-01-01',
            },
            {'round_id': 2, 'export_round_id': 2, 'status': 'started', 'destination_address': 'unmined'},
        ]),
        encoding='utf-8',
    )
    write_joinmarket_manifest(tmp_path, events_file, positive_count=1)
    raw_tx_db = {'txA': {'txid': 'txA', 'vin': [], 'vout': []}}

    coinjoins, rounds = joinmarket_parse_round_events(str(tmp_path), raw_tx_db)

    assert list(coinjoins) == ['txA']
    assert list(rounds) == ['1']


def test_joinmarket_manifest_rejects_modified_round_events(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    events_file = data_dir / 'joinmarket_round_events.json'
    events_file.write_text('[]', encoding='utf-8')
    write_joinmarket_manifest(tmp_path, events_file, positive_count=0)
    events_file.write_text('{}', encoding='utf-8')

    with pytest.raises(ValueError, match='source hash does not match manifest'):
        joinmarket_parse_round_events(str(tmp_path), {})


def test_joinmarket_manifest_rejects_cleanly_truncated_round_events(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    events_file = data_dir / 'joinmarket_round_events.json'
    events = [
        {
            'round_id': 1,
            'export_round_id': 1,
            'status': 'confirmed',
            'destination_matches': [{'txid': 'txA', 'block_height': 7}],
        },
        {
            'round_id': 2,
            'export_round_id': 2,
            'status': 'confirmed',
            'destination_matches': [{'txid': 'txB', 'block_height': 8}],
        },
    ]
    events_file.write_text(json.dumps(events), encoding='utf-8')
    write_joinmarket_manifest(tmp_path, events_file, positive_count=2)
    events_file.write_text(json.dumps(events[:1]), encoding='utf-8')

    with pytest.raises(ValueError, match='source size does not match manifest'):
        joinmarket_parse_round_events(str(tmp_path), {})


def test_joinmarket_manifest_rejects_incomplete_capture(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    events_file = data_dir / 'joinmarket_round_events.json'
    events_file.write_text('[]', encoding='utf-8')
    write_joinmarket_manifest(tmp_path, events_file, positive_count=0, complete=False)

    with pytest.raises(ValueError, match='manifest is incomplete'):
        joinmarket_parse_round_events(str(tmp_path), {})


def test_joinmarket_manifest_rejects_another_engine(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    events_file = data_dir / 'joinmarket_round_events.json'
    events_file.write_text('[]', encoding='utf-8')
    write_joinmarket_manifest(tmp_path, events_file, positive_count=0, engine='wasabi')

    with pytest.raises(ValueError, match='engine is not JoinMarket'):
        joinmarket_parse_round_events(str(tmp_path), {})


def test_joinmarket_manifest_positive_count_must_match_processed_events(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    events_file = data_dir / 'joinmarket_round_events.json'
    events_file.write_text(
        json.dumps([
            {
                'round_id': 1,
                'export_round_id': 1,
                'status': 'confirmed',
                'destination_matches': [{'txid': 'missing', 'block_height': 7}],
                'timestamp': '2026-01-01',
            },
        ]),
        encoding='utf-8',
    )
    write_joinmarket_manifest(tmp_path, events_file, positive_count=1)

    with pytest.raises(ValueError, match='parsed 0, expected 1') as excinfo:
        joinmarket_parse_round_events(str(tmp_path), {})

    assert "missing=['missing']" in str(excinfo.value)
    assert 'unexpected' not in str(excinfo.value)


def test_joinmarket_manifest_positive_count_must_match_its_source(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    events_file = data_dir / 'joinmarket_round_events.json'
    events_file.write_text(
        json.dumps([
            {
                'round_id': 1,
                'export_round_id': 1,
                'status': 'confirmed',
                'destination_matches': [{'txid': 'txA', 'block_height': 7}],
                'timestamp': '2026-01-01',
            },
        ]),
        encoding='utf-8',
    )
    write_joinmarket_manifest(tmp_path, events_file, positive_count=0)

    with pytest.raises(ValueError, match='round-event source: parsed 1, expected 0'):
        joinmarket_parse_round_events(str(tmp_path), {})


def test_joinmarket_manifest_rejects_unexpected_source(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    events_file = data_dir / 'other-events.json'
    events_file.write_text('[]', encoding='utf-8')
    write_joinmarket_manifest(
        tmp_path,
        events_file,
        positive_count=0,
        source_path='other-events.json',
    )

    with pytest.raises(ValueError, match='unexpected source'):
        joinmarket_parse_round_events(str(tmp_path), {})


def test_dropped_joinmarket_events_do_not_create_rounds(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    events_file = data_dir / "joinmarket_round_events.json"
    events_file.write_text(
        json.dumps(
            [
                {"round_id": 1, "export_round_id": 1, "status": "failed"},
                {
                    "round_id": 2,
                    "export_round_id": 2,
                    "status": "confirmed",
                    "destination_matches": [
                        {"txid": "not-in-exported-blocks", "block_height": 7}
                    ],
                    "timestamp": "2026-06-13 09:10:00.000",
                },
            ]
        ),
        encoding="utf-8",
    )

    with mock.patch('cj_process.parse_cj_logs.als.run_command') as run_command:
        coinjoins, rounds = joinmarket_parse_round_events(str(tmp_path), {})

    assert coinjoins == {}
    assert rounds == {}
    run_command.assert_not_called()


def test_empty_joinmarket_log_directory_is_not_usable(tmp_path):
    (tmp_path / 'data' / 'jcs-000' / 'joinmarket').mkdir(parents=True)

    assert find_joinmarket_client_log_files(str(tmp_path)) == []


def test_joinmarket_event_timestamp_is_normalized(tmp_path):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    (data_dir / 'joinmarket_round_events.json').write_text(
        json.dumps([
            {
                'round_id': 1,
                'export_round_id': 1,
                'status': 'confirmed',
                'destination_matches': [{'txid': 'txA', 'block_height': 7}],
                'timestamp': '2026-01-01T12:34:56Z',
            },
        ]),
        encoding='utf-8',
    )
    raw_tx_db = {'txA': {'txid': 'txA', 'vin': [], 'vout': []}}

    coinjoins, rounds = joinmarket_parse_round_events(str(tmp_path), raw_tx_db)

    assert coinjoins['txA']['broadcast_time'] == '2026-01-01 12:34:56.000'
    assert rounds['1']['round_start_timestamp'] == '2026-01-01 12:34:56.000'


def test_duplicate_export_round_id_with_different_txids_is_rejected(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    events_file = data_dir / "joinmarket_round_events.json"
    events_file.write_text(
        json.dumps([
            {
                "round_id": 3,
                "export_round_id": 1,
                "status": "confirmed",
                "destination_matches": [{"txid": "txA", "block_height": 7}],
                "timestamp": "2026-01-01",
            },
            {
                "round_id": 3,
                "export_round_id": 1,
                "status": "confirmed",
                "destination_matches": [{"txid": "txB", "block_height": 8}],
                "timestamp": "2026-01-02",
            },
        ]),
        encoding="utf-8",
    )
    raw_tx_db = {
        txid: {
            "txid": txid,
            "vin": [],
            "vout": [],
        }
        for txid in ("txA", "txB")
    }

    with pytest.raises(ValueError, match="round ids map to multiple txids"):
        joinmarket_parse_round_events(str(tmp_path), raw_tx_db)


def test_export_round_ids_disambiguate_per_taker_round_ids(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    events_file = data_dir / "joinmarket_round_events.json"
    events_file.write_text(
        json.dumps([
            {
                "round_id": 3,
                "export_round_id": 7,
                "status": "confirmed",
                "destination_matches": [{"txid": "txA", "block_height": 7}],
                "timestamp": "2026-01-01",
            },
            {
                "round_id": 3,
                "export_round_id": 8,
                "status": "confirmed",
                "destination_matches": [{"txid": "txB", "block_height": 8}],
                "timestamp": "2026-01-02",
            },
        ]),
        encoding="utf-8",
    )
    write_joinmarket_manifest(tmp_path, events_file, positive_count=2)
    raw_tx_db = {
        txid: {"txid": txid, "vin": [], "vout": []}
        for txid in ("txA", "txB")
    }

    coinjoins, rounds = joinmarket_parse_round_events(str(tmp_path), raw_tx_db)

    assert coinjoins["txA"]["round_id"] == "7"
    assert coinjoins["txB"]["round_id"] == "8"
    assert set(rounds) == {"7", "8"}


def test_duplicate_txid_with_different_round_ids_is_rejected(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    events_file = data_dir / "joinmarket_round_events.json"
    events_file.write_text(
        json.dumps([
            {
                "round_id": 1,
                "export_round_id": 1,
                "status": "confirmed",
                "destination_matches": [{"txid": "txA", "block_height": 7}],
                "timestamp": "2026-01-01",
            },
            {
                "round_id": 2,
                "export_round_id": 2,
                "status": "confirmed",
                "destination_matches": [{"txid": "txA", "block_height": 8}],
                "timestamp": "2026-01-02",
            },
        ]),
        encoding="utf-8",
    )
    raw_tx_db = {
        "txA": {
            "txid": "txA",
            "vin": [],
            "vout": [],
        }
    }

    with pytest.raises(ValueError, match="txids map to multiple round ids"):
        joinmarket_parse_round_events(str(tmp_path), raw_tx_db)
