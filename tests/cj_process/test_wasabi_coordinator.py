"""Cross-check between coordinator broadcast records and parsed CoinJoins.

parse_wasabi_coordinator_coinjoins must fail a run whose coordinator log and
parser output disagree, instead of returning a quietly incomplete result.
"""
from unittest import mock

import pytest

from cj_process import parse_cj_logs, wasabi_coordinator
from cj_process import wasabi_manifest
from utils import write_manifest


def test_wasabi_manifest_compatibility_api_is_preserved(tmp_path):
    log_path = tmp_path / 'Logs.txt'
    log_path.write_text('coordinator log\n', encoding='utf-8')
    source = {
        'path': 'Logs.txt',
        'size_bytes': log_path.stat().st_size,
        'sha256': wasabi_manifest.sha256_file(log_path),
    }

    assert wasabi_manifest.PRODUCER_LABEL_MANIFEST_SCHEMA_VERSION == '1.0'
    assert wasabi_manifest.verify_wasabi_manifest_source(source, str(tmp_path)) == str(log_path)


def test_manifest_positive_count_must_match_coordinator_log(tmp_path):
    log_path = tmp_path / "data" / "wasabi-coordinator" / "coordinator" / "Logs.txt"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("round started but not completed\n", encoding="utf-8")
    write_manifest(tmp_path, log_path, positive_count=1)

    with pytest.raises(ValueError, match="parsed 0, expected 1"):
        wasabi_coordinator.parse_wasabi_coordinator_coinjoins(str(tmp_path), {})


def test_legacy_empty_round_without_manifest_is_supported(tmp_path):
    log_path = tmp_path / "data" / "wasabi-backend" / "backend" / "Logs.txt"
    log_path.parent.mkdir(parents=True)
    round_id = "a" * 64
    log_path.write_text(
        "2026-01-01 00:00:00.000 [1] INFO Arena.CreateRound (1) "
        f"Round ({round_id}): Created round with parameters: "
        "MaxSuggestedAmount:'1.0' BTC.\n",
        encoding="utf-8",
    )

    assert wasabi_coordinator.parse_wasabi_coordinator_coinjoins(str(tmp_path), {}) == (
        {}, [str(log_path.resolve())]
    )


def test_legacy_complete_round_without_manifest_is_supported(tmp_path):
    log_path = tmp_path / "data" / "wasabi-backend" / "backend" / "Logs.txt"
    log_path.parent.mkdir(parents=True)
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

    with mock.patch.object(parse_cj_logs, "extract_tx_info", return_value=transaction):
        coinjoins, coordinator_log_files = wasabi_coordinator.parse_wasabi_coordinator_coinjoins(
            str(tmp_path),
            {},
        )

    assert list(coinjoins) == [txid]
    assert coordinator_log_files == [str(log_path.resolve())]


def test_incomplete_coordinator_logs_are_returned_for_error_parsing(tmp_path):
    complete_log = tmp_path / "data" / "wasabi-coordinator-1" / "coordinator" / "Logs.txt"
    incomplete_log = tmp_path / "data" / "wasabi-coordinator-2" / "coordinator" / "Logs.txt"
    complete_log.parent.mkdir(parents=True)
    incomplete_log.parent.mkdir(parents=True)
    txid = "b" * 64
    complete_log.write_text(
        f"Round ({'a' * 64}): Successfully broadcast the coinjoin: {txid}.\n",
        encoding="utf-8",
    )
    incomplete_log.write_text(
        f"Round ({'c' * 64}): Created round with parameters: MaxSuggestedAmount:'1.0' BTC.\n",
        encoding="utf-8",
    )

    def parse_log(log_path, *_args, **_kwargs):
        return {txid: {}} if log_path == str(complete_log.resolve()) else {}

    with mock.patch.object(parse_cj_logs, "parse_backend_coinjoin_logs", side_effect=parse_log):
        coinjoins, coordinator_log_files = wasabi_coordinator.parse_wasabi_coordinator_coinjoins(
            str(tmp_path), {}
        )

    assert list(coinjoins) == [txid]
    assert coordinator_log_files == [
        str(complete_log.resolve()),
        str(incomplete_log.resolve()),
    ]


def test_legacy_unrecognized_log_is_rejected(tmp_path):
    log_path = tmp_path / "data" / "wasabi-backend" / "backend" / "Logs.txt"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("truncated or unrelated content\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no recognizable round activity"):
        wasabi_coordinator.parse_wasabi_coordinator_coinjoins(str(tmp_path), {})


def test_missing_coordinator_log_lists_the_searched_paths(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        wasabi_coordinator.parse_wasabi_coordinator_coinjoins(str(tmp_path), {})

    message = str(excinfo.value)
    assert 'Searched:' in message
    assert str(tmp_path / 'data' / 'coinjoin_label_manifest.json') in message
    assert str(tmp_path / 'WalletWasabi' / 'Backend' / 'Logs.txt') in message
    assert str(tmp_path / 'data' / 'wasabi-coordinator*' / '**' / 'Logs.txt') in message
    assert str(tmp_path / 'data' / 'wasabi-backend*' / '**' / 'Logs.txt') in message


def test_manifest_broadcast_must_be_processed_by_analyzer(tmp_path):
    log_path = tmp_path / "data" / "wasabi-coordinator" / "coordinator" / "Logs.txt"
    log_path.parent.mkdir(parents=True)
    txid = "b" * 64
    log_path.write_text(
        f"Round ({'a' * 64}): Successfully broadcast the coinjoin: {txid}.\n",
        encoding="utf-8",
    )
    write_manifest(tmp_path, log_path, positive_count=1)

    with (
        mock.patch.object(parse_cj_logs, "parse_backend_coinjoin_logs", return_value={}),
        pytest.raises(ValueError, match="labels do not match parsed CoinJoins"),
    ):
        wasabi_coordinator.parse_wasabi_coordinator_coinjoins(str(tmp_path), {})
