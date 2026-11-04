"""Locate Wasabi coordinator artifacts and cross-check what they claim.

Two related jobs live here. Discovery resolves which coordinator log files
belong to a run, across the manifest, legacy and emulator-export layouts. The
cross-check then compares the broadcast records in those logs against the parser
output, so a silently incomplete run - one where a CoinJoin was dropped because
its transaction is missing from the exported data - becomes a hard failure.

``parse_cj_logs`` is imported lazily inside the one function that needs it: it
imports this module at module level, and the reference must be resolved at call
time so tests can patch ``parse_cj_logs`` attributes.
"""
import logging
import os
import re
from pathlib import Path

from cj_process.wasabi_manifest import verify_wasabi_manifest_log_files

logger = logging.getLogger(__name__)

WASABI_SUCCESSFUL_BROADCAST_RE = re.compile(
    r"successfully\s+broadcast\s+the\s+coinjoin:\s*([0-9a-f]{64})",
    re.IGNORECASE,
)
WASABI_ROUND_ACTIVITY_RE = re.compile(
    r"\b(?:Blame\s+)?Round\s+\([^)]+\):\s+(?:Created round|Blame round created)\b",
    re.IGNORECASE,
)


def find_wasabi_exported_log_files(data_path):
    """Find Wasabi coordinator logs in legacy emulator exports.

    This fallback is used only when the run has no producer-label manifest.
    The emulator exports container directories as archives, so extraction may add
    intermediate directory levels below the known coordinator or backend root.

    Patterns are checked in order; logs from the first matching layout are used:

        <data_path>/wasabi-coordinator/coordinator/**/Logs.txt
        <data_path>/wasabi-backend/backend/**/Logs.txt
        <data_path>/wasabi-backend-2.6/backend/**/Logs.txt

    ``**`` matches zero or more nested directories. This accepts both the usual
    export layout and equivalent archive layouts with additional nesting, while
    preserving the preference for the Wasabi 2.6 coordinator log.
    """
    data_root = Path(data_path)
    if not data_root.is_dir():
        return []

    # Split Wasabi stores coordinator logs separately; legacy Wasabi keeps
    # them in the backend export. Prefer the coordinator when both exist.
    for directory_prefix in ('wasabi-coordinator', 'wasabi-backend'):
        export_roots = sorted(
            path
            for path in data_root.iterdir()
            if path.is_dir() and path.name.startswith(directory_prefix)
        )
        log_files = sorted(
            str(log_file.resolve())
            for export_root in export_roots
            for log_file in export_root.rglob('Logs.txt')
            if log_file.is_file()
        )
        if log_files:
            return log_files
    return []


def find_legacy_wasabi_coordinator_log_files(base_path):
    # Historical manual-Wasabi layout. This is checked first for an analysis
    # input assembled outside coinjoin-emulator; the emulator never writes here.
    local_log_file = os.path.join(base_path, 'WalletWasabi', 'Backend', 'Logs.txt')
    if os.path.isfile(local_log_file):
        return [local_log_file]

    # Emulator-export layout. Reached when the local layout above is absent:
    # EngineBase.store_logs() creates <base_path>/data and stores artifacts there.
    data_path = os.path.join(base_path, 'data')
    return find_wasabi_exported_log_files(data_path)


def wasabi_coordinator_log_search_patterns(base_path):
    """Return the manifest path and filesystem patterns used for discovery."""
    return [
        os.path.join(base_path, 'data', 'coinjoin_label_manifest.json'),
        os.path.join(base_path, 'WalletWasabi', 'Backend', 'Logs.txt'),
        os.path.join(base_path, 'data', 'wasabi-coordinator*', '**', 'Logs.txt'),
        os.path.join(base_path, 'data', 'wasabi-backend*', '**', 'Logs.txt'),
    ]


def find_wasabi_coordinator_log_files(base_path):
    data_path = os.path.join(base_path, 'data')
    # A present Wasabi manifest is authoritative. Only runs without one use legacy discovery.
    manifest_logs_and_count = verify_wasabi_manifest_log_files(data_path)
    if manifest_logs_and_count is not None:
        log_files, _ = manifest_logs_and_count
        return log_files
    return find_legacy_wasabi_coordinator_log_files(base_path)


def find_wasabi_prison_file(base_path):
    # Prison.txt is stored relative to the coordinator log in every supported layout.
    for log_file in find_wasabi_coordinator_log_files(base_path):
        prison_file = os.path.join(os.path.dirname(log_file), 'WabiSabi', 'Prison.txt')
        if os.path.isfile(prison_file):
            return prison_file
    return None


def resolve_wasabi_coordinator_logs(base_path):
    """Return coordinator logs plus the producer positive count declared for them.

    A present manifest is authoritative; the count is None only for legacy runs.

    This deliberately differs from find_wasabi_coordinator_log_files above: the
    cross-check needs the declared count, and it rejects a manifest written by
    another engine rather than silently falling back to legacy discovery.
    """
    data_path = os.path.join(base_path, 'data')
    manifest_logs_and_count = verify_wasabi_manifest_log_files(data_path, reject_other_engine=True)
    if manifest_logs_and_count is None:
        return find_legacy_wasabi_coordinator_log_files(base_path), None
    return manifest_logs_and_count


def read_wasabi_coordinator_log(coordinator_log_file):
    try:
        with open(coordinator_log_file, 'r', encoding='utf-8') as log_file:
            return log_file.read()
    except (OSError, UnicodeError) as error:
        raise ValueError(f'Cannot read Wasabi coordinator log {coordinator_log_file}: {error}') from error


def read_wasabi_producer_positive_txids(coordinator_log_files, has_manifest):
    """Collect the CoinJoin txids the coordinator logs claim to have broadcast."""
    producer_positive_txids = set()
    for coordinator_log_file in coordinator_log_files:
        log_text = read_wasabi_coordinator_log(coordinator_log_file)
        log_positive_txids = {
            match.group(1).lower()
            for match in WASABI_SUCCESSFUL_BROADCAST_RE.finditer(log_text)
        }
        # Without a manifest, an empty log is only credible when it shows round activity.
        if not has_manifest and not log_positive_txids and WASABI_ROUND_ACTIVITY_RE.search(log_text) is None:
            raise ValueError(
                'Legacy Wasabi coordinator log contains no recognizable round activity: '
                f'{coordinator_log_file}'
            )
        producer_positive_txids.update(log_positive_txids)
    return producer_positive_txids


def parse_wasabi_logs_by_round_completeness(coordinator_log_files, raw_txs, allow_rpc):
    """Parse every log and split the files by whether they yielded complete rounds."""
    from cj_process import parse_cj_logs

    coinjoins = {}
    complete_round_logs = []
    incomplete_round_logs = []

    for coordinator_log_file in coordinator_log_files:
        parsed_coinjoins = parse_cj_logs.parse_backend_coinjoin_logs(
            coordinator_log_file, raw_txs, allow_rpc=allow_rpc
        )
        if parsed_coinjoins:
            coinjoins.update(parsed_coinjoins)
            complete_round_logs.append(coordinator_log_file)
        else:
            incomplete_round_logs.append(coordinator_log_file)

    return coinjoins, complete_round_logs, incomplete_round_logs


def check_wasabi_producer_positive_count(producer_positive_txids, expected_positive_count):
    """Reject a manifest whose declared count disagrees with its own logs."""
    if (
        expected_positive_count is not None
        and len(producer_positive_txids) != expected_positive_count
    ):
        raise ValueError(
            'Wasabi producer positive count does not match coordinator logs: '
            f'parsed {len(producer_positive_txids)}, expected {expected_positive_count}'
        )


def check_wasabi_parsed_positives(coinjoins, producer_positive_txids):
    """Reject logs whose broadcast records and parsed CoinJoins describe different sets."""
    parsed_positive_txids = {str(txid).lower() for txid in coinjoins}
    if parsed_positive_txids != producer_positive_txids:
        missing_txids = sorted(producer_positive_txids - parsed_positive_txids)
        unexpected_txids = sorted(parsed_positive_txids - producer_positive_txids)
        raise ValueError(
            'Wasabi coordinator labels do not match parsed CoinJoins: '
            f'missing={missing_txids}, unexpected={unexpected_txids}'
        )


def parse_wasabi_coordinator_coinjoins(base_path, raw_txs, allow_rpc: bool = True):
    """Parse integrity-checked Wasabi CoinJoins and return all coordinator logs.

    Current runs must have a complete, hash-verified producer manifest whose
    positive count matches the log and parser output. Runs from older emulator
    versions remain supported without a manifest, but their logs must be valid
    UTF-8 and contain recognizable Wasabi round activity.
    """
    coordinator_log_files, expected_positive_count = resolve_wasabi_coordinator_logs(base_path)
    if not coordinator_log_files:
        searched = '\n  '.join(wasabi_coordinator_log_search_patterns(base_path))
        raise FileNotFoundError(
            'No Wasabi coordinator log file found for emulation analysis. '
            f'Searched:\n  {searched}'
        )

    # Cheap pass over the raw text first, so a manifest mismatch fails before the full parse.
    producer_positive_txids = read_wasabi_producer_positive_txids(
        coordinator_log_files, expected_positive_count is not None
    )
    check_wasabi_producer_positive_count(producer_positive_txids, expected_positive_count)

    coinjoins, complete_round_logs, incomplete_round_logs = parse_wasabi_logs_by_round_completeness(
        coordinator_log_files, raw_txs, allow_rpc
    )
    check_wasabi_parsed_positives(coinjoins, producer_positive_txids)

    # No complete round means no log yielded one, so incomplete_round_logs holds them all.
    if not coinjoins:
        parsed_files = '\n  '.join(incomplete_round_logs)
        logger.warning(
            'No complete Wasabi CoinJoin transactions were found in coordinator logs. '
            'Parsed files:\n  %s',
            parsed_files,
        )
        return {}, coordinator_log_files

    if len(complete_round_logs) > 1:
        logger.warning(
            'Merged complete Wasabi CoinJoins from multiple coordinator logs: %s',
            ', '.join(complete_round_logs),
        )
    return coinjoins, coordinator_log_files
