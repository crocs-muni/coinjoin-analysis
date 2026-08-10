"""Locate the JoinMarket artifacts of a run and parse the CoinJoins they describe.

Current emulator runs declare an integrity-checked round-event file in their
producer-label manifest. Older runs without a manifest remain compatible: their
jcs-*/joinmarket client logs take precedence over an optional round-event file.

``parse_cj_logs`` is imported lazily inside the functions that need it: it
imports this module at module level, and the references (extract_tx_info,
format_mine_time, SM) must be resolved at call time so tests can patch
``parse_cj_logs`` attributes.
"""
import glob
import json
import logging
import os
import re

import cj_process.cj_analysis as als
from cj_process.joinmarket_manifest import verify_joinmarket_manifest_round_events_file

logger = logging.getLogger(__name__)


def find_joinmarket_round_events_file(base_path: str):
    candidate_paths = [
        os.path.join(base_path, 'data', 'joinmarket_round_events.json'),
        os.path.join(base_path, 'joinmarket_round_events.json'),
    ]
    for candidate_path in candidate_paths:
        if os.path.isfile(candidate_path):
            return candidate_path
    return None


def find_joinmarket_client_log_files(base_path: str):
    pattern = os.path.join(base_path, 'data', 'jcs-*', 'joinmarket', 'jmwalletd.log')
    return sorted(path for path in glob.glob(pattern) if os.path.isfile(path))


# A derivation path next to an address is proof of ownership: a JoinMarket wallet
# never knows the path of a counterparty's address, so peer addresses logged during
# a round (for example in 'Makers responded with') are not matched here.
JOINMARKET_OWN_ADDRESS_PATTERN = re.compile(
    r"path:\s*(?P<path>m/[0-9']+[0-9'/]*)\s*,\s*address:\s*(?P<address>\w+)"
)


def joinmarket_wallet_addresses(wallet_path: str):
    """Collect the addresses a single JoinMarket wallet derived for itself.

    JoinMarket has no 'listkeys' RPC, so keys.json holds a sentinel string and the
    wallet would stay unattributed.  Ownership is recovered from the two places
    where the wallet reports a derivation path next to an address: the structured
    unspent-coin export, and the wallet display lines in its own logs.  The export
    alone covers only currently unspent coins, so the logs are needed to recover
    addresses that were already spent.

    :param wallet_path: path of one jcs-* client directory
    :return: mapping {address: {'address': ..., 'path': ...}}, matching the shape
        that the keys.json branch of obtain_wallets_info builds
    """
    addresses = {}

    unspent_file = os.path.join(wallet_path, 'unspent_coins.json')
    if os.path.isfile(unspent_file):
        with open(unspent_file, 'r') as file:
            unspent_coins = json.load(file)
        if isinstance(unspent_coins, dict):
            for utxo in unspent_coins.get('utxos') or []:
                address = utxo.get('address')
                if address:
                    addresses.setdefault(address, {'address': address, 'path': utxo.get('path')})

    log_patterns = [
        os.path.join(wallet_path, 'joinmarket', 'jmwalletd.log'),
        os.path.join(wallet_path, 'joinmarket', '.joinmarket', 'logs', '*.log'),
    ]
    for log_pattern in log_patterns:
        for log_file in sorted(glob.glob(log_pattern)):
            with open(log_file, 'r', errors='replace') as file:
                for match in JOINMARKET_OWN_ADDRESS_PATTERN.finditer(file.read()):
                    address = match.group('address')
                    addresses.setdefault(address, {'address': address, 'path': match.group('path')})

    if not addresses:
        logger.warning('No JoinMarket wallet addresses recovered from %s', wallet_path)
    return addresses


def joinmarket_parse_coinjoin_logs(base_path: str, raw_tx_db: dict, allow_rpc: bool = True):
    """
    Obtain information about coinjoins from collated logs from all separate clients
    :param base_path: base path where docker client images are stored
    :param raw_tx_db: database of all transaction loaded from btc-node
    :return: cjtx_stats structure with information about all detected coinjoins
    """
    # Resolved at call time so tests can patch ``parse_cj_logs`` attributes; see module docstring.
    from cj_process import parse_cj_logs

    print(f'Parsing coinjoin-relevant data from JoinMarket client logs {base_path}...', end='')
    # 1. Parse logs for each client separately
    # 2. Collate client logs into complete view

    success_coinjoins = {}
    for log_file in find_joinmarket_client_log_files(base_path):
        success_coinjoins[log_file] = als.joinmarket_find_coinjoins(log_file)

    all_coinjoins_duplicities = [success_coinjoins[path][txid] for path in success_coinjoins.keys() for txid in success_coinjoins[path].keys()]
    all_coinjoins = {txid: success_coinjoins[path][txid] for path in success_coinjoins.keys() for txid in success_coinjoins[path].keys()}

    parse_cj_logs.SM.print(
        f'Total JoinMarket coinjoins detected (with duplicities: {len(all_coinjoins_duplicities)})'
    )
    parse_cj_logs.SM.print(f'Total fully finished JoinMarket coinjoins found: {len(all_coinjoins)}')
    print('Parsing separate coinjoin transactions ', end='')
    cjtx_stats = {}
    for cjtxid in all_coinjoins.keys():
        # extract input and output addresses
        tx_record = parse_cj_logs.extract_tx_info(cjtxid, raw_tx_db, allow_rpc=allow_rpc)
        if tx_record is not None:
            tx_record['round_id'] = cjtxid
            tx_record['round_start_time'] = all_coinjoins[cjtxid]['timestamp']  # BUGBUG: bad round start time, needs to be extracted from logs better
            tx_record['broadcast_time'] = all_coinjoins[cjtxid]['timestamp']
            tx_record['is_blame_round'] = False
            tx_record['is_cjtx'] = True
            cjtx_stats[cjtxid] = tx_record
        else:
            logger.warning('Could not decode JoinMarket CoinJoin transaction tx=%s', cjtxid)
        print('.', end='')
    print('done')

    parse_cj_logs.SM.print(f'Total fully finished JoinMarket coinjoins processed: {len(cjtx_stats)}')

    return cjtx_stats


def joinmarket_parse_round_events(base_path: str, raw_tx_db: dict):
    """
    Obtain JoinMarket coinjoins from emulator-provided round labels.
    Newer emulator runs may not contain copied jcs-*/joinmarket client logs, but
    they do contain data/joinmarket_round_events.json with matched txids.
    """
    # Resolved at call time so tests can patch ``parse_cj_logs`` attributes; see module docstring.
    from cj_process import parse_cj_logs

    manifest_source_and_count = verify_joinmarket_manifest_round_events_file(base_path)
    if manifest_source_and_count is None:
        events_file = find_joinmarket_round_events_file(base_path)
        expected_positive_count = None
    else:
        events_file, expected_positive_count = manifest_source_and_count
    if events_file is None:
        return None

    print(f'Parsing CoinJoin-relevant data from JoinMarket round events {events_file}...', end='')
    with open(events_file, 'r') as file:
        round_events = json.load(file)
    if not isinstance(round_events, list):
        raise ValueError(f'JoinMarket round event source must contain a JSON list: {events_file}')
    if any(not isinstance(event, dict) for event in round_events):
        invalid_event = next(event for event in round_events if not isinstance(event, dict))
        raise ValueError(f'JoinMarket round event must contain a JSON object: {invalid_event!r}')

    producer_positive_txids = None
    if expected_positive_count is not None:
        producer_positive_txids = {
            str(event['txid'])
            for event in round_events
            if event.get('status') == 'confirmed' and event.get('txid')
        }
        if len(producer_positive_txids) != expected_positive_count:
            raise ValueError(
                'JoinMarket producer positive count does not match its round-event source: '
                f'parsed {len(producer_positive_txids)}, expected {expected_positive_count}'
            )

    cjtx_stats = {}
    parsed_rounds = {}
    dropped_decode_failures = 0
    dropped_missing_txids = 0
    missing_time_txids = []
    round_txids = {}
    txid_round_ids = {}
    conflicting_round_ids = []
    conflicting_txids = []
    for event in round_events:
        # The current producer counts only reconciled, confirmed transactions as
        # positives. Legacy files predate this manifest contract and retain the
        # previous behavior of accepting any record carrying a txid.
        if expected_positive_count is not None and event.get('status') != 'confirmed':
            continue
        raw_txid = event.get('txid')
        txid = str(raw_txid) if raw_txid else None
        event_round_id = event.get('round_id')
        round_id = str(event_round_id if event_round_id is not None else txid or len(parsed_rounds) + 1)
        if not txid:
            dropped_missing_txids += 1
            continue

        if txid not in raw_tx_db:
            dropped_decode_failures += 1
            logger.warning('Dropping JoinMarket round event for tx=%s missing from exported blocks', txid)
            continue

        timestamp = (
            event.get('broadcast_time')
            or event.get('timestamp')
            or event.get('round_start_time')
            or raw_tx_db[txid].get('mine_time')
        )
        if not timestamp:
            missing_time_txids.append(txid)
            continue
        try:
            timestamp = parse_cj_logs.format_mine_time(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'JoinMarket round event has invalid timestamp for txid {txid}: {timestamp!r}') from exc

        tx_record = parse_cj_logs.extract_tx_info(txid, raw_tx_db, allow_rpc=False)
        if tx_record is not None:
            tx_record['round_id'] = round_id
            tx_record['round_start_time'] = timestamp
            tx_record['broadcast_time'] = timestamp
            tx_record['is_blame_round'] = False
            tx_record['is_cjtx'] = True
            tx_record['joinmarket_round_event'] = event
            previous_txid = round_txids.get(round_id)
            if previous_txid is not None and previous_txid != txid:
                conflicting_round_ids.append((round_id, previous_txid, txid))
                continue
            previous_round_id = txid_round_ids.get(txid)
            if previous_round_id is not None and previous_round_id != round_id:
                conflicting_txids.append((txid, previous_round_id, round_id))
                continue
            round_txids[round_id] = txid
            txid_round_ids[txid] = round_id
            cjtx_stats[txid] = tx_record
            parsed_rounds[round_id] = {'round_start_timestamp': timestamp}
        else:
            dropped_decode_failures += 1
            logger.warning('Dropping JoinMarket round event with undecodable tx=%s', txid)
        print('.', end='')
    print('done')

    if missing_time_txids:
        missing_time_txids_text = ', '.join(str(txid) for txid in missing_time_txids)
        raise ValueError(
            f'JoinMarket round events missing broadcast/mine time for txids: {missing_time_txids_text}'
        )
    if conflicting_round_ids:
        conflicting_round_ids_text = ', '.join(
            f'{round_id} ({first_txid} and {second_txid})'
            for round_id, first_txid, second_txid in conflicting_round_ids
        )
        raise ValueError(
            f'JoinMarket round ids map to multiple txids: {conflicting_round_ids_text}'
        )
    if conflicting_txids:
        conflicting_txids_text = ', '.join(
            f'{txid} ({first_round_id} and {second_round_id})'
            for txid, first_round_id, second_round_id in conflicting_txids
        )
        raise ValueError(
            f'JoinMarket txids map to multiple round ids: {conflicting_txids_text}'
        )
    if producer_positive_txids is not None:
        missing_txids = sorted(producer_positive_txids - set(cjtx_stats))
        if missing_txids:
            raise ValueError(
                'JoinMarket producer labels were not all parsed as CoinJoins: '
                f'parsed {len(cjtx_stats)}, expected {expected_positive_count}, '
                f'missing={missing_txids}'
            )
    parse_cj_logs.SM.print(f'Total JoinMarket round events loaded: {len(round_events)}')
    parse_cj_logs.SM.print(
        'JoinMarket round events dropped: '
        f'missing_txid={dropped_missing_txids}, decode_failures={dropped_decode_failures}'
    )
    parse_cj_logs.SM.print(
        f'Total fully finished JoinMarket coinjoins processed from round events: {len(cjtx_stats)}'
    )
    return cjtx_stats, parsed_rounds
