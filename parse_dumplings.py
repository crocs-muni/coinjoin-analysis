import copy
import os
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from jsonpickle import json
import logging

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

# Configure the logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

VerboseTransactionInfoLineSeparator = ':::'
VerboseInOutInfoInLineSeparator = '}'
SATS_IN_BTC = 100000000

# If True, difference between assigned and existing cluster id is checked and failed upon if different
# If False, only warning is printed, but execution continues.
# TODO: Systematic solution requires merging and resolving different cluster ids
CLUSTER_ID_CHECK_HARD_ASSERT = False

SAVE_BASE_FILES_JSON = False


class ClusterIndex:
    NEW_CLUSTER_INDEX = 0

    def __init__(self, initial_cluster_index):
        self.NEW_CLUSTER_INDEX = initial_cluster_index

    def get_new_index(self):
        self.NEW_CLUSTER_INDEX += 1
        return self.NEW_CLUSTER_INDEX

    def get_current_index(self):
        return self.NEW_CLUSTER_INDEX


CLUSTER_INDEX = ClusterIndex(0)


class MIX_EVENT_TYPE(Enum):
    MIX_ENTER = 'MIX_ENTER'  # New liquidity coming to mix
    MIX_LEAVE = 'MIX_LEAVE'  # Liquidity leaving mix (postmix spend)
    MIX_REMIX = 'MIX_REMIX'  # Remixed value within mix
    MIX_STAY = 'MIX_STAY'    # Mix output not yet spend (may be remixed or leave mix later)


class SummaryMessages:
    summary_messages = []

    def print(self, message: str):
        print(message)
        self.summary_messages.append(message)

    def print_summary(self):
        for message in self.summary_messages:
            print(message)

SM = SummaryMessages()


def set_key_value_assert(data, key, value, hard_assert):
    if key in data:
        if hard_assert:
            assert data[key] == value, f"Key '{key}' already exists with a different value {data[key]} vs. {value}."
        else:
            if data[key] != value:
                logging.warning(f"Key '{key}' already exists with a different value {data[key]} vs. {value}.")

    else:
        data[key] = value


def get_ratio_string(numerator, denominator) -> str:
    if denominator != 0:
        return f'{numerator}/{denominator} ({round(numerator/float(denominator) * 100, 1)}%)'
    else:
        return f'{numerator}/{0} (0%)'


def get_input_name_string(txid, index):
    return f'vin_{txid}_{index}'


def get_output_name_string(txid, index):
    return f'vout_{txid}_{index}'


def extract_txid_from_inout_string(inout_string):
    if inout_string.startswith('vin') or inout_string.startswith('vout'):
        return inout_string[inout_string.find('_') + 1: inout_string.rfind('_')], inout_string[inout_string.rfind('_')+1:]
    else:
        assert False, f'Invalid inout string {inout_string}'


def get_synthetic_address(create_txid, vout_index):
    """
    Synthetic unique address from creating transaction and its vout index
    :param create_txid: tx which created this output
    :param vout_index: index of output
    :return: formatted string with synthetic address
    """
    return f'synbc1{create_txid[:16]}_{vout_index}'


def load_coinjoin_stats_from_file(target_file, start_date: str = None, stop_date: str = None):
    cj_stats = {}
    logging.debug(f'Processing file {target_file}')
    with open(target_file, "r") as file:
        for line in file.readlines():
            parts = line.split(VerboseTransactionInfoLineSeparator)
            record = {}

            # Be careful, broadcast time and blocktime can be significantly different
            block_time = None if parts[3] is None else datetime.fromtimestamp(int(parts[3]))
            record['broadcast_time'] = block_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            if start_date and stop_date:
                if record['broadcast_time'] < start_date or record['broadcast_time'] > stop_date:
                    # Skip this record as it is outside of observed period
                    continue

            tx_id = None if parts[0] is None else parts[0]
            record['txid'] = tx_id
            block_hash = None if parts[1] is None else parts[1]
            record['block_hash'] = block_hash
            block_index = None if parts[2] is None else int(parts[2])
            record['block_index'] = block_index

            inputs = [input.strip('{') for input in parts[4].split(VerboseInOutInfoInLineSeparator)] if parts[4] else None
            record['inputs'] = {}
            index = 0
            for input in inputs:
                # Split to segments using - and + separators
                segments_pipe = input.split("-")
                segments = [segment.split("+") for segment in segments_pipe]
                segments = [item for sublist in segments for item in sublist]

                this_input = {}
                this_input['spending_tx'] = get_output_name_string(segments[0], segments[1])
                this_input['value'] = int(segments[2])
                this_input['wallet_name'] = 'real_unknown'
                #this_input['script'] = segments[3]
                #this_input['script_type'] = segments[4]
                # TODO: generate proper address from script, now replaced by synthetic
                this_input['address'] = get_synthetic_address(segments[0], segments[1])

                record['inputs'][f'{index}'] = this_input
                index += 1

            outputs = [output.strip('{') for output in parts[5].split(VerboseInOutInfoInLineSeparator)] if parts[5] else None
            record['outputs'] = {}
            index = 0
            for output in outputs:
                segments = output.split('+')
                this_output = {}
                this_output['value'] = int(segments[0])
                this_output['wallet_name'] = 'real_unknown'
                this_output['script'] = segments[1]
                this_output['script_type'] = segments[2]
                this_output['address'] = get_synthetic_address(tx_id, index)  # TODO: Compute proper address from script

                record['outputs'][f'{index}'] = this_output
                index += 1

            cj_stats[tx_id] = record

    # backward reference to spending transaction output is already set ('spending_tx'), now set also forward link ('spend_by_tx')
    for txid in cj_stats.keys():
        for index in cj_stats[txid]['inputs'].keys():
            input = cj_stats[txid]['inputs'][index]

            if 'spending_tx' in input.keys():
                tx, vout = extract_txid_from_inout_string(input['spending_tx'])
                # Try to find transaction and set its record
                if tx in cj_stats.keys() and vout in cj_stats[tx]['outputs'].keys():
                    cj_stats[tx]['outputs'][vout]['spend_by_tx'] = get_input_name_string(txid, index)

    return cj_stats


def load_coinjoin_stats(base_path):
    coinjoin_stats = {}
    files = []
    if os.path.exists(base_path):
        files = os.listdir(base_path)
    else:
        logging.error('Path {} does not exists'.format(base_path))

    for file in files:
        target_file = os.path.join(base_path, file)
        coinjoin_stats[target_file]['coinjoins'] = load_coinjoin_stats_from_file(target_file)

    return coinjoin_stats


def analyze_input_out_liquidity(coinjoins, postmix_spend):
    liquidity_events = []
    total_inputs = 0
    total_mix_entering = 0
    total_outputs = 0
    total_mix_leaving = 0
    total_mix_staying = []
    total_utxos = 0
    for cjtx in coinjoins:
        for input in coinjoins[cjtx]['inputs']:
            total_inputs += 1
            spending_tx, index = extract_txid_from_inout_string(coinjoins[cjtx]['inputs'][input]['spending_tx'])
            if spending_tx not in coinjoins.keys():
                # Previous transaction is from outside the mix => new fresh liquidity entered
                total_mix_entering += 1
                coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_ENTER.name
            else:
                coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_REMIX.name

        for output in coinjoins[cjtx]['outputs']:
            total_outputs += 1
            if 'spend_by_tx' not in coinjoins[cjtx]['outputs'][output].keys():
                # This output is not spend by any tx => still utxo (stays within mixing pool)
                total_utxos += 1
                total_mix_staying.append(coinjoins[cjtx]['outputs'][output]['value'])
                coinjoins[cjtx]['outputs'][output]['mix_event_type'] = MIX_EVENT_TYPE.MIX_STAY.name
            else:
                # This output is spend, figure out if by other mixing transaction or postmix spend
                spend_by_tx, index = extract_txid_from_inout_string(coinjoins[cjtx]['outputs'][output]['spend_by_tx'])
                if spend_by_tx not in coinjoins.keys():
                    # Postmix spend: the spending transaction is outside mix => liquidity out
                    if spend_by_tx not in postmix_spend.keys():
                        logging.warning(f'Could not find spend_by_tx {spend_by_tx} in postmix_spend txs')
                    total_mix_leaving += 1
                    coinjoins[cjtx]['outputs'][output]['mix_event_type'] = MIX_EVENT_TYPE.MIX_LEAVE.name
                else:
                    # Mix spend: The output is spent by next coinjoin tx => stays in mix
                    coinjoins[cjtx]['outputs'][output]['mix_event_type'] = MIX_EVENT_TYPE.MIX_REMIX.name

    SM.print(f'  {get_ratio_string(total_mix_entering, total_inputs)} Inputs entering mix / total inputs used by mix transactions')
    SM.print(f'  {get_ratio_string(total_mix_leaving, total_outputs)} Outputs leaving mix / total outputs by mix transactions')
    SM.print(f'  {get_ratio_string(len(total_mix_staying), total_outputs)} Outputs staying in mix / total outputs by mix transactions')
    SM.print(f'  {sum(total_mix_staying) / SATS_IN_BTC} btc, total value staying in mix')


def extract_wallets_info(data):
    wallets_info = {}
    wallets_coins_info = {}
    txs_data = data['coinjoins']

    if len(txs_data) == 0:
        return wallets_info, wallets_coins_info

    # Compute artificial min and max times
    min_cj_time = min([txs_data[cjtxid]['broadcast_time'] for cjtxid in txs_data.keys()])  # Time of the earliest coinjoin
    max_cj_time = max([txs_data[cjtxid]['broadcast_time'] for cjtxid in txs_data.keys()])  # Time of the latest coinjoin
    # Use it as the earliest creation of coin
    datetime_obj = datetime.strptime(min_cj_time, "%Y-%m-%d %H:%M:%S.%f")
    datetime_obj = datetime_obj - timedelta(minutes=60)
    artificial_min_cj_time = datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    datetime_obj = datetime.strptime(max_cj_time, "%Y-%m-%d %H:%M:%S.%f")
    datetime_obj = datetime_obj + timedelta(minutes=60)
    artificial_max_cj_time = datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # 1. Extract all information from outputs and create also corresponding coins
    for cjtxid in txs_data.keys():
        for index in txs_data[cjtxid]['outputs'].keys():
            target_addr = txs_data[cjtxid]['outputs'][index]['address']
            wallet_name = txs_data[cjtxid]['outputs'][index]['wallet_name']
            if wallet_name not in wallets_info.keys():
                wallets_info[wallet_name] = {}
                wallets_coins_info[wallet_name] = []
            wallets_info[wallet_name][target_addr] = {'address': target_addr}

            # Create new coin with information derived from output and transaction info
            coin = {'txid': cjtxid, 'index': index, 'amount': txs_data[cjtxid]['outputs'][index]['value'],
                    'anonymityScore': -1, 'address': target_addr, 'create_time': txs_data[cjtxid]['broadcast_time'],
                    'wallet_name': wallet_name}
            #coin.update({'confirmed': True, 'confirmations': 1, 'keyPath': '', 'block_hash': txs_data[cjtxid]['block_hash']})
            if 'spend_by_tx' in txs_data[cjtxid]['outputs'][index].keys():
                spent_tx, spend_index = extract_txid_from_inout_string(txs_data[cjtxid]['outputs'][index]['spend_by_tx'])
                coin['spentBy'] = spent_tx
                if spent_tx in txs_data.keys():
                    coin['destroy_time'] = txs_data[spent_tx]['broadcast_time']
            wallets_coins_info[wallet_name].append(coin)

    num_outputs = sum([len(txs_data[cjtxid]['outputs']) for cjtxid in txs_data.keys()])
    num_coins = sum([len(wallets_coins_info[wallet_name]) for wallet_name in wallets_coins_info.keys()])
    assert num_outputs == num_coins, f'Mismatch in number of identified coins {num_outputs} vs {num_coins}'

    # 2. Extract all information from inputs and update corresponding coins (destroy_time)
    all_coins = []
    for wallet_name in wallets_coins_info.keys():
        all_coins.extend(wallets_coins_info[wallet_name])
    coins = {coin['address']: coin for coin in all_coins}

    for cjtxid in txs_data.keys():
        for index in txs_data[cjtxid]['inputs'].keys():
            target_addr = txs_data[cjtxid]['inputs'][index]['address']
            wallet_name = txs_data[cjtxid]['inputs'][index]['wallet_name']
            if wallet_name not in wallets_info.keys():
                wallets_info[wallet_name] = {}
            wallets_info[wallet_name][target_addr] = {'address': target_addr}

            # Update coin destroy time for this specific input (if coin already exists)
            if target_addr not in coins.keys():
                # Coin record was not found in any of the previous outputs of all analyzed transactions,
                # Create new coin with information derived from output and transaction info
                # Coin creation time set to artificial_min_cj_time . TODO: change to real value from blockchain
                txid, vout = extract_txid_from_inout_string(txs_data[cjtxid]['inputs'][index]['spending_tx'])
                coin = {'txid': txid, 'index': vout, 'amount': txs_data[cjtxid]['inputs'][index]['value'],
                        'anonymityScore': -1, 'address': target_addr, 'create_time': artificial_min_cj_time,
                        'wallet_name': wallet_name}
                # coin.update({'confirmed': True, 'confirmations': 1, 'keyPath': '', 'block_hash': txs_data[cjtxid]['block_hash']})
                coin['destroy_time'] = txs_data[cjtxid]['broadcast_time']
                coin['spentBy'] = cjtxid
                coins[target_addr] = coin
            else:
                assert coins[target_addr]['amount'] == txs_data[cjtxid]['inputs'][index]['value'], f'Inconsistent value found for {target_addr}'
                # We have found the coin, update destroy_time
                coins[target_addr]['destroy_time'] = txs_data[cjtxid]['broadcast_time']
                if 'spentBy' not in coins[target_addr].keys():
                    coins[target_addr]['spentBy'] = cjtxid
                else:
                    assert coins[target_addr]['spentBy'] == cjtxid, f'Inconsistent spentBy mapping for {coins[target_addr]['address']}'

    wallets_coins_info_updated = {}
    for address in coins.keys():
        coin = coins[address]
        if coin['wallet_name'] not in wallets_coins_info_updated.keys():
            wallets_coins_info_updated[coin['wallet_name']] = []
        wallets_coins_info_updated[coin['wallet_name']].append(coin)

    return wallets_info, wallets_coins_info_updated


def extract_rounds_info(data):
    rounds_info = {}
    txs_data = data['coinjoins']
    for cjtxid in txs_data.keys():
        # Create basic round info from coinjoin data
        rounds_info[cjtxid] = {"cj_tx_id": cjtxid, "round_start_timestamp": txs_data[cjtxid]['broadcast_time'],
                               "logs": [{"round_id": cjtxid, "timestamp": txs_data[cjtxid]['broadcast_time'],
                                         "type": "ROUND_STARTED"}]
                               }
    return rounds_info


def compute_mix_postmix_link(data: dict):
    """
    Set explicit link between mix transactions (coinjoins) and postmix txs
    :param data: dictionary with all transactions
    :return: modified dictionary with all transactions
    """
    # backward reference to spending transaction output is already set ('spending_tx'),
    # now set also forward link ('spend_by_tx')
    for txid in data['postmix'].keys():
        for index in data['postmix'][txid]['inputs'].keys():
            input = data['postmix'][txid]['inputs'][index]
            if 'spending_tx' in input.keys():
                tx, vout = extract_txid_from_inout_string(input['spending_tx'])
                # Try to find transaction in mix (coinjoins) and set its record
                if tx in data['coinjoins'].keys() and vout in data['coinjoins'][tx]['outputs'].keys():
                    data['coinjoins'][tx]['outputs'][vout]['spend_by_tx'] = get_input_name_string(txid, index)

    if 'premix' in data.keys():
        # backward reference from coinjoin to premix is already set ('spending_tx')
        # now set also forward link ('spend_by_tx')
        for txid in data['coinjoins'].keys():
            for index in data['coinjoins'][txid]['inputs'].keys():
                input = data['coinjoins'][txid]['inputs'][index]
                if 'spending_tx' in input.keys():
                    tx, vout = extract_txid_from_inout_string(input['spending_tx'])
                    # Try to find transaction in mix (coinjoins) and set its record
                    if tx in data['premix'].keys() and vout in data['premix'][tx]['outputs'].keys():
                        data['premix'][tx]['outputs'][vout]['spend_by_tx'] = get_input_name_string(txid, index)

    return data


def load_coinjoins(target_path: str, mix_filename: str, postmix_filename: str, premix_filename: str,
                   start_date: str, stop_date: str) -> dict:
    # All mixes are having mixing coinjoins and postmix spends
    data = {'rounds': {}, 'filename': os.path.join(target_path, mix_filename),
            'coinjoins': load_coinjoin_stats_from_file(os.path.join(target_path, mix_filename), start_date, stop_date),
            'postmix': load_coinjoin_stats_from_file(os.path.join(target_path, postmix_filename), start_date, stop_date)}

    # Only Samourai Whirlpool is having premix tx (TX0)
    if premix_filename is not None:
        data['premix'] = load_coinjoin_stats_from_file(os.path.join(target_path, premix_filename), start_date, stop_date)
        for txid in list(data['premix'].keys()):
            if is_whirlpool_coinjoin_tx(data['premix'][txid]):
                # Misclassified mix transaction, move between groups
                data['coinjoins'][txid] = data['premix'][txid]
                data['premix'].pop(txid)
                logging.info(f'{txid} is mix transaction, removing from premix and putting to mix')

    # Set spending transactions also between mix and postmix
    data = compute_mix_postmix_link(data)

    data['wallets_info'], data['wallets_coins'] = extract_wallets_info(data)
    data['rounds'] = extract_rounds_info(data)

    return data


def propagate_cluster_name_for_all_inputs(cluster_name, postmix_txs, txid, mix_txs):
    # Set same cluster id for all inputs
    for input in postmix_txs[txid]['inputs']:
        set_key_value_assert(postmix_txs[txid]['inputs'][input], 'cluster_id', cluster_name, CLUSTER_ID_CHECK_HARD_ASSERT)
        # Set also for outputs connected to these inputs
        if 'spending_tx' in postmix_txs[txid]['inputs'][input]:
            tx, vout = extract_txid_from_inout_string(postmix_txs[txid]['inputs'][input]['spending_tx'])
            # Try to find transaction and set its record (postmix txs, coinjoin txs)
            if tx in postmix_txs.keys() and vout in postmix_txs[tx]['outputs'].keys():
                # This is suspicious, one premix propagates to another premix (maybe badbank merged into next TX0?)
                logging.warning(f'Potentially suspicious link between two premixes (badbank/peelchain?) from {postmix_txs[txid]['inputs'][input]['spending_tx']} to {get_input_name_string(txid, input)}')
                set_key_value_assert(postmix_txs[tx]['outputs'][vout], 'cluster_id', cluster_name, CLUSTER_ID_CHECK_HARD_ASSERT)
            if tx in mix_txs.keys() and vout in mix_txs[tx]['outputs'].keys():
                set_key_value_assert(mix_txs[tx]['outputs'][vout], 'cluster_id', cluster_name, CLUSTER_ID_CHECK_HARD_ASSERT)


def propagate_cluster_name_for_all_outputs(cluster_name, premix_txs, txid, mix_txs):
    # Set same cluster id for all outputs
    for output in premix_txs[txid]['outputs']:
        # Set for output
        set_key_value_assert(premix_txs[txid]['outputs'][output], 'cluster_id', cluster_name, CLUSTER_ID_CHECK_HARD_ASSERT)

        # set for inputs which are spending this output
        if 'spend_by_tx' in premix_txs[txid]['outputs'][output]:
            tx, vin = extract_txid_from_inout_string(premix_txs[txid]['outputs'][output]['spend_by_tx'])
            # Try to find transaction and set its record (premix txs, coinjoin txs)
            if tx in premix_txs.keys() and vin in premix_txs[tx]['inputs'].keys():
                # This is suspicious, one premix propagates to another premix
                # (maybe badbank/peelchain merged into next TX0?)
                logging.warning(f'Potentially suspicious link between two premixes (badbank/peelchain?) from {premix_txs[txid]['outputs'][output]['spend_by_tx']} to {get_output_name_string(txid, output)}')
                set_key_value_assert(premix_txs[tx]['inputs'][vin], 'cluster_id', cluster_name, CLUSTER_ID_CHECK_HARD_ASSERT)
            if tx in mix_txs.keys() and vin in mix_txs[tx]['inputs'].keys():
                set_key_value_assert(mix_txs[tx]['inputs'][vin], 'cluster_id', cluster_name, CLUSTER_ID_CHECK_HARD_ASSERT)


def analyze_postmix_spends(tx_dict: dict) -> dict:
    """
    Simple chain analysis heuristics:
    1. N:1 Merges (many inputs, single output)
    2. 1:1 Resend (one input, one output)
    :param tx_dict: input dict with transactions
    :return: updated dict with transactions
    """
    postmix_txs = tx_dict['postmix']
    mix_txs = tx_dict['coinjoins']

    # N:1 Merge (many inputs, one output), including # 1:1 Resend (one input, one output)
    cluster_name = 'unassigned'  # Index to use
    offset = CLUSTER_INDEX.get_current_index()   # starting offset of cluster index used to compute number of assigned indexes
    total_inputs_merged = 0
    for txid in postmix_txs.keys():
        if len(postmix_txs[txid]['outputs']) == 1:
            # Find or use existing cluster index
            if 'cluster_id' in postmix_txs[txid]['outputs']['0']:
                cluster_name = postmix_txs[txid]['outputs']['0']['cluster_id']
            else:
                # New cluster index
                cluster_name = f'c_{CLUSTER_INDEX.get_new_index()}'

            # Set output cluster id
            postmix_txs[txid]['outputs']['0']['cluster_id'] = cluster_name
            # Set same cluster id for all merged inputs
            propagate_cluster_name_for_all_inputs(cluster_name, postmix_txs, txid, mix_txs)
            # Count number of inputs merged
            total_inputs_merged += len(postmix_txs[txid]['inputs'])

    # Compute total number of inputs used in postmix spending
    total_inputs = sum([len(postmix_txs[txid]['inputs']) for txid in postmix_txs.keys()])
    SM.print(f'  {get_ratio_string(total_inputs_merged, total_inputs)} '
             f'N:1 postmix merges detected (merged inputs / all inputs)')
    SM.print(f'  {get_ratio_string(CLUSTER_INDEX.get_current_index() - offset, len(postmix_txs))} '
             f'N:1 unique postmix clusters detected (clusters / all postmix txs)')

    return tx_dict


def is_whirlpool_coinjoin_tx(premix_tx):
    # The transaction is whirlpool coinjoin transaction if number of inputs is bigger than 4
    if len(premix_tx['inputs']) >= 5:
        # ... number of inputs and outputs is the same
        if len(premix_tx['inputs']) == len(premix_tx['outputs']):
            # ... all outputs are the same value
            if all(premix_tx['outputs'][vout]['value'] == premix_tx['outputs']['0']['value']
                   for vout in premix_tx['outputs'].keys()):
                return True

    return False


def analyze_premix_spends(tx_dict: dict) -> dict:
    """
    Assign cluster information for outputs of Whirlpool's premix TX0
    1. N:M preparation of mix utxos (many inputs, many outputs), assume same user
    :param tx_dict: input dict with transactions
    :return: updated dict with transactions
    """
    if 'premix' not in tx_dict.keys():  # No analysis if premix not present
        return tx_dict

    premix_txs = tx_dict['premix']
    mix_txs = tx_dict['coinjoins']

    # N:M preparation of mix utxos
    offset = CLUSTER_INDEX.get_current_index()  # starting offset of cluster index used to compute number of assigned indexes
    for txid in premix_txs.keys():
        # Check if any of the premix inputs are labeled with cluster id. If yes, use it, generate new otherwise
        cluster_name = None
        for input in premix_txs[txid]['inputs']:
            if 'cluster_id' in premix_txs[txid]['inputs'][input]:
                cluster_name = premix_txs[txid]['inputs'][input]['cluster_id']
                break
        for output in premix_txs[txid]['outputs']:
            if 'cluster_id' in premix_txs[txid]['outputs'][output]:
                cluster_name = premix_txs[txid]['outputs'][output]['cluster_id']
                break
        if cluster_name is None:
            # New cluster index
            cluster_name = f'c_{CLUSTER_INDEX.get_new_index()}'

        # Set cluster id for all inputs (assuming same owner of premix tx inputs)
        for input in premix_txs[txid]['inputs']:
            set_key_value_assert(premix_txs[txid]['inputs'][input], 'cluster_id', cluster_name,
                                 CLUSTER_ID_CHECK_HARD_ASSERT)
        # Propagate to all outputs and spending inputs
        propagate_cluster_name_for_all_outputs(cluster_name, premix_txs, txid, mix_txs)

    # Compute total number of new premix clusters
    total_outputs = sum([len(premix_txs[txid]['outputs']) for txid in premix_txs.keys()])
    SM.print(f'  {get_ratio_string(CLUSTER_INDEX.get_current_index() - offset, total_outputs)} '
             f'N:M new premix clusters detected (number clusters / total outputs in premix)')

    return tx_dict


def analyze_coinjoin_blocks(data):
    same_block_coinjoins = defaultdict(list)
    for txid in data['coinjoins'].keys():
        same_block_coinjoins[data['coinjoins'][txid]['block_hash']].append(txid)
    filtered_dict = {key: value for key, value in same_block_coinjoins.items() if len(value) > 1}
    SM.print(f'  {get_ratio_string(len(filtered_dict), len(data['coinjoins']))} coinjoins in same block')


def visualize_coinjoins_in_time(data, ax_num_coinjoins):
    #
    # Number of coinjoins per given time interval (e.g., day)
    #
    coinjoins = data['coinjoins']
    SLOT_WIDTH_SECONDS = 3600 * 24 * 7
    broadcast_times = [datetime.strptime(coinjoins[item]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for item in
                       coinjoins.keys()]
    experiment_start_time = min(broadcast_times)
    slot_start_time = experiment_start_time
    slot_last_time = max(broadcast_times)
    diff_seconds = (slot_last_time - slot_start_time).total_seconds()
    num_slots = int(diff_seconds // SLOT_WIDTH_SECONDS)
    cjtx_in_hours = {hour: [] for hour in range(0, num_slots + 1)}
    rounds_started_in_hours = {hour: [] for hour in range(0, num_slots + 1)}
    for cjtx in coinjoins.keys():  # go over all coinjoin transactions
        timestamp = datetime.strptime(coinjoins[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")
        cjtx_hour = int((timestamp - slot_start_time).total_seconds() // SLOT_WIDTH_SECONDS)
        cjtx_in_hours[cjtx_hour].append(cjtx)
    # remove last slot(s) if no coinjoins are available there
    while cjtx_in_hours[len(cjtx_in_hours.keys()) - 1] == []:
        del cjtx_in_hours[len(cjtx_in_hours.keys()) - 1]
    ax_num_coinjoins.plot([len(cjtx_in_hours[cjtx_hour]) for cjtx_hour in cjtx_in_hours.keys()],
                      label='All coinjoins finished', color='green')
    ax_num_coinjoins.legend()
    x_ticks = []
    for slot in cjtx_in_hours.keys():
        x_ticks.append(
            (experiment_start_time + slot * timedelta(seconds=SLOT_WIDTH_SECONDS)).strftime("%Y-%m-%d"))
    ax_num_coinjoins.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    num_xticks = 30
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_xticks))
    ax_num_coinjoins.set_ylim(0)
    ax_num_coinjoins.set_ylabel('Number of coinjoin transactions')
    ax_num_coinjoins.set_title('Number of coinjoin transactions in given time period')


def visualize_liquidity_in_time(events, ax_number, ax_boxplot, ax_input_values_boxplot, ax_output_values_boxplot,
                                ax_input_values_bar, ax_output_values_bar, ax_burn_time, legend_labels: list):
    #
    # Number of coinjoins per given time interval (e.g., day)
    #
    coinjoins = events
    SLOT_WIDTH_SECONDS = 600 * 24 * 7
    broadcast_times_cjtxs = {item: datetime.strptime(coinjoins[item]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for item in
                       coinjoins.keys()}
    broadcast_times = list(broadcast_times_cjtxs.values())
    experiment_start_time = min(broadcast_times)
    slot_start_time = experiment_start_time
    slot_last_time = max(broadcast_times)
    diff_seconds = (slot_last_time - slot_start_time).total_seconds()
    num_slots = int(diff_seconds // SLOT_WIDTH_SECONDS)
    inputs_cjtx_in_slot = {hour: [] for hour in range(0, num_slots + 1)}
    outputs_cjtx_in_slot = {hour: [] for hour in range(0, num_slots + 1)}
    inputs_remixed_cjtx_in_slot = {hour: [] for hour in range(0, num_slots + 1)}
    inputs_values_in_slot = {hour: [] for hour in range(0, num_slots + 1)}
    outputs_values_in_slot = {hour: [] for hour in range(0, num_slots + 1)}
    inputs_burned_time_in_slot = {hour: [] for hour in range(0, num_slots + 1)}
    for cjtx in coinjoins.keys():  # go over all coinjoin transactions
        timestamp = datetime.strptime(coinjoins[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")
        cjtx_hour = int((timestamp - slot_start_time).total_seconds() // SLOT_WIDTH_SECONDS)
        inputs_cjtx_in_slot[cjtx_hour].append(len(coinjoins[cjtx]['inputs']))
        outputs_cjtx_in_slot[cjtx_hour].append(len(coinjoins[cjtx]['outputs']))
        inputs_remixed_cjtx_in_slot[cjtx_hour].append(len([coinjoins[cjtx]['inputs'][index] for index in coinjoins[cjtx]['inputs'].keys()
                                                           if coinjoins[cjtx]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name]))
        inputs_values_in_slot[cjtx_hour].extend([coinjoins[cjtx]['inputs'][index]['value'] for index in coinjoins[cjtx]['inputs'].keys()])
        outputs_values_in_slot[cjtx_hour].extend([coinjoins[cjtx]['outputs'][index]['value'] for index in coinjoins[cjtx]['outputs'].keys()])

        # Extract difference in time between output creation and destruction in this cjtx
        if ax_burn_time:
            destruct_time = timestamp
            for index in coinjoins[cjtx]['inputs'].keys():
                if 'spending_tx' in coinjoins[cjtx]['inputs'][index].keys():
                    txid, vout = extract_txid_from_inout_string(coinjoins[cjtx]['inputs'][index]['spending_tx'])
                    if txid in broadcast_times_cjtxs.keys():
                        create_time = broadcast_times_cjtxs[txid]
                        time_diff = destruct_time - create_time
                        hours_diff = time_diff.total_seconds() / 3600
                        inputs_burned_time_in_slot[cjtx_hour].append(hours_diff)

    # remove last slot(s) if no coinjoins are available there
    while inputs_cjtx_in_slot[len(inputs_cjtx_in_slot.keys()) - 1] == []:
        del inputs_cjtx_in_slot[len(inputs_cjtx_in_slot.keys()) - 1]
    while outputs_cjtx_in_slot[len(outputs_cjtx_in_slot.keys()) - 1] == []:
        del outputs_cjtx_in_slot[len(outputs_cjtx_in_slot.keys()) - 1]
    while inputs_values_in_slot[len(inputs_values_in_slot.keys()) - 1] == []:
        del inputs_values_in_slot[len(inputs_values_in_slot.keys()) - 1]
    while outputs_cjtx_in_slot[len(outputs_cjtx_in_slot.keys()) - 1] == []:
        del outputs_values_in_slot[len(outputs_values_in_slot.keys()) - 1]

    ax_number.plot(range(0, len(inputs_cjtx_in_slot)), [sum(inputs_cjtx_in_slot[cjtx_hour]) for cjtx_hour in inputs_cjtx_in_slot.keys()],
                   label=legend_labels[0], alpha=0.5)
    ax_number.plot(range(0, len(outputs_cjtx_in_slot)), [sum(outputs_cjtx_in_slot[cjtx_hour]) for cjtx_hour in outputs_cjtx_in_slot.keys()],
                   label=legend_labels[1], alpha=0.5)
    ax_number.plot(range(0, len(inputs_remixed_cjtx_in_slot)), [sum(inputs_remixed_cjtx_in_slot[cjtx_hour]) for cjtx_hour in inputs_remixed_cjtx_in_slot.keys()],
                   label=legend_labels[2], alpha=0.5, linestyle='-.')

    # Create a boxplot
    data = [series[1] for series in inputs_cjtx_in_slot.items()]
    ax_boxplot.boxplot(data)
    # data = [series[1] for series in outputs_cjtx_in_slot.items()]
    # ax_boxplot.boxplot(data)
    data = [series[1] for series in inputs_values_in_slot.items()]
    ax_input_values_boxplot.boxplot(data)
    data = [series[1] for series in outputs_values_in_slot.items()]
    ax_output_values_boxplot.boxplot(data)
    if ax_burn_time:
        data = [series[1] for series in inputs_burned_time_in_slot.items()]
        ax_burn_time.boxplot(data)
        ax_burn_time.set_yscale('log')

    # Plot distribution of input values (bar height corresponding to number of occurences)
    if ax_input_values_bar:
        flat_data = [item for index in inputs_values_in_slot.keys() for item in inputs_values_in_slot[index]]
        input_values_count = Counter(flat_data)
        sorted_input_values_count = sorted(input_values_count.items(), key=lambda x: x[0])
        sorted_values, counts = zip(*sorted_input_values_count)
        #ax_input_values_bar.bar(sorted_values, counts)

        log_values = np.log(flat_data)
        ax_input_values_bar.hist(log_values, 100)
        ax_input_values_bar.set_xscale('log')

        #ax_input_values_bar.set_xscale('linear')
        #ax_input_values_bar.set_xticks(np.linspace(min(log_values), max(log_values), 30))
        #ax_input_values_bar.set_xticks(np.arange(min(log_values), max(log_values) + 1, 100))

    # Plot distribution of output values (bar height corresponding to number of occurences)
    if ax_output_values_bar:
        flat_data = [item for index in outputs_values_in_slot.keys() for item in outputs_values_in_slot[index]]
        output_values_count = Counter(flat_data)
        sorted_input_values_count = sorted(output_values_count.items(), key=lambda x: x[0])
        sorted_values, counts = zip(*sorted_input_values_count)
        #ax_output_values_bar.bar(sorted_values, counts)

        log_values = np.log(flat_data)
        ax_output_values_bar.hist(log_values, 100)
        ax_output_values_bar.set_xscale('log')
        # ax_output_values_bar.set_xscale('linear')
        #ax_output_values_bar.set_xticks(np.linspace(min(flat_data), max(flat_data), 30))

    # if ax_burn_time:
    #     flat_data = [item for index in inputs_burned_time_in_slot.keys() for item in inputs_burned_time_in_slot[index]]
    #     ax_burn_time.bar(flat_data)

    ax_number.legend()
    ax_boxplot.legend()
    x_ticks = []
    for slot in inputs_cjtx_in_slot.keys():
        time_delta_format = "%Y-%m-%d %H:%M:%S" if SLOT_WIDTH_SECONDS < 600 * 24 else "%Y-%m-%d"
        x_ticks.append(
            (experiment_start_time + slot * timedelta(seconds=SLOT_WIDTH_SECONDS)).strftime(time_delta_format))
    ax_number.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    ax_boxplot.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    ax_input_values_boxplot.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    ax_output_values_boxplot.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    if ax_burn_time:
        ax_burn_time.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    num_xticks = 30
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_xticks))
    ax_number.set_ylim(0)
    ax_boxplot.set_ylim(0)
    if ax_input_values_bar:
        ax_input_values_bar.set_xlim(0)
    if ax_output_values_bar:
        ax_output_values_bar.set_xlim(0)
    if ax_burn_time:
        ax_burn_time.set_ylim(0)


def visualize_coinjoins(data, events, base_path, experiment_name):
    fig = plt.figure(figsize=(30, 20))
    ax_num_coinjoins = fig.add_subplot(4, 3, 1)
    ax_inputs_outputs = fig.add_subplot(4, 3, 2)
    ax_liquidity = fig.add_subplot(4, 3, 3)
    ax_input_time_to_burn = fig.add_subplot(4, 3, 4)
    ax_inputs_outputs_boxplot = fig.add_subplot(4, 3, 5)
    ax_liquidity_boxplot = fig.add_subplot(4, 3, 6)
    ax_inputs_value_bar = fig.add_subplot(4, 3, 7)
    ax_inputs_value_boxplot = fig.add_subplot(4, 3, 8)
    ax_fresh_inputs_value_boxplot = fig.add_subplot(4, 3, 9)
    ax_outputs_value_bar = fig.add_subplot(4, 3, 10)
    ax_outputs_value_boxplot = fig.add_subplot(4, 3, 11)
    ax_fresh_outputs_value_boxplot = fig.add_subplot(4, 3, 12)

    # Coinjoins in time
    visualize_coinjoins_in_time(data, ax_num_coinjoins)

    # All inputs and outputs
    visualize_liquidity_in_time(data['coinjoins'], ax_inputs_outputs, ax_inputs_outputs_boxplot, ax_inputs_value_boxplot,
                                ax_outputs_value_boxplot, None, None, ax_input_time_to_burn, ['all inputs', 'all outputs', 'remixed inputs'])
    ax_inputs_outputs.set_ylabel('Number of inputs / outputs')
    ax_inputs_outputs.set_title('Number of all inputs and outputs in cjtx')
    ax_inputs_outputs_boxplot.set_ylabel('Number of inputs / outputs')
    ax_inputs_outputs_boxplot.set_title('Distribution of inputs of single coinjoins')

    ax_inputs_value_boxplot.set_ylabel('Value of inputs (sats)')
    ax_inputs_value_boxplot.set_yscale('log')
    ax_inputs_value_boxplot.set_title('Distribution of value of inputs (log scale)')
    ax_outputs_value_boxplot.set_ylabel('Value of outputs (sats)')
    ax_outputs_value_boxplot.set_yscale('log')
    ax_outputs_value_boxplot.set_title('Distribution of value of outputs (log scale)')

    ax_input_time_to_burn.set_ylabel('Input coin burn time (hours)')
    ax_input_time_to_burn.set_yscale('log')
    ax_input_time_to_burn.set_title('Distribution of coin burn times (log scale)')

    # Fresh liquidity in/out of mix
    visualize_liquidity_in_time(events, ax_liquidity, ax_liquidity_boxplot, ax_fresh_inputs_value_boxplot,
                                ax_fresh_outputs_value_boxplot, ax_inputs_value_bar, ax_outputs_value_bar,
                                None, ['fresh inputs mixed', 'outputs leaving mix', ''])
    ax_liquidity.set_ylabel('Number of new inputs / outputs')
    ax_liquidity.set_title('Number of fresh liquidity in and out of cjtx')
    ax_liquidity_boxplot.set_ylabel('Number of new inputs / outputs')
    ax_liquidity_boxplot.set_title('Distribution of fresh liquidity inputs of single coinjoins')
    ax_fresh_inputs_value_boxplot.set_ylabel('Value of inputs (sats)')
    ax_fresh_inputs_value_boxplot.set_yscale('log')
    ax_fresh_inputs_value_boxplot.set_title('Distribution of value of fresh inputs (log scale)')
    ax_fresh_outputs_value_boxplot.set_ylabel('Value of outputs (sats)')
    ax_fresh_outputs_value_boxplot.set_yscale('log')
    ax_fresh_outputs_value_boxplot.set_title('Distribution of value of fresh outputs (log scale)')

    ax_inputs_value_bar.set_ylabel('Number of inputs')
    ax_inputs_value_bar.set_title('Histogram of frequencies of specific values of fresh inputs.\n(x is log scale)')
    ax_outputs_value_bar.set_ylabel('Number of outputs')
    ax_outputs_value_bar.set_title('Histogram of frequencies of specific values of fresh outputs.\n(x is log scale)')

    # TODO: Add distribution of time-to-burn for remixed utxos
    # TODO: Add detection of any non-standard output values for WW2 and WW1

    # save graph
    plt.suptitle('{}'.format(experiment_name), fontsize=16)  # Adjust the fontsize and y position as needed
    plt.subplots_adjust(bottom=0.1, wspace=0.5, hspace=0.5)
    save_file = os.path.join(base_path, f'{experiment_name}_coinjoin_stats.png')
    plt.savefig(save_file, dpi=300)
    plt.close()
    print('Basic coinjoins statistics saved into {}'.format(save_file))


def process_coinjoins(target_path, mix_filename, postmix_filename, premix_filename, start_date: str, stop_date: str):
    data = load_coinjoins(target_path, mix_filename, postmix_filename, premix_filename, start_date, stop_date)
    if len(data['coinjoins']) == 0:
        return data

    SM.print('*******************************************')
    SM.print(f'{mix_filename} coinjoins: {len(data['coinjoins'])}')
    min_date = min([data['coinjoins'][txid]['broadcast_time'] for txid in data['coinjoins'].keys()])
    max_date = max([data['coinjoins'][txid]['broadcast_time'] for txid in data['coinjoins'].keys()])
    SM.print(f'Dates from {min_date} to {max_date}')

    SM.print('### Simple chain analysis')
    analyze_input_out_liquidity(data['coinjoins'], data['postmix'])
    analyze_postmix_spends(data)
    analyze_premix_spends(data)
    analyze_coinjoin_blocks(data)

    return data


def filter_liquidity_events(data):
    events = {}
    for txid in data['coinjoins']:
        events[txid] = copy.deepcopy(data['coinjoins'][txid])
        events[txid].pop('block_hash')
        # Process inputs
        events[txid]['num_inputs'] = len(events[txid]['inputs'])
        for input in list(events[txid]['inputs'].keys()):
            if ('mix_event_type' not in events[txid]['inputs'][input]
                    or events[txid]['inputs'][input]['mix_event_type'] not in [MIX_EVENT_TYPE.MIX_ENTER.name, MIX_EVENT_TYPE.MIX_LEAVE.name]):
                # Remove whole given input
                events[txid]['inputs'].pop(input)
            else:
                # Remove all unnecessary data
                for item in events[txid]['inputs'][input].copy():
                    if item not in ['value', 'wallet_name', 'mix_event_type']:
                        events[txid]['inputs'][input].pop(item)
        # Process outputs
        events[txid]['num_outputs'] = len(events[txid]['outputs'])
        for output in list(events[txid]['outputs'].keys()):
            if ('mix_event_type' not in events[txid]['outputs'][output]
                    or events[txid]['outputs'][output]['mix_event_type'] not in [MIX_EVENT_TYPE.MIX_ENTER.name, MIX_EVENT_TYPE.MIX_LEAVE.name]):
                # Remove whole given output
                events[txid]['outputs'].pop(output)
            else:
                # Remove all unnecessary data
                for item in events[txid]['outputs'][output].copy():
                    if item not in ['value', 'wallet_name', 'mix_event_type']:
                        events[txid]['outputs'][output].pop(item)
    return events


def process_and_save_coinjoins(mix_id: str, target_path: os.path, mix_filename: str, postmix_filename: str,
                               premix_filename: str, start_date: str, stop_date: str, target_save_path: os.path=None):
    if not target_save_path:
        target_save_path = target_path
    # Process and save full conjoin information
    data = process_coinjoins(target_path, mix_filename, postmix_filename, premix_filename, start_date, stop_date)

    if SAVE_BASE_FILES_JSON:
        with open(os.path.join(target_save_path, f'coinjoin_tx_info.json'), "w") as file:
            file.write(json.dumps(dict(sorted(data.items())), indent=4))

    # Filter only liquidity-relevant events to maintain smaller file
    events = filter_liquidity_events(data)
    with open(os.path.join(target_save_path, f'{mix_id}_events.json'), "w") as file:
        file.write(json.dumps(dict(sorted(events.items())), indent=4))

    # Visualize coinjoins
    if len(data['coinjoins']) > 0:
        visualize_coinjoins(data, events, target_save_path, mix_filename)

    return data


def process_and_save_intervals_onload(mix_id: str, target_path: os.path, start_date: str, stop_date: str, mix_filename: str,
                                      postmix_filename: str, premix_filename: str=None):

    # Create directory structure with files split per month (around 1000 subsequent coinjoins)

    # Find first day of a month when first coinjoin ocured
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S.%f")
    start_date = datetime(start_date_obj.year, start_date_obj.month, 1)

    # Month After the last coinjoin occured
    last_date_obj = datetime.strptime(stop_date, "%Y-%m-%d %H:%M:%S.%f")
    last_date_obj = last_date_obj + timedelta(days=32)
    last_date_str = last_date_obj.strftime("%Y-%m-%d %H:%M:%S")

    # Previously used stop date (will become start date for next interval)
    last_stop_date = start_date
    last_stop_date_str = last_stop_date.strftime("%Y-%m-%d %H:%M:%S")

    # Current stop date
    current_stop_date = start_date + timedelta(days=32)
    current_stop_date = datetime(current_stop_date.year, current_stop_date.month, 1)
    current_stop_date_str = current_stop_date.strftime("%Y-%m-%d %H:%M:%S")

    while current_stop_date_str <= last_date_str:
        print(f'Processing interval {last_stop_date_str} - {current_stop_date_str}')
        interval_path = os.path.join(target_path, f'{last_stop_date_str.replace(':', '-')}--{current_stop_date_str.replace(':', '-')}_unknown-static-100-1utxo')
        if not os.path.exists(interval_path):
            os.makedirs(interval_path.replace('\\', '/'))
            os.makedirs(os.path.join(interval_path, 'data').replace('\\', '/'))
        process_and_save_coinjoins(mix_id, target_path, mix_filename, postmix_filename, premix_filename,
                                          last_stop_date_str, current_stop_date_str, interval_path)
        # Move to the next month
        last_stop_date_str = current_stop_date_str

        current_stop_date = current_stop_date + timedelta(days=32)
        current_stop_date = datetime(current_stop_date.year, current_stop_date.month, 1)
        current_stop_date_str = current_stop_date.strftime("%Y-%m-%d %H:%M:%S")


def process_and_save_intervals_filter(mix_id: str, target_path: os.path, start_date: str, stop_date: str, mix_filename: str,
                                      postmix_filename: str, premix_filename: str=None, save_base_files=True):
    # Create directory structure with files split per month (around 1000 subsequent coinjoins)
    # Load all coinjoins first, then filter based on intervals
    SAVE_BASE_FILES_JSON = False
    data = process_and_save_coinjoins(mix_id, target_path, mix_filename, postmix_filename, premix_filename, None, None)
    SAVE_BASE_FILES_JSON = save_base_files

    # Find first day of a month when first coinjoin ocured
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S.%f")
    start_date = datetime(start_date_obj.year, start_date_obj.month, 1)

    # Month After the last coinjoin occured
    last_date_obj = datetime.strptime(stop_date, "%Y-%m-%d %H:%M:%S.%f")
    last_date_obj = last_date_obj + timedelta(days=32)
    last_date_str = last_date_obj.strftime("%Y-%m-%d %H:%M:%S")

    # Previously used stop date (will become start date for next interval)
    last_stop_date = start_date
    last_stop_date_str = last_stop_date.strftime("%Y-%m-%d %H:%M:%S")

    # Current stop date
    current_stop_date = start_date + timedelta(days=32)
    current_stop_date = datetime(current_stop_date.year, current_stop_date.month, 1)
    current_stop_date_str = current_stop_date.strftime("%Y-%m-%d %H:%M:%S")

    while current_stop_date_str <= last_date_str:
        print(f'Processing interval {last_stop_date_str} - {current_stop_date_str}')

        # Create folder structure compatible with ww2 coinjoin simulation for further processing
        interval_path = os.path.join(target_path, mix_id, f'{last_stop_date_str.replace(':', '-')}--{current_stop_date_str.replace(':', '-')}_unknown-static-100-1utxo')
        if not os.path.exists(interval_path):
            os.makedirs(interval_path.replace('\\', '/'))
            os.makedirs(os.path.join(interval_path, 'data').replace('\\', '/'))

        # Filter only data relevant for given interval and save
        interval_data = {}
        interval_data['coinjoins'] = {txid: data['coinjoins'][txid] for txid in data['coinjoins'].keys()
                                      if last_stop_date_str < data['coinjoins'][txid]['broadcast_time'] < current_stop_date_str}
        interval_data['postmix'] = {}
        interval_data['rounds'] = {roundid: data['rounds'][roundid] for roundid in data['rounds'].keys()
                                   if last_stop_date_str < data['rounds'][roundid]['round_start_timestamp'] < current_stop_date_str}
        interval_data['wallets_coins'] = {wallet_name: [] for wallet_name in data['wallets_coins'].keys()}
        for wallet_name in data['wallets_coins'].keys():
            interval_data['wallets_coins'][wallet_name] = [coin for coin in data['wallets_coins'][wallet_name]
                    if last_stop_date_str < coin['create_time'] < current_stop_date_str
                    or 'destroy_time' in coin.keys() and last_stop_date_str < coin['destroy_time'] < current_stop_date_str]
        interval_data['wallets_info'] = data['wallets_info']

        with open(os.path.join(interval_path, f'coinjoin_tx_info.json'), "w") as file:
            file.write(json.dumps(dict(sorted(interval_data.items())), indent=4))
        # Filter only liquidity-relevant events to maintain smaller file
        events = filter_liquidity_events(interval_data)
        with open(os.path.join(interval_path, f'{mix_id}_events.json'), "w") as file:
            file.write(json.dumps(dict(sorted(events.items())), indent=4))

        # Visualize coinjoins
        if len(interval_data['coinjoins']) > 0:
            visualize_coinjoins(interval_data, events, interval_path, os.path.basename(interval_path))

        # Move to the next month
        last_stop_date_str = current_stop_date_str

        current_stop_date = current_stop_date + timedelta(days=32)
        current_stop_date = datetime(current_stop_date.year, current_stop_date.month, 1)
        current_stop_date_str = current_stop_date.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    FULL_TX_SET = False

    #target_base_path = 'c:\\!blockchains\\CoinJoin\\Dumplings_Stats_20231113\\'
    target_base_path = 'c:\\!blockchains\\CoinJoin\\Dumplings_Stats_20240215\\'
    target_path = os.path.join(target_base_path, 'Scanner')
    SM.print(f'Starting analysis of {target_path}, FULL_TX_SET={FULL_TX_SET}, SAVE_BASE_FILES_JSON={SAVE_BASE_FILES_JSON}')

    # process_and_save_intervals('wasabi2', target_path, '2022-06-18 01:38:07.000', '2024-02-15 01:38:07.000',
    #                            'Wasabi2CoinJoins.txt', 'Wasabi2PostMixTxs.txt')

    process_and_save_intervals_filter('wasabi2_test', target_path, '2023-12-01 01:38:07.000', '2024-02-15 01:38:07.000',
                               'wasabi2_mix_test.txt', 'wasabi2_postmix_test.txt', None, SAVE_BASE_FILES_JSON)

    # process_and_save_intervals_filter('wasabi2', target_path, '2022-06-18 01:38:07.000', '2024-02-15 01:38:07.000',
    #                            'Wasabi2CoinJoins.txt', 'Wasabi2PostMixTxs.txt', None, SAVE_BASE_FILES_JSON)
    #
    # process_and_save_intervals_filter('wasabi1', target_path, '2018-07-19 01:38:07.000', '2024-02-15 01:38:07.000',
    #                            'WasabiCoinJoins.txt', 'WasabiPostMixTxs.txt', None, SAVE_BASE_FILES_JSON)
    #
    # process_and_save_intervals_filter('whirlpool', target_path, '2019-04-17 01:38:07.000', '2024-02-15 01:38:07.000',
    #                            'SamouraiCoinJoins.txt', 'SamouraiPostMixTxs.txt', 'SamouraiTx0s.txt', SAVE_BASE_FILES_JSON)



    exit(1)

    if FULL_TX_SET:
        # All transactions
        process_and_save_coinjoins('whirlpool', target_path, 'SamouraiCoinJoins.txt', 'SamouraiPostMixTxs.txt', 'SamouraiTx0s.txt')
        process_and_save_coinjoins('wasabi', target_path, 'WasabiCoinJoins.txt', 'WasabiPostMixTxs.txt')
        process_and_save_coinjoins('wasabi2', target_path, 'Wasabi2CoinJoins.txt', 'Wasabi2PostMixTxs.txt')
    else:
        # Smaller set for debugging
        # process_and_save_coinjoins('whirlpool_test', target_path, True, 'sam_mix_test.txt', 'sam_postmix_test.txt', 'sam_premix_test.txt')
        # process_and_save_coinjoins('wasabi_test', target_path, True, 'wasabi_mix_test.txt', 'wasabi_postmix_test.txt')
        data = process_and_save_coinjoins('wasabi2_test', target_path,'wasabi2_mix_test.txt',
                                          'wasabi2_postmix_test.txt', None, None, None)
        first_coinjoin_date_str = min([data['coinjoins'][cjtx]['broadcast_time'] for cjtx in data['coinjoins'].keys()])
        last_coinjoin_date_str = max([data['coinjoins'][cjtx]['broadcast_time'] for cjtx in data['coinjoins'].keys()])
        process_and_save_intervals('wasabi2_test', target_path, first_coinjoin_date_str, last_coinjoin_date_str,
                                   'wasabi2_mix_test.txt', 'wasabi2_postmix_test.txt')

        # 2022-06-18 01:38:07.000
        # 2024-02-15 01:38:07.000


    print('### SUMMARY #############################')
    SM.print_summary()
    print('### END SUMMARY #########################')
