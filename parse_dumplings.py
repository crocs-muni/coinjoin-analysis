import copy
import os
import pickle
import random
import shutil
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import seaborn as sns
from scipy import stats
import logging
from cj_analysis import MIX_EVENT_TYPE, get_output_name_string, get_input_name_string
from cj_analysis import MIX_PROTOCOL
from cj_analysis import precomp_datetime
import cj_analysis as als
import mpl_toolkits.axisartist as AA
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import orjson

# Configure the logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_to_disable = logging.getLogger("mathplotlib")
logger_to_disable.setLevel(logging.WARNING)

VerboseTransactionInfoLineSeparator = ':::'
VerboseInOutInfoInLineSeparator = '}'
SATS_IN_BTC = 100000000

# If True, difference between assigned and existing cluster id is checked and failed upon if different
# If False, only warning is printed, but execution continues.
# TODO: Systematic solution requires merging and resolving different cluster ids
CLUSTER_ID_CHECK_HARD_ASSERT = False

SAVE_BASE_FILES_JSON = True

# SLOT_WIDTH_SECONDS = 3600 * 24 * 7  # week
#SLOT_WIDTH_SECONDS = 3600 * 24  # day
SLOT_WIDTH_SECONDS = 3600   # hour


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


SM = als.SummaryMessages()
als.SM = SM


class CoinMixInfo:
    num_coins = 0
    num_mixes = 0
    pool_size = -1

    def clear(self):
        self.num_coins = 0
        self.num_mixes = 0
        self.pool_size = -1


class CoinJoinStats:
    pool_100k = CoinMixInfo()
    pool_1M = CoinMixInfo()
    pool_5M = CoinMixInfo()
    no_pool = CoinMixInfo()
    cj_type = MIX_PROTOCOL.UNSET

    def clear(self):
        self.pool_100k.clear()
        self.pool_100k.pool_size = 100000
        self.pool_1M.clear()
        self.pool_1M.pool_size = 1000000
        self.pool_5M.clear()
        self.pool_5M.pool_size = 5000000
        self.no_pool.clear()

        self.cj_type = MIX_PROTOCOL.UNSET

WHIRLPOOL_FUNDING_TXS = {}
WHIRLPOOL_FUNDING_TXS[100000] = {'start_date': '2021-03-05 23:50:59.000', 'funding_txs': ['ac9566a240a5e037471b1a58ea50206062c13e1a75c0c2de3f21c7053573330a']}
WHIRLPOOL_FUNDING_TXS[1000000] = {'start_date': '2019-05-23 20:54:27.000', 'funding_txs': ['c6c27bef217583cca5f89de86e0cd7d8b546844f800da91d91a74039c3b40fba', 'a42596825352055841949a8270eda6fb37566a8780b2aec6b49d8035955d060e', '4c906f897467c7ed8690576edfcaf8b1fb516d154ef6506a2c4cab2c48821728']}
WHIRLPOOL_FUNDING_TXS[5000000] = {'start_date': '2019-04-17 16:20:09.000', 'funding_txs': ['a554db794560458c102bab0af99773883df13bc66ad287c29610ad9bac138926', '792c0bfde7f6bf023ff239660fb876315826a0a52fd32e78ea732057789b2be0', '94b0da89431d8bd74f1134d8152ed1c7c4f83375e63bc79f19cf293800a83f52', 'e04e5a5932e8d42e4ef641c836c6d08d9f0fff58ab4527ca788485a3fceb2416']}
WHIRLPOOL_FUNDING_TXS[50000000] = {'start_date': '2019-08-02 17:45:23.000', 'funding_txs': ['b42df707a3d876b24a22b0199e18dc39aba2eafa6dbeaaf9dd23d925bb379c59']}

WASABI2_FUNDING_TXS = {}
WASABI2_FUNDING_TXS['zksnacks'] = {'start_date': '2022-06-18 01:38:00.000', 'funding_txs': ['d31c2b4d71eb143b23bb87919dda7fdfecee337ffa1468d1c431ece37698f918']}
WASABI2_FUNDING_TXS['kruw.io'] = {'start_date': '2024-05-31 10:33:00.000', 'funding_txs': ['8c28787b4f5014cab20be23204c1889191dacd2e4622e1250ca16dba78cf915e', 'f861aa534a5efe7212a0c1bdb61f7a581b0d262452a79e807afaa2d20d73c8f5', 'b5e839299bfc0e50ed6b6b6c932a38b544d9bb6541cd0ab0b8ddcc44255bfb78']}



def set_key_value_assert(data, key, value, hard_assert):
    if key in data:
        if hard_assert:
            assert data[key] == value, f"Key '{key}' already exists with a different value {data[key]} vs. {value}."
        else:
            if data[key] != value:
                logging.warning(f"Key '{key}' already exists with a different value {data[key]} vs. {value}.")

    else:
        data[key] = value


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
    json_file = target_file + '.json'
    if os.path.exists(json_file):
        with open(json_file, "rb") as file:
            cj_stats = pickle.load(file)
    else:
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
                record['is_cjtx'] = True
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
                    this_input['script'] = segments[3]
                    this_input['script_type'] = segments[4]
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

                # Add this record as coinjoin
                cj_stats[tx_id] = record

        # backward reference to spending transaction output is already set ('spending_tx'),
        # now set also forward link ('spend_by_tx')
        update_spend_by_reference(cj_stats, cj_stats)

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


def extract_rounds_info(data):
    rounds_info = {}
    txs_data = data['coinjoins']
    for cjtxid in txs_data.keys():
        # Create basic round info from coinjoin data
        rounds_info[cjtxid] = {"cj_tx_id": cjtxid, "round_start_time": txs_data[cjtxid]['broadcast_time'],
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
    update_spend_by_reference(data['postmix'], data['coinjoins'])

    if 'premix' in data.keys():
        # backward reference from coinjoin to premix is already set ('spending_tx')
        # now set also forward link ('spend_by_tx')
        update_spend_by_reference(data['coinjoins'], data['premix'])

    return data


def filter_false_coinjoins(data, mix_protocol):
    false_cjtxs = {}
    cjtxids = list(data['coinjoins'].keys())
    for cjtx in cjtxids:
        if mix_protocol == MIX_PROTOCOL.WHIRLPOOL:
            if not is_whirlpool_coinjoin_tx(data['coinjoins'][cjtx]):
                logging.info(f'{cjtx} is not whirlpool coinjoin, removing from coinjoin list')
                false_cjtxs[cjtx] = data['coinjoins'][cjtx]
                data['coinjoins'].pop(cjtx)

        #if cj_type == MIX_PROTOCOL.WASABI2:
            # Not WW2 coinjoin if:
            # P2SH, P2PKH addresses,
            # large number of repeated same addresses
            # large number of non-standard denominations for outputs
            # if len(data['coinjoins'][cjtx]['inputs']) < 100:
            #     print(f'{cjtx} is false coinjoin, removing....')
            #     data['coinjoins'].pop(cjtx)

    return data, false_cjtxs


def update_spend_by_reference(updating: dict, updated: dict):
    updating_keys = updating.keys()  # Create copy for case when updating == updated
    total_updated = 0
    for txid in updating_keys:  # 'coinjoin' by 'coinjoin'
        for index in updating[txid]['inputs'].keys():
            input = updating[txid]['inputs'][index]

            if 'spending_tx' in input.keys():
                tx, vout = als.extract_txid_from_inout_string(input['spending_tx'])
                # Try to find transaction and set its record
                if tx in updated.keys() and vout in updated[tx]['outputs'].keys():
                    updated[tx]['outputs'][vout]['spend_by_tx'] = get_input_name_string(txid, index)
                    total_updated += 1

    return total_updated


def update_all_spend_by_reference(data: dict):
    # backward reference to spending transaction output is already set ('spending_tx'),
    # now set also forward link ('spend_by_tx')

    # Update 'premix' based on 'coinjoin' tx 'spending_tx'
    total_updated = update_spend_by_reference(data['coinjoins'], data['premix'])
    logging.debug(f'Update premix based on coinjoins: {total_updated}')
    # Update 'coinjoin' based on 'coinjoin'
    total_updated = update_spend_by_reference(data['coinjoins'], data['coinjoins'])
    logging.debug(f'Update coinjoins based on coinjoins: {total_updated}')
    # Update 'coinjoin' based on 'postmix'
    total_updated = update_spend_by_reference(data['coinjoins'], data['postmix'])
    logging.debug(f'Update coinjoins based on postmix: {total_updated}')

    return data


def load_coinjoins(target_path: str, mix_protocol: MIX_PROTOCOL, mix_filename: str, postmix_filename: str, premix_filename: str,
                   start_date: str, stop_date: str) -> dict:
    # All mixes are having mixing coinjoins and postmix spends
    data = {'rounds': {}, 'filename': os.path.join(target_path, mix_filename),
            'coinjoins': load_coinjoin_stats_from_file(os.path.join(target_path, mix_filename), start_date, stop_date),
            'postmix': load_coinjoin_stats_from_file(os.path.join(target_path, postmix_filename), start_date, stop_date)}

    # Only Samourai Whirlpool is having premix tx (TX0)
    cjtxs_fixed = 0
    if mix_protocol == MIX_PROTOCOL.WHIRLPOOL:
        data['premix'] = load_coinjoin_stats_from_file(os.path.join(target_path, premix_filename), start_date, stop_date)
        for txid in list(data['premix'].keys()):
            if is_whirlpool_coinjoin_tx(data['premix'][txid]):
                # Misclassified mix transaction, move between groups
                data['coinjoins'][txid] = data['premix'][txid]
                data['premix'].pop(txid)
                logging.info(f'{txid} is mix transaction, removing from premix and putting to mix')
                cjtxs_fixed += 1
    else:
        data['premix'] = {}
    logging.info(f'{cjtxs_fixed} total premix txs moved into coinjoins')

    # Detect misclassified Whirlpool coinjoin transactions found in Dumpling's postmix txs
    cjtxs_fixed = 0
    if mix_protocol == MIX_PROTOCOL.WHIRLPOOL:
        for txid in list(data['postmix'].keys()):
            if is_whirlpool_coinjoin_tx(data['postmix'][txid]):
                # Misclassified mix transaction, move between groups
                data['coinjoins'][txid] = data['postmix'][txid]
                data['postmix'].pop(txid)
                logging.info(f'{txid} is mix transaction, removing from postmix and putting to mix')
                cjtxs_fixed += 1
    logging.info(f'{cjtxs_fixed} total postmix txs moved into coinjoins')

    # Filter mistakes in Dumplings analysis of coinjoins
    data, false_cjtxs = filter_false_coinjoins(data, mix_protocol)

    data = update_all_spend_by_reference(data)

    # Set spending transactions also between mix and postmix
    data = compute_mix_postmix_link(data)

    data['wallets_info'], data['wallets_coins'] = als.extract_wallets_info(data)
    data['rounds'] = extract_rounds_info(data)

    return data, false_cjtxs


def propagate_cluster_name_for_all_inputs(cluster_name, postmix_txs, txid, mix_txs):
    # Set same cluster id for all inputs
    for input in postmix_txs[txid]['inputs']:
        set_key_value_assert(postmix_txs[txid]['inputs'][input], 'cluster_id', cluster_name, CLUSTER_ID_CHECK_HARD_ASSERT)
        # Set also for outputs connected to these inputs
        if 'spending_tx' in postmix_txs[txid]['inputs'][input]:
            tx, vout = als.extract_txid_from_inout_string(postmix_txs[txid]['inputs'][input]['spending_tx'])
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
            tx, vin = als.extract_txid_from_inout_string(premix_txs[txid]['outputs'][output]['spend_by_tx'])
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
    SM.print(f'  {als.get_ratio_string(total_inputs_merged, total_inputs)} '
             f'N:1 postmix merges detected (merged inputs / all inputs)')
    SM.print(f'  {als.get_ratio_string(CLUSTER_INDEX.get_current_index() - offset, len(postmix_txs))} '
             f'N:1 unique postmix clusters detected (clusters / all postmix txs)')

    return tx_dict


def is_whirlpool_coinjoin_tx(test_tx):
    # The transaction is whirlpool coinjoin transaction if number of inputs is bigger than 4
    if len(test_tx['inputs']) >= 5:
        # ... number of inputs and outputs is the same
        if len(test_tx['inputs']) == len(test_tx['outputs']):
            # ... all outputs are the same value
            if all(test_tx['outputs'][vout]['value'] == test_tx['outputs']['0']['value']
                   for vout in test_tx['outputs'].keys()):
                # ... and output sizes are one of the pool sizes [100k, 1M, 5M, 50M]
                if all(test_tx['outputs'][vout]['value'] in [100000, 1000000, 5000000, 50000000]
                       for vout in test_tx['outputs'].keys()):
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
    SM.print(f'  {als.get_ratio_string(CLUSTER_INDEX.get_current_index() - offset, total_outputs)} '
             f'N:M new premix clusters detected (number clusters / total outputs in premix)')

    return tx_dict


def analyze_coinjoin_blocks(data):
    same_block_coinjoins = defaultdict(list)
    for txid in data['coinjoins'].keys():
        same_block_coinjoins[data['coinjoins'][txid]['block_hash']].append(txid)
    filtered_dict = {key: value for key, value in same_block_coinjoins.items() if len(value) > 1}
    SM.print(f'  {als.get_ratio_string(len(filtered_dict), len(data['coinjoins']))} coinjoins in same block')


def visualize_coinjoins_in_time(data, ax_num_coinjoins):
    #
    # Number of coinjoins per given time interval (e.g., day)
    #
    coinjoins = data['coinjoins']
    broadcast_times = [precomp_datetime.strptime(coinjoins[item]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for item in
                       coinjoins.keys()]
    experiment_start_time = min(broadcast_times)
    slot_start_time = experiment_start_time
    slot_last_time = max(broadcast_times)
    diff_seconds = (slot_last_time - slot_start_time).total_seconds()
    num_slots = int(diff_seconds // SLOT_WIDTH_SECONDS)
    cjtx_in_hours = {hour: [] for hour in range(0, num_slots + 1)}
    rounds_started_in_hours = {hour: [] for hour in range(0, num_slots + 1)}
    for cjtx in coinjoins.keys():  # go over all coinjoin transactions
        timestamp = precomp_datetime.strptime(coinjoins[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")
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
        if SLOT_WIDTH_SECONDS < 3600:
            time_delta_format = "%Y-%m-%d %H:%M:%S"
        elif SLOT_WIDTH_SECONDS < 3600 * 24:
            time_delta_format = "%Y-%m-%d %H:%M:%S"
        else:
            time_delta_format = "%Y-%m-%d"
        x_ticks.append(
            (experiment_start_time + slot * timedelta(seconds=SLOT_WIDTH_SECONDS)).strftime(time_delta_format))
    ax_num_coinjoins.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    num_xticks = 30
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_xticks))
    ax_num_coinjoins.set_ylim(0)
    ax_num_coinjoins.set_ylabel('Number of coinjoin transactions')
    ax_num_coinjoins.set_title('Number of coinjoin transactions in given time period')


def visualize_liquidity_in_time(events, ax_number, ax_boxplot, ax_input_values_boxplot, ax_output_values_boxplot,
                                ax_input_values_bar, ax_output_values_bar, ax_burn_time, legend_labels: list, events_premix: dict = None):
    #
    # Number of coinjoins per given time interval (e.g., day)
    #
    coinjoins = events
    broadcast_times_cjtxs = {item: precomp_datetime.strptime(coinjoins[item]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for item in
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
    tx0_inputs_values_in_slot = None
    outputs_values_in_slot = {hour: [] for hour in range(0, num_slots + 1)}
    inputs_burned_time_in_slot = {hour: [] for hour in range(0, num_slots + 1)}
    for cjtx in coinjoins.keys():  # go over all coinjoin transactions
        timestamp = precomp_datetime.strptime(coinjoins[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")
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
                    txid, vout = als.extract_txid_from_inout_string(coinjoins[cjtx]['inputs'][index]['spending_tx'])
                    if txid in broadcast_times_cjtxs.keys():
                        create_time = broadcast_times_cjtxs[txid]
                        time_diff = destruct_time - create_time
                        hours_diff = time_diff.total_seconds() / 3600
                        inputs_burned_time_in_slot[cjtx_hour].append(hours_diff)

    # If provided, process also TX0 premix
    if events_premix and len(events_premix) > 0:
        tx0_broadcast_times_cjtxs = {item: precomp_datetime.strptime(events_premix[item]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for
                                 item in events_premix.keys()}
        broadcast_times = list(tx0_broadcast_times_cjtxs.values())
        experiment_start_time = min(broadcast_times)
        slot_start_time = experiment_start_time
        slot_last_time = max(broadcast_times)
        diff_seconds = (slot_last_time - slot_start_time).total_seconds()
        num_slots = int(diff_seconds // SLOT_WIDTH_SECONDS)
        tx0_inputs_values_in_slot = {hour: [] for hour in range(0, num_slots + 1)}

        for cjtx in events_premix.keys():
            timestamp = precomp_datetime.strptime(events_premix[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")
            cjtx_hour = int((timestamp - slot_start_time).total_seconds() // SLOT_WIDTH_SECONDS)
            tx0_inputs_values_in_slot[cjtx_hour].extend(
                [events_premix[cjtx]['inputs'][index]['value'] for index in events_premix[cjtx]['inputs'].keys()])
        while tx0_inputs_values_in_slot[len(tx0_inputs_values_in_slot.keys()) - 1] == []:
            del tx0_inputs_values_in_slot[len(tx0_inputs_values_in_slot.keys()) - 1]

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
        # For whirlpooll, use distribution of inputs to TX0 (which splits inputs to premix), otherwise inputs to coinjoins
        if tx0_inputs_values_in_slot and len(tx0_inputs_values_in_slot) > 0:
            input_data = tx0_inputs_values_in_slot
        else:
            input_data = inputs_values_in_slot
        flat_data = [item for index in input_data.keys() for item in input_data[index]]
        log_data = np.log(flat_data)
        hist, bins = np.histogram(log_data, bins=100)
        ax_input_values_bar.bar(bins[:-1], hist, width=np.diff(bins))
        xticks = np.linspace(min(log_data), max(log_data), num=10)
        ax_input_values_bar.set_xticks(xticks, np.round(np.exp(xticks), decimals=0), rotation=45, fontsize=6)
        ax_input_values_bar.set_xlim(0, max(log_data))

    # Plot distribution of output values (bar height corresponding to number of occurences)
    if ax_output_values_bar:
        flat_data = [item for index in outputs_values_in_slot.keys() for item in outputs_values_in_slot[index]]
        log_data = np.log(flat_data)
        hist, bins = np.histogram(log_data, bins=100)
        ax_output_values_bar.bar(bins[:-1], hist, width=np.diff(bins))
        xticks = np.linspace(min(log_data), max(log_data), num=10)
        ax_output_values_bar.set_xticks(xticks, np.round(np.exp(xticks), decimals=0), rotation=45, fontsize=6)
        ax_output_values_bar.set_xlim(0, max(log_data))

        #
        # output_values_count = Counter(flat_data)
        # sorted_input_values_count = sorted(output_values_count.items(), key=lambda x: x[0])
        # sorted_values, counts = zip(*sorted_input_values_count)
        # #ax_output_values_bar.bar(sorted_values, counts)
        #
        # log_values = np.log(flat_data)
        # ax_output_values_bar.hist(log_values, 100)
        # ax_output_values_bar.set_xscale('log')
        # # ax_output_values_bar.set_xscale('linear')
        # #ax_output_values_bar.set_xticks(np.linspace(min(flat_data), max(flat_data), 30))

    # if ax_burn_time:
    #     flat_data = [item for index in inputs_burned_time_in_slot.keys() for item in inputs_burned_time_in_slot[index]]
    #     ax_burn_time.bar(flat_data)

    ax_number.legend()
    ax_boxplot.legend()
    x_ticks = []
    for slot in inputs_cjtx_in_slot.keys():
        if SLOT_WIDTH_SECONDS < 3600:
            time_delta_format = "%Y-%m-%d %H:%M:%S"
        elif SLOT_WIDTH_SECONDS < 3600 * 24:
            time_delta_format = "%Y-%m-%d %H:%M:%S"
        else:
            time_delta_format = "%Y-%m-%d"

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
                                None, ['fresh inputs mixed', 'outputs leaving mix', ''], data.get('premix', None))
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
    save_file = os.path.join(base_path, f'{experiment_name}_coinjoin_stats')
    plt.savefig(f'{save_file}.png', dpi=300)
    plt.savefig(f'{save_file}.pdf', dpi=300)
    plt.close()
    logging.info('Basic coinjoins statistics saved into {}'.format(save_file))


def process_coinjoins(target_path, mix_protocol: MIX_PROTOCOL, mix_filename, postmix_filename, premix_filename, start_date: str, stop_date: str):
    data, false_cjtxs = load_coinjoins(target_path, mix_protocol, mix_filename, postmix_filename, premix_filename, start_date, stop_date)
    if len(data['coinjoins']) == 0:
        return data

    false_cjtxs_file = os.path.join(target_path, f'{mix_protocol.name}_false_filtered_cjtxs.json')
    als.save_json_to_file_pretty(false_cjtxs_file, false_cjtxs)

    SM.print('*******************************************')
    SM.print(f'{mix_filename} coinjoins: {len(data['coinjoins'])}')
    min_date = min([data['coinjoins'][txid]['broadcast_time'] for txid in data['coinjoins'].keys()])
    max_date = max([data['coinjoins'][txid]['broadcast_time'] for txid in data['coinjoins'].keys()])
    SM.print(f'Dates from {min_date} to {max_date}')

    SM.print('### Simple chain analysis')
    cj_relative_order = als.analyze_input_out_liquidity(data['coinjoins'], data['postmix'], data.get('premix', {}), mix_protocol)

    analyze_postmix_spends(data)
    analyze_premix_spends(data)
    analyze_coinjoin_blocks(data)
    analyze_coordinator_fees(mix_filename, data, mix_protocol)
    analyze_mining_fees(mix_filename, data)

    return data, cj_relative_order


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


def process_and_save_coinjoins(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: os.path, mix_filename: str, postmix_filename: str,
                               premix_filename: str, start_date: str, stop_date: str, target_save_path: os.path=None):
    if not target_save_path:
        target_save_path = target_path
    # Process and save full conjoin information
    data, cj_relative_order = process_coinjoins(target_path, mix_protocol, mix_filename, postmix_filename, premix_filename, start_date, stop_date)
    als.save_json_to_file_pretty(os.path.join(target_save_path, f'cj_relative_order.json'), cj_relative_order)

    if SAVE_BASE_FILES_JSON:
        als.save_json_to_file(os.path.join(target_save_path, f'coinjoin_tx_info.json'), data)

    # Filter only liquidity-relevant events to maintain smaller file
    events = filter_liquidity_events(data)
    als.save_json_to_file_pretty(os.path.join(target_save_path, f'{mix_id}_events.json'), events)

    # # Visualize coinjoins
    # if len(data['coinjoins']) > 0:
    #     visualize_coinjoins(data, events, target_save_path, mix_filename)

    return data


def process_and_save_intervals_onload(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: os.path, start_date: str, stop_date: str, mix_filename: str,
                                      postmix_filename: str, premix_filename: str=None):

    # Create directory structure with files split per month (around 1000 subsequent coinjoins)

    # Find first day of a month when first coinjoin ocured
    start_date_obj = precomp_datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S.%f")
    start_date = datetime(start_date_obj.year, start_date_obj.month, 1)

    # Month After the last coinjoin occured
    last_date_obj = precomp_datetime.strptime(stop_date, "%Y-%m-%d %H:%M:%S.%f")
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
        logging.info(f'Processing interval {last_stop_date_str} - {current_stop_date_str}')
        interval_path = os.path.join(target_path, f'{last_stop_date_str.replace(':', '-')}--{current_stop_date_str.replace(':', '-')}_unknown-static-100-1utxo')
        if not os.path.exists(interval_path):
            os.makedirs(interval_path.replace('\\', '/'))
            os.makedirs(os.path.join(interval_path, 'data').replace('\\', '/'))
        process_and_save_coinjoins(mix_id, mix_protocol, target_path, mix_filename, postmix_filename, premix_filename,
                                          last_stop_date_str, current_stop_date_str, interval_path)
        # Move to the next month
        last_stop_date_str = current_stop_date_str

        current_stop_date = current_stop_date + timedelta(days=32)
        current_stop_date = datetime(current_stop_date.year, current_stop_date.month, 1)
        current_stop_date_str = current_stop_date.strftime("%Y-%m-%d %H:%M:%S")


def process_interval(mix_id: str, data: dict, mix_filename: str, premix_filename: str, target_save_path: str, last_stop_date_str: str, current_stop_date_str: str):
    logging.info(f'Processing interval {last_stop_date_str} - {current_stop_date_str}')

    # Create folder structure compatible with ww2 coinjoin simulation for further processing
    interval_path = os.path.join(target_save_path, f'{last_stop_date_str.replace(':', '-')}--{current_stop_date_str.replace(':', '-')}_unknown-static-100-1utxo')
    if not os.path.exists(interval_path):
        os.makedirs(interval_path.replace('\\', '/'))
        os.makedirs(os.path.join(interval_path, 'data').replace('\\', '/'))

    # Filter only data relevant for given interval and save
    interval_data = als.extract_interval(data, last_stop_date_str, current_stop_date_str)

    als.save_json_to_file(os.path.join(interval_path, f'coinjoin_tx_info.json'), interval_data)
    # Filter only liquidity-relevant events to maintain smaller file
    events = filter_liquidity_events(interval_data)
    als.save_json_to_file_pretty(os.path.join(interval_path, f'{mix_id}_events.json'), events)

    # extract liquidity for given interval
    if premix_filename:
        # Whirlpool
        extract_inputs_distribution(mix_id, target_path, premix_filename, interval_data['premix'], True)
    else:
        # WW1, WW2
        extract_inputs_distribution(mix_id, target_path, mix_filename, interval_data['coinjoins'], True)

    # Moved under separate command
    # # Visualize coinjoins
    # if len(interval_data['coinjoins']) > 0:
    #     visualize_coinjoins(interval_data, events, interval_path, os.path.basename(interval_path))


def process_and_save_intervals_filter(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: os.path, start_date: str, stop_date: str, mix_filename: str,
                                      postmix_filename: str, premix_filename: str=None, save_base_files=True, load_base_files=False):
    # Create directory structure with files split per month (around 1000 subsequent coinjoins)
    # Load all coinjoins first, then filter based on intervals
    target_save_path = os.path.join(target_path, mix_id)
    if not os.path.exists(target_save_path):
        os.makedirs(target_save_path.replace('\\', '/'))

    if load_base_files:
        # Load base files from already stored json
        logging.info(f'Loading {target_save_path}/coinjoin_tx_info.json ...')

        data = als.load_json_from_file(os.path.join(target_save_path, f'coinjoin_tx_info.json'))

        logging.info(f'{target_save_path}/coinjoin_tx_info.json loaded with {len(data['coinjoins'])} conjoins')
    else:
        #
        # Convert all Dumplings files into json (time intensive)
        SAVE_BASE_FILES_JSON = False
        data = process_and_save_coinjoins(mix_id, mix_protocol, target_path, mix_filename, postmix_filename, premix_filename, None, None, target_save_path)
        SAVE_BASE_FILES_JSON = save_base_files

    if mix_protocol == MIX_PROTOCOL.WHIRLPOOL:
        # Whirlpool
        extract_inputs_distribution(mix_id, target_path, premix_filename, data['premix'], True)
    else:
        # WW1, WW2
        extract_inputs_distribution(mix_id, target_path, mix_filename, data['coinjoins'], True)

    # Find first day of a month when first coinjoin occured
    start_date_obj = precomp_datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S.%f")
    start_date = datetime(start_date_obj.year, start_date_obj.month, 1)

    # Month After the last coinjoin occured
    last_date_obj = precomp_datetime.strptime(stop_date, "%Y-%m-%d %H:%M:%S.%f")
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
        process_interval(mix_id, data, mix_filename, premix_filename, target_save_path, last_stop_date_str, current_stop_date_str)

        # Move to the next month
        last_stop_date_str = current_stop_date_str

        current_stop_date = current_stop_date + timedelta(days=32)
        current_stop_date = datetime(current_stop_date.year, current_stop_date.month, 1)
        current_stop_date_str = current_stop_date.strftime("%Y-%m-%d %H:%M:%S")

    # Backup corresponding log file
    backup_log_files(target_path)

    return data


def visualize_interval(mix_id: str, data: dict, mix_filename: str, premix_filename: str, target_save_path: str, last_stop_date_str: str, current_stop_date_str: str):
    logging.info(f'Processing interval {last_stop_date_str} - {current_stop_date_str}')

    interval_path = os.path.join(target_save_path, f'{last_stop_date_str.replace(':', '-')}--{current_stop_date_str.replace(':', '-')}_unknown-static-100-1utxo')
    assert os.path.exists(interval_path), f'{interval_path} does not exist'

    interval_data = als.load_json_from_file(os.path.join(interval_path, f'coinjoin_tx_info.json'))
    events = filter_liquidity_events(interval_data)

    # Visualize coinjoins
    if len(interval_data['coinjoins']) > 0:
        visualize_coinjoins(interval_data, events, interval_path, os.path.basename(interval_path))


def visualize_intervals(mix_id: str, target_path: os.path, start_date: str, stop_date: str):
    # Create directory structure with files split per month (around 1000 subsequent coinjoins)
    # Load all coinjoins first, then filter based on intervals
    target_save_path = os.path.join(target_path, mix_id)
    if not os.path.exists(target_save_path):
        os.makedirs(target_save_path.replace('\\', '/'))

    # Load base files from already stored json
    logging.info(f'Loading {target_save_path}/coinjoin_tx_info.json ...')

    data = als.load_json_from_file(os.path.join(target_save_path, f'coinjoin_tx_info.json'))

    logging.info(f'{target_save_path}/coinjoin_tx_info.json loaded with {len(data['coinjoins'])} conjoins')

    # Find first day of a month when first coinjoin occured
    start_date_obj = precomp_datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S.%f")
    start_date = datetime(start_date_obj.year, start_date_obj.month, 1)

    # Month After the last coinjoin occured
    last_date_obj = precomp_datetime.strptime(stop_date, "%Y-%m-%d %H:%M:%S.%f")
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
        visualize_interval(mix_id, data, target_save_path, last_stop_date_str, current_stop_date_str)

        # Move to the next month
        last_stop_date_str = current_stop_date_str

        current_stop_date = current_stop_date + timedelta(days=32)
        current_stop_date = datetime(current_stop_date.year, current_stop_date.month, 1)
        current_stop_date_str = current_stop_date.strftime("%Y-%m-%d %H:%M:%S")


def process_and_save_single_interval(mix_id: str, data: dict, mix_protocol: MIX_PROTOCOL, target_path: os.path, start_date: str, stop_date: str):
    # Create directory structure for target interval
    # Load all coinjoins first, then filter based on intervals
    target_save_path = os.path.join(target_path, mix_id)
    if not os.path.exists(target_save_path):
        os.makedirs(target_save_path.replace('\\', '/'))

    process_interval(mix_id, data, None, None, target_save_path, start_date, stop_date)


def find_whirlpool_tx0_reuse(mix_id: str, target_path: Path, premix_filename: str):
    """
    Detects all address reuse in Whirlpool TX0 transactions
    :param mix_id:
    :param target_path:
    :param premix_filename:
    :return:
    """
    txs = load_coinjoin_stats_from_file(os.path.join(target_path, premix_filename))
    # If potential reuse detected, check if not coordinator address for fees
    # pools are 100k (=>5000), 1M (=>50000), 5M (=>175000), 50M (=> 1750000)
    return find_address_reuse(mix_id, txs, target_path, [0, 5000, 50000, 175000, 1750000, 250000, 2500000])


def find_txs_address_reuse(mix_id: str, target_path: Path, tx_filename: str, save_outputs = False):
    """
    Detects all address reuse in Whirlpool mix transactions
    :param mix_id:
    :param target_path:
    :param tx_filename:
    :return:
    """
    txs = load_coinjoin_stats_from_file(os.path.join(target_path, tx_filename))
    return find_address_reuse(mix_id, txs, target_path, [], save_outputs)


def find_address_reuse(mix_id: str, txs: dict, target_path: Path = None, ignore_denominations: list = [], save_outputs = False):
    """
    Detects all address reuse in given list of transactions
    :param mix_id:
    :param txs: dictionary of transactions
    :param target_path: path used for saving results
    :param premix_filename:
    :return:
    """
    logging.info(f'Processing {mix_id}')
    seen_addresses = defaultdict(list)
    reused_addresses = defaultdict(list)
    for txid in list(txs.keys()):
        for index in txs[txid]['outputs']:
            address = txs[txid]['outputs'][index]['script']
            value = txs[txid]['outputs'][index]['value']
            if address in seen_addresses.keys():
                #print(f'Detected address reuse {txid}_{index} and {seen_addresses[address][0][0]['txid']}')
                if value not in ignore_denominations:
                    #print(f'{value}')
                    # Add this address as seen and reused
                    reused_addresses[address].append((txs[txid], index))
                    # Add previous record we now know was reused in this output
                    reused_addresses[address].append(seen_addresses[address][0])
            seen_addresses[address].append((txs[txid], index))

    total_txs = len(txs)
    total_out_addresses = sum([len(txs[txid]['outputs']) for txid in txs.keys()])
    single_reuse = {address: reused_addresses[address] for address in reused_addresses.keys() if len(reused_addresses[address]) == 2}
    multiple_reuse = {address: reused_addresses[address] for address in reused_addresses.keys() if len(reused_addresses[address]) > 2}
    logging.info(f'{mix_id} total txs: {total_txs}, total out addresses {total_out_addresses}')
    logging.info(f'Total reused addresses: {len(reused_addresses)} ({round(len(reused_addresses) / total_out_addresses, 4)}%), {sum([len(reused_addresses[addr]) for addr in reused_addresses])} times')
    logging.info(f'Total single reuse addresses: {len(single_reuse)} ({round(len(single_reuse) / total_out_addresses, 4)}%), {sum([len(single_reuse[addr]) for addr in single_reuse])} times')
    logging.info(f'Total multiple reuse addresses: {len(multiple_reuse)} ({round(len(multiple_reuse) / total_out_addresses, 4)}%), {sum([len(multiple_reuse[addr]) for addr in multiple_reuse])} times')

    if target_path and save_outputs:
        target_save_path = target_path
        als.save_json_to_file_pretty(os.path.join(target_save_path, f'{mix_id}_reused_addresses.json'), reused_addresses)
        als.save_json_to_file_pretty(os.path.join(target_save_path, f'{mix_id}_reused_addresses_single.json'), single_reuse)
        als.save_json_to_file_pretty(os.path.join(target_save_path, f'{mix_id}_reused_addresses_multiple.json'), multiple_reuse)

    # TODO: Plot characteristics of address reuse (time between reuse, ocurence in real time...)


def extract_coinjoin_interval(mix_id: str, target_path: Path, txs: dict, start_date: str, stop_date: str, save_outputs=False):
    #print(f'Processing {mix_id}')
    inputs = {txid: txs[txid] for txid in txs.keys() if start_date <= txs[txid]['broadcast_time'] <= stop_date}
    logging.info(f'  Interval extracted for {start_date} to {stop_date}, total {len(inputs.keys())} coinjoins found')
    interval_data = {'coinjoins': inputs, 'start_date': start_date, 'stop_date': stop_date}
    if save_outputs:
        als.save_json_to_file(os.path.join(target_path, f'{mix_id}_conjoins_interval_{start_date[:start_date.find(' ') - 1]}-{stop_date[:stop_date.find(' ') - 1]}.json'), interval_data)

    return interval_data


def print_interval_data_stats(pool_stats: dict, client_stats: CoinMixInfo, results: dict):
    num_inputs = [len(pool_stats[txid]['inputs']) for txid in pool_stats.keys()]
    assert max(num_inputs) < 9, 'Whirpool shall not have more than 9 inputs in mix tx'
    #print(num_inputs)
    num_inputs_pool = sum(num_inputs)
    logging.info(
        f'  {round(client_stats.pool_size / 100000000, 3)} pool total inputs={num_inputs_pool}, client mixes={client_stats.num_mixes}, '
        f'client coins={client_stats.num_coins}')
    if client_stats.num_mixes > 0:
        ratio = round((num_inputs_pool / client_stats.num_mixes) * client_stats.num_coins, 1)
        logging.info(f'    {round(client_stats.pool_size / 100000000, 3)} pool participation rate= 1:{ratio}')
        results[str(client_stats.pool_size)].append(ratio)
        logging.info(f'    present in % of mixes (per single coin)= {round(client_stats.num_mixes / len(pool_stats.keys()) * 100 / client_stats.num_coins, 2)}%')


def analyze_interval_data(interval_data, stats: CoinJoinStats, results: dict):
    if stats.cj_type == MIX_PROTOCOL.WHIRLPOOL:
        # Count number of coinjoins in different pools
        pool_100k = {txid: interval_data['coinjoins'][txid] for txid in interval_data['coinjoins'].keys() if interval_data['coinjoins'][txid]['outputs']['0']['value'] == 100000}
        pool_1M = {txid: interval_data['coinjoins'][txid] for txid in interval_data['coinjoins'].keys() if interval_data['coinjoins'][txid]['outputs']['0']['value'] == 1000000}
        pool_5M = {txid: interval_data['coinjoins'][txid] for txid in interval_data['coinjoins'].keys() if interval_data['coinjoins'][txid]['outputs']['0']['value'] == 5000000}

        logging.info(f'  Total cjs={len(interval_data['coinjoins'].keys())}, 100k pool={len(pool_100k)}, 1M pool={len(pool_1M)}, 5M pool={len(pool_5M)}')
        logging.info(f'  {interval_data['start_date']} - {interval_data['stop_date']}')
        print_interval_data_stats(pool_100k, stats.pool_100k, results)
        print_interval_data_stats(pool_1M, stats.pool_1M, results)
        print_interval_data_stats(pool_5M, stats.pool_5M, results)

    if stats.cj_type == MIX_PROTOCOL.WASABI2:
        logging.info(f'  Total cjs in interval= {len(interval_data['coinjoins'].keys())}')
        logging.info(f'  {interval_data['start_date']} - {interval_data['stop_date']}')
        logging.info(f'  Used cjs= {stats.no_pool.num_mixes}, skipped= {len(interval_data['coinjoins']) - stats.no_pool.num_mixes}')
        logging.info(f'  #cjs per one input coin= {round(stats.no_pool.num_mixes / stats.no_pool.num_coins, 2)}')


def process_inputs_distribution(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: Path, tx_filename: str, save_outputs: bool= False):
    logging.info(f'Processing {mix_id}')
    txs = load_coinjoin_stats_from_file(os.path.join(target_path, tx_filename))
    als.analyze_input_out_liquidity(txs, [], [], mix_protocol)
    # inputs_not_enter = [(txid, index) for txid in txs.keys() for index in txs[txid]['inputs'].keys()
    #           if txs[txid]['inputs'][index]['mix_event_type'] != MIX_EVENT_TYPE.MIX_ENTER.name]

    inputs_info, inputs = extract_inputs_distribution(mix_id, target_path, tx_filename, txs, save_outputs)

    log_data = np.log(inputs)
    hist, bins = np.histogram(log_data, bins=100)
    plt.bar(bins[:-1], hist, width=np.diff(bins))
    xticks = np.linspace(min(log_data), max(log_data), num=10)
    plt.xscale('log')
    plt.xticks(xticks, np.round(np.exp(xticks), decimals=0), rotation=45, fontsize=6)
    plt.title(f'{mix_id} inputs histogram (x axis is log)')
    plt.xlabel(f'Size of input')
    plt.ylabel(f'Number of inputs')
    plt.show()


def process_estimated_wallets_distribution(mix_id: str, target_path: Path, inputs_wallet_factor: list, save_outputs: bool= True):
    logging.info(f'Processing process_estimated_wallets_distribution({mix_id})')
    # Load txs for all pools
    target_load_path = os.path.join(target_path, mix_id)

    data = als.load_coinjoins_from_file(target_load_path, None, True)

    # For each cjtx compute rough number of wallets present based on the inputs_wallet_factor
    num_wallets = [len(data['coinjoins'][txid]['inputs'].keys()) for txid in data['coinjoins'].keys()]

    for factor in inputs_wallet_factor:
        logging.info(f' Processing factor={factor}')
        wallets_distrib = Counter([round(item / factor) for item in num_wallets])
        wallets_distrib = dict(sorted(wallets_distrib.items(), key=lambda item: (-item[1], item[0])))
        wallets_info = {'mix_id': mix_id, 'path': target_load_path, 'wallets_distrib': wallets_distrib, 'wallets_distrib_factor': factor}
        logging.info(f'  Distribution of walets extracted, total {len(wallets_info['wallets_distrib'])} different input values found')
        if save_outputs:
            als.save_json_to_file_pretty(os.path.join(target_path, mix_id, f'{mix_id}_wallets_distribution_factor{factor}.json'), wallets_info)

        labels = list(wallets_distrib.keys())
        values = list(wallets_distrib.values())
        plt.figure(figsize=(10, 3))
        plt.bar(labels, values)
        plt.title(f'{mix_id}: distribution of number of wallets in coinjoins (est. by factor {factor})')
        plt.xlabel(f'Number of wallets')
        plt.ylabel(f'Number of occurences')
        save_file = os.path.join(target_path, mix_id, f'{mix_id}_wallets_distribution_factor{factor}')
        plt.subplots_adjust(bottom=0.17)
        plt.savefig(f'{save_file}.png', dpi=300)
        plt.savefig(f'{save_file}.pdf', dpi=300)
        plt.close()


def process_inputs_distribution2(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: Path, tx_filename: str, save_outputs: bool= True):
    logging.info(f'Processing {mix_id} process_inputs_distribution2()')
    # Load txs for all pools
    target_load_path = os.path.join(target_path, mix_id)

    data = als.load_coinjoins_from_file(target_load_path, None, True)
    inputs_info, inputs = extract_inputs_distribution(mix_id, target_path, tx_filename, data['coinjoins'], save_outputs)

    log_data = np.log(inputs)
    hist, bins = np.histogram(log_data, bins=100)
    plt.bar(bins[:-1], hist, width=np.diff(bins))
    xticks = np.linspace(min(log_data), max(log_data), num=10)
    plt.xscale('log')
    plt.xticks(xticks, np.round(np.exp(xticks), decimals=0), rotation=45, fontsize=6)
    plt.title(f'{mix_id} inputs histogram (x axis is log)')
    plt.xlabel(f'Size of input')
    plt.ylabel(f'Number of inputs')
    plt.show()


def extract_inputs_distribution(mix_id: str, target_path: Path, tx_filename: str, txs: dict, save_outputs = False):
    inputs = [txs[txid]['inputs'][index]['value'] for txid in txs.keys() for index in txs[txid]['inputs'].keys()
              if 'mix_event_type' in txs[txid]['inputs'][index].keys() and
              txs[txid]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_ENTER.name]
    inputs_distrib = Counter(inputs)
    inputs_distrib = dict(sorted(inputs_distrib.items(), key=lambda item: (-item[1], item[0])))
    inputs_info = {'mix_id': mix_id, 'path': tx_filename, 'distrib': inputs_distrib}
    logging.info(f'  Distribution extracted, total {len(inputs_info['distrib'])} different input values found')
    if save_outputs:
        als.save_json_to_file_pretty(os.path.join(target_path, mix_id, f'{mix_id}_inputs_distribution.json'), inputs_info)

    return inputs_info, inputs


def analyze_address_reuse(target_path):
    # find_whirlpool_tx0_reuse('whirlpool_tx0_test', target_path, 'whirlpool_tx0_test.txt')
    find_whirlpool_tx0_reuse('whirlpool_tx0', target_path, 'SamouraiTx0s.txt')
    find_txs_address_reuse('whirlpool_mix', target_path, 'SamouraiCoinJoins.txt')
    find_txs_address_reuse('whirlpool_postmix', target_path, 'SamouraiPostMixTxs.txt')

    find_txs_address_reuse('wasabi1_mix', target_path, 'WasabiCoinJoins.txt')
    find_txs_address_reuse('wasabi1_postmix', target_path, 'WasabiPostMixTxs.txt')
    find_txs_address_reuse('wasabi2_mix', target_path, 'Wasabi2CoinJoins.txt')
    find_txs_address_reuse('wasabi2_mix', target_path, 'Wasabi2CoinJoins.txt', False)
    find_txs_address_reuse('wasabi2_postmix', target_path, 'Wasabi2PostMixTxs.txt')


def burntime_histogram(mix_id: str, data: dict):
    cjtxs = data['coinjoins']
    burn_times = [cjtxs[cjtx]['inputs'][index]['burn_time_cjtxs']
                  for cjtx in cjtxs.keys() for index in cjtxs[cjtx]['inputs']
                  if 'burn_time_cjtxs' in cjtxs[cjtx]['inputs'][index].keys()]

    # plt.hist(burn_times, bins=100, edgecolor='black')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Data')
    # plt.show()

    # log_data = np.log(burn_times)
    # hist, bins = np.histogram(log_data, bins=100)
    # plt.bar(bins[:-1], hist, width=np.diff(bins))
    # plt.gca().get_xaxis().set_major_formatter(ScalarFormatter())
    # plt.gca().get_xaxis().set_minor_formatter(NullFormatter())
    # xticks = np.linspace(min(log_data), max(log_data), num=10)
    # plt.xticks(xticks, np.round(np.exp(xticks), decimals=0), rotation=45, fontsize=6)
    # plt.show()

    NUM_BINS = 1000
    # Compute standard histogram
    plt.figure()
    plt.hist(burn_times, NUM_BINS)
    plt.title(f'{mix_id} Histogram of burn times for all inputs')
    plt.xlabel('Burn time (num of coinjoins executed in meantime)')
    plt.ylabel('Frequency')
    plt.show()

    # Compute histogram in log scale
    plt.figure()
    hist, bins = np.histogram(burn_times, bins=np.logspace(np.log10(min(burn_times)), np.log10(max(burn_times)), NUM_BINS))
    plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='black')
    xticks = np.linspace(min(burn_times), max(burn_times), num=10)
    plt.xticks(xticks, np.round(np.exp(xticks), decimals=0), rotation=45, fontsize=6)
    plt.xscale('log')
    plt.title(f'{mix_id} Frequency of different burn times for all inputs')
    plt.xlabel('Burn time (num of coinjoins executed in meantime)')
    plt.ylabel('Frequency')
    plt.show()

    # # Create histogram
    # hist, bins = np.histogram(burn_times, bins=np.logspace(np.log10(min(burn_times)), np.log10(max(burn_times)), 100))
    # # Plot histogram
    # plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor='black')
    # # Set x-axis scale to logarithmic
    # plt.xscale('log')
    # # Set x-axis ticks
    # xticks = np.linspace(np.log10(min(burn_times)), np.log10(max(burn_times)), num=10)
    # xticks_labels = np.round(np.power(10, xticks)).astype(int)  # Convert tick values from log to integer
    # plt.xticks(10 ** xticks, xticks_labels, rotation=45, fontsize=6)
    # # Set labels and title
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Data (Log Scale)')
    # # Show plot
    # plt.show()

    # fig, ax = plt.subplots(1, 1)
    # hist, bins = np.histogram(burn_times, bins=100)
    # logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    # ax.set_xscale('log')
    # ax.get_xaxis().set_major_formatter(ScalarFormatter())
    # ax.get_xaxis().set_minor_formatter(NullFormatter())
    # ax.hist(burn_times, bins=logbins)
    # fig.show()


def plot_analyze_liquidity(mix_id: str, cjtxs):
    plt.figure()

    short_exp_name = 'remix'
    sorted_cj_time = als.sort_coinjoins(cjtxs, als.SORT_COINJOINS_BY_RELATIVE_ORDER)
    
    #
    # num_unmixed_utxos_per_cj = [len(cjtxs[cjtx['txid']]['analysis2']['unmixed_utxos_in_wallets']) for cjtx in sorted_cj_time]
    # num_mixed_data_per_cj = [len(cjtxs[cjtx['txid']]['analysis2']['mixed_utxos_in_wallets']) for cjtx in sorted_cj_time]
    # num_finished_data_per_cj = [len(cjtxs[cjtx['txid']]['analysis2']['finished_utxos_in_wallets']) for cjtx in sorted_cj_time]
    # if ax:
    #     bar_width = 0.1
    #     categories = range(0, len(outputs_data))
    #     ax.bar(categories, num_unmixed_utxos_per_cj, bar_width,
    #                                  label=f'unmixed {short_exp_name}', alpha=0.3, color='blue')
    #     ax.bar(categories, num_mixed_data_per_cj, bar_width, bottom=num_unmixed_utxos_per_cj,
    #                                label=f'mixed {short_exp_name}', color='orange', alpha=0.3)
    #     ax.bar(categories, num_finished_data_per_cj, bar_width,
    #                                bottom=np.array(num_unmixed_utxos_per_cj) + np.array(num_mixed_data_per_cj),
    #                                label=f'finished {short_exp_name}', color='green', alpha=0.3)
    #     ax.plot(categories, num_unmixed_utxos_per_cj, label=f'unmixed {short_exp_name}',
    #                                   linewidth=3, color='blue', linestyle='--', alpha=0.3)
    #     ax.plot(categories, num_mixed_data_per_cj, label=f'mixed {short_exp_name}',
    #                                  linewidth=3, color='orange', linestyle='-.', alpha=0.3)
    #     ax.plot(categories, num_finished_data_per_cj, label=f'finished {short_exp_name}',
    #                                  linewidth=3, color='green', linestyle=':', alpha=0.3)
    #
    #     ax.set_xlabel('Coinjoin in time')
    #     ax.set_ylabel('Number of txos')
    #     ax.legend(loc='lower left')
    #     ax.set_title(f'Number of txos available in wallets when given cjtx is starting (all transactions)\n{experiment_name}')
    return None


def inputs_value_burntime_heatmap(mix_id: str, data: dict):
    cjtxs = data['coinjoins']
    # Create logarithmic range for values and digitize it
    # (we have too many different values, compute log bins then assign each precise value its bin number)
    NUM_BINS = 40  # Number of total bins to scale x and y axis to (logarithmically)

    # Sample list of tuples containing x and y coordinates
    points = [(cjtxs[cjtx]['inputs'][index]['burn_time_cjtxs'], cjtxs[cjtx]['inputs'][index]['value'])
              for cjtx in cjtxs.keys() for index in cjtxs[cjtx]['inputs'].keys()
              if 'burn_time_cjtxs' in cjtxs[cjtx]['inputs'][index].keys()]

    # Extract x and y coordinates from points list
    x_coords, y_coords = zip(*points)

    # Compute logarithmic bins for each axis separate (value of input, burn time in cjtxs)
    bins_x = np.logspace(np.log10(min(x_coords)), np.log10(max(x_coords)), num=NUM_BINS)
    bins_y = np.logspace(np.log10(min(y_coords)), np.log10(max(y_coords)), num=NUM_BINS)

    # Assign original precise values into range of log bins
    # np.digitize will compute corresponding bin for given precise value (5000 sats will go into first bin => 1...)
    x_coords_digitized = np.digitize(x_coords, bins_x)
    y_coords_digitized = np.digitize(y_coords, bins_y)
    points_digitized = zip(x_coords_digitized, y_coords_digitized)

    # Determine the dimensions of the heatmap (shall be close to NUM_BINS)
    x_min, x_max = min(x_coords_digitized), max(x_coords_digitized)
    y_min, y_max = min(y_coords_digitized), max(y_coords_digitized)
    x_range = np.arange(x_min, x_max + 1)
    y_range = np.arange(y_min, y_max + 1)
    # Create a grid representing the heatmap (initially empty)
    heatmap = np.zeros((len(y_range), len(x_range)))
    # Fill the grid with counts of points based in digitized inputs (value, burn_time)
    for x, y in points_digitized:
        heatmap[y - y_min, x - x_min] += 1

    # Plot the heatmap (no approximation)
    plt.figure()
    plt.hist2d(x_coords_digitized, y_coords_digitized, bins=NUM_BINS, cmap='plasma')
    plt.colorbar()

    # Add ticks labels from original non-log range
    custom_xticks = np.linspace(min(x_coords_digitized), max(x_coords_digitized), 10)
    custom_xticklabels = [f'{int(round(bins_x[int(tick-1)], 0))}' for tick in custom_xticks]  # Customize labels as needed
    plt.gca().set_xticklabels(custom_xticklabels, rotation=45, fontsize=6)
    custom_yticks = np.linspace(min(y_coords_digitized), max(y_coords_digitized), 10)
    custom_yticklabels = [f'{int(round(bins_y[int(tick)], 0))}' for tick in custom_yticks if int(tick) < len(bins_y)]  # Customize labels as needed
    plt.gca().set_yticklabels(custom_yticklabels, rotation=45, fontsize=6)
    plt.title(f'{mix_id} Input value to burn time heatmap for remixed coinjoin inputs')
    plt.xlabel('Burn time (num coinjoins)')
    plt.ylabel('Value of inputs (sats)')
    plt.show()


def whirlpool_analyze_fees(mix_id: str, cjtxs):
    whirlpool_analyze_coordinator_fees(mix_id, cjtxs)
    analyze_mining_fees(mix_id, cjtxs)


def wasabi2_analyze_fees(mix_id: str, cjtxs):
    wasabi_analyze_coordinator_fees(mix_id, cjtxs)
    analyze_mining_fees(mix_id, cjtxs)


def wasabi1_analyze_fees(mix_id: str, cjtxs):
    wasabi_analyze_coordinator_fees(mix_id, cjtxs)
    analyze_mining_fees(mix_id, cjtxs)


def analyze_mining_fees(mix_id: str, data: dict):
    cjtxs = data['coinjoins']
    sorted_cj_time = als.sort_coinjoins(cjtxs, als.SORT_COINJOINS_BY_RELATIVE_ORDER)

    cjtxs_mining_fee = []
    for index in sorted_cj_time:
        cjtx = index['txid']
        inputs_val = sum([cjtxs[cjtx]['inputs'][index]['value'] for index in cjtxs[cjtx]['inputs'].keys()])
        outputs_val = sum([cjtxs[cjtx]['outputs'][index]['value'] for index in cjtxs[cjtx]['outputs'].keys()])
        cjtxs_mining_fee.append(inputs_val - outputs_val)

    print(f'Total mining fee: {sum(cjtxs_mining_fee) / 100000000} btc ({sum(cjtxs_mining_fee)} sats)')

    plt.figure()
    plt.plot(cjtxs_mining_fee)
    plt.title(f'{mix_id} Mining fee spent on coinjoin transactions in time')
    plt.xlabel('Index of coinjoin in time')
    plt.ylabel('Mining fee (sats)')
    plt.show()

    plt.figure()
    plt.hist(cjtxs_mining_fee, 100)
    plt.title(f'{mix_id} Histogram of mining fees spent on coinjoin transactions')
    plt.xlabel('Mining fee (sats)')
    plt.ylabel('Frequency')
    plt.show()

    #FEE_THRESHOLD = 1500000 # For WW2, almost all are below 1500000
    threshold = np.percentile(cjtxs_mining_fee, 95)
    filter_below_threshold = [value for value in cjtxs_mining_fee if value < threshold]

    plt.figure()
    plt.hist(filter_below_threshold, 100)
    plt.title(f'{mix_id} Histogram of mining fees spent on coinjoin transactions (95 percentil)')
    plt.xlabel('Mining fee (sats)')
    plt.ylabel('Frequency')
    plt.show()

    return cjtxs_mining_fee


def analyze_coordinator_fees(mix_id: str, data, mix_protocol):
    if mix_protocol == MIX_PROTOCOL.WASABI1 or mix_protocol == MIX_PROTOCOL.WASABI2:
        return wasabi_analyze_coordinator_fees(mix_id, data)
    elif mix_protocol == MIX_PROTOCOL.WHIRLPOOL:
        return whirlpool_analyze_fees(mix_id, data)
    else:
        assert False, f'Unexpected value of mix_protocol provided: {mix_protocol.name}'


def wasabi_analyze_coordinator_fees(mix_id: str, cjtxs: dict):
    only_cjtxs = cjtxs['coinjoins']
    sorted_cj_time = als.sort_coinjoins(only_cjtxs, als.SORT_COINJOINS_BY_RELATIVE_ORDER)

    PLEBS_SATS_LIMIT = 1000000
    WW2_COORD_FEE = 0.003
    cjtxs_coordinator_fee = []
    for index in sorted_cj_time:
        cjtx = index['txid']
        coord_fee = sum([only_cjtxs[cjtx]['inputs'][index]['value'] * WW2_COORD_FEE for index in only_cjtxs[cjtx]['inputs'].keys()
                         if only_cjtxs[cjtx]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_ENTER.name and only_cjtxs[cjtx]['inputs'][index]['value'] >= PLEBS_SATS_LIMIT])
        cjtxs_coordinator_fee.append(coord_fee)

    # TODO: analyze plebs-do-not-pay frequency

    print(f'Total coordination fee: {sum(cjtxs_coordinator_fee) / 100000000} btc ({sum(cjtxs_coordinator_fee)} sats)')

    plt.figure()
    plt.plot(cjtxs_coordinator_fee)
    plt.title(f'{mix_id} Coordination fee spent on coinjoin transactions in time')
    plt.xlabel('Index of coinjoin in time')
    plt.ylabel('Coordinator fee (sats)')
    plt.show()

    return cjtxs_coordinator_fee


def whirlpool_analyze_coordinator_fees(mix_id: str, data: dict):
    tx0s = data['premix']
    cjtxs = data['coinjoins']
    tx0_time = [{'txid': cjtxid, 'broadcast_time': precomp_datetime.strptime(tx0s[cjtxid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")}
        for cjtxid in tx0s.keys()]
    sorted_tx0_time = sorted(tx0_time, key=lambda x: x['broadcast_time'])

    # Compute coordinator fee (5% of the size of the pool)
    WHIRLPOOL_COORD_FEE = 0.05
    WHIRLPOOL_POOLS = [100000, 1000000, 500000, 5000000]

    # For each whirlpool coinjoin transaction find targeted pool based on size of outputs
    cjtxs_coordinator_fees = {}
    for pool in WHIRLPOOL_POOLS:
        cjtxs_coordinator_fees[pool] = []

    for index in sorted_tx0_time:
        tx0 = index['txid']
        # Identify pool used based on size and presence in subsequent coinjoin
        pool = 0
        for index in tx0s[tx0]['outputs'].keys():
            if 'spend_by_tx' in tx0s[tx0]['outputs'][index]:
                txid, vin = als.extract_txid_from_inout_string(tx0s[tx0]['outputs'][index]['spend_by_tx'])
                if txid in cjtxs.keys():
                    for pool_size in WHIRLPOOL_POOLS:
                        if abs(tx0s[tx0]['outputs'][index]['value'] - pool_size) < pool_size * 0.1:
                            pool = pool_size
                            break
            if pool != 0:
                break
        if pool != 0:
            # Fee is computed from size of the pool choosen, not size of input
            coord_fee = int(pool * WHIRLPOOL_COORD_FEE)
            cjtxs_coordinator_fees[pool].append(coord_fee)
        else:
            logging.debug(f'No whirlpool poolsize identified for TX0: {tx0}')
    print(f'Total coordination fee: {sum(cjtxs_coordinator_fees) / 100000000} btc ({sum(cjtxs_coordinator_fees)} sats)')

    plt.figure()
    for pool in cjtxs_coordinator_fees.keys():
        plt.plot(cjtxs_coordinator_fees[pool], label=f'{pool}')
    plt.title(f'{mix_id} Coordination fee spent by TX0 transactions in time')
    plt.xlabel('Index of coinjoin in time')
    plt.ylabel('Coordinator fee (sats)')
    plt.show()

    return cjtxs_coordinator_fees


def whirlpool_analyse_remixes(mix_id: str, target_path: str):
    data = als.load_json_from_file(os.path.join(target_path, mix_id, 'coinjoin_tx_info.json'))
    als.analyze_input_out_liquidity(data['coinjoins'], data['postmix'], data['premix'], MIX_PROTOCOL.WHIRLPOOL)
    whirlpool_analyze_fees(mix_id, data)
    inputs_value_burntime_heatmap(mix_id, data)
    burntime_histogram(mix_id, data)


def wasabi2_analyse_remixes(mix_id: str, target_path: str):
    data = als.load_json_from_file(os.path.join(target_path, mix_id, 'coinjoin_tx_info.json'))
    cj_relative_order = als.analyze_input_out_liquidity(data['coinjoins'], data['postmix'], [], MIX_PROTOCOL.WASABI2)
    als.save_json_to_file_pretty(os.path.join(target_path, mix_id, f'cj_relative_order.json'), cj_relative_order)

    wasabi2_analyze_fees(mix_id, data)
    inputs_value_burntime_heatmap(mix_id, data)
    burntime_histogram(mix_id, data)


def wasabi_plot_remixes(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: Path, tx_file: str, analyze_values: bool = True, normalize_values: bool = True, restrict_to_out_size: int = None, restrict_to_in_size = None, plot_multigraph: bool = True):
    files = os.listdir(target_path) if os.path.exists(target_path) else print(
        f'Path {target_path} does not exist')

    # Load fee rates
    mining_fee_rates = als.load_json_from_file(os.path.join(target_path, 'fee_rates.json'))

    # Load false positives
    fp_file = os.path.join(target_path, 'false_cjtxs.json')
    false_cjtxs = als.load_json_from_file(fp_file)

    # Compute number of required month subgraphs
    num_months = sum([1 for dir_name in files
                      if os.path.isdir(os.path.join(target_path, dir_name)) and
                      os.path.exists(os.path.join(target_path, dir_name, f'{tx_file}'))])

    NUM_COLUMNS = 3
    NUM_ADDITIONAL_GRAPHS = 1 + NUM_COLUMNS
    NUM_ROWS = int((num_months + NUM_ADDITIONAL_GRAPHS) / NUM_COLUMNS + 1)
    fig = plt.figure(figsize=(40, NUM_ROWS * 5))

    ax_index = 1
    changing_liquidity = [0]  # Cummulative liquidity in mix from the perspective of given coinjoin (can go up and down)
    stay_liquidity = [0]  # Absolute cummulative liquidity staying in the mix outputs (mixed, but untouched)
    mining_fee_rate = []  # Mining fee rate
    remix_liquidity = [0] # Liquidity that is remixed in time despite likely reaching target anonscore
    changing_liquidity_timecutoff = [0]
    stay_liquidity_timecutoff = [0]
    coord_fee_rate = []  # Coordinator fee payments
    input_types = {}
    num_wallets = []
    initial_cj_index = 0
    time_liquidity = {}  # If MIX_LEAVE is detected, out liquidity is put into dictionary for future display
    no_remix_all = {'inputs': [], 'outputs': [], 'both': []}

    prev_year = files[0][0:4]
    #new_month_indices = [('placeholder', 0, files[0][0:7])]  # Start with the first index
    new_month_indices = []
    next_month_index = 0
    weeks_dict = defaultdict(dict)
    days_dict = defaultdict(dict)
    for dir_name in files:
        target_base_path = os.path.join(target_path, dir_name)
        tx_json_file = os.path.join(target_base_path, f'{tx_file}')
        current_year = dir_name[0:4]
        if os.path.isdir(target_base_path) and os.path.exists(tx_json_file):
            data = als.load_coinjoins_from_file(target_base_path, false_cjtxs, True)

            # If required, filter only coinjoins with specific size (whirlpool pools)
            if restrict_to_out_size is not None:
                before_len = len(data['coinjoins'])
                data['coinjoins'] = {cjtx: item for cjtx, item in data['coinjoins'].items() if item['outputs']['0']['value'] == restrict_to_out_size}
                print(f'Length after / before filtering {len(data['coinjoins'])} / {before_len} ({restrict_to_out_size/100000000})')
                if len(data['coinjoins']) == 0:
                    print(f'No coinjoins of specified value {restrict_to_out_size/100000000} found in given interval, skipping')
                    continue

            ax = fig.add_subplot(NUM_ROWS, NUM_COLUMNS, ax_index, axes_class=AA.Axes)  # Get next subplot
            ax_index += 1

            # Plot lines as separators corresponding to days
            dates = sorted([precomp_datetime.strptime(data['coinjoins'][cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for cjtx in data['coinjoins'].keys()])
            new_day_indices = [('day', 0)]  # Start with the first index
            for i in range(1, len(dates)):
                if dates[i].day != dates[i - 1].day:
                    new_day_indices.append(('day', i))
            print(new_day_indices)
            for pos in new_day_indices:
                ax.axvline(x=pos[1], color='gray', linewidth=1, alpha=0.2)

            # Store index of coinjoins within this month (to be printed later in cummulative graph)
            if current_year == prev_year:
                new_month_indices.append(('month', next_month_index, dir_name[0:7]))
            else:
                new_month_indices.append(('year', next_month_index, dir_name[0:7]))
            next_month_index += len(data['coinjoins'])  # Store index of start fo next month (right after last index of current month)

            # Detect transactions with no remixes on input/out or both
            no_remix = als.detect_no_inout_remix_txs(data['coinjoins'])
            for key in no_remix.keys():
                no_remix_all[key].extend(no_remix[key])

            # Plot bars corresponding to different input types
            input_types_interval = als.plot_inputs_type_ratio(f'{mix_id} {dir_name}', data, initial_cj_index, ax, analyze_values, normalize_values, restrict_to_in_size)
            for input_type in input_types_interval:
                if input_type not in input_types.keys():
                    input_types[input_type] = []
                input_types[input_type].extend(input_types_interval[input_type])

            # Add current total mix liquidity into the same graph
            ax2 = ax.twinx()
            changing_liquidity_interval, stay_liquidity_interval, remix_liquidity_interval, changing_liquidity_timecutoff_interval, stay_liquidity_timecutoff_interval = (
                als.plot_mix_liquidity(f'{mix_id} {dir_name}', data, (changing_liquidity[-1], stay_liquidity[-1], remix_liquidity[-1], changing_liquidity_timecutoff[-1], stay_liquidity_timecutoff[-1]), time_liquidity, initial_cj_index, ax2))
            changing_liquidity.extend(changing_liquidity_interval)
            stay_liquidity.extend(stay_liquidity_interval)
            remix_liquidity.extend(remix_liquidity_interval)
            changing_liquidity_timecutoff.extend(changing_liquidity_timecutoff_interval)
            stay_liquidity_timecutoff.extend(stay_liquidity_timecutoff_interval)

            # Add fee rate into the same graph
            PLOT_FEERATE = True
            if PLOT_FEERATE:
                ax3 = ax.twinx()
                ax3.spines['right'].set_position(('outward', -30))  # Adjust position of the third axis
            else:
                ax3 = None
            mining_fee_rate_interval = als.plot_mining_fee_rates(f'{mix_id} {dir_name}', data, mining_fee_rates, ax3)
            mining_fee_rate.extend(mining_fee_rate_interval)

            PLOT_NUM_WALLETS = False
            if PLOT_NUM_WALLETS:
                ax3 = ax.twinx()
                ax3.spines['right'].set_position(('outward', -28))  # Adjust position of the third axis
            else:
                ax3 = None
            num_wallets_interval = als.plot_num_wallets(f'{mix_id} {dir_name}', data, ax3)
            num_wallets.extend(num_wallets_interval)

            initial_cj_index = initial_cj_index + len(data['coinjoins'])
            ax.set_title(f'Type of inputs for given cjtx ({'values' if analyze_values else 'number'})\n{mix_id} {dir_name}')
            logging.info(f'{target_base_path} inputs analyzed')

            # Compute liquidity inflows (sum of weeks)
            # Split cjtxs into weeks, then compute sum of MIX_ENTER
            for key, record in data['coinjoins'].items():
                # Parse the 'broadcast_time/virtual' string into a datetime object
                if mix_protocol == MIX_PROTOCOL.WASABI2:
                    dt = datetime.strptime(record['broadcast_time_virtual'], '%Y-%m-%d %H:%M:%S.%f')
                else:
                    dt = datetime.strptime(record['broadcast_time'], '%Y-%m-%d %H:%M:%S.%f')
                year, week_num, _ = dt.isocalendar()
                weeks_dict[(year, week_num)][key] = record
                day_key = (dt.year, dt.month, dt.day)
                days_dict[day_key][key] = record

            # Extend the y-limits to ensure the vertical lines go beyond the plot edges
            y_range = ax.get_ylim()
            padding = 0.02 * (y_range[1] - y_range[0])
            ax.set_ylim(y_range[0] - padding, y_range[1] + padding)

        prev_year = current_year

    def plot_allcjtxs_cummulative(ax, new_month_indices, changing_liquidity, changing_liquidity_timecutoff, stay_liquidity, remix_liquidity, mining_fee_rate, separators_to_plot: list):
        # Plot mining fee rate
        PLOT_FEERATE = False
        if PLOT_FEERATE:
            ax.plot(mining_fee_rate, color='gray', alpha=0.3, linewidth=1, linestyle=':', label='Mining fee (90th percentil)')
            ax.tick_params(axis='y', colors='gray', labelsize=6)
            ax.set_ylabel('Mining fee rate sats/vB (90th percentil)', color='gray', fontsize='6')

        def plot_bars_downscaled(values, downscalefactor, color, ax):
            downscaled_values = [sum(values[i:i + downscalefactor]) for i in range(0, len(values), downscalefactor)]
            downscaled_indices = range(0, len(values), downscalefactor)
            ax.bar(downscaled_indices, downscaled_values, color=color, width=downscalefactor, alpha=0.2, edgecolor='none')

        # Create artificial limits if not provided
        if restrict_to_in_size is None:
            limit_size = (0, 1000000000000)
            print(f'No limits for inputs value')
        else:
            limit_size = restrict_to_in_size
            print(f'Limits for inputs value is {limit_size[0]} - {limit_size[1]}')

        # Decide on resolution of liquidity display
        #interval_to_display = weeks_dict
        interval_to_display = days_dict

        def compute_aggregated_interval_liquidity(interval_to_display):
            liquidity = [0]
            for interval in sorted(interval_to_display.keys()):
                records = interval_to_display[interval]
                mix_enter_values = [records[cjtx]['inputs'][index]['value'] for cjtx in records.keys() for index in
                                    records[cjtx]['inputs'].keys()
                                    if records[cjtx]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_ENTER.name or
                                    records[cjtx]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name and
                                    limit_size[0] <= records[cjtx]['inputs'][index]['value'] <= limit_size[1]]
                liquidity.extend([sum(mix_enter_values) / SATS_IN_BTC] * len(records))
                print(f"Interval {interval}: {sum(mix_enter_values)}sats, num_cjtxs={len(records)}")
            return liquidity

        new_liquidity = compute_aggregated_interval_liquidity(interval_to_display)
        assert len(new_liquidity) == len(changing_liquidity), f'Incorrect enter_liquidity length: expected: {len(changing_liquidity)}, got {len(new_liquidity)}'
        plot_bars_downscaled(new_liquidity, 1, 'gray', ax)
        ax.set_title(f'{mix_id}: Liquidity dynamics in time')
        label = f'{'Fresh liquidity (btc)' if analyze_values else 'Number of inputs'} {'normalized' if normalize_values else ''}'
        ax.set_ylabel(label, color='gray', fontsize='6')
        ax.tick_params(axis='y', colors='gray')

        # Outflows
        # out_liquidity = [input_types[MIX_EVENT_TYPE.MIX_LEAVE.name][i] for i in range(len(input_types[MIX_EVENT_TYPE.MIX_LEAVE.name]))]
        # plot_bars_downscaled(out_liquidity, 1, 'red', ax)

        # Remix ratio
        remix_ratios_all = [input_types[MIX_EVENT_TYPE.MIX_REMIX.name][i] * 100 for i in
                            range(len(input_types[MIX_EVENT_TYPE.MIX_REMIX.name]))]  # All remix including nonstandard
        remix_ratios_nonstd = [input_types['MIX_REMIX_nonstd'][i] * 100 for i in
                               range(len(input_types['MIX_REMIX_nonstd']))]  # Nonstd remixes
        remix_ratios_std = [remix_ratios_all[i] - remix_ratios_nonstd[i] for i in
                            range(len(remix_ratios_all))]  # Only standard remixes
        WINDOWS_SIZE = round(len(remix_ratios_all) / 1000)  # Set windows size to get 1000 points total (unless short, then only 5)
        WINDOWS_SIZE = 1 if WINDOWS_SIZE < 1 else WINDOWS_SIZE
        remix_ratios_avg = [np.average(remix_ratios_std[i:i + WINDOWS_SIZE]) for i in
                            range(0, len(remix_ratios_std), WINDOWS_SIZE)]
        ax2 = ax.twinx()
        ax2.plot(range(0, len(remix_ratios_std), WINDOWS_SIZE), remix_ratios_avg, label=f'MIX_REMIX avg({WINDOWS_SIZE})',
                 color='brown', linewidth=1, linestyle='--', alpha=0.5)
        ax2.set_ylim(0, 100)  # Force whole range of yaxis
        ax2.tick_params(axis='y', colors='brown', labelsize=6)
        ax2.set_ylabel('Median remix rate %', color='brown', fontsize='6')
        ax2.spines['right'].set_position(('outward', -30))  # Adjust position of the third axis

        # Save computed remixes to file
        save_file = os.path.join(target_path,
                         f'{mix_id}_remixrate_{'values' if analyze_values else 'nums'}_{'norm' if normalize_values else 'notnorm'}{'' if restrict_to_in_size is None else f'{round(restrict_to_in_size[1] / SATS_IN_BTC, 2)}btc'}')
        als.save_json_to_file_pretty(f'{save_file}.json', {'remix_ratios_all': remix_ratios_all, 'remix_ratios_nonstd': remix_ratios_nonstd, 'remix_ratios_std': remix_ratios_std})

        # Plot changing liquidity in time
        ax2 = ax.twinx()
        changing_liquidity_btc = [item / SATS_IN_BTC for item in changing_liquidity]
        remix_liquidity_btc = [item / SATS_IN_BTC for item in remix_liquidity]
        stay_liquidity_btc = [item / SATS_IN_BTC for item in stay_liquidity]
        ax2.plot(changing_liquidity_btc, color='royalblue', alpha=0.6, label='Changing liquidity (cjtx centric, MIX_ENTER - MIX_LEAVE)')
        ax2.plot(stay_liquidity_btc, color='darkgreen', alpha=0.6, linestyle='--', label='Unmoved outputs (MIX_STAY)')
        #ax2.plot(remix_liquidity_btc, color='black', alpha=0.6, linestyle='--', label='Cummulative remix liquidity, MIX_ENTER - MIX_LEAVE - MIX_STAY')
        PLOT_CHAINANALYSIS_TIMECUTOFF = False
        if PLOT_CHAINANALYSIS_TIMECUTOFF:
            ax2.plot([a - b for a, b in zip([item / SATS_IN_BTC for item in changing_liquidity_timecutoff], [item / SATS_IN_BTC for item in stay_liquidity_timecutoff])], color='red', alpha=0.6, linestyle='-.', label='Actively remixed liquidity (Changing - Unmoved)')
        else:
            ax2.plot([a - b for a, b in zip(changing_liquidity_btc, stay_liquidity_btc)], color='red', alpha=0.6, linestyle='-.', label='Actively remixed liquidity (Changing - Unmoved)')
        ax2.set_ylabel('btc in mix', color='royalblue')
        ax2.tick_params(axis='y', colors='royalblue')

        ax3 = None
        PLOT_ESTIMATED_WALLETS = False
        if PLOT_ESTIMATED_WALLETS:
            # TODO: Compute wallets estimation based on inputs per time interval, not directly conjoins
            AVG_WINDOWS = 10
            num_wallets_avg = als.compute_averages(num_wallets, AVG_WINDOWS)
            AVG_WINDOWS_100 = 100
            num_wallets_avg100 = als.compute_averages(num_wallets, AVG_WINDOWS_100)
            ax3 = ax.twinx()
            ax3.spines['right'].set_position(('outward', -28))  # Adjust position of the third axis
            ax3.plot(num_wallets_avg, color='green', alpha=0.4, label=f'Estimated # wallets ({AVG_WINDOWS} avg)')
            ax3.plot(num_wallets_avg100, color='green', alpha=0.8, label=f'Estimated # wallets ({AVG_WINDOWS_100} avg)')
            ax3.set_ylabel('Estimated number of active wallets', color='green')
            ax3.tick_params(axis='y', colors='green')

        # Plot lines as separators corresponding to months
        for pos in new_month_indices:
            if pos[0] in separators_to_plot:
                PLOT_DAYS_MONTHS = False
                if pos[0] == 'day' or pos[0] == 'month' and PLOT_DAYS_MONTHS:
                    ax2.axvline(x=pos[1], color='gray', linewidth=0.5, alpha=0.1, linestyle='--')
                if pos[0] == 'year':
                    ax2.axvline(x=pos[1], color='gray', linewidth=1, alpha=0.4, linestyle='--')
        ax2.set_xticks([x[1] for x in new_month_indices])
        labels = []
        prev_year_offset = -10000
        for x in new_month_indices:
            if x[0] == 'year':
                if x[1] - prev_year_offset > 1000:
                    labels.append(f'{x[2][0:4]}')
                    prev_year_offset = x[1]
                else:
                    labels.append('')
            else:
                labels.append('')
        ax2.set_xticklabels(labels, rotation=45, fontsize=6)

        # if ax:
        #     ax.legend(loc='center left')
        if ax2:
            ax2.legend(loc='upper left', fontsize='small', bbox_to_anchor=(0.01, 0.85), borderaxespad=0)
        if ax3:
            ax3.legend()

    # Add additional cummulative plots for all coinjoin in one
    ax = fig.add_subplot(NUM_ROWS, NUM_COLUMNS, ax_index, axes_class=AA.Axes)  # Get next subplot
    ax_index += 1
    plot_allcjtxs_cummulative(ax, new_month_indices, changing_liquidity, stay_liquidity, stay_liquidity_timecutoff, remix_liquidity, mining_fee_rate, ['month', 'year'])

    # Finalize multigraph graph
    if plot_multigraph:
        plt.subplots_adjust(bottom=0.1, wspace=0.15, hspace=0.4)
        save_file = os.path.join(target_path, f'{mix_id}_input_types_{'values' if analyze_values else 'nums'}_{'norm' if normalize_values else 'notnorm'}{'' if restrict_to_in_size is None else f'{round(restrict_to_in_size[1]/SATS_IN_BTC, 2)}btc'}')
        plt.savefig(f'{save_file}.png', dpi=300)
        plt.savefig(f'{save_file}.pdf', dpi=300)
    plt.close()

    # Save generate and save cummulative results separately
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(1, 1, 1, axes_class=AA.Axes)  # Get next subplot
    plot_allcjtxs_cummulative(ax, new_month_indices, changing_liquidity, changing_liquidity_timecutoff, stay_liquidity, remix_liquidity, mining_fee_rate, ['month', 'year'])
    plt.subplots_adjust(bottom=0.1, wspace=0.15, hspace=0.4)
    save_file = os.path.join(target_path, f'{mix_id}_cummul_{'values' if analyze_values else 'nums'}_{'norm' if normalize_values else 'notnorm'}{'' if restrict_to_in_size is None else f'{round(restrict_to_in_size[1]/SATS_IN_BTC, 2)}btc'}')
    plt.savefig(f'{save_file}.png', dpi=300)
    plt.savefig(f'{save_file}.pdf', dpi=300)
    plt.close()

    # save detected no transactions with no remixes (potentially false positives)
    als.save_json_to_file_pretty(os.path.join(target_path, 'no_remix_txs.json'), no_remix_all)
    # Backup corresponding log file
    backup_log_files(target_path)


def wasabi_detect_false(target_path: Path, tx_file: str):
    files = os.listdir(target_path) if os.path.exists(target_path) else print(
        f'Path {target_path} does not exist')

    # Load false positives
    fp_file = os.path.join(target_path, 'false_cjtxs.json')
    false_cjtxs = als.load_json_from_file(fp_file)

    no_remix_all = {'inputs': [], 'outputs': [], 'both': []}
    for dir_name in files:
        target_base_path = os.path.join(target_path, dir_name)
        tx_json_file = os.path.join(target_base_path, f'{tx_file}')
        if os.path.isdir(target_base_path) and os.path.exists(tx_json_file):
            data = als.load_json_from_file(tx_json_file)

            # Filter false positives
            for false_tx in false_cjtxs:
                if false_tx in data['coinjoins'].keys():
                    data['coinjoins'].pop(false_tx)

            # Detect transactions with no remixes on input/out or both
            no_remix = als.detect_no_inout_remix_txs(data['coinjoins'])
            for key in no_remix.keys():
                no_remix_all[key].extend(no_remix[key])

    # save detected no transactions with no remixes (potentially false positives)
    als.save_json_to_file_pretty(os.path.join(target_path, 'no_remix_txs.json'), no_remix_all)

    #


def wasabi1_analyse_remixes(mix_id: str, target_path: str):
    data = als.load_json_from_file(os.path.join(target_path, mix_id, 'coinjoin_tx_info.json'))
    als.analyze_input_out_liquidity(data['coinjoins'], data['postmix'], [], MIX_PROTOCOL.WASABI1)

    wasabi1_analyze_fees(mix_id, data)
    inputs_value_burntime_heatmap(mix_id, data)
    burntime_histogram(mix_id, data)


def fix_ww2_for_fdnp_ww1(mix_id: str, target_path: Path):
    """
    Detects and corrects all information of WW2 extracted from coinjoin_tx_info.json based on WW1 inflows.
    Process also subfolders with monthly intervals
    :param mix_id:
    :param target_path:
    :return:
    """
    #'wasabi2', target_path, os.path.join(target_path, 'wasabi1_burn', 'coinjoin_tx_info.json.full'))
    # Load Wasabi1 files, then update MIX_ENTER for Wasabi2 where friends-do-not-pay rule does not apply
    ww1_coinjoins = load_coinjoin_stats_from_file(os.path.join(target_path, 'WasabiCoinJoins.txt'))
    ww1_postmix_spend = load_coinjoin_stats_from_file(os.path.join(target_path, 'WasabiPostMixTxs.txt'))

    target_path = os.path.join(target_path, mix_id)  # Go into target ww2 folder

    paths_to_process = [target_path]  # Always process 'coinjoin_tx_info.json' with all transactions
    # paths_to_process = []  # Do not process large .json with all txs
    # Add subpaths for months if present
    files = os.listdir(target_path)
    for file_name in files:
        target_base_path = os.path.join(target_path, file_name)
        tx_json_file = os.path.join(target_base_path, f'coinjoin_tx_info.json')
        if os.path.isdir(target_base_path) and os.path.exists(tx_json_file):
            paths_to_process.append(target_base_path)

    # Now fix all prepared paths
    for path in paths_to_process:
        logging.info(f'Processing {path}...')
        ww2_data = als.load_json_from_file(os.path.join(path, f'coinjoin_tx_info.json'))

        # For all values with mix_event_type equal to MIX_ENTER check if they are not from WW1
        # with friends-do-not-pay rule
        coinjoins = ww2_data['coinjoins']

        total_ww1_inputs = 0
        for cjtx in coinjoins:
            for input in coinjoins[cjtx]['inputs']:
                if coinjoins[cjtx]['inputs'][input]['mix_event_type'] == MIX_EVENT_TYPE.MIX_ENTER.name:
                    if 'spending_tx' in coinjoins[cjtx]['inputs'][input].keys():
                        spending_tx, index = als.extract_txid_from_inout_string(coinjoins[cjtx]['inputs'][input]['spending_tx'])
                        if spending_tx in ww1_coinjoins.keys() or spending_tx in ww1_postmix_spend.keys():
                            # Friends do not pay rule tx - change to MIX_REMIX_FRIENDS_WW1
                            coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name
                            total_ww1_inputs += 1

        print(f'Total WW1 inputs with friends-do-not-pay rule: {total_ww1_inputs}')

        als.save_json_to_file(os.path.join(path, f'coinjoin_tx_info.json'), ww2_data)


def extract_flows_blocksci(flows: dict):
    start_year = 2019
    end_year = 2024

    flow_types = sorted(set([item['flow_direction'] for item in flows]))
    flows_in_year = {'broadcast_time_mix1': {}, 'broadcast_time_mix2': {}, 'broadcast_time_bridge': {}}
    for time_type in flows_in_year.keys():
        flows_in_year[time_type] = {flow_type: {} for flow_type in flow_types}
        for flow_type in flow_types:
            for year in range(start_year, end_year + 1):
                flows_in_year[time_type][flow_type][year] = {}
                for month in range(1, 12 + 1):
                    flows_in_year[time_type][flow_type][year][month] = {}

    for flow_type in flow_types:
        for year in range(start_year, end_year + 1):
            for month in range(1, 12 + 1):
                # Aggregated by time when bridging transaction was send
                flows_in_year['broadcast_time_bridge'][flow_type][year][month] = sum(
                    [item['sats_moved'] for item in flows if item['flow_direction'] == flow_type and
                     precomp_datetime.strptime(item['broadcast_time'], "%Y-%m-%dT%H:%M:%S").year == year and
                     precomp_datetime.strptime(item['broadcast_time'], '%Y-%m-%dT%H:%M:%S').month == month
                     ])
                # Aggregated by time when tx from mix2 was executed
                flows_in_year['broadcast_time_mix2'][flow_type][year][month] = sum(
                    [item['sats_moved'] for item in flows if item['flow_direction'] == flow_type and
                     precomp_datetime.strptime(item['out_cjs'][list(item['out_cjs'].keys())[0]]['broadcast_time'], "%Y-%m-%dT%H:%M:%S").year == year and
                     precomp_datetime.strptime(item['out_cjs'][list(item['out_cjs'].keys())[0]]['broadcast_time'], '%Y-%m-%dT%H:%M:%S').month == month
                     ])
                # # Aggregated by time when tx from mix2 was executed
                # flows_in_year['broadcast_time_mix2'][flow_type][year][month] = sum(
                #     [item['out_cjs'][txid]['value'] for item in flows for txid in item['out_cjs'].keys()
                #      if item['flow_direction'] == flow_type and
                #      precomp_datetime.strptime(item['out_cjs'][txid]['broadcast_time'], "%Y-%m-%dT%H:%M:%S").year == year and
                #      precomp_datetime.strptime(item['out_cjs'][txid]['broadcast_time'], '%Y-%m-%dT%H:%M:%S').month == month
                #      ])
                # # Aggregated by time when tx from mix1 was executed
                # Do not use, as not all outflows from mix1 are necessarily going to mix2
                # flows_in_year['broadcast_time_mix1'][flow_type][year][month] = sum(
                #     [item['in_cjs'][txid]['value'] for item in flows for txid in item['in_cjs'].keys()
                #      if item['flow_direction'] == flow_type and
                #      precomp_datetime.strptime(item['in_cjs'][txid]['broadcast_time'], "%Y-%m-%dT%H:%M:%S").year == year and
                #      precomp_datetime.strptime(item['in_cjs'][txid]['broadcast_time'], '%Y-%m-%dT%H:%M:%S').month == month
                #      ])

    return flows_in_year


def extract_flows_dumplings(flows: dict):
    start_year = 2019
    end_year = 2024

    flow_in_year = {}
    for flow_type in flows.keys():
        flow_in_year[flow_type] = {}
        for year in range(start_year, end_year + 1):
            flow_in_year[flow_type][year] = {}
            for month in range(1, 12 + 1):
                flow_in_year[flow_type][year][month] = sum(
                    [flows[flow_type][txid]['value'] for txid in flows[flow_type].keys()
                     if precomp_datetime.strptime(flows[flow_type][txid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f").year == year and
                     precomp_datetime.strptime(flows[flow_type][txid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f").month == month
                     ])

    return flow_in_year


def plot_flows_steamgraph(flow_in_year: dict, title: str):
    start_year = 2019
    end_year = 2024

    flow_types = sorted([flow_type for flow_type in flow_in_year.keys()])

    COLORS = ['gray', 'green', 'olive', 'black', 'red', 'orange']
    fig, ax = plt.subplots(figsize=(10, 5))
    # end_year in x_axis must be + 1 to correct for 0 index in flow_data
    x_axis = np.linspace(start_year, end_year + 1, num=(end_year - start_year + 1) * 12)

    DRAW_WW1_WW2_FLOW = False
    if DRAW_WW1_WW2_FLOW:
        flow_data_1 = []
        flow_types_process_1 = ['Wasabi -> Wasabi2']
        for flow_type in flow_types_process_1:
            case_data = [round(flow_in_year[flow_type][year][month] / SATS_IN_BTC, 2) for year in flow_in_year[flow_type].keys()
                         for month in range(1, 13)]
            flow_data_1.append(case_data)
        ax.stackplot(x_axis, flow_data_1, labels=list(flow_types_process_1), colors=COLORS, baseline="sym", alpha=0.4)

    if DRAW_WW1_WW2_FLOW:
        flow_types_process_2 = [item for item in flow_types if item != 'Wasabi -> Wasabi2']
    else:
        flow_types_process_2 = [item for item in flow_types]
    flow_data_2 = []
    flow_data_labels_2 = []
    for flow_type in flow_types_process_2:
        case_data = [round(flow_in_year[flow_type][year][month] / SATS_IN_BTC, 2) for year in flow_in_year[flow_type].keys() for month in range(1, 13)]
        flow_data_2.append(case_data)
        flow_data_labels_2.append(f'{flow_type} ({sum(case_data)} btc)')
        assert len(case_data) == (end_year - start_year + 1) * 12
    if DRAW_WW1_WW2_FLOW:
        ax.stackplot(x_axis, flow_data_2, labels=flow_data_labels_2, colors=COLORS[1:], baseline="sym", alpha=0.7)
    else:
        ax.stackplot(x_axis, flow_data_2, labels=flow_data_labels_2, colors=COLORS, baseline="sym", alpha=0.7)

    ax.legend(loc="lower left")
    ax.set_title(title)
    #ax.set_yscale('log')  # If enabled, it does not plot correctly, possibly bug in mathplotlib
    plt.show()

    PLOT_SMOOTH = False
    if PLOT_SMOOTH:
        def gaussian_smooth(x, y, grid, sd):
            weights = np.transpose([stats.norm.pdf(grid, m, sd) for m in x])
            weights = weights / weights.sum(0)
            return (weights * y).sum(1)

        fig, ax = plt.subplots(figsize=(10, 5))
        grid = np.linspace(start_year, end_year + 1, num=1000)
        y_smoothed = [gaussian_smooth(x_axis, y_, grid, 0.05) for y_ in flow_data_1]
        ax.stackplot(grid, y_smoothed, labels=list(flow_types_process_1), colors=COLORS, baseline="sym", alpha=0.3)
        y_smoothed = [gaussian_smooth(x_axis, y_, grid, 0.05) for y_ in flow_data_2]
        ax.stackplot(grid, y_smoothed, labels=list(flow_types_process_2), colors=COLORS[1:], baseline="sym", alpha=0.7)
        ax.legend(loc="lower left")
        ax.set_title(title)
        plt.show()


def plot_flows_dumplings(flows: dict):
    num_flow_types = len(flows.keys())
    start_year = 2018
    end_year = 2024
    # num_months = (end_year - start_year)*12
    x = np.arange(start_year, end_year, 12)  # (N,) array-like
    np.random.seed(42)
    y = [np.random.randint(0, 5, size=end_year - start_year) for _ in range(num_flow_types)]

    flow_in_year = {}
    for flow_type in flows.keys():
        flow_in_year[flow_type] = {}
        for year in range(start_year, end_year + 1):
            flow_in_year[flow_type][year] = {}
            for month in range(1, 12 + 1):
                flow_in_year[flow_type][year][month] = sum(
                    [flows[flow_type][txid]['value'] for txid in flows[flow_type].keys()
                     if precomp_datetime.strptime(flows[flow_type][txid]['broadcast_time'],
                                                  "%Y-%m-%d %H:%M:%S.%f").year == year and
                     precomp_datetime.strptime(flows[flow_type][txid]['broadcast_time'],
                                               "%Y-%m-%d %H:%M:%S.%f").month == month
                     ])
    def gaussian_smooth(x, y, grid, sd):
        weights = np.transpose([stats.norm.pdf(grid, m, sd) for m in x])
        weights = weights / weights.sum(0)
        return (weights * y).sum(1)

    flow_data = []
    for flow_type in flows.keys():
        case_data = [flow_in_year[flow_type][year][month] for year in flow_in_year[flow_type].keys() for month in range(1, 13)]
        flow_data.append(case_data)
        assert len(case_data) == (end_year - start_year + 1) * 12
    #COLORS = sns.color_palette("twilight_shifted", n_colors=len(flow_data))
    COLORS = sns.color_palette("RdYlGn", n_colors=len(flow_data))
    COLORS = ['red', 'orange', 'green', 'olive', 'gray', 'black']

    fig, ax = plt.subplots(figsize=(10, 5))
    # end_year in x_axis must be + 1 to correct for 0 index in flow_data
    x_axis = np.linspace(start_year, end_year + 1, num=(end_year - start_year + 1) * 12)
    ax.stackplot(x_axis, flow_data, labels=list(flows.keys()), colors=COLORS, baseline="sym", alpha=1)
    ax.legend(loc="lower left")
    #ax.set_yscale('log')  # If enabled, it does not plot correctly, possibly bug in mathplotlib
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    grid = np.linspace(start_year - 1, end_year + 1, num=500)
    y_smoothed = [gaussian_smooth(x, y_, grid, 0.1) for y_ in flow_data]
    ax.stackplot(grid, y_smoothed, labels=list(flows.keys()), colors=COLORS, baseline="sym", alpha=0.7)

    ax.legend()
    plt.show()


def plot_steamgraph_example():
    x = np.arange(1990, 2020)  # (N,) array-like
    y = [np.random.randint(0, 5, size=30) for _ in range(5)]  # (M, N) array-like

    def gaussian_smooth(x, y, grid, sd):
        weights = np.transpose([stats.norm.pdf(grid, m, sd) for m in x])
        weights = weights / weights.sum(0)
        return (weights * y).sum(1)

    COLORS = ["#D0D1E6", "#A6BDDB", "#74A9CF", "#2B8CBE", "#045A8D"]
    fig, ax = plt.subplots(figsize=(10, 7))
    grid = np.linspace(1985, 2025, num=500)
    y_smoothed = [gaussian_smooth(x, y_, grid, 1) for y_ in y]
    ax.stackplot(grid, y_smoothed, colors=COLORS, baseline="sym")
    plt.show()


def analyze_mixes_flows(target_path):
    flows_file = os.path.join(target_path, 'one_hop_flows_misclassifications.json')
    if os.path.exists(flows_file):
        flows = als.load_json_from_file(flows_file)
        print(f'Total misclassifications: {len(flows.keys())}')

    # Visualization of results from BlockSci
    flows_file = os.path.join(target_path, 'one_hop_flows.json')
    if os.path.exists(flows_file):
        flows = als.load_json_from_file(flows_file)
        flows_in_time = extract_flows_blocksci(flows)
        #plot_flows_steamgraph(flows_in_time['broadcast_time_mix1'], 'BlockSci flows (1 hop), mix1')
        plot_flows_steamgraph(flows_in_time['broadcast_time_bridge'], 'BlockSci flows (1 hop), bridge tx time')
        plot_flows_steamgraph(flows_in_time['broadcast_time_mix2'], 'BlockSci flows (1 hop), mix2 tx time')

    TWO_HOPS = False
    if TWO_HOPS:
        flows_file = os.path.join(target_path, 'two_hops_flows.json')
        if os.path.exists(flows_file):
            flows = als.load_json_from_file(flows_file)
            flows_in_time = extract_flows_blocksci(flows)
            plot_flows_steamgraph(flows_in_time, 'BlockSci flows (2 hops)')

    # Visualization of results from Dumplings
    flows_file = os.path.join(target_path, 'mix_flows.json')
    if os.path.exists(flows_file):
        flows = als.load_json_from_file(flows_file)
        flows_in_time = extract_flows_dumplings(flows)
        plot_flows_steamgraph(flows_in_time, 'Dumplings flows (1 hop)')
    else:
        whirlpool_postmix = load_coinjoin_stats_from_file(os.path.join(target_path, 'SamouraiPostMixTxs.txt'))
        wasabi1_postmix = load_coinjoin_stats_from_file(os.path.join(target_path, 'WasabiPostMixTxs.txt'))
        wasabi2_postmix = load_coinjoin_stats_from_file(os.path.join(target_path, 'Wasabi2PostMixTxs.txt'))

        wasabi1_cj = load_coinjoin_stats_from_file(os.path.join(target_path, 'WasabiCoinJoins.txt'))
        wasabi2_cj = load_coinjoin_stats_from_file(os.path.join(target_path, 'Wasabi2CoinJoins.txt'))
        whirlpool_cj = load_coinjoin_stats_from_file(os.path.join(target_path, 'SamouraiCoinJoins.txt'))

        whirlpool_premix = load_coinjoin_stats_from_file(os.path.join(target_path, 'SamouraiTx0s.txt'))

        def load_premix_tx_dict(target_path, file_name, full_tx_dict):
            """
            Optimized computation or loading of precomputed list of premix transaction ids extracted from all inputs
            :param target_path: folder path for loading/saving
            :param file_name: target file name
            :param full_tx_dict: dictionary with all transactions and inputs from which premix txs are extracted
            :return: dictionary with unique premix txs
            """
            json_file = os.path.join(target_path, file_name)
            if os.path.exists(json_file):
                with open(json_file, "rb") as file:
                    return pickle.load(file)
            else:
                txs = list({full_tx_dict[txid]['inputs'][index]['spending_tx'] for txid in full_tx_dict.keys() for
                                       index in full_tx_dict[txid]['inputs'].keys()})
                tx_dict = {als.extract_txid_from_inout_string(item)[0]: [] for item in txs}
                with open(json_file, "wb") as file:
                    pickle.dump(tx_dict, file)
                return tx_dict

        wasabi1_premix_dict = load_premix_tx_dict(target_path, 'wasabi1_premix_dict.json', wasabi1_cj)
        wasabi2_premix_dict = load_premix_tx_dict(target_path, 'wasabi2_premix_dict.json', wasabi2_cj)

        # Precompute dictionary with full name (vout_txid_index and vin_txid_index) for quick queries if given 'spending_tx' and 'spend_by_tx' are included
        # Precompute for quick queries 'spending_tx' existence
        wasabi1_vout_txid_index = {als.get_output_name_string(txid, index) for txid in wasabi1_cj.keys() for index in wasabi1_cj[txid]['outputs'].keys()}
        wasabi2_vout_txid_index = {als.get_output_name_string(txid, index) for txid in wasabi2_cj.keys() for index in wasabi2_cj[txid]['outputs'].keys()}
        whirlpool_vout_txid_index = {als.get_output_name_string(txid, index) for txid in whirlpool_cj.keys() for index in whirlpool_cj[txid]['outputs'].keys()}
        # Precompute for quick queries 'spend_by_tx' existence
        wasabi1_vin_txid_index = {wasabi1_cj[txid]['inputs'][index]['spending_tx'] for txid in wasabi1_cj.keys() for index in wasabi1_cj[txid]['inputs'].keys()}
        wasabi2_vin_txid_index = {wasabi2_cj[txid]['inputs'][index]['spending_tx'] for txid in wasabi2_cj.keys() for index in wasabi2_cj[txid]['inputs'].keys()}
        whirlpool_vin_txid_index = {whirlpool_cj[txid]['inputs'][index]['spending_tx'] for txid in whirlpool_cj.keys() for index in whirlpool_cj[txid]['inputs'].keys()}

        # Analyze flows
        flows = {}
        flows['Whirlpool -> Wasabi1'] = analyze_extramix_flows('Whirlpool -> Wasabi1', target_path, whirlpool_vout_txid_index, whirlpool_postmix, wasabi1_premix_dict, wasabi1_vin_txid_index)
        flows['Whirlpool -> Wasabi2'] = analyze_extramix_flows('Whirlpool -> Wasabi2', target_path, whirlpool_vout_txid_index, whirlpool_postmix, wasabi2_premix_dict, wasabi2_vin_txid_index)
        flows['Wasabi1 -> Whirlpool'] = analyze_extramix_flows('Wasabi1 -> Whirlpool', target_path, wasabi1_vout_txid_index, wasabi1_postmix, whirlpool_premix, whirlpool_vin_txid_index)
        flows['Wasabi -> Wasabi2'] = analyze_extramix_flows('Wasabi1 -> Wasabi2', target_path, wasabi1_vout_txid_index, wasabi1_postmix, wasabi2_premix_dict, wasabi2_vin_txid_index)
        flows['Wasabi2 -> Whirlpool'] = analyze_extramix_flows('Wasabi2 -> Whirlpool', target_path, wasabi2_vout_txid_index, wasabi2_postmix, whirlpool_premix, whirlpool_vin_txid_index)
        flows['Wasabi2 -> Wasabi1'] = analyze_extramix_flows('Wasabi2 -> Wasabi1', target_path, wasabi2_vout_txid_index, wasabi2_postmix, wasabi1_premix_dict, wasabi1_vin_txid_index)
        # analyze_extramix_flows('Wasabi1 -> Wasabi1', target_path, wasabi1_postmix, wasabi1_premix_dict)
        # analyze_extramix_flows('Wasabi2 -> Wasabi2', target_path, wasabi2_postmix, wasabi2_premix_dict)
        # analyze_extramix_flows('Whirlpool -> Whirlpool', target_path, whirlpool_postmix, whirlpool_premix)

        als.save_json_to_file_pretty(os.path.join(target_path, 'mix_flows.json'), flows)
        flows_in_time = extract_flows_dumplings(flows)
        plot_flows_steamgraph(flows_in_time, 'Dumplings flows')


def analyze_extramix_flows(experiment_id: str, target_path: Path, mix1_precomp_vout_txid_index: dict, mix1_postmix: dict, mix2_premix: dict, mix2_precomp_vin_txid_index: dict):
    # (non-strict, 1-hop case): Mix1 coinjoin output (mix1_coinjoin_file) -> Mix2 wallet (mix1_postmix_file, mix2_premix_file) -> Mix2 coinjoin input (mix2_coinjoin_file)
    logging.info(f'{experiment_id} (non-strict, 1-hop case): #mix1 postmix txs = {len(mix1_postmix.keys())}, #mix2 premix txs {len(mix2_premix)}')
    mix1_mix2_txs = list(set(list(mix1_postmix.keys())).intersection(list(mix2_premix.keys())))
    logging.info(f'{experiment_id} (non-strict, 1-hop case): {len(mix1_mix2_txs)} txs')

    # Iterate over shared bridging transactions (mix1->shared_tx->mix2), take minimum from (outflow_first_mix, inflow_second_mix)
    # Compute sum of values for all inputs taking only these inputs coming from mix1
    flow_sizes = {}
    for inter_txid in mix1_mix2_txs:
        from_mix1 = sum([mix1_postmix[inter_txid]['inputs'][index]['value'] for index in mix1_postmix[inter_txid]['inputs'].keys()
                            if mix1_postmix[inter_txid]['inputs'][index]['spending_tx'] in mix1_precomp_vout_txid_index])
        to_mix2 = sum([mix1_postmix[inter_txid]['outputs'][index]['value'] for index in mix1_postmix[inter_txid]['outputs'].keys()
                            if als.get_output_name_string(inter_txid, index) in mix2_precomp_vin_txid_index])
        assert from_mix1 > 0 and to_mix2 > 0, f'Invalid sum of intermix inputs/outputs for {inter_txid}:  {from_mix1} vs {to_mix2}'
        # Fill record
        flow_sizes[inter_txid] = {'broadcast_time': mix1_postmix[inter_txid]['broadcast_time'],
                                  'value': min(from_mix1, to_mix2)}

        # Inflows are always bit smaller than inflows due to mining fees. Detect and print bridging txs with significant difference
        MINING_FEE_LIMIT = 0.01  # 1%
        if from_mix1 - to_mix2 > from_mix1 * MINING_FEE_LIMIT:
            logging.debug(f'Mix2 inflow significantly SMALLER than mix1 outflow for {inter_txid}: {from_mix1} vs {to_mix2}')
        if (to_mix2 - from_mix1) > to_mix2 * MINING_FEE_LIMIT:
            logging.debug(f'Mix2 inflow significantly LARGER than mix1 outflow for {inter_txid}: {from_mix1} vs {to_mix2}')

    sum_all_flows = sum([flow_sizes[txid]['value'] for txid in flow_sizes.keys()])
    logging.info(f'{experiment_id} (non-strict, 1-hop case): {sum_all_flows} sats / {round(sum_all_flows / SATS_IN_BTC, 2)} btc')

    return flow_sizes



def whirlpool_extract_pool(full_data: dict, mix_id: str, target_path: Path, pool_id: str, pool_size: int):
    # Start from initial tx for specific pool size
    # Add iteratively additional transactions if connected to already included ones
    all_cjtxs_keys = full_data['coinjoins'].keys()
    # Initial seeding for given pool size
    pool_txs = {cjtx: full_data['coinjoins'][cjtx] for cjtx in WHIRLPOOL_FUNDING_TXS[pool_size]['funding_txs']}
    # Initial premix txs
    pool_premix_txs = {}
    txs_to_probe = list(pool_txs.keys())
    while len(txs_to_probe) > 0:
        next_txs_to_probe = []
        for cjtx in txs_to_probe:
            for output in pool_txs[cjtx]['outputs'].keys():
                if 'spend_by_tx' in pool_txs[cjtx]['outputs'][output].keys():
                    txid, index = als.extract_txid_from_inout_string(pool_txs[cjtx]['outputs'][output]['spend_by_tx'])
                    if txid not in pool_txs.keys() and txid in all_cjtxs_keys:
                        next_txs_to_probe.append(txid)
                        pool_txs[txid] = full_data['coinjoins'][txid]

                        # If Whirlpool, check all inputs for this tx if is premix
                        if 'premix' in full_data.keys():
                            for input in full_data['coinjoins'][txid]['inputs'].keys():
                                if 'spending_tx' in full_data['coinjoins'][txid]['inputs'][input].keys():
                                    txid_premix, index = als.extract_txid_from_inout_string(full_data['coinjoins'][txid]['inputs'][input]['spending_tx'])
                                    if txid_premix not in pool_premix_txs.keys() and txid_premix in full_data['premix'].keys():
                                        pool_premix_txs[txid_premix] = full_data['premix'][txid_premix]

        if len(pool_txs.keys()) % 1000 == 0:
            logging.info(f'Discovered {len(pool_txs)} cjtxs for pool {pool_size}')

        txs_to_probe = next_txs_to_probe
    logging.info(f'Total cjtxs extracted for pool {pool_size}: {len(pool_txs)}')

    target_save_path = os.path.join(target_path, pool_id)
    logging.info(f'Saving to {target_save_path}/coinjoin_tx_info.json ...')
    if not os.path.exists(target_save_path):
        os.makedirs(target_save_path.replace('\\', '/'))
    als.save_json_to_file(os.path.join(target_save_path, 'coinjoin_tx_info.json'), {'coinjoins': pool_txs, 'premix': pool_premix_txs})

    # Backup corresponding log file
    backup_log_files(target_path)

    return {'coinjoins': pool_txs, 'premix': pool_premix_txs}


def backup_log_files(target_path: str):
    """
    This code runs before exiting
    :return:
    """
    # Copy logs file into base
    print(os.path.abspath(__file__))
    log_file_path = f'{os.path.abspath(__file__)}.log'
    if os.path.exists(log_file_path):
        file_name = os.path.basename(log_file_path)
        shutil.copy(os.path.join(log_file_path), os.path.join(target_path, f'{file_name}.{random.randint(10000, 99999)}.txt'))
    else:
        logging.warning(f'Log file {log_file_path} does not found, not copied.')


def compute_stats(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: Path):
    data = als.load_coinjoins_from_file(target_path, None, True)

    sorted_cjtxs = als.sort_coinjoins(data['coinjoins'], True)
    num_cjtxs = [len(data['coinjoins'][cjtx['txid']]['inputs']) for cjtx in sorted_cjtxs]

    def compute_corr(input_series: list, window_size: int):
        input_series_windowed = [np.sum(input_series[i:i+window_size]) for i in range(0, len(input_series), window_size)]
        data = pd.Series(input_series_windowed)
        # Shift the series by one position
        shifted_data = data.shift(1)
        # Drop the NaN value
        original_data = data[1:]
        shifted_data = shifted_data[1:]
        # Calculate the Pearson correlation
        correlation = original_data.corr(shifted_data)
        print(f'Correlation {window_size} = {correlation}')

        data = np.array(input_series_windowed)
        # Compute autocorrelation using numpy's correlate function
        autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')

        # Normalize the result
        autocorr = autocorr / (np.var(data) * len(data))

        # We only need the second half of the result (non-negative lags)
        autocorr = autocorr[len(autocorr) // 2:]

        # Print the autocorrelation values
        print("Autocorrelation values:", autocorr)

        # Optionally, plot the autocorrelation
        plt.plot(autocorr)
        plt.title('Autocorrelation')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.show()

    for i in range(1, 5):
        compute_corr(num_cjtxs, i)


if __name__ == "__main__":
    # Limit analysis only to specific coinjoin type
    CONSIDER_WW1 = False
    CONSIDER_WW2 = False
    CONSIDER_WHIRLPOOL = True

    # Sorting strategy for coinjoins in time.
    # If False, coinjoins are sorted using 'broadcast_time' (which is equal to mining_time for on-chain cjtxs where we lack real broadcast time)
    # If True, then relative ordering based on connections in graph formed by remix inputs/outputs is used
    # BUGBUG: SORT_COINJOINS_BY_RELATIVE_ORDER prevents corrects use of CONSIDER_WW2 and CONSIDER_WW1 / CONSIDER_WHIRLPOOL at the same time
    if CONSIDER_WW2:
        SORT_COINJOINS_BY_RELATIVE_ORDER = True
    else:
        SORT_COINJOINS_BY_RELATIVE_ORDER = False
    als.SORT_COINJOINS_BY_RELATIVE_ORDER = SORT_COINJOINS_BY_RELATIVE_ORDER

    FULL_TX_SET = False

    ANALYSIS_PROCESS_ALL_COINJOINS_INTERVALS = False
    PROCESS_NOTABLE_INTERVALS = False
    DETECT_FALSE_POSITIVES = False
    SPLIT_WHIRLPOOL_POOLS = False
    PLOT_REMIXES = True
    PLOT_REMIXES_FLOWS = False
    ANALYSIS_ADDRESS_REUSE = False
    ANALYSIS_PROCESS_ALL_COINJOINS = False
    ANALYSIS_PROCESS_ALL_COINJOINS_DEBUG = False
    ANALYSIS_PROCESS_ALL_COINJOINS_INTERVALS_DEBUG = False
    ANALYSIS_INPUTS_DISTRIBUTION = False  # OK
    ANALYSIS_BURN_TIME = False
    PLOT_INTERMIX_FLOWS = False
    VISUALIZE_ALL_COINJOINS_INTERVALS = False

    target_base_path = 'c:\\!blockchains\\CoinJoin\\Dumplings_Stats_20240215\\'
    target_base_path = 'c:\\!blockchains\\CoinJoin\\Dumplings_Stats_20240417\\'
    target_base_path = 'c:\\!blockchains\\CoinJoin\\Dumplings_Stats_20240509\\'
    target_base_path = 'c:\\!blockchains\\CoinJoin\\Dumplings_Stats_20240605\\'
    target_base_path = 'c:\\!blockchains\\CoinJoin\\Dumplings_Stats_20240701\\'
    target_base_path = 'c:\\!blockchains\\CoinJoin\\Dumplings_Stats_20240802\\'
    interval_stop_date = '2024-08-03 00:00:07.000'

    target_path = os.path.join(target_base_path, 'Scanner')
    SM.print(f'Starting analysis of {target_path}, FULL_TX_SET={FULL_TX_SET}, SAVE_BASE_FILES_JSON={SAVE_BASE_FILES_JSON}')

    DEBUG = True
    if DEBUG:
        wasabi_plot_remixes('whirlpool_5M_test', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool_5M_test'), 'coinjoin_tx_info.json',
                            True, False, None, None, False)
        exit(42)
        SORT_COINJOINS_BY_RELATIVE_ORDER = False
        als.SORT_COINJOINS_BY_RELATIVE_ORDER = SORT_COINJOINS_BY_RELATIVE_ORDER
        target_path = 'c:\\!blockchains\\CoinJoin\\Dumplings_Stats_20240701\\Scanner\\'
        wasabi_plot_remixes('wasabi1', MIX_PROTOCOL.WASABI1, os.path.join(target_path, 'wasabi1'), 'coinjoin_tx_info.json',
                            True, False, None, None, False)
        exit(42)
        SORT_COINJOINS_BY_RELATIVE_ORDER = True
        als.SORT_COINJOINS_BY_RELATIVE_ORDER = SORT_COINJOINS_BY_RELATIVE_ORDER
        wasabi_plot_remixes('wasabi2', MIX_PROTOCOL.WASABI2, os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json',
                            True, False, None, None, False)
        SORT_COINJOINS_BY_RELATIVE_ORDER = False
        als.SORT_COINJOINS_BY_RELATIVE_ORDER = SORT_COINJOINS_BY_RELATIVE_ORDER
        wasabi_plot_remixes('whirlpool_100k', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool_100k'), 'coinjoin_tx_info.json',
                            True, False, None, None, False)
        wasabi_plot_remixes('whirlpool_1M', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool_1M'), 'coinjoin_tx_info.json',
                            True, False, None, None, False)
        wasabi_plot_remixes('whirlpool_5M', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool_5M'), 'coinjoin_tx_info.json',
                            True, False, None, None, False)
        wasabi_plot_remixes('whirlpool_50M', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool_50M'), 'coinjoin_tx_info.json',
                            True, False, None, None, False)
        wasabi_plot_remixes('whirlpool', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool'), 'coinjoin_tx_info.json',
                            True, False, None, None, False)
        target_path = 'c:\\!blockchains\\CoinJoin\\Dumplings_Stats_20240701\\Scanner\\'
        wasabi_plot_remixes('wasabi1', MIX_PROTOCOL.WASABI1, os.path.join(target_path, 'wasabi1'), 'coinjoin_tx_info.json',
                            True, False, None, None, False)
        exit(42)
        wasabi_plot_remixes('wasabi2', MIX_PROTOCOL.WASABI2, os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json',
                            True, False, None, None, False)
        wasabi_plot_remixes('whirlpool_5M', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool_50M'), 'coinjoin_tx_info.json',
                            True, False, None, None, False)
        exit(42)
        wasabi_plot_remixes('whirlpool_50M', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool_50M'), 'coinjoin_tx_info.json',
                            True, False, None, None, False)
        wasabi_plot_remixes('wasabi2', MIX_PROTOCOL.WASABI2, os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json',
                            True, False, None, None, False)
        exit(42)
        #process_inputs_distribution2('wasabi2_test', MIX_PROTOCOL.WASABI2, target_path, 'Wasabi2CoinJoins.txt', True)
        process_estimated_wallets_distribution('wasabi2', target_path, [1.8, 2.0, 2.3, 2.7], True)
        exit(42)

        process_and_save_intervals_filter('wasabi2_as25', MIX_PROTOCOL.WASABI2, target_path, '2024-05-01 00:00:07.000',
                                          '2024-06-01 00:00:07.000',
                                          'Wasabi2CoinJoins.txt', 'Wasabi2PostMixTxs.txt', None, SAVE_BASE_FILES_JSON,
                                          False)
        fix_ww2_for_fdnp_ww1('wasabi2_as25', target_path)  # WW2 requires detection of WW1 inflows as friends
        exit(42)
        wasabi_plot_remixes('wasabi1', os.path.join(target_path, 'wasabi1'), 'coinjoin_tx_info.json', False, False, None,
                            None, True)
        wasabi_plot_remixes('wasabi2', os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json', False, False, None,
                            None, True)
        exit(42)
        wasabi_plot_remixes('whirlpool_100k', os.path.join(target_path, 'whirlpool_100k'), 'coinjoin_tx_info.json',
                            True, False, None, None, True)
        wasabi_plot_remixes('whirlpool_1M', os.path.join(target_path, 'whirlpool_1M'), 'coinjoin_tx_info.json',
                            True, False, None, None, True)
        wasabi_plot_remixes('whirlpool_5M', os.path.join(target_path, 'whirlpool_5M'), 'coinjoin_tx_info.json',
                            True, False, None, None, True)
        wasabi_plot_remixes('whirlpool_50M', os.path.join(target_path, 'whirlpool_50M'), 'coinjoin_tx_info.json',
                            True, False, None, None, True)

        exit(42)
        wasabi_plot_remixes('wasabi1', os.path.join(target_path, 'wasabi1'), 'coinjoin_tx_info.json', True, False, None,
                            None, False)
        exit(42)

        wasabi_plot_remixes('wasabi2_select', os.path.join(target_path, 'wasabi2_select'), 'coinjoin_tx_info.json', True, False, None,
                            None, False)
        exit(42)

        wasabi_plot_remixes('wasabi2', os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json', True, False, None,
                            None, False)
        exit(42)


        wasabi_plot_remixes('whirlpool', os.path.join(target_path, 'whirlpool'), 'coinjoin_tx_info.json', True, False)

        exit(42)
        wasabi_plot_remixes('whirlpool_100k', os.path.join(target_path, 'whirlpool'), 'coinjoin_tx_info.json',
                            True, False)
        wasabi_plot_remixes('whirlpool_1M', os.path.join(target_path, 'whirlpool'), 'coinjoin_tx_info.json',
                            True, False)
        wasabi_plot_remixes('whirlpool_50M', os.path.join(target_path, 'whirlpool'), 'coinjoin_tx_info.json',
                            True, False)

        exit(42)
        #wasabi_plot_remixes('wasabi1', os.path.join(target_path, 'wasabi1'), 'coinjoin_tx_info.json', False, True)
        #wasabi_plot_remixes('wasabi2', os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json', False, True)
        wasabi_plot_remixes('whirlpool', os.path.join(target_path, 'whirlpool'), 'coinjoin_tx_info.json', False, True)
        #wasabi_plot_remixes('wasabi2_select', os.path.join(target_path, 'wasabi2_select'), 'coinjoin_tx_info.json', False, True)
        exit(42)
        wasabi_plot_remixes_multilevels()
        exit(42)

        limits = [1000000, 11000000, 100000000, 1000000000, 10000000000]
        for limit in limits:
            # wasabi_plot_remixes('wasabi1', os.path.join(target_path, 'wasabi1'), 'coinjoin_tx_info.json', True, False, None, (0, limit))
            # wasabi_plot_remixes('wasabi2', os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json', True, False, None, (0, limit))
            wasabi_plot_remixes('wasabi1', os.path.join(target_path, 'wasabi1'), 'coinjoin_tx_info.json', True, True, None,
                                (0, limit))
            wasabi_plot_remixes('wasabi2', os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json', True, True, None,
                                (0, limit))
        exit(42)
        wasabi_plot_remixes('wasabi2', os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json', True, True, None, (0, 10000000))
        exit(42)
        # process_and_save_single_interval('wasabi2_select', MIX_PROTOCOL.WASABI2, target_path, '2024-05-01 00:00:07.000',
        #                                   '2024-06-01 23:38:07.000',
        #                                   'Wasabi2CoinJoins.txt', 'Wasabi2PostMixTxs.txt', None, SAVE_BASE_FILES_JSON,
        #                                   True)
        # exit(42)

        # process_and_save_single_interval('wasabi2_select', MIX_PROTOCOL.WASABI2, target_path, '2024-06-01 00:00:07.000',
        #                                   '2024-06-07 23:38:07.000',
        #                                   'Wasabi2CoinJoins.txt', 'Wasabi2PostMixTxs.txt', None, SAVE_BASE_FILES_JSON,
        #                                   True)
        # exit(42)

        # wasabi2_analyse_remixes('wasabi2_select', target_path)
        # exit(42)

        wasabi_plot_remixes('wasabi1', os.path.join(target_path, 'wasabi1_select'), 'coinjoin_tx_info.json',
                            True, False)
        exit(42)

        wasabi_plot_remixes('wasabi2', os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json',
                            True, False)
        wasabi_plot_remixes('wasabi2', os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json',
                            True, True)
        wasabi_plot_remixes('wasabi2', os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json', False, True)
        exit(42)

        wasabi_plot_remixes('wasabi2_select', os.path.join(target_path, 'wasabi2_select'), 'coinjoin_tx_info.json', False, True)
        wasabi_plot_remixes('wasabi2_select', os.path.join(target_path, 'wasabi2_select'), 'coinjoin_tx_info.json', True, False)
        wasabi_plot_remixes('wasabi2_select', os.path.join(target_path, 'wasabi2_select'), 'coinjoin_tx_info.json', True, True)

    if PROCESS_NOTABLE_INTERVALS:
        def process_joint_interval(mix_origin_name, interval_name, all_data, mix_type, target_path, start_date: str,
                                   end_date: str):
            process_and_save_single_interval(interval_name, all_data, mix_type, target_path, start_date, end_date)
            shutil.copyfile(os.path.join(target_path, mix_origin_name, 'fee_rates.json'),
                            os.path.join(target_path, interval_name, 'fee_rates.json'))
            shutil.copyfile(os.path.join(target_path, mix_origin_name, 'false_cjtxs.json'),
                            os.path.join(target_path, interval_name, 'false_cjtxs.json'))
            wasabi_plot_remixes(interval_name, MIX_PROTOCOL.WASABI1, os.path.join(target_path, interval_name),
                                'coinjoin_tx_info.json', True, False)

        if CONSIDER_WW1:
            target_load_path = os.path.join(target_path, 'wasabi1')
            all_data = als.load_coinjoins_from_file(os.path.join(target_load_path, f'coinjoin_tx_info.json'), None, True)

            # Large inflows into WW1 in 2019-08-09, mixed and the all taken out
            process_joint_interval('wasabi1', 'wasabi1__2019_08-09', all_data, MIX_PROTOCOL.WASABI1, target_path, '2019-08-01 00:00:07.000', '2019-09-30 23:59:59.000')

            # Large single inflow with long remixing continously taken out
            process_joint_interval('wasabi1', 'wasabi1__2020_03-04', all_data, MIX_PROTOCOL.WASABI1, target_path, '2020-03-26 00:00:07.000','2020-04-20 23:59:59.000')

            # Two inflows, subsequent remixing
            process_joint_interval('wasabi1', 'wasabi1__2022_04-05', all_data, MIX_PROTOCOL.WASABI1, target_path, '2022-04-23 00:00:07.000', '2022-05-06 23:59:59.000')

        if CONSIDER_WW2:
            target_load_path = os.path.join(target_path, 'wasabi2')
            all_data = als.load_coinjoins_from_file(os.path.join(target_load_path), None, True)

            # Large inflow, in 2023-12, slightly mixed, send out, received as friend, then remixed
            process_joint_interval('wasabi2', 'wasabi2__2023_12-01', all_data, MIX_PROTOCOL.WASABI2, target_path, '2023-12-20 00:00:07.000', '2024-01-30 23:59:59.000')


    #
    #
    #
    if ANALYSIS_PROCESS_ALL_COINJOINS_INTERVALS:
        if CONSIDER_WHIRLPOOL:
            all_data = process_and_save_intervals_filter('whirlpool', MIX_PROTOCOL.WHIRLPOOL, target_path, '2019-04-17 01:38:07.000', interval_stop_date,
                                       'SamouraiCoinJoins.txt', 'SamouraiPostMixTxs.txt', 'SamouraiTx0s.txt',
                                              SAVE_BASE_FILES_JSON, False)

            # Split Whirlpool based on pools
            whirlpool_extract_pool(all_data, 'whirlpool', target_path, 'whirlpool_100k', 100000)
            whirlpool_extract_pool(all_data, 'whirlpool', target_path, 'whirlpool_1M', 1000000)
            whirlpool_extract_pool(all_data, 'whirlpool', target_path, 'whirlpool_5M', 5000000)
            whirlpool_extract_pool(all_data, 'whirlpool', target_path, 'whirlpool_50M', 50000000)

            # 100k pool
            process_and_save_intervals_filter('whirlpool_100k', MIX_PROTOCOL.WHIRLPOOL, target_path,
                                              WHIRLPOOL_FUNDING_TXS[100000]['start_date'], interval_stop_date,
                                              None, None, None, SAVE_BASE_FILES_JSON, True)
            # 1M pool
            process_and_save_intervals_filter('whirlpool_1M', MIX_PROTOCOL.WHIRLPOOL, target_path,
                                              WHIRLPOOL_FUNDING_TXS[1000000]['start_date'], interval_stop_date,
                                              None, None, None, SAVE_BASE_FILES_JSON, True)
            # 5M pool
            process_and_save_intervals_filter('whirlpool_5M', MIX_PROTOCOL.WHIRLPOOL, target_path,
                                              WHIRLPOOL_FUNDING_TXS[5000000]['start_date'], interval_stop_date,
                                              None, None, None, SAVE_BASE_FILES_JSON, True)
            # 50M pool
            process_and_save_intervals_filter('whirlpool_50M', MIX_PROTOCOL.WHIRLPOOL, target_path,
                                              WHIRLPOOL_FUNDING_TXS[50000000]['start_date'], interval_stop_date,
                                              None, None, None, SAVE_BASE_FILES_JSON, True)

        if CONSIDER_WW1:
            process_and_save_intervals_filter('wasabi1', MIX_PROTOCOL.WASABI1, target_path, '2018-07-19 01:38:07.000', interval_stop_date,
                                       'WasabiCoinJoins.txt', 'WasabiPostMixTxs.txt', None, SAVE_BASE_FILES_JSON, False)

        if CONSIDER_WW2:
            process_and_save_intervals_filter('wasabi2', MIX_PROTOCOL.WASABI2, target_path, '2022-06-01 00:00:07.000', interval_stop_date,
                                       'Wasabi2CoinJoins.txt', 'Wasabi2PostMixTxs.txt', None, SAVE_BASE_FILES_JSON, False)
            # WW2 needs additional treatment - detect and fix origin of WW1 inflows as friends
            fix_ww2_for_fdnp_ww1('wasabi2', target_path)

    if VISUALIZE_ALL_COINJOINS_INTERVALS:
        if CONSIDER_WHIRLPOOL:
            visualize_intervals('whirlpool', target_path, '2019-04-17 01:38:07.000', interval_stop_date)

        if CONSIDER_WW1:
            visualize_intervals('wasabi1', target_path, '2018-07-19 01:38:07.000', interval_stop_date)

        if CONSIDER_WW2:
            visualize_intervals('wasabi2', target_path, '2022-06-01 00:00:07.000', interval_stop_date)

    if DETECT_FALSE_POSITIVES:
        if CONSIDER_WW1:
            wasabi_detect_false(os.path.join(target_path, 'wasabi1'), 'coinjoin_tx_info.json')
        if CONSIDER_WW2:
            wasabi_detect_false(os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json')
        if CONSIDER_WHIRLPOOL:
            wasabi_detect_false(os.path.join(target_path, 'whirlpool'), 'coinjoin_tx_info.json')
            wasabi_detect_false(os.path.join(target_path, 'whirlpool_100k'), 'coinjoin_tx_info.json')
            wasabi_detect_false(os.path.join(target_path, 'whirlpool_1M'), 'coinjoin_tx_info.json')
            wasabi_detect_false(os.path.join(target_path, 'whirlpool_5M'), 'coinjoin_tx_info.json')
            wasabi_detect_false(os.path.join(target_path, 'whirlpool_50M'), 'coinjoin_tx_info.json')

    if SPLIT_WHIRLPOOL_POOLS:
        if CONSIDER_WHIRLPOOL:
            # Load txs for all pools
            target_load_path = os.path.join(target_path, 'whirlpool')
            logging.info(f'Loading {target_load_path}/coinjoin_tx_info.json ...')
            data = als.load_json_from_file(os.path.join(target_load_path, f'coinjoin_tx_info.json'))

            # Separate per pool
            pool_100k = whirlpool_extract_pool(data, 'whirlpool', target_path, 'whirlpool_100k', 100000)
            pool_1M = whirlpool_extract_pool(data, 'whirlpool', target_path, 'whirlpool_1M', 1000000)
            pool_5M = whirlpool_extract_pool(data, 'whirlpool', target_path, 'whirlpool_5M', 5000000)
            pool_50M = whirlpool_extract_pool(data, 'whirlpool', target_path, 'whirlpool_50M', 50000000)

            # Detect transactions which were not assigned to any pool
            missed_cjtxs = list(set(data['coinjoins'].keys()) - set(pool_100k['coinjoins'].keys()) - set(pool_1M['coinjoins'].keys())
                                - set(pool_5M['coinjoins'].keys()) - set(pool_50M['coinjoins'].keys()))
            als.save_json_to_file_pretty(os.path.join(target_load_path, f'coinjoin_tx_info__missed.json'), missed_cjtxs)
            print(f'Total transactions not separated into pools: {len(missed_cjtxs)}')
            print(missed_cjtxs)

    if PLOT_INTERMIX_FLOWS:
        analyze_mixes_flows(target_path)
        exit(42)

    if PLOT_REMIXES:
        if CONSIDER_WW1:
            wasabi_plot_remixes('wasabi1', MIX_PROTOCOL.WASABI1, os.path.join(target_path, 'wasabi1'), 'coinjoin_tx_info.json', True, False)
            wasabi_plot_remixes('wasabi1', MIX_PROTOCOL.WASABI1, os.path.join(target_path, 'wasabi1'), 'coinjoin_tx_info.json', False, True)
            wasabi_plot_remixes('wasabi1', MIX_PROTOCOL.WASABI1, os.path.join(target_path, 'wasabi1'), 'coinjoin_tx_info.json', True, True)

        if CONSIDER_WW2:
            wasabi_plot_remixes('wasabi2', MIX_PROTOCOL.WASABI2, os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json', True, False)
            wasabi_plot_remixes('wasabi2', MIX_PROTOCOL.WASABI2, os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json', True, True)
            wasabi_plot_remixes('wasabi2', MIX_PROTOCOL.WASABI2, os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json', False, True)
            wasabi_plot_remixes('wasabi2', MIX_PROTOCOL.WASABI2, os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json', False, False)

        if CONSIDER_WHIRLPOOL:
            PLOT_OPTIONS = []  # (analyze_values, normalize_values, plot_multigraph)
            PLOT_OPTIONS.append((True, False, False))  # values, not normalized
            PLOT_OPTIONS.append((False, True, False))  # number of inputs, normalized
            PLOT_OPTIONS.append((True, True, False))  # values, normalized
            PLOT_OPTIONS.append((False, False, False))  # number of inputs, not normalized
            for option in PLOT_OPTIONS:
                # Plotting remixes separately for different Whirlpool pools
                wasabi_plot_remixes('whirlpool_100k', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool_100k'), 'coinjoin_tx_info.json',
                                    option[0], option[1], None, None, option[2])
                wasabi_plot_remixes('whirlpool_1M', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool_1M'), 'coinjoin_tx_info.json',
                                    option[0], option[1], None, None, option[2])
                wasabi_plot_remixes('whirlpool_5M', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool_5M'), 'coinjoin_tx_info.json',
                                    option[0], option[1], None, None, option[2])
                wasabi_plot_remixes('whirlpool_50M', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool_50M'), 'coinjoin_tx_info.json',
                                    option[0], option[1], None, None, option[2])

            wasabi_plot_remixes('whirlpool', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool'), 'coinjoin_tx_info.json', True, False, None, None, False)
            wasabi_plot_remixes('whirlpool', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool'), 'coinjoin_tx_info.json', False, True, None, None, False)
            wasabi_plot_remixes('whirlpool', MIX_PROTOCOL.WHIRLPOOL, os.path.join(target_path, 'whirlpool'), 'coinjoin_tx_info.json', True, True, None, None, False)

    if PLOT_REMIXES_FLOWS:
        wasabi_plot_remixes_flows('wasabi2_select',
                             os.path.join(target_path, 'wasabi2_select'),
                             'coinjoin_tx_info.json', False, True)
        exit(42)

    if ANALYSIS_BURN_TIME:
        if CONSIDER_WW1:
            wasabi1_analyse_remixes('Wasabi1', target_path)
        if CONSIDER_WW2:
            wasabi2_analyse_remixes('Wasabi2', target_path)
        if CONSIDER_WHIRLPOOL:
            whirlpool_analyse_remixes('Whirlpool', target_path)

    # Extract distribution of mix fresh input sizes
    if ANALYSIS_INPUTS_DISTRIBUTION:
        # TODO: Compute input / output distributions from processed coinjoin_tx_info.json files including removal of false positives
        # Produce figure with distribution of diffferent pools merged
        if CONSIDER_WW1:
            process_inputs_distribution('Wasabi1', MIX_PROTOCOL.WASABI1,  target_path, 'WasabiCoinJoins.txt', True)
        if CONSIDER_WHIRLPOOL:
            #process_inputs_distribution('whirlpool_tx0_test', MIX_PROTOCOL.WHIRLPOOL, target_path, 'sam_premix_test.txt', True)
            process_inputs_distribution('whirlpool_tx0', MIX_PROTOCOL.WHIRLPOOL,  target_path, 'SamouraiTx0s.txt', True)
        if CONSIDER_WW2:
            process_inputs_distribution('Wasabi2', MIX_PROTOCOL.WASABI2,  target_path, 'Wasabi2CoinJoins.txt', True)

    #
    # Analyze address reuse in all mixes
    #
    if ANALYSIS_ADDRESS_REUSE:
        analyze_address_reuse(target_path)

    # if ANALYSIS_PROCESS_ALL_COINJOINS_INTERVALS_DEBUG:
    #     process_and_save_intervals_filter('wasabi2_feb24', target_path, '2024-02-11 01:38:07.000', '2024-02-11 23:38:07.000',
    #                            'wasabi2_mix_test.txt', 'wasabi2_postmix_test.txt')
    #
    #     process_and_save_intervals_filter('wasabi2_test', target_path, '2023-12-01 01:38:07.000', '2024-02-15 01:38:07.000',
    #                                'wasabi2_mix_test.txt', 'wasabi2_postmix_test.txt', None, SAVE_BASE_FILES_JSON)
    #
    #     process_and_save_intervals_filter('wasabi2false', target_path, '2022-06-18 01:38:07.000', '2024-02-15 01:38:07.000',
    #                                'Wasabi2CoinJoins_false.txt', 'Wasabi2PostMixTxs.txt', None, SAVE_BASE_FILES_JSON)

    # TODO: Analyze difference of unmoved and dynamic liquidity for Whirlpool between 2024-04-24 and 2024-08-24 (impact of knowledge of whirlpool seizure). Show last 1 year.

    print('### SUMMARY #############################')
    SM.print_summary()
    print('### END SUMMARY #########################')

    # TODO: Fee payed by cjtxs => ratio to remixed inputs => cost for chainalaysis
    # TODO: Fraction of inputs remixed / fraction of outputs remixed
    # TODO: Set x labels for histogram of frequencies to rounded denominations
    # TODO: Detect likely cases of WW2 round split due to more than 400 inputs registered
    #   (two coinjoins broadcasted close to each other, sum of inputs is close or higher than 400)
    # TODO: Compute miner and coordinator fee rate per participant / byte of tx
    # TODO: Detect if multiple rounds were happening in parallel (coinjoin time close to each other)
    # TODO: Filter false positives for hits coinjoin detection (Standa?)
    # BUGBUG: Finish detection of friends do not pay

    # TODO; Analyze Whirlpool consolidations. e.g. https://mempool.space/tx/890fb564ce500739a1566b29de9d53a8a0e545e8d1243224dfbed336219d8889
    #  is huge consolidation of whirlpool coins then resend 4 times, likely to confuse filters detecting whirlpool
    #  Consolidation https://mempool.space/tx/52e36136c4b45eb6b9effcdf7fa79997e4b5efb611675356702bcbb6ce9d5e8c#flow=&vin=6

    # TODO: Huge consolidation of Whirlpool coins: https://mempool.space/tx/d463b35b3d18dda4e59f432728c7a365eaefd50b24a6596ab42a077868e9d7e5
    #  (>60btc total, payjoin (possibly fake) attempted, 140+ inputs from various )
    # https://mempool.space/tx/8f59577b2dfa88e7d7fdd206a17618893db7559007a15658872b665bc16417c5
    # https://mempool.space/tx/d463b35b3d18dda4e59f432728c7a365eaefd50b24a6596ab42a077868e9d7e5
    # https://mempool.space/tx/57a8ea3ba1568fed4d9f7d7b3b84cdec552d9c49d4849bebf77a1053c180d0d1
    #

    # Analyze dominance cost:
    # 1. Coordinator fee to maintain X% pool liquidity at the time (put new input in if current liquidity below X%)
    # 2. Mining fees to maintain X% control of all inputs / outputs of each coinjoin. Disregard outliers with large sudden
    # incoming liquidity which will not be completely mixed anyway
    # - Stay in pool if already there (not to pay coordination fee again)
    # - Maximize impact of X% presence (WW2 outputs computation deviation)
    # Have X% control of all standard output denominations
    # (=> for whirlpool, have X% of all active remixing liquidity => will be selected )

    # Fee rates download
    # curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > fee_rates.json.all
    # curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/3y" > fee_rates.json.3y

    # TODO: Filter overall smaller and bigger cjtxs and plot separately

    # TODO: Plot graph of remix rates (values, num_inputs) as line plot for all months into single graph

