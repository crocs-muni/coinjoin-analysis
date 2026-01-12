import copy
import csv
import os
import subprocess
from collections import Counter, defaultdict
import sqlite3
import logging
from pathlib import Path
from statistics import median
#import msgpack
import orjson
import json
import time
import numpy as np
from datetime import timedelta, datetime, UTC
import re
import math
#from  txstore import TxStore, TxStoreMsgPack

from bitcoin.core import CTransaction, CMutableTransaction, CTxWitness
# from bitcoin.core import CScript, x
# from bitcoin import SelectParams
# from bitcoin.core.script import OP_HASH160, OP_EQUAL
# from bitcoin.wallet import P2WPKHBitcoinAddress, CBitcoinAddressError, P2SHBitcoinAddress, P2WSHBitcoinAddress

from bitcoinlib.transactions import Output

from cj_process.cj_consts import SATS_IN_BTC, MAX_SATS, VerboseTransactionInfoLineSeparator
from cj_process.cj_structs import MIX_EVENT_TYPE, precomp_datetime, MIX_PROTOCOL, SM, CJ_LOG_TYPES, CJ_ALICE_TYPES


# Sorting option for transactions.
# If False, then mining time of block with the transaction is used.
# If True then relative ordering of transactions based on remix connections.
# Important: SORT_COINJOINS_BY_RELATIVE_ORDER=True is causing reordering of transactions wrt their mining time
#            (broadcast_time_virtual is used instead). This may cause unexpected (seemingly incorrect) situations like transactions being
#            placed into previous month when splitting into monthly intervals is performed.
SORT_COINJOINS_BY_RELATIVE_ORDER = True

PERF_USE_COMPACT_CJTX_STRUCTURE = False  # If True, more compacted dictionary with coinjoin records is used
PERF_USE_SHORT_TXID = False
PERF_TX_SHORT_LEN = 16


def load_json_from_file(file_path: str | Path) -> dict:
    with open(file_path, "rb") as file:
        return orjson.loads(file.read())


def save_json_to_file(file_path: str, data: dict | list):
    with open(file_path, "wb") as file:
        file.write(orjson.dumps(data))


def save_json_to_file_pretty(file_path: str, data: dict, sort: bool = False):
    with open(file_path, "w") as file:
        if sort:
            file.write(json.dumps(dict(sorted(data.items())), indent=4))
        else:
            file.write(json.dumps(data, indent=4))


def save_json_to_csv_file_filtered(file_path: str | Path, data: dict, filter_columns: list=None):
    """
    Creates csv file from provided dictionary with columns specified in wanted_columns
    :param data:
    :param file_path:
    :param filter_columns:
    :return:
    """
    if filter_columns is None:
        subkeys = set()
        for inner in data.values():
            if isinstance(inner, dict):
                subkeys.update(inner.keys())
        filter_columns = sorted(subkeys)
    else:
        filter_columns = list(filter_columns)

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key"] + filter_columns)
        for key, inner in data.items():
            writer.writerow([key] + [inner.get(x, "") for x in filter_columns])


def detect_no_inout_remix_txs(coinjoins):
    no_remix = {'inputs_noremix': {}, 'outputs_noremix': {}}
    for cjtx in coinjoins.keys():
        if sum([1 for index in coinjoins[cjtx]['inputs'].keys()
                if coinjoins[cjtx]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name]) == 0:
            logging.warning(f'No input remix detected for {cjtx}')
            no_remix['inputs_noremix'][cjtx] = coinjoins[cjtx]['broadcast_time']
        if sum([1 for index in coinjoins[cjtx]['outputs'].keys()
             if coinjoins[cjtx]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name]) == 0:
            logging.warning(f'No output remix detected for {cjtx}')
            no_remix['outputs_noremix'][cjtx] = coinjoins[cjtx]['broadcast_time']

    noremix_txs = set(no_remix['inputs_noremix'].keys()).intersection(set(no_remix.get('outputs_noremix').keys()))
    no_remix['both_noremix'] = {cjtx: coinjoins[cjtx]['broadcast_time'] for cjtx in noremix_txs}
    logging.warning(f'Txs with no input&output remix: {no_remix["both_noremix"]}')
    return no_remix


def detect_stdenom_rbf_notap_onechange_txs(coinjoins):
    hits = {'stdenom_rbf_notap_onechange': {}}
    for cjtx in coinjoins.keys():
        # Is RBF?
        if coinjoins[cjtx].get('isRbf', 'unknown') == 'yes':
            script_freq = coinjoins[cjtx].get('script_frequencies', None)

            # Is zero Taproot?
            isZeroTaproot = False
            if script_freq:
                if len(script_freq['inputs']) == 1 and script_freq['inputs'].get('TxWitnessV1Taproot', 0) == 0:
                    #and len(script_freq['outputs']) == 1 and script_freq['outputs'].get('TxWitnessV1Taproot', 0) == 0):
                    isZeroTaproot = True
            if isZeroTaproot:
                # Has exactly one change output?
                output_denoms = [coinjoins[cjtx]['outputs'][index]['value'] for index in coinjoins[cjtx]['outputs'].keys()]
                counts = Counter(output_denoms).values()
                isExactlyOneChange = (min(counts, default=0) == 1) and (sum(c == 1 for c in counts) == 1)

                if isExactlyOneChange:
                    hits['stdenom_rbf_notap_onechange'][cjtx] = coinjoins[cjtx]['broadcast_time']

    return hits


def detect_local_outliers_txs(coinjoins, window_size: int, outlier_threshold: float):
    hits = {'local_outliers': {}}

    sorted_cjtxs = sort_coinjoins(coinjoins, SORT_COINJOINS_BY_RELATIVE_ORDER)
    effective_window_size = min(window_size, len(sorted_cjtxs))
    for index in range(0, len(sorted_cjtxs) - effective_window_size + 1):
        # BUGBUG: inefficient implementation, use more clever sliding window instead
        local_flag_string = [coinjoins[item['txid']]['flags_str'] for item in sorted_cjtxs[index:index + effective_window_size]]
        flags_counts = Counter(local_flag_string)
        for value, count in flags_counts.items():
            if count == 1 or count / effective_window_size < outlier_threshold:
                outlier_flags = value
                for record in sorted_cjtxs[index:index+window_size]:
                    if coinjoins[record['txid']]['flags_str'] == outlier_flags:
                        # hits['local_outliers'][record['txid']] = f"{coinjoins[record['txid']]['broadcast_time']}__{coinjoins[record['txid']]['flags_str']}"
                        hits['local_outliers'][record['txid']] = f"{coinjoins[record['txid']]['broadcast_time']}"

    return hits


def detect_address_reuse_txs(coinjoins, reuse_threshold: float):
    """
    Detect addresses reusing.
    :param coinjoins: structire with all coinjoins
    :param reuse_threshold: value between 0 and 1. Higher the threshold, more addresses needs to be reused (=> less size of set())
    :return: detected txs with addresses reusing
    """
    addr_reuse = {'inputs_address_reuse': {}, 'outputs_address_reuse': {}}
    for cjtx in coinjoins.keys():
        in_addressses = set([coinjoins[cjtx]['inputs'][index]['script'] for index in coinjoins[cjtx]['inputs'].keys()])
        ratio = len(in_addressses) / len(coinjoins[cjtx]['inputs'])
        if ratio < (1 - reuse_threshold):
            logging.warning(f'Input address reuse above threshold {ratio} detected for {cjtx}')
            #addr_reuse['inputs_address_reuse'].append(cjtx)
            addr_reuse['inputs_address_reuse'][cjtx] = coinjoins[cjtx]['broadcast_time']
        out_addressses = set([coinjoins[cjtx]['outputs'][index]['script'] for index in coinjoins[cjtx]['outputs'].keys()])
        ratio = len(out_addressses) / len(coinjoins[cjtx]['outputs'])
        if ratio < (1 - reuse_threshold):
            logging.warning(f'Output address reuse above threshold {ratio} detected for {cjtx}')
            #addr_reuse['outputs_address_reuse'].append(cjtx)
            addr_reuse['outputs_address_reuse'][cjtx] = coinjoins[cjtx]['broadcast_time']

    reused_txs = set(addr_reuse['inputs_address_reuse'].keys()).intersection(set(addr_reuse['outputs_address_reuse'].keys()))
    addr_reuse['both_reuse'] = {cjtx: coinjoins[cjtx]['broadcast_time'] for cjtx in reused_txs}
    logging.warning(f'Txs with no input&output remix: {addr_reuse["both_reuse"]}')
    return addr_reuse


def detect_unbalanced_inout_txs(coinjoins, unbalance_threshold: float):
    """
    Detect transactions with unbalanced number of inputs and outputs.
    :param coinjoins: structure with all coinjoins
    :param unbalance_threshold: value between 0 and 1. Higher the threshold, more significant ratio between inputs and outputs is required to classify as hit
    :return: detected txs with unbalanced number of inputs to outputs
    """
    results = {'unbalanced_inouts': {}}
    for cjtx in coinjoins.keys():
        num_ins = len(coinjoins[cjtx]['inputs'])
        num_outs = len(coinjoins[cjtx]['outputs'])

        denom = max(abs(num_ins), abs(num_outs))
        if denom != 0 and abs(num_ins - num_outs) / denom >= unbalance_threshold:
            results['unbalanced_inouts'][cjtx] = coinjoins[cjtx]['broadcast_time']

    return results


def detect_specific_cj_denoms(coinjoins: dict, specific_denoms_list: list, min_times_most_frequent_denom: int, exact_times_least_frequent_denom: int):
    specific_denoms = {'specific_denoms': {}}
    for cjtx in coinjoins.keys():
        output_denoms = [coinjoins[cjtx]['outputs'][index]['value'] for index in coinjoins[cjtx]['outputs'].keys()]
        out_counts = Counter(output_denoms)

        used_denoms = [coinjoins[cjtx]['inputs'][index]['value'] for index in coinjoins[cjtx]['inputs'].keys()]
        used_denoms.extend(output_denoms)

        a_counts = Counter(used_denoms)
        result = {b: a_counts[b] for b in specific_denoms_list if b in a_counts}

        if len(result) > 0 and max(result.values()) >= min_times_most_frequent_denom and min(out_counts.values()) == exact_times_least_frequent_denom:
            specific_denoms['specific_denoms'][cjtx] = coinjoins[cjtx]['broadcast_time']

    #logging.warning(f'Txs with specific input/output values: {specific_denoms["specific_denoms"]}')
    return specific_denoms


txid_precomp = {}  # Precomputed list of values to save on string extraction operations


def extract_txid_from_inout_string(inout_string):
    if isinstance(inout_string, str):
        if inout_string not in txid_precomp:
            if inout_string.startswith('vin') or inout_string.startswith('vout'):
                txid_precomp[inout_string] = (inout_string[inout_string.find('_') + 1: inout_string.rfind('_')], inout_string[inout_string.rfind('_') + 1:])
            else:
                assert False, f'Invalid inout string {inout_string}'
        return txid_precomp[inout_string]
    else:
        return inout_string[0], inout_string[1]


def get_ratio(numerator, denominator) -> int:
    if denominator != 0:
        return round(numerator/float(denominator) * 100, 1)
    else:
        return 0

def get_ratio_string(numerator, denominator) -> str:
    if denominator != 0:
        if isinstance(numerator, int):
            return f'{numerator}/{denominator} ({get_ratio(numerator, denominator)}%)'
        else:
            return f'{numerator:.2f}/{denominator:.2f} ({get_ratio(numerator, denominator)}%)'
    else:
        return f'{numerator:.2f}/{0} (0%)'


def get_inputs_type_list(coinjoins, sorted_cj_time, event_type, in_or_out: str, burn_time_from, burn_time_to, analyze_values, restrict_to_in_size: (int, int), only_standard_denoms: False):
    if analyze_values:
        return [sum([coinjoins[cjtx['txid']][in_or_out][index]['value'] for index in coinjoins[cjtx['txid']][in_or_out].keys()
                     if coinjoins[cjtx['txid']][in_or_out][index]['mix_event_type'] == event_type.name and
                     coinjoins[cjtx['txid']][in_or_out][index].get('burn_time_cjtxs', -1) in range(burn_time_from, burn_time_to + 1) and
                     restrict_to_in_size[0] <= coinjoins[cjtx['txid']][in_or_out][index]['value'] <= restrict_to_in_size[1] and
                     coinjoins[cjtx['txid']][in_or_out][index].get('is_standard_denom', False) == only_standard_denoms])
            for cjtx in sorted_cj_time]
    else:
        return [sum([1 for index in coinjoins[cjtx['txid']][in_or_out].keys()
                     if coinjoins[cjtx['txid']][in_or_out][index]['mix_event_type'] == event_type.name and
                     coinjoins[cjtx['txid']][in_or_out][index].get('burn_time_cjtxs', -1) in range(burn_time_from, burn_time_to + 1) and
                     restrict_to_in_size[0] <= coinjoins[cjtx['txid']][in_or_out][index]['value'] <= restrict_to_in_size[1] and
                     coinjoins[cjtx['txid']][in_or_out][index].get('is_standard_denom', False) == only_standard_denoms])
        for cjtx in sorted_cj_time]



def get_wallets_prediction_ratios(mix_id: str, prediction_matrix: dict=None):
    # NOTE: Based on real wallet experiments, average number of outputs (AVG_NUM_OUTPUTS) is significantly more
    # independent of number of coins in wallet and stable => take it as fixed point and compute synthetic value for AVG_NUM_INPUTS

    # AVG_NUM_INPUTS = 1.765  # value taken from simulations for all distributions
    # AVG_NUM_INPUTS = 3.18  # value taken from simulations for all distributions

    # Default values (if not more specific found)
    AVG_NUM_INPUTS = 3.65  # real value taken from kruw.io as38 experiment (use for kruw)
    AVG_NUM_OUTPUTS = 4.05  # synthetic value minimizing euclidean distance between output and input factors for kruw.io

    # kruw.io
    if 'kruw' in mix_id:
        AVG_NUM_OUTPUTS = 4.92 # real value taken from kruw.io as38 experiment (use for kruw)
        #AVG_NUM_OUTPUTS = 4.04 #  synthetic value minimizing euclidean distance for  kruw.io for interval 02/2025 if AVG_NUM_INPUTS = 3.65
        #AVG_NUM_OUTPUTS = 4.45  # synthetic value minimizing euclidean distance for  kruw.io for interval 03/2025 if AVG_NUM_INPUTS = 3.65

        #AVG_NUM_INPUTS = 3.65  # real value taken from kruw.io as38 experiment (use for kruw)
        AVG_NUM_INPUTS = 4.44 #  synthetic value minimizing euclidean distance for  kruw.io for interval 02/2025 if AVG_NUM_OUTPUTS = 4.92
        #AVG_NUM_INPUTS = 4.03 #  synthetic value minimizing euclidean distance for  kruw.io for interval 03/2025 if AVG_NUM_OUTPUTS = 4.92
        if prediction_matrix:
            AVG_NUM_INPUTS = prediction_matrix['inputs']['1']['mu_hat']
            AVG_NUM_OUTPUTS = prediction_matrix['outputs']['1']['mu_hat']

    # zksnacks
    if 'zksnacks' in mix_id:
        AVG_NUM_OUTPUTS = 4.17 # real value taken from zksnacks as25 experiment (use for zksnacks)
        AVG_NUM_INPUTS = 2.72  # real value taken from zksnacks as25 experiment (use for zksnacks)
#        AVG_NUM_OUTPUTS = 2.91 # synthetic median value minimizing euclidean distance between output and input factors for zksnacks if AVG_NUM_INPUTS = 2.72
        if prediction_matrix:
            AVG_NUM_INPUTS = prediction_matrix['inputs']['1']['mu_hat']
            AVG_NUM_OUTPUTS = prediction_matrix['outputs']['1']['mu_hat']

    # Wasabi 1.x
    if 'wasabi1' in mix_id:
        AVG_NUM_OUTPUTS = 2.31 # real value taken from wasabi1 experiments (typically one standard denomination, one change output)
        AVG_NUM_INPUTS = 1.15  # synthetic value

    # Whirlpool
    if 'whirlpool' in mix_id:
        AVG_NUM_OUTPUTS = 1 # real value taken from implementation of Whirlpool clients
        AVG_NUM_INPUTS = 1  # real value taken from implementation of Whirlpool clients


    return AVG_NUM_INPUTS, AVG_NUM_OUTPUTS


def compute_cjtxs_relative_ordering(coinjoins):
    coinjoins_relative_distance = {}
    cj_time = [{'txid': cjtxid, 'broadcast_time': precomp_datetime.strptime(coinjoins[cjtxid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")} for cjtxid in coinjoins.keys()]
    sorted_cj_times = sorted(cj_time, key=lambda x: x['broadcast_time'])

    # 1. Initialize relative distance from first coinjoin tx to 0
    for i in range(0, len(sorted_cj_times)):
        coinjoins_relative_distance[sorted_cj_times[i]['txid']] = 0

    # Process from very first coinjoin, update relative distance to be higher (+1)
    # than the distance of maximal distance of any of the inputs
    for i in range(1, len(sorted_cj_times)):  # skip the very first transaction
        txid = sorted_cj_times[i]['txid']
        prev_distances = []
        for input in coinjoins[txid]['inputs']:
            prev_tx_str = coinjoins[txid]['inputs'][input].get('spending_tx', None)
            if prev_tx_str:
                prev_tx, prev_tx_index = extract_txid_from_inout_string(prev_tx_str)
                if prev_tx in coinjoins_relative_distance.keys():  # Consider only inputs from previous mixes
                    prev_distances.append(coinjoins_relative_distance[prev_tx])
        coinjoins_relative_distance[txid] = max(prev_distances) + 1 if len(prev_distances) > 0 else 0

    return coinjoins_relative_distance


def compute_liquidity_summary(coinjoins: dict, prepare_strings: bool):
    if len(coinjoins) == 0:
        return None

    total_inputs_len = [len(coinjoins[cjtx]['inputs']) for cjtx in coinjoins.keys()]
    total_inputs = [coinjoins[cjtx]['inputs'][input]['value'] for cjtx in coinjoins.keys() for input in
                    coinjoins[cjtx]['inputs']]
    total_outputs_len = [len(coinjoins[cjtx]['outputs']) for cjtx in coinjoins.keys()]
    total_inputs_number = len(total_inputs)
    total_inputs_value = sum(total_inputs)

    total_outputs = [coinjoins[cjtx]['outputs'][output]['value'] for cjtx in coinjoins.keys() for output in
                     coinjoins[cjtx]['outputs']]
    total_outputs_number = len(total_outputs)
    total_outputs_value = sum(total_outputs)

    total_mix_entering = [coinjoins[cjtx]['inputs'][input]['value'] for cjtx in coinjoins.keys() for input in
                          coinjoins[cjtx]['inputs']
                          if coinjoins[cjtx]['inputs'][input]['mix_event_type'] == MIX_EVENT_TYPE.MIX_ENTER.name]
    total_mix_entering_number = len(total_mix_entering)
    total_mix_entering_value = sum(total_mix_entering)

    total_mix_friends = [coinjoins[cjtx]['inputs'][input]['value'] for cjtx in coinjoins.keys() for input in
                         coinjoins[cjtx]['inputs']
                         if coinjoins[cjtx]['inputs'][input][
                             'mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name]
    total_mix_friends_ww1 = [coinjoins[cjtx]['inputs'][input]['value'] for cjtx in coinjoins.keys() for input in
                             coinjoins[cjtx]['inputs']
                             if coinjoins[cjtx]['inputs'][input][
                                 'mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name]
    total_mix_friends_number = len(total_mix_friends) + len(total_mix_friends_ww1)
    total_mix_friends_value = sum(total_mix_friends) + sum(total_mix_friends_ww1)

    total_mix_remix = [coinjoins[cjtx]['inputs'][input]['value'] for cjtx in coinjoins.keys() for input in
                       coinjoins[cjtx]['inputs']
                       if coinjoins[cjtx]['inputs'][input]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name]
    total_mix_remix_number = len(total_mix_remix)
    total_mix_remix_value = sum(total_mix_remix)

    total_mix_remix_out = [coinjoins[cjtx]['outputs'][input]['value'] for cjtx in coinjoins.keys() for input in
                           coinjoins[cjtx]['outputs']
                           if coinjoins[cjtx]['outputs'][input]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name]
    total_mix_remix_out_number = len(total_mix_remix_out)
    total_mix_remix_out_value = sum(total_mix_remix_out)

    total_mix_leaving = [coinjoins[cjtx]['outputs'][output]['value'] for cjtx in coinjoins.keys() for output in
                         coinjoins[cjtx]['outputs']
                         if coinjoins[cjtx]['outputs'][output]['mix_event_type'] == MIX_EVENT_TYPE.MIX_LEAVE.name]
    total_mix_leaving_number = len(total_mix_leaving)
    total_mix_leaving_value = sum(total_mix_leaving)

    total_mix_leaving_nonstd = [coinjoins[cjtx]['outputs'][output]['value'] for cjtx in coinjoins.keys() for output in
                                coinjoins[cjtx]['outputs']
                                if coinjoins[cjtx]['outputs'][output][
                                    'mix_event_type'] == MIX_EVENT_TYPE.MIX_LEAVE.name and
                                coinjoins[cjtx]['outputs'][output]['is_standard_denom'] == False]
    total_mix_leaving_nonstd_number = len(total_mix_leaving_nonstd)
    total_mix_leaving_nonstd_value = sum(total_mix_leaving_nonstd)

    total_mix_staying = [coinjoins[cjtx]['outputs'][output]['value'] for cjtx in coinjoins.keys() for output in
                         coinjoins[cjtx]['outputs']
                         if coinjoins[cjtx]['outputs'][output]['mix_event_type'] == MIX_EVENT_TYPE.MIX_STAY.name]
    total_mix_staying_number = len(total_mix_staying)
    total_mix_staying_value = sum(total_mix_staying)

    def parse_broadcast_time(cjtx):
        return precomp_datetime.strptime(coinjoins[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")

    # Find earliest and latest
    earliest_cjtx = min(coinjoins, key=parse_broadcast_time)
    latest_cjtx = max(coinjoins, key=parse_broadcast_time)

    earliest_time = coinjoins[earliest_cjtx]['broadcast_time']
    latest_time = coinjoins[latest_cjtx]['broadcast_time']

    lr = {}
    lr['earliest_cjtx'] = earliest_cjtx
    lr['earliest_time'] = earliest_time
    lr['latest_cjtx'] = latest_cjtx
    lr['latest_time'] = latest_time
    lr['total_coinjoins'] = len(coinjoins.keys())
    lr['min_inputs'] = min(total_inputs_len)
    lr['max_inputs'] = max(total_inputs_len)
    lr['avg_inputs'] = np.average(total_inputs_len)
    lr['median_inputs'] = np.median(total_inputs_len)
    lr['min_outputs'] = min(total_outputs_len)
    lr['max_outputs'] = max(total_outputs_len)
    lr['avg_outputs'] = np.average(total_outputs_len)
    lr['median_outputs'] = np.median(total_outputs_len)

    lr['total_outputs_number'] = total_outputs_number
    lr['total_mix_entering_number'] = total_mix_entering_number
    lr['total_mix_staying_number'] = total_mix_staying_number
    lr['total_mix_remix_out_number'] = total_mix_remix_out_number

    lr['total_inputs_value_sats'] = total_inputs_value
    lr['total_inputs_value'] = lr['total_inputs_value_sats'] / SATS_IN_BTC
    lr['total_leaving_outputs_value_sats'] = total_mix_leaving_value
    lr['total_leaving_outputs_value'] = lr['total_leaving_outputs_value_sats'] / SATS_IN_BTC

    lr['total_mix_remix_value_sats'] = total_mix_remix_value
    lr['total_mix_remix_value'] = lr['total_mix_remix_value_sats'] / SATS_IN_BTC
    lr['total_unmoved_outputs_value_sats'] = total_mix_staying_value
    lr['total_unmoved_outputs_value'] = lr['total_unmoved_outputs_value_sats'] / SATS_IN_BTC
    lr['total_friends_inputs_value_sats'] = total_mix_friends_value
    lr['total_friends_inputs_value'] = lr['total_friends_inputs_value_sats'] / SATS_IN_BTC
    lr['total_fresh_inputs_value_sats'] = total_mix_entering_value
    lr['total_fresh_inputs_value'] = lr['total_fresh_inputs_value_sats'] / SATS_IN_BTC
    lr['total_nonstandard_leaving_outputs_value_sats'] = total_mix_leaving_nonstd_value
    lr['total_nonstandard_leaving_outputs_value'] = lr['total_nonstandard_leaving_outputs_value_sats'] / SATS_IN_BTC
    lr['total_leaving_outputs_without_nonstandard_leaving_outputs_value_sats'] = total_mix_leaving_value - total_mix_leaving_nonstd_value
    lr['total_leaving_outputs_without_nonstandard_leaving_outputs_value'] = lr['total_leaving_outputs_without_nonstandard_leaving_outputs_value_sats'] / SATS_IN_BTC
    lr['total_fresh_inputs_without_nonstandard_outputs_value_sats'] = total_mix_entering_value - total_mix_leaving_nonstd_value
    lr['total_fresh_inputs_without_nonstandard_outputs_value'] = lr['total_fresh_inputs_without_nonstandard_outputs_value_sats'] / SATS_IN_BTC
    lr['total_fresh_inputs_and_friends_without_nonstandard_outputs_value_sats'] = total_mix_entering_value + total_mix_friends_value - total_mix_leaving_nonstd_value
    lr['total_fresh_inputs_and_friends_without_nonstandard_outputs_value'] = lr['total_fresh_inputs_and_friends_without_nonstandard_outputs_value_sats'] / SATS_IN_BTC

    #
    # Shortcuts
    #
    # All inputs of coinjoin
    lr['inputs_value_type1_sats'] = lr['total_inputs_value_sats']
    lr['inputs_value_type1'] = lr['inputs_value_type1_sats'] / SATS_IN_BTC
    # All MIX_ENTER inputs of coinjoin (source is outside coinjoins)
    lr['inputs_value_type2_sats'] = lr['total_fresh_inputs_value_sats']
    lr['inputs_value_type2'] = lr['inputs_value_type2_sats'] / SATS_IN_BTC
    # All MIX_ENTER inputs of coinjoin without non-standard outputs (large utxo in, but only part of is mixed)
    lr['inputs_value_type3_sats'] = lr['total_fresh_inputs_without_nonstandard_outputs_value_sats']
    lr['inputs_value_type3'] = lr['inputs_value_type3_sats'] / SATS_IN_BTC
    # All MIX_ENTER + MIX_FRIENDS + MIX_FRIENDS_WW1 (friends are treated as fresh inputs)
    lr['inputs_value_type4_sats'] = lr['total_fresh_inputs_and_friends_without_nonstandard_outputs_value_sats']
    lr['inputs_value_type4'] = lr['inputs_value_type4_sats'] / SATS_IN_BTC

    # All outputs of coinjoin
    lr['outputs_value_type1_sats'] = lr['total_leaving_outputs_value_sats']
    lr['outputs_value_type1'] = lr['outputs_value_type1_sats'] / SATS_IN_BTC
    # All MIX_LEAVE outputs of coinjoin (target is outside coinjoins)
    lr['outputs_value_type2_sats'] = lr['total_leaving_outputs_value_sats']
    lr['outputs_value_type2'] = lr['outputs_value_type2_sats'] / SATS_IN_BTC
    # All MIX_LEAVE without non-standard outputs
    lr['outputs_value_type3_sats'] = lr['total_leaving_outputs_without_nonstandard_leaving_outputs_value_sats']
    lr['outputs_value_type3'] = lr['outputs_value_type3_sats'] / SATS_IN_BTC

    if prepare_strings:
        lr['ratio_fresh_inputs_2_total_inputs'] = get_ratio_string(total_mix_entering_number, total_inputs_number)
        lr['ratio_friends_inputs_2_total_inputs'] = get_ratio_string(total_mix_friends_number, total_inputs_number)
        lr['ratio_leaving_outputs_2_total_outputs'] = get_ratio_string(total_mix_leaving_number, total_outputs_number)
        lr['ratio_staying_outputs_2_total_outputs'] = get_ratio_string(total_mix_staying_number, total_outputs_number)
        lr['ratio_staying_outputs_2_nonremix_outputs'] = get_ratio_string(total_mix_staying_number,
                                                                          total_outputs_number - total_mix_remix_out_number)
        lr['ratio_remixed_inputs_2_total_inputs_numbers'] = get_ratio_string(total_mix_remix_number, total_inputs_number)
        lr['ratio_remixed_inputs_2_total_inputs_values'] = get_ratio_string(total_mix_remix_value / SATS_IN_BTC,
                                                                            total_inputs_value / SATS_IN_BTC)

    return lr


def print_liquidity_summary(coinjoins: dict, mix_id: str):
    lr = compute_liquidity_summary(coinjoins, True)

    if lr:
        mix_id_latex = mix_id.replace('_', '\\_' )
        lr['latex_summary'] = f"\\hline   " \
                 + f"{mix_id_latex} & {lr['earliest_time']}--{lr['latest_time']} & " \
                 + f"{lr['total_coinjoins']} & {lr['total_mix_entering_number']} / {round(lr['total_fresh_inputs_without_nonstandard_outputs_value'], 1)}~\\bitcoinSymbol" + "{} & " \
                 + f"{get_ratio(lr['total_mix_remix_value_sats'], lr['total_inputs_value_sats'])}\\% & " \
                 + f"{get_ratio(lr['total_mix_staying_number'], lr['total_outputs_number'] - lr['total_mix_remix_out_number'])}\\%, {round(lr['total_unmoved_outputs_value_sats'] / SATS_IN_BTC, 1)}~\\bitcoinSymbol" + "{} & " \
                 + f"{lr['min_inputs']} / {round(lr['avg_inputs'], 1)} / {lr['max_inputs']} \\\\"

        # Print summary results
        SM.print(f"  Earliest broadcast: {lr['earliest_time']} from {lr['earliest_cjtx']}")
        SM.print(f"  Latest broadcast: {lr['latest_time']} from {lr['latest_cjtx']}")
        SM.print(f"  Total coinjoin transactions: {lr['total_coinjoins']}")
        SM.print(f"  Number of inputs: min={lr['min_inputs']}, max={lr['max_inputs']}, avg={lr['avg_inputs']}, median={lr['median_inputs']}")
        SM.print(f"  Number of outputs: min={lr['min_outputs']}, max={lr['max_outputs']}, avg={lr['avg_outputs']}, median={lr['median_outputs']}")
        SM.print(f"  {lr['ratio_fresh_inputs_2_total_inputs']} Inputs entering mix / total inputs used by mix transactions")
        SM.print(f"  {lr['ratio_friends_inputs_2_total_inputs']} Friends inputs re-entering mix / total inputs used by mix transactions")
        SM.print(f"  {lr['ratio_leaving_outputs_2_total_outputs']} Outputs leaving mix / total outputs by mix transactions")
        SM.print(f"  {lr['ratio_staying_outputs_2_total_outputs']} Outputs staying in mix / total outputs by mix transactions")
        SM.print(f"  {lr['ratio_staying_outputs_2_nonremix_outputs']} Outputs staying in mix / non-remix outputs")
        SM.print(f"  {lr['ratio_remixed_inputs_2_total_inputs_numbers']} Inputs remixed / total inputs based on number of inputs")
        SM.print(f"  {lr['ratio_remixed_inputs_2_total_inputs_values']} Inputs remixed / total inputs based on value of inputs")
        SM.print(f"  {lr['total_fresh_inputs_value']} btc, total fresh entering mix")
        SM.print(f"  {lr['total_friends_inputs_value']} btc, total friends entering mix")
        SM.print(f"  {lr['total_unmoved_outputs_value']} btc, total value staying unmoved in mix")
        SM.print(f"  {lr['total_leaving_outputs_value']} btc, total value leaving mix")
        SM.print(f"  {lr['total_nonstandard_leaving_outputs_value']} btc, total non-standard value leaving mix (not mixed)")
        SM.print(f"  {lr['total_fresh_inputs_without_nonstandard_outputs_value']} btc, total fresh entering mix without non-standard leaving")

        SM.print(f"  {lr['latex_summary']}")

    return lr


def print_coordinators_counts(coord_txs: dict, min_print_txs: int):
    print('*********')
    coord_tx_counts = {id: len(coord_txs[id]) for id in coord_txs.keys()}
    sorted_counts = sorted(coord_tx_counts, key=coord_tx_counts.get, reverse=True)
    # sorted_counts = coord_tx_counts.keys()
    for id in sorted_counts:
        if len(coord_txs[id]) >= min_print_txs:
            #print(f"Coordinator {id} has {coord_tx_counts[id]} txs")
            print(f'  coord. {id}: {len(coord_txs[id])} txs')
    print(f'Total non-small coordinators (min={min_print_txs}): {len([1 for x in coord_txs.keys() if len(coord_txs[x]) >= min_print_txs])}')
    print(f'Theoretical total coordinators (incl. very small ones) detected: {len(coord_txs)}')
    print('*********')



def recompute_enter_remix_liquidity_after_removed_cjtxs(coinjoins, mix_protocol: MIX_PROTOCOL):
    """
    Call after some changes to existing set of coinjoins were made to update MIX_ENTER and MIX_REMIX values.
    Expected to be called after full analysis by analyze_input_out_liquidity()
    :param coinjoins: dictionary with coinjoins
    :param mix_protocol: type of protocol
    :return:
    """
    logging.debug('recompute_enter_remix_liquidity_after_removed_cjtxs() started')

    # Idea: Coinjoins may have been removed from the set of coinjoins, changing MIX_REMIX -> MIX_ENTER for inputs
    # and MIX_REMIX -> MIX_LEAVE for outputs
    # Detect these cases and rectify.

    for cjtx in coinjoins:
        for input in coinjoins[cjtx]['inputs']:
            if 'spending_tx' in coinjoins[cjtx]['inputs'][input].keys():
                spending_tx, index = extract_txid_from_inout_string(coinjoins[cjtx]['inputs'][input]['spending_tx'])
                if coinjoins[cjtx]['inputs'][input]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name and spending_tx not in coinjoins.keys():
                    # Change to MIX_ENTER as original cjtx is no longer in coinjoin set
                    logging.debug(f'Changing MIX_REMIX -> MIX_ENTER for input {cjtx}[{input}]')
                    coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_ENTER.name
                    coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs_as_mined'] = 0
                    coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs_relative'] = 0
                    coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs'] = 0

        for output in coinjoins[cjtx]['outputs']:
            if 'spend_by_tx' in coinjoins[cjtx]['outputs'][output].keys():
                spend_by_tx, index = extract_txid_from_inout_string(coinjoins[cjtx]['outputs'][output]['spend_by_tx'])
                if coinjoins[cjtx]['outputs'][output]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name and spend_by_tx not in coinjoins.keys():
                    # Change to MIX_LEAVE as original spending tx is no longer in coinjoin set
                    logging.debug(f'Changing MIX_REMIX -> MIX_LEAVE for output {cjtx}[{output}]')
                    coinjoins[cjtx]['outputs'][output]['mix_event_type'] = MIX_EVENT_TYPE.MIX_LEAVE.name
                    coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs_as_mined'] = 0
                    coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs_relative'] = 0
                    coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs'] = 0


def unfinished_recompute_enter_remix_liquidity_after_added_cjtxs(coinjoins, mix_protocol: MIX_PROTOCOL):
    """
    Problem:  we need to set also ['mix_event_type'] for newly added transactions
    Call after some changes to existing set of coinjoins were made to update MIX_ENTER and MIX_REMIX values.
    Expected to be called after full analysis by analyze_input_out_liquidity()
    :param coinjoins: dictionary with coinjoins
    :param mix_protocol: type of protocol
    :return:
    """
    logging.debug('recompute_enter_remix_liquidity_after_added_cjtxs() started')

    # Idea: Coinjoins may have been added from the set of coinjoins, changing MIX_ENTER -> MIX_REMIX for inputs
    # and MIX_LEAVE -> MIX_REMIX for outputs
    # Detect these cases and rectify.

    broadcast_times = {cjtx: precomp_datetime.strptime(coinjoins[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for cjtx in coinjoins.keys()}
    # Sort coinjoins based on mining time
    cj_time = [{'txid': cjtxid, 'broadcast_time': precomp_datetime.strptime(coinjoins[cjtxid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")} for cjtxid in coinjoins.keys()]
    sorted_cj_times = sorted(cj_time, key=lambda x: x['broadcast_time'])
    # Precomputed mapping of txid to index for fast burntime computation
    coinjoins_index = {}
    for i in range(0, len(sorted_cj_times)):
        coinjoins_index[sorted_cj_times[i]['txid']] = i
    coinjoins_relative_order = compute_cjtxs_relative_ordering(coinjoins)

    for cjtx in coinjoins:
        for input in coinjoins[cjtx]['inputs']:
            if 'spending_tx' in coinjoins[cjtx]['inputs'][input].keys():
                update_item = False
                spending_tx, index = extract_txid_from_inout_string(coinjoins[cjtx]['inputs'][input]['spending_tx'])
                if 'mix_event_type' in coinjoins[cjtx]['inputs'][input]:
                    if coinjoins[cjtx]['inputs'][input]['mix_event_type'] == MIX_EVENT_TYPE.MIX_ENTER.name and spending_tx in coinjoins.keys():
                        # Change to MIX_ENTER as original cjtx is no longer in coinjoin set
                        logging.debug(f'Changing MIX_ENTER -> MIX_REMIX for input {cjtx}[{input}]')
                        update_item = True
                else:
                    if spending_tx in coinjoins.keys():
                        # Set to MIX_ENTER as original cjtx is no longer in coinjoin set
                        logging.debug(f'Setting MIX_REMIX for input {cjtx}[{input}]')
                        update_item = True
                    else:
                        # Not in coinjoin, fresh liquidity
                        coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_ENTER.name


                if update_item:
                    coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_REMIX.name
                    coinjoins[cjtx]['inputs'][input]['burn_time'] = round((broadcast_times[cjtx] - broadcast_times[spending_tx]).total_seconds(), 0)
                    coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs_as_mined'] = coinjoins_index[cjtx] - coinjoins_index[spending_tx]
                    coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs_relative'] = coinjoins_relative_order[cjtx] - coinjoins_relative_order[spending_tx]
                    coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs'] = coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs_relative']


        for output in coinjoins[cjtx]['outputs']:
            if 'spend_by_tx' in coinjoins[cjtx]['outputs'][output].keys():
                update_item = False
                spend_by_tx, index = extract_txid_from_inout_string(coinjoins[cjtx]['outputs'][output]['spend_by_tx'])

                if 'mix_event_type' in coinjoins[cjtx]['outputs'][output]:
                    if coinjoins[cjtx]['outputs'][output]['mix_event_type'] == MIX_EVENT_TYPE.MIX_LEAVE.name and spend_by_tx in coinjoins.keys():
                        logging.debug(f'Changing MIX_LEAVE -> MIX_REMIX for output {cjtx}[{output}]')
                        update_item = True
                else:
                    if spend_by_tx in coinjoins.keys():
                        logging.debug(f'Setting MIX_REMIX for input {cjtx}[{output}]')
                        update_item = True
                    else:
                        # Spent outside coinjoin
                        coinjoins[cjtx]['outputs'][output]['mix_event_type'] = MIX_EVENT_TYPE.MIX_LEAVE.name
                        # BUGBUG: we need to find spending tx to compute burn time properly

                if update_item:
                    coinjoins[cjtx]['outputs'][output]['mix_event_type'] = MIX_EVENT_TYPE.MIX_REMIX.name
                    coinjoins[cjtx]['outputs'][output]['burn_time'] = round((broadcast_times[spend_by_tx] - broadcast_times[cjtx]).total_seconds(), 0)
                    coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs_as_mined'] = coinjoins_index[spend_by_tx] - coinjoins_index[cjtx]
                    coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs_relative'] = coinjoins_relative_order[spend_by_tx] - coinjoins_relative_order[cjtx]
                    coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs'] = coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs_relative']
            else:
                if 'mix_event_type' not in coinjoins[cjtx]['outputs'][output]:
                    # BUGBUG: newly added transaction without 'mix_event_type' for this specific output set.
                    # May be MIX_EVENT_TYPE.MIX_STAY, MIX_EVENT_TYPE.MIX_LEAVE as well as MIX_EVENT_TYPE.MIX_REMIX
                    logging.error(f"Missing status of output for {cjtx}['outputs']{output}")


    return coinjoins


def analyze_input_out_liquidity(target_path: str, coinjoins, postmix_spend, premix_spend, mix_protocol: MIX_PROTOCOL, ww1_coinjoins:dict = None, ww1_postmix_spend:dict = None, warn_if_not_found_in_postmix:bool = True):
    """
    Requires performance speedup, will not finish (after 8 hours) for Whirlpool with very large number of coins
    :param coinjoins:
    :param postmix_spend:
    :param premix_spend:
    :param mix_protocol:
    :param ww1_coinjoins:
    :param ww1_postmix_spend:
    :param warn_if_not_found_in_postmix: If True warning is emmited if spending_tx is not found in set of postmix txs
    :return:
    """
    logging.debug('analyze_input_out_liquidity() started')

    if ww1_coinjoins is None:
        ww1_coinjoins = {}
    if ww1_postmix_spend is None:
        ww1_postmix_spend = {}

    liquidity_events = []
    total_inputs = 0
    total_mix_entering = 0
    total_mix_friends = 0
    total_outputs = 0
    total_mix_leaving = 0
    total_mix_staying = []
    total_utxos = 0
    broadcast_times = {cjtx: precomp_datetime.strptime(coinjoins[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for cjtx in coinjoins.keys()}
    if postmix_spend:
        broadcast_times.update({tx: precomp_datetime.strptime(postmix_spend[tx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for tx in postmix_spend.keys()})
    # Sort coinjoins based on mining time
    cj_time = [{'txid': cjtxid, 'broadcast_time': precomp_datetime.strptime(coinjoins[cjtxid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")} for cjtxid in coinjoins.keys()]
    sorted_cj_times = sorted(cj_time, key=lambda x: x['broadcast_time'])

    # Precomputed mapping of txid to index for fast burntime computation
    coinjoins_index = {}
    for i in range(0, len(sorted_cj_times)):
        coinjoins_index[sorted_cj_times[i]['txid']] = i

    # Compute sorting of coinjoins based on their interconnections
    # Assumptions made:
    #   1. At least one input is from freshest previous coinjoin (given large number of wallets and remixes, that is expected case)
    #   2. Output from previous coinjoin X can be registered to next coinjoin as input only after X is mined to block (enforced by coordinator)
    coinjoins_relative_order = compute_cjtxs_relative_ordering(coinjoins)

    for cjtx in coinjoins:
        coinjoins[cjtx]['relative_order'] = coinjoins_relative_order[cjtx]  # Save computed relative order
        if coinjoins_index[cjtx] % 10000 == 0:
            print(f'  {coinjoins_index[cjtx]} coinjoins processed')
        for input in coinjoins[cjtx]['inputs']:
            total_inputs += 1
            if 'spending_tx' in coinjoins[cjtx]['inputs'][input].keys():
                spending_tx, index = extract_txid_from_inout_string(coinjoins[cjtx]['inputs'][input]['spending_tx'])
                if spending_tx not in coinjoins.keys():
                    # Direct previous transaction is from outside the mix => potentially new input liquidity
                    if mix_protocol == MIX_PROTOCOL.WASABI2:
                        # Either: 1. New fresh liquidity entered or 2. Friend-do-not-pay rule (if WW2/WW1, one or two hops)
                        # If fresh input is coming from WW1, friends-do-not-pay may also still apply, check
                        if (spending_tx in postmix_spend.keys() or
                                spending_tx in ww1_coinjoins.keys() or
                                spending_tx in ww1_postmix_spend.keys()):
                            # Friends do not pay rule tx
                            coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name
                            total_mix_friends += 1
                        else:
                            # Fresh input coming from outside
                            total_mix_entering += 1
                            coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_ENTER.name
                    else:
                        # All other protocols than WW2 do not have 'friends do not pay'
                        total_mix_entering += 1
                        coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_ENTER.name
                else:  # Direct mix to mix transaction
                    coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_REMIX.name
                    coinjoins[cjtx]['inputs'][input]['burn_time'] = round((broadcast_times[cjtx] - broadcast_times[spending_tx]).total_seconds(), 0)
                    coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs_as_mined'] = coinjoins_index[cjtx] - coinjoins_index[spending_tx]
                    coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs_relative'] = coinjoins_relative_order[cjtx] - coinjoins_relative_order[spending_tx]
                    coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs'] = coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs_relative']
                    if mix_protocol != MIX_PROTOCOL.JOINMARKET: # JoinMarket may end-up with schuffled transactions??
                        assert coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs'] >= 0, f"Invalid burn time computed for {cjtx}:{input}; got {coinjoins[cjtx]['inputs'][output]['burn_time_cjtxs']}; {cjtx} - {spending_tx}"
            else:
                total_mix_entering += 1
                coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_ENTER.name

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
                        if warn_if_not_found_in_postmix:
                            logging.warning(f'Could not find spend_by_tx {spend_by_tx} in postmix_spend txs')
                    else:
                        coinjoins[cjtx]['outputs'][output]['burn_time'] = round((broadcast_times[spend_by_tx] - broadcast_times[cjtx]).total_seconds(), 0)
                    total_mix_leaving += 1
                    coinjoins[cjtx]['outputs'][output]['mix_event_type'] = MIX_EVENT_TYPE.MIX_LEAVE.name
                else:
                    # Mix spend: The output is spent by next coinjoin tx => stays in mix
                    coinjoins[cjtx]['outputs'][output]['mix_event_type'] = MIX_EVENT_TYPE.MIX_REMIX.name
                    coinjoins[cjtx]['outputs'][output]['burn_time'] = round((broadcast_times[spend_by_tx] - broadcast_times[cjtx]).total_seconds(), 0)
                    coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs_as_mined'] = coinjoins_index[spend_by_tx] - coinjoins_index[cjtx]
                    coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs_relative'] = coinjoins_relative_order[spend_by_tx] - coinjoins_relative_order[cjtx]
                    coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs'] = coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs_relative']
                    if mix_protocol != MIX_PROTOCOL.JOINMARKET:  # JoinMarket may end-up with shuffled transactions??
                        assert coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs'] >= 0, \
                            f"Invalid burn time computed for {cjtx}:{output}; got {coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs']}; {spend_by_tx} - {cjtx}"

    # Establish standard denominations for this coinjoin (depends on coinjoin design)
    # Heuristics: standard denomination is denomination which is repeated at least two times in outputs (anonset>=2)
    # Needs to be computed for each coinjoin again, as standard denominations may change in time
    # Compute first for all outputs, then assign to related inputs (if remix)
    for cjtx in coinjoins_relative_order:
        denom_frequencies = Counter([coinjoins[cjtx]['outputs'][output]['value'] for output in coinjoins[cjtx]['outputs']])
        std_denoms = {value: count for value, count in denom_frequencies.items() if count > 1}
        for output in coinjoins[cjtx]['outputs']:
            coinjoins[cjtx]['outputs'][output]['is_standard_denom'] = coinjoins[cjtx]['outputs'][output]['value'] in std_denoms.keys()
    # Now set to spending inputs retrospectively
    for cjtx in coinjoins_relative_order:
        for input in coinjoins[cjtx]['inputs']:
            if 'spending_tx' in coinjoins[cjtx]['inputs'][input].keys():
                spending_tx, index = extract_txid_from_inout_string(coinjoins[cjtx]['inputs'][input]['spending_tx'])
                if spending_tx in coinjoins.keys():
                    coinjoins[cjtx]['inputs'][input]['is_standard_denom'] = coinjoins[spending_tx]['outputs'][index]['is_standard_denom']

    # Fix broadcast time based on relative ordering
    # Set artificial broadcast time base on minimum broadcast time of all txs with same relative order
    cj_ordering = [{'txid': cjtxid, 'relative_order': coinjoins[cjtxid]['relative_order'], 'broadcast_time': precomp_datetime.strptime(coinjoins[cjtxid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")} for cjtxid in coinjoins.keys()]
    sorted_cj_ordering = sorted(cj_ordering, key=lambda item: (item['relative_order'], item['broadcast_time']), reverse=False)

    # Print all transactions with 0 relative order (the "first" transaction(s))
    print('Transactions with relative order 0 ("first"):')
    for index in range(0, len(sorted_cj_ordering)):
        if sorted_cj_ordering[index]['relative_order'] == 0:
            print(f'  {sorted_cj_ordering[index]["broadcast_time"]}:{sorted_cj_ordering[index]["txid"]}')
        else:
            break

    min_broadcast_time = sorted_cj_ordering[0]['broadcast_time']
    min_broadcast_time_order = sorted_cj_ordering[0]['relative_order']
    broadcast_times_observed = [min_broadcast_time]
    for tx in sorted_cj_ordering:
        if min_broadcast_time_order < tx['relative_order']:
            # Next chuck of cjtxs as sorted by 'relative_order' going to be processed

            # Sanity check on broadcast_times_observed - shall be roughly same
            sorted_datetimes = sorted(broadcast_times_observed)
            time_difference = sorted_datetimes[-1] - sorted_datetimes[0]
            if time_difference > timedelta(days=1):
                print(f'WARNING: Coinjoins with same relative ordering \'{min_broadcast_time_order}\' differ too much \'{time_difference}\'. {tx["txid"]} ')

            # Set min_broadcast_time as a broadcast_time of first from this chunk
            min_broadcast_time = tx['broadcast_time']
            min_broadcast_time_order = tx['relative_order']
            broadcast_times_observed = [min_broadcast_time]  # Start new broadcast_times_observed for this chunk
        else:
            broadcast_times_observed.append(tx['broadcast_time'])  # Save broadcast_time of this cjtx

        # Set virtual time as minimum from the chunk if distance is more than 120 minutes
        # (do not correct cases where difference is too big and is not caused by delay in mining, but start of new pool instead)
        # (do not correct cases where difference is small and no delay in mining was introduced)
        time_difference = abs(precomp_datetime.strptime(coinjoins[tx['txid']]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") - min_broadcast_time)
        if time_difference > timedelta(days=14) or time_difference < timedelta(minutes=120):
            coinjoins[tx['txid']]['broadcast_time_virtual'] = coinjoins[tx['txid']]['broadcast_time']  # Use original time
        else:
            coinjoins[tx['txid']]['broadcast_time_virtual'] = precomp_datetime.strftime(min_broadcast_time)[:-3]  # Use corrected time

    # Compute ['broadcast_time']
    broadcast_reorder_times_diff_mins = [int(abs((precomp_datetime.strptime(coinjoins[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") - precomp_datetime.strptime(coinjoins[cjtx]['broadcast_time_virtual'], "%Y-%m-%d %H:%M:%S.%f")).total_seconds() / 60)) for cjtx in coinjoins.keys()]
    difference_counts = dict(Counter(broadcast_reorder_times_diff_mins))
    print(f'Broadcast time differences: {difference_counts}')
    difference_counts_str = {str(key): item for key, item in difference_counts.items()}
    save_json_to_file(os.path.join(target_path, 'tx_reordering_stats.json'), difference_counts_str)

    # Print summary results
    print_liquidity_summary(coinjoins, '')
    SM.print(f'  {get_ratio_string(total_mix_entering, total_inputs)} Inputs entering mix / total inputs used by mix transactions')
    SM.print(f'  {get_ratio_string(total_mix_friends, total_inputs)} Friends inputs re-entering mix / total inputs used by mix transactions')
    SM.print(f'  {get_ratio_string(total_mix_leaving, total_outputs)} Outputs leaving mix / total outputs by mix transactions')
    SM.print(f'  {get_ratio_string(len(total_mix_staying), total_outputs)} Outputs staying in mix / total outputs by mix transactions')
    SM.print(f'  {sum(total_mix_staying) / SATS_IN_BTC} btc, total value staying in mix')

    logging.debug('analyze_input_out_liquidity() finished')

    return coinjoins_relative_order


def smooth_interval(lst, window_size):
    #return compute_medians(lst, window_size)
    return compute_averages(lst, window_size)


def compute_averages(lst, window_size):
    averages = []
    window_sum = sum(lst[:window_size])  # Initialize the sum of the first window
    averages.append(window_sum / window_size)  # Compute and store the average of the first window

    # Slide the window and compute averages
    for i in range(1, len(lst) - window_size - 1):
        # Add the next element to the window sum and subtract the first element of the previous window
        window_sum += lst[i + window_size - 1] - lst[i - 1]
        averages.append(window_sum / window_size)  # Compute and store the average of the current window

    return averages


def compute_medians(lst, window_size):
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if window_size > len(lst):
        return []

    medians = []
    for i in range(len(lst) - window_size + 1):
        window = lst[i:i + window_size]
        medians.append(median(window))

    return medians


def get_output_name_string(txid, index):
    return f'vout_{txid}_{index}'


def get_input_name_string(txid, index):
    return f'vin_{txid}_{index}'


def extract_interval(data: dict, start_date: str, end_date: str):
    interval_data = {}
    if SORT_COINJOINS_BY_RELATIVE_ORDER:
        interval_data['coinjoins'] = {txid: data['coinjoins'][txid] for txid in data['coinjoins'].keys()
                                      if start_date < data['coinjoins'][txid]['broadcast_time_virtual'] < end_date}
    else:
        interval_data['coinjoins'] = {txid: data['coinjoins'][txid] for txid in data['coinjoins'].keys()
                                      if start_date < data['coinjoins'][txid]['broadcast_time'] < end_date}
    interval_data['postmix'] = {}
    if 'rounds' in data.keys():
        interval_data['rounds'] = {roundid: data['rounds'][roundid] for roundid in data['rounds'].keys()
                                   if
                                   start_date < data['rounds'][roundid]['round_start_time'] < end_date}
    interval_data['wallets_info'], interval_data['wallets_coins'] = extract_wallets_info(interval_data)

    if 'premix' in data.keys():  # Only for Whirlpool
        interval_data['premix'] = {txid: data['premix'][txid] for txid in data['premix'].keys()
                                   if start_date < data['premix'][txid]['broadcast_time'] < end_date}

    return interval_data


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
    datetime_obj = precomp_datetime.strptime(min_cj_time, "%Y-%m-%d %H:%M:%S.%f")
    datetime_obj = datetime_obj - timedelta(minutes=60)
    artificial_min_cj_time = datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    datetime_obj = precomp_datetime.strptime(max_cj_time, "%Y-%m-%d %H:%M:%S.%f")
    datetime_obj = datetime_obj + timedelta(minutes=60)
    artificial_max_cj_time = datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # 1. Extract all information from outputs and create also corresponding coins
    for cjtxid in txs_data.keys():
        for index in txs_data[cjtxid]['outputs'].keys():
            target_addr = txs_data[cjtxid]['outputs'][index]['address']
            wallet_name = txs_data[cjtxid]['outputs'][index].get('wallet_name', 'real_unknown')
            if wallet_name not in wallets_info.keys():
                wallets_info[wallet_name] = {}
                wallets_coins_info[wallet_name] = []
            wallets_info[wallet_name][target_addr] = {'address': target_addr}

            # Create new coin with information derived from output and transaction info
            coin = {'txid': cjtxid, 'index': index, 'amount': txs_data[cjtxid]['outputs'][index]['value'],
                    'anonymityScore': -1, 'address': target_addr, 'create_time': txs_data[cjtxid]['broadcast_time'],
                    'wallet_name': wallet_name, 'is_from_cjtx': False, 'is_spent_by_cjtx': False}
            #coin.update({'confirmed': True, 'confirmations': 1, 'keyPath': '', 'block_hash': txs_data[cjtxid]['block_hash']})
            coin['is_from_cjtx'] = txs_data[cjtxid].get('is_cjtx', False)
            if 'spend_by_tx' in txs_data[cjtxid]['outputs'][index].keys():
                spent_tx, spend_index = extract_txid_from_inout_string(txs_data[cjtxid]['outputs'][index]['spend_by_tx'])
                coin['spentBy'] = spent_tx
                coin['is_spent_by_cjtx'] = False if spent_tx not in txs_data.keys() else txs_data[spent_tx].get('is_cjtx', False)
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
    coins = {coin['address']: coin for coin in all_coins}  # BUGBUG: Will not work in case of address reuse!!!

    for cjtxid in txs_data.keys():
        for index in txs_data[cjtxid]['inputs'].keys():
            target_addr = txs_data[cjtxid]['inputs'][index]['address']
            wallet_name = txs_data[cjtxid]['inputs'][index].get('wallet_name', 'real_unknown')
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
                        'wallet_name': wallet_name, 'is_from_cjtx': False, 'is_spent_by_cjtx': False}
                # coin.update({'confirmed': True, 'confirmations': 1, 'keyPath': '', 'block_hash': txs_data[cjtxid]['block_hash']})
                coin['is_from_cjtx'] = False if txid not in txs_data.keys() else txs_data[txid].get('is_cjtx', False)

                coin['destroy_time'] = txs_data[cjtxid]['broadcast_time']
                coin['spentBy'] = cjtxid
                coin['is_spent_by_cjtx'] = False if cjtxid not in txs_data.keys() else txs_data[cjtxid].get('is_cjtx', False)
                coins[target_addr] = coin
            else:
                if coins[target_addr]['amount'] != txs_data[cjtxid]['inputs'][index]['value']:
                    print(f'Number of items in coins map: {len(coins)}')
                    print(f'{coins[target_addr]}')
                    assert coins[target_addr]['amount'] == txs_data[cjtxid]['inputs'][index]['value'], f'Inconsistent value found for {cjtxid}/{index}/{target_addr} {coins[target_addr]["amount"]} != {txs_data[cjtxid]["inputs"][index]["value"]}'
                # We have found the coin, update destroy_time
                coins[target_addr]['destroy_time'] = txs_data[cjtxid]['broadcast_time']
                if 'spentBy' not in coins[target_addr].keys():
                    coins[target_addr]['spentBy'] = cjtxid
                    coins[target_addr]['is_spent_by_cjtx'] = False if cjtxid not in txs_data.keys() else txs_data[cjtxid].get('is_cjtx', False)
                else:
                    assert coins[target_addr]['spentBy'] == cjtxid, f'Inconsistent spentBy mapping for {coins[target_addr]["address"]}'

    wallets_coins_info_updated = {}
    for address in coins.keys():
        coin = coins[address]
        if coin['wallet_name'] not in wallets_coins_info_updated.keys():
            wallets_coins_info_updated[coin['wallet_name']] = []
        wallets_coins_info_updated[coin['wallet_name']].append(coin)

    return wallets_info, wallets_coins_info_updated


def merge_dicts(source: dict, dest: dict):
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = dest.setdefault(key, {})
            merge_dicts(value, node)
        else:
            dest[key] = value

    return dest


def joinmarket_find_coinjoins(filename):
    """
    Extracts all coinjoin transactions stored as json fragment in logs.
    :param filename: name of file with logs
    :return: list of dictionaries for all specified group_names
    """
    hits = {}
    try:
        with (open(filename, 'r') as file):
            lines = file.readlines()
            line_index = 0
            while line_index < len(lines):
                regex_pattern = "(?P<timestamp>.*) \[INFO\]  obtained tx"
                match = re.search(regex_pattern, lines[line_index])
                line_index = line_index + 1
                if match is None:
                    continue
                else:
                    cjtx_lines = []
                    # After 'obtained tx', json is pasted in logs. Find its end by '}'
                    while lines[line_index] != '}\n':
                        cjtx_lines.append(lines[line_index])
                        line_index = line_index + 1
                    cjtx_lines.append(lines[line_index])
                    # Reconstruct json
                    cjtx_json = json.loads("".join(cjtx_lines))
                    # read next line to extract timestamp
                    line_index = line_index + 1
                    regex_pattern = "(?P<timestamp>.*) \[INFO\]"
                    match = re.search(regex_pattern, lines[line_index])
                    # Extract timestamp, replace , by . before fraction of seconds
                    cjtx_json['timestamp'] = match.group('timestamp').strip().replace(',', '.')

                    # # Extract cjtx json
                    # cjtx_lines = []
                    # regex_pattern = "(?P<timestamp>.*) \[INFO\]  INFO:Built tx, sending to counterparties."
                    # match = None
                    # while match is None:
                    #     match = re.search(regex_pattern, lines[line_index])
                    #     if match is None:
                    #         cjtx_lines.append(lines[line_index])
                    #     line_index = line_index + 1
                    # cjtx_json = json.loads("".join(cjtx_lines))
                    # cjtx_json['timestamp'] = match.group('timestamp').strip()

                    hits[cjtx_json['txid']] = cjtx_json

    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return hits


def find_round_ids(filename, regex_pattern, group_names):
    """
    Extracts all round_ids which from provided file which match regexec pattern and its specified part given by group_name.
    Function is more generic as any group_name from regex_pattern can be specified, not only round_id
    :param filename: name of file with logs
    :param regex_pattern: regex pattern which is matched to every line
    :param group_names: name of items specified in regex pattern, which are extracted
    :return: list of dictionaries for all specified group_names
    """
    hits = {}

    try:
        with open(filename, 'r') as file:
            for line in file:
                for match in re.finditer(regex_pattern, line):
                    hit_group = {}
                    for group_name in group_names:  # extract all provided group names
                        if group_name in match.groupdict():
                            hit_group[group_name] = match.group(group_name).strip()
                    # insert into dictionary with key equal to value of first hit group
                    key_name = match.group(group_names[0]).strip()
                    if key_name not in hits.keys():
                        hits[key_name] = []
                    hits[key_name].append(hit_group)

    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return hits


def find_round_cjtx_mapping(filename, regex_pattern, round_id, cjtx):
    """
    Extracts mapping between round id and its coinjoin tx id.
    :param filename: name of file with logs
    :param regex_pattern: regex pattern to match log line where mapping is found
    :param round_id: name in regex for round id item
    :param cjtx: name in regex for coinjointx id item
    :return: dictionary of mapping between round_id and coinjoin tx id
    """
    mapping = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                for match in re.finditer(regex_pattern, line):
                    if round_id in match.groupdict() and cjtx in match.groupdict():
                        mapping[match.group(round_id).strip()] = match.group(cjtx).strip()
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return mapping


def insert_type(items, type_info):
    for round_id, value in items.items():
        for index in value:
            index.update({'type': type_info.name})


def insert_by_round_id(rounds_logs, events):
    for round_id, value in events.items():
        if round_id not in rounds_logs:
            rounds_logs[round_id] = {}
        if 'logs' not in rounds_logs[round_id]:
            rounds_logs[round_id]['logs'] = []
        rounds_logs[round_id]['logs'].extend(value)


def parse_client_coinjoin_logs(base_directory):
    # Client logs parsing

    rounds_logs = {}

    # TODO: client log parsing
      # Wallet (XXX): CoinJoinClient finished. Coinjoin transaction was broadcast.  # 218

      # CoinJoinClient finished. Coinjoin transaction was not broadcast.    # 289
      # Aborted. Not enough participants.   # 143
      # Aborted. Not enough participants signed the coinjoin transaction.   #22
      # Aborted. Some Alices didn't confirm.        #47
      # Aborted. Some Alices didn't sign. Go to blame round.    # 931
      # Aborted. Load balancing registrations.      #77


      # Failed to handle the HTTP request via Tor       #45

      # ZKSNACKS IS NOW BLOCKING U.S. RESIDENTS AND CITIZENS    #5

      # ): Successfully registered X inputs
      # X out of Y Alices have signed the coinjoin tx.

    # 2023-10-23 16:23:30.303 [40] INFO	AliceClient.RegisterInputAsync (121)	Round (5455291d82b748469b5eb2e63d3859370c1f3823d4b8ca5cea7322f93b98af05), Alice (6eb340fe-d153-ac10-0246-252e8a866fbc): Registered 95cdc75886465b7e0a95b7f7e41a92c0ff92a8d2d075d426b92f0ca1b8424d2c-4.
    # 2023-10-23 16:23:38.053 [41] INFO	AliceClient.CreateRegisterAndConfirmInputAsync (77)	Round (5455291d82b748469b5eb2e63d3859370c1f3823d4b8ca5cea7322f93b98af05), Alice (6eb340fe-d153-ac10-0246-252e8a866fbc): Connection was confirmed.
    # 2023-10-23 16:24:05.939 [27] INFO	AliceClient.ReadyToSignAsync (223)	Round (5455291d82b748469b5eb2e63d3859370c1f3823d4b8ca5cea7322f93b98af05), Alice (6eb340fe-d153-ac10-0246-252e8a866fbc): Ready to sign.
    # 2023-10-23 16:24:46.110 [41] INFO	AliceClient.SignTransactionAsync (217)	Round (5455291d82b748469b5eb2e63d3859370c1f3823d4b8ca5cea7322f93b98af05), Alice (6eb340fe-d153-ac10-0246-252e8a866fbc): Posted a signature.
    client_input_file = os.path.join(base_directory, 'Logs.txt')

    print('Parsing coinjoin-relevant data from client logs {}...'.format(client_input_file), end='')

    # 2024-05-14 22:44:23.438 [35] INFO	CoinJoinManager.HandleCoinJoinFinalizationAsync (507)	Wallet (Wallet_mix_research): CoinJoinClient finished. Coinjoin transaction was broadcast.
    regex_pattern = r"(?P<timestamp>.*) INFO.+CoinJoinManager\.HandleCoinJoinFinalizationAsync.*Wallet \((?P<wallet_name>.*)\): CoinJoinClient finished. Coinjoin transaction was broadcast."
    broadcast_coinjoin_txs = find_round_ids(client_input_file, regex_pattern, ['timestamp', 'wallet_name'])
    insert_type(broadcast_coinjoin_txs, CJ_LOG_TYPES.COINJOIN_BROADCASTED)
    rounds_logs['no_round'].append(broadcast_coinjoin_txs)

    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Not enough inputs \((?P<num_participants>[0-9]+)\) in InputRegistration phase\. The minimum is \((?P<min_participants_required>[0-9]+)\)\. MaxSuggestedAmount was '([0-9\.]+)' BTC?"
    not_enough_participants = find_round_ids(client_input_file, regex_pattern,
                                             ['round_id', 'timestamp', 'num_participants', 'min_participants_required'])
    insert_type(not_enough_participants, CJ_LOG_TYPES.NOT_ENOUGH_PARTICIPANTS)
    insert_by_round_id(rounds_logs, not_enough_participants)

    alice_events_log = {}
    regex_pattern = r"(?P<timestamp>.*) \[.+(AliceClient.RegisterInputAsync.*) \(.*Round \((?P<round_id>.*)\), Alice \((?P<alice_id>.*)\): Registered (?P<tx_id>.*)-(?P<tx_out_index>[0-9]+)\.?"
    alice_events = find_round_ids(client_input_file, regex_pattern, ['round_id', 'timestamp', 'alice_id', 'tx_id', 'tx_out_index'])
    for round_id in alice_events.keys():
        for alice_event in alice_events[round_id]:
            alice_id = alice_event['alice_id']
            if alice_id not in alice_events_log.keys():
                alice_events_log[alice_id] = {}

            alice_events_log[alice_id][CJ_ALICE_TYPES.ALICE_REGISTERED.name] = alice_event

    regex_pattern = r"(?P<timestamp>.*) \[.+(AliceClient.CreateRegisterAndConfirmInputAsync.*) \(.*Round \((?P<round_id>.*)\), Alice \((?P<alice_id>.*)\): Connection was confirmed\.?"
    alice_events = find_round_ids(client_input_file, regex_pattern, ['round_id', 'timestamp', 'alice_id'])
    for round_id in alice_events.keys():
        for alice_event in alice_events[round_id]:
            alice_id = alice_event['alice_id']
            alice_events_log[alice_id][CJ_ALICE_TYPES.ALICE_CONNECTION_CONFIRMED.name] = alice_event

    regex_pattern = r"(?P<timestamp>.*) \[.+(AliceClient.ReadyToSignAsync.*) \(.*Round \((?P<round_id>.*)\), Alice \((?P<alice_id>.*)\): Ready to sign\.?"
    alice_events = find_round_ids(client_input_file, regex_pattern, ['round_id', 'timestamp', 'alice_id'])
    for round_id in alice_events.keys():
        for alice_event in alice_events[round_id]:
            alice_id = alice_event['alice_id']
            alice_events_log[alice_id][CJ_ALICE_TYPES.ALICE_READY_TO_SIGN.name] = alice_event

    regex_pattern = r"(?P<timestamp>.*) \[.+(AliceClient.SignTransactionAsync.*) \(.*Round \((?P<round_id>.*)\), Alice \((?P<alice_id>.*)\): Posted a signature\.?"
    alice_events = find_round_ids(client_input_file, regex_pattern, ['round_id', 'timestamp', 'alice_id'])
    for round_id in alice_events.keys():
        for alice_event in alice_events[round_id]:
            alice_id = alice_event['alice_id']
            alice_events_log[alice_id][CJ_ALICE_TYPES.ALICE_POSTED_SIGNATURE.name] = alice_event

    # Find and pair alice event logs to the right input
    #for cjtx_id in cjtx_stats['coinjoins'].keys():

    print('finished')

    return rounds_logs




def remove_link_between_inputs_and_outputs(coinjoins):
    for txid in coinjoins.keys():
        for index, input in coinjoins[txid]['inputs'].items():
            coinjoins[txid]['inputs'][index].pop('spending_tx', None)
        for index, output in coinjoins[txid]['outputs'].items():
            coinjoins[txid]['outputs'][index].pop('spend_by_txid', None)


def compute_link_between_inputs_and_outputs(coinjoins, sorted_cjs_in_scope):
    """
    Compute backward and forward connection between all transactions in sorted_cjs_in_scope list. As a result,
    for every input, 'spending_tx' record is inserted pointing to transaction and index of its output spent.
    For every output, 'spend_by_txid' is inserted pointing to transaction and its index which spents this output.
    :param coinjoins: structure with coinjoins
    :param sorted_cjs_in_scope: list of cj transactions to be used for calculating connections. Can be subset of
    coinjoins parameter - in such case, not all inputs and outputs will have 'spending_tx' and spend_by_txid' filled.
    :return: Updated structure with coinjoins
    """
    all_outputs = {}
    # Obtain all outputs as (address, value) tuples
    for tx_index in range(0, len(sorted_cjs_in_scope)):
        txid = sorted_cjs_in_scope[tx_index]
        for index, output in coinjoins[txid]['outputs'].items():
            all_outputs[output['address']] = (txid, index, output)  # (txid, output['address'], output['value'])

    # Check if such combination is in inputs of any other transaction in the scope
    for tx_index in range(0, len(sorted_cjs_in_scope)):
        txid = sorted_cjs_in_scope[tx_index]
        for index, input in coinjoins[txid]['inputs'].items():
            if input['address'] in all_outputs.keys() and input['value'] == all_outputs[input['address']][2]['value']:
                # we found corresponding input, mark it as used (tuple (txid, index))
                # Set also corresponding output 'spend_by_txid'
                target_output = all_outputs[input['address']]
                coinjoins[target_output[0]]['outputs'][target_output[1]]['spend_by_txid'] = (txid, index)
                coinjoins[txid]['inputs'][index]['spending_tx'] = (target_output[0], target_output[1])

    #
    # Update 'anon_score' item for inputs from previous outputs where exist
    #
    # Start with outputs - if spent, then fill anon_score
    for cjtx in coinjoins.keys():
        record = coinjoins[cjtx]
        for index in record['outputs'].keys():
            if 'spend_by_txid' in record['outputs'][index].keys():
                txid, tx_index = record['outputs'][index]['spend_by_txid']
                #tx_index = str(tx_index)
                if txid in coinjoins.keys():
                    if (txid in coinjoins.keys() and
                            'anon_score' in coinjoins[txid]['inputs'][tx_index].keys()):
                        assert math.isclose(coinjoins[txid]['inputs'][tx_index]['anon_score'], record['outputs'][index]['anon_score'], rel_tol=1e-9)
                    else:
                        coinjoins[txid]['inputs'][tx_index]['anon_score'] = record['outputs'][index]['anon_score']
    # Fill all non-set inputs to anonscore 1.0
    for cjtx in coinjoins.keys():
        record = coinjoins[cjtx]
        for index in record['inputs'].keys():
            if 'anon_score' not in record['inputs'][index].keys():
                record['inputs'][index]['anon_score'] = 1.0


    return coinjoins


def sort_coinjoins(cjtxs: dict, sort_by_order: bool = False):
    """
    Sort coinjoins based on time of mining or relative order
    :param cjtxs: coinjoins dictionary
    :param sort_by_order: if true, then sorted by relative order, by time otherwise
    :return: sorted list of cjtx ids
    """
    if sort_by_order:
        # Sort based on relative order
        cj_order = [{'txid': cjtxid, 'relative_order': cjtxs[cjtxid]['relative_order']} for cjtxid in cjtxs.keys()]
        sorted_cj_order = sorted(cj_order, key=lambda x: x['relative_order'])
        return sorted_cj_order
    else:
        # sort based on broadcast/mining time
        cj_time = [{'txid': cjtxid, 'broadcast_time': precomp_datetime.strptime(cjtxs[cjtxid]['broadcast_time'],
                                                                                "%Y-%m-%d %H:%M:%S.%f")}
                   for cjtxid in cjtxs.keys()]
        sorted_cj_time = sorted(cj_time, key=lambda x: x['broadcast_time'])
        return sorted_cj_time


def dump_json_to_db(cjtx_dict, db_path):
    # Dump to sqlite db
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")  # faster + concurrent reads
    con.execute("""
        CREATE TABLE IF NOT EXISTS txs (
            txid TEXT PRIMARY KEY,
            data BLOB        
        )
    """)

    tic = time.perf_counter()

    BATCH = 1_000
    it = iter(cjtx_dict.items())
    with con:
        batch = []
        for txid, tx in it:
            batch.append((txid, orjson.dumps(tx)))
            #batch.append((txid, msgpack.packb(tx, use_bin_type=True)))

            if len(batch) == BATCH:
                con.executemany("INSERT OR REPLACE INTO txs VALUES (?, ?)", batch)
                batch.clear()
        if batch:  # leftovers
            con.executemany("INSERT OR REPLACE INTO txs VALUES (?, ?)", batch)

    print(f"Wrote {len(cjtx_dict):,d} rows in {time.perf_counter() - tic:.1f}s")


def load_coinjoins_from_file_sqlite(target_load_path: str, false_cjtxs: dict, filter_false_positives: bool) -> dict:
    logging.debug(f'load_coinjoins_from_file_sqlite {target_load_path}/coinjoin_tx_info.json ...')
    data = load_json_from_file(os.path.join(target_load_path, f'coinjoin_tx_info.json'))
    logging.debug(f'  ... loaded.')
    db_path = os.path.join(target_load_path, f'coinjoin_tx_info.sqlite')
    logging.debug(f'Transforming to sqlite...')
    dump_json_to_db(data['coinjoins'], db_path)
    logging.debug(f'   ... done')
    del(data['coinjoins'])
    assert False, 'Unfinished, enable TxStore for use'
    #data['coinjoins'] = TxStore(db_path)
    #data['coinjoins'] = TxStoreMsgPack(db_path)
    assert False, 'Missing filtering of false positives, add'
    # # Filter false positives if required
    # if filter_false_positives:
    #     if false_cjtxs is None:
    #         fp_file = os.path.join(target_load_path, 'false_cjtxs.json')
    #         false_cjtxs = load_json_from_file(fp_file)
    #     for false_tx in false_cjtxs:
    #         if false_tx in data['coinjoins'].keys():
    #             data['coinjoins'].pop(false_tx)

    return data


def load_false_cjtxs(base_path: Path):
    """
    Loads false positives transactions from all files with 'false_cjtxs.json.*' format,
    then merge together
    :param base_path: path where to search for 'false_cjtxs.json.*' files
    :return: list of false positives transactions
    """
    false_cjtxs = set()
    # Add original file 'false_cjtxs.json'
    fp_files = [os.path.join(base_path, 'false_cjtxs.json')]
    # List all files with 'false_cjtxs.json.*' format and merge
    fp_files.extend(list(Path(base_path).glob('false_cjtxs.json.*')))
    for fp_file in fp_files:
        logging.debug(f"Reading false positives from file {fp_file}")
        false_cjtxs.update(load_false_cjtxs_from_file(fp_file))

    return list(false_cjtxs)


def load_false_cjtxs_from_file(fp_file):
    """
    Loads all false positive transactions from structured json (section->list of false cjtxs),
    then merge it together.
    :param fp_file: target file with false positives
    :return:
    """
    data = load_json_from_file(fp_file)
    false_cjtxs = [item for sublist in data.values() for item in sublist]
    if PERF_USE_SHORT_TXID:
        logging.warning(f'Loading load_false_cjtxs_from_file() making short {PERF_TX_SHORT_LEN} txid')
        return [txid[0:PERF_TX_SHORT_LEN] for txid in false_cjtxs]
    else:
        return false_cjtxs


def load_coinjoin_txids_from_file(target_file, start_date: str = None, stop_date: str = None):
    cjtxs = {}
    logging.debug(f'load_coinjoin_txids_from_file() Processing file {target_file}')
    with open(target_file, "r") as file:
        for line in file.readlines():
            parts = line.split(VerboseTransactionInfoLineSeparator)
            tx_id = None if parts[0] is None else parts[0]
            if tx_id:
                if PERF_USE_SHORT_TXID:
                    tx_id = tx_id[0:PERF_TX_SHORT_LEN]
                cjtxs[tx_id] = None

    return cjtxs


def load_coinjoins_from_file(target_load_path: str | Path, false_cjtxs: dict | None, filter_false_positives: bool, filtered_false_coinjoins: dict=None) -> dict:
    logging.info(f'Loading {target_load_path}/coinjoin_tx_info.json ...')
    data = load_json_from_file(os.path.join(target_load_path, f'coinjoin_tx_info.json'))

    non_cjtxs_keys = [cjtx for cjtx in data['coinjoins'].keys() if not data['coinjoins'][cjtx]['is_cjtx']]
    assert len(non_cjtxs_keys) == 0, f'Coinjoin list contains {len(non_cjtxs_keys)} unexpected non-coinjoin transactions, e.g., {non_cjtxs_keys[0]} '

    if PERF_USE_COMPACT_CJTX_STRUCTURE:
        logging.warning(f'IMPORTANT: PERF_USE_COMPACT_CJTX_STRUCTURE==True => compacting in-memory data structure')
        streamline_coinjoins_structure(data)

    # Filter false positives if required
    if filter_false_positives:
        if not filtered_false_coinjoins:
            filtered_false_coinjoins = {}
        if false_cjtxs is None:
            false_cjtxs = load_false_cjtxs(target_load_path)
        for false_tx in false_cjtxs:
            if false_tx in data['coinjoins'].keys():
                # Remove false transaction and place it into separate list (if provided)
                removed = data['coinjoins'].pop(false_tx)
                if filtered_false_coinjoins:
                    filtered_false_coinjoins[false_tx] = removed

        if filtered_false_coinjoins:
            # Store transactions filtered based on false positives file
            false_cjtxs_file = os.path.join(target_load_path, f'false_filtered_cjtxs_manual.json')
            save_json_to_file_pretty(false_cjtxs_file, filtered_false_coinjoins)

    return data


def load_coordinator_mapping_from_file(target_load_path: str | Path, filter_source_str: str=None):
    # Load initial data (two-level dictionary)
    data = load_json_from_file(target_load_path)
    # If required, filter out all keys which does not contain filter_source_str (if required)
    if filter_source_str is not None:
        for data_source in list(data.keys()):
            if filter_source_str not in data_source:
                data.pop(data_source)

    # Collapse into single-level dictionary with conflicts checking
    collapsed = {}
    for source in data.keys():
        for txid in data[source].keys():
            if txid not in collapsed.keys():
                collapsed[txid] = data[source][txid]
            else:
                assert collapsed[txid] == data[source][txid], f'Conflict in mapped coordinators for {txid} between {collapsed[txid]} and {data[source][txid]}'
                # if collapsed[txid] != data[source][txid]:
                #     print(f'Conflict in mapped coordinators for {txid} between {collapsed[txid]} and {data[source][txid]}')

    return collapsed


def compute_partial_vsize(tx_hex: str, input_indices: list[int], output_indices: list[int]):
    """
    Compute the exact virtual size (vsize) contribution of selected inputs and outputs
    into a Bitcoin transaction.

    :param tx_hex: Hexadecimal string of the raw Bitcoin transaction
    :param input_indices: List of input indices to include in the computation
    :param output_indices: List of output indices to include in the computation
    :return: Exact virtual size (vsize) in vbytes for the selected parts, total vsize for whole tx
    """
    # Deserialize transaction
    tx_bytes = bytes.fromhex(tx_hex)
    original_tx = CTransaction.deserialize(tx_bytes)
    orig_vsize = math.ceil(original_tx.calc_weight() / 4)

    # Turn original transaction into mutable and remove specified inputs and outputs
    mutable_tx = CMutableTransaction.from_tx(original_tx)
    # Filter out inputs and outputs we want to compute (tx2 is smaller tx without inputs and outputs to be evaluated)
    mutable_tx.vin = [mutable_tx.vin[index] for index in range(0, len(mutable_tx.vin)) if index not in input_indices]
    filtered_tx2_witness = tuple(item for index, item in enumerate(mutable_tx.wit.vtxinwit) if index not in input_indices)
    mutable_tx.vout = [mutable_tx.vout[index] for index in range(0, len(mutable_tx.vout)) if index not in output_indices]

    # Create new transaction with specified inputs and outputs removed
    filtered_tx = CMutableTransaction(mutable_tx.vin, mutable_tx.vout, mutable_tx.nLockTime, mutable_tx.nVersion, CTxWitness(filtered_tx2_witness))

    # Difference between original and filtered transaction is the contribution by the specified inputs and outputs
    filtered_weight = original_tx.calc_weight() - filtered_tx.calc_weight() if len(filtered_tx.vin) > 0 else original_tx.calc_weight()
    filtered_vsize = math.ceil(filtered_weight / 4)

    return filtered_vsize, orig_vsize


def get_address(script_hex: str):
    """
    Create an Output object from the script
    @param script_hex: hex string representation of the script
    """
    output = Output(lock_script=bytes.fromhex(script_hex), value=0)
    address = output.address

    return address, output.script_type


def detect_bybit_hack(target_path: str, interval: str, bybit_hack_addresses: dict):
    results = {'hits': {}}
    data = load_coinjoins_from_file(os.path.join(target_path, interval), {}, True)
    sorted_cjtxs = sort_coinjoins(data["coinjoins"], True)

    print('Bybit hack address detected')
    mixed_values = []
    for tx in sorted_cjtxs:
        cjtx = tx['txid']
        for index in data['coinjoins'][cjtx]['inputs'].keys():
            #script_type = data['coinjoins'][cjtx]['inputs'][index]['script_type']
            address, _ = get_address(data['coinjoins'][cjtx]['inputs'][index]['script'])
            # print(address)
            if address in bybit_hack_addresses:
                mixed_values.append(data['coinjoins'][cjtx]['inputs'][index]['value'])
                if address not in results['hits']:
                    results['hits'][address] = []
                results['hits'][address].append({'txid': cjtx, 'input_index': index,
                                         'value': data['coinjoins'][cjtx]['inputs'][index]['value'],
                                         'broadcast_time': data['coinjoins'][cjtx]['broadcast_time']})
                print(
                    f"{data['coinjoins'][cjtx]['broadcast_time']} {cjtx}:input[{index}]: {data['coinjoins'][cjtx]['inputs'][index]['value'] / float(SATS_IN_BTC)} btc")

        for index in data['coinjoins'][cjtx]['outputs'].keys():
            #script_type = data['coinjoins'][cjtx]['outputs'][index]['script_type']
            address, _ = get_address(data['coinjoins'][cjtx]['outputs'][index]['script'])
            # print(address)
            if address in bybit_hack_addresses:
                if address not in results['hits']:
                    results['hits'][address] = []
                results['hits'][address].append({'txid': cjtx, 'output_index': index,
                                                 'value': data['coinjoins'][cjtx]['outputs'][index]['value'],
                                                 'broadcast_time': data['coinjoins'][cjtx]['broadcast_time']})
                print(
                    f"{data['coinjoins'][cjtx]['broadcast_time']} {cjtx}:output[{index}]: {data['coinjoins'][cjtx]['outputs'][index]['value'] / float(SATS_IN_BTC)} btc")

    return results


def generate_tx_download_script(txids: list, file_name, target_path: Path | str):
    curl_lines = []
    for cjtx in txids:
        # Generate download record only for transactions not yet downloaded
        if not os.path.exists(os.path.join(target_path, f'{cjtx}.json')):
            curl_str = "curl --user user:password --data-binary \'{\"jsonrpc\": \"1.0\", \"id\": \"curltest\", \"method\": \"getrawtransaction\", \"params\": [\"" + cjtx + "\", true]}\' -H \'Content-Type: application/json\' http://127.0.0.1:8332/" + f" > {cjtx}.json\n"
            curl_lines.append(curl_str)
    with open(file_name, 'w') as f:
        f.writelines(curl_lines)

    return file_name


def get_input_address(txid, txid_in_out, raw_txs: dict = None):
    """
    Returns address which was used in transaction given by 'txid' as 'txid_in_out' output index
    :param txid: transaction id to read input address from
    :param txid_in_out: index in vout to read input address from
    :param raw_txs: pre-computed database of transactions
    :return:
    """
    if raw_txs is None:
        raw_txs = {}

    tx_info = raw_txs[txid]
    try:
        outputs = tx_info['vout']
        for output in outputs:
            if output['n'] == txid_in_out:
                return output['scriptPubKey']['address'], tx_info

    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)

    return None, None


def extract_tx_info(txid: str, raw_txs: dict):
    """
    Extract input and output addresses
    :param txid: transaction to parse
    :param raw_txs: dictionary with pre-loaded transactions
    :return: parsed transaction record
    """

    tx_info = raw_txs[txid]

    input_addresses = {}
    output_addresses = {}
    try:
        parsed_data = tx_info
        tx_record = {}

        tx_record['txid'] = txid
        # tx_record['raw_tx_json'] = parsed_data
        tx_record['inputs'] = {}
        tx_record['outputs'] = {}
        datetime_obj = datetime.fromtimestamp(tx_info['blocktime'], tz=UTC)
        tx_record['broadcast_time'] = datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        inputs = parsed_data['vin']
        index = 0
        for input in inputs:
            # we need to read and parse previous transaction to obtain address and other information
            in_address, in_full_info = get_input_address(input['txid'], input['vout'], raw_txs)

            tx_record['inputs'][str(index)] = {}
            tx_record['inputs'][str(index)]['address'] = in_address
            tx_record['inputs'][str(index)]['txid'] = input['txid']
            tx_record['inputs'][str(index)]['value'] = int(in_full_info['vout'][input['vout']]['value'] * SATS_IN_BTC)
            tx_record['inputs'][str(index)]['spending_tx'] = get_output_name_string(input['txid'], input['vout'])
            tx_record['inputs'][str(index)]['wallet_name'] = 'real_unknown'

            input_addresses[str(index)] = in_address  # store address to index of the input
            index = index + 1

        outputs = parsed_data['vout']
        for output in outputs:
            index = output['n']
            output_addresses[str(index)] = output['scriptPubKey']['address']
            tx_record['outputs'][str(index)] = {}
            tx_record['outputs'][str(index)]['address'] = output['scriptPubKey']['address']
            tx_record['outputs'][str(index)]['value'] = int(output['value'] * SATS_IN_BTC)
            # tx_record['outputs'][str(index)]['spend_by_tx'] = get_input_name_string(output['txid'], output['vout'])
            tx_record['outputs'][str(index)]['wallet_name'] = 'real_unknown'

    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None

    return tx_record


def run_command(command, verbose):
    """
    Execute shell command and return results
    :param command: command line to be executed
    :param verbose: if True, print intermediate results
    :return: command results with stdout, stderr and returncode (see subprocess CompletedProcess for documentation)
    """
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        if verbose:
            if result.returncode == 0:
                print("Command executed successfully.")
                print("Output:")
                print(result.stdout)
            else:
                print("Command failed.")
                print("Error:")
                print(result.stderr)
    except Exception as e:
        print("An error occurred:", e)

    return result


def streamline_coinjoins_structure(all_data:dict, compact_strong: bool=False):
    """
    Prune all_data dictionary of all (currently) unused data structures and shrinks unnecessarily long items like 32B txids.
    :param all_data: Initial dictionary with full
    :param compact_strong:
    :return:
    """
    full_txid_mapping = {'full_txid_map': {}}

    cjtxs_list = list(all_data['coinjoins'].keys())
    for cjtx in cjtxs_list:
        short_cjtx = cjtx[0:PERF_TX_SHORT_LEN] if compact_strong else cjtx
        full_txid_mapping['full_txid_map'][short_cjtx] = cjtx
        full_txid_mapping['full_txid_map'][cjtx] = short_cjtx

        all_data['coinjoins'][short_cjtx] = all_data['coinjoins'][cjtx]
        # Shorten
        all_data['coinjoins'][short_cjtx]['txid'] = short_cjtx
        # Remove
        all_data['coinjoins'][short_cjtx].pop('block_hash', None)
        all_data['coinjoins'][short_cjtx].pop('block_index', None)

        for index in all_data['coinjoins'][short_cjtx]['inputs'].keys():
            # Remove
            #all_data['coinjoins'][short_cjtx]['inputs'][index].pop('script', None)
            all_data['coinjoins'][short_cjtx]['inputs'][index].pop('script_type', None)
            all_data['coinjoins'][short_cjtx]['inputs'][index].pop('wallet_name', None)
            # Shorten
            if compact_strong:
                if 'spending_tx' in all_data['coinjoins'][short_cjtx]['inputs'][index]:
                    id = all_data['coinjoins'][short_cjtx]['inputs'][index]['spending_tx']
                    shorter = id[0:5 + PERF_TX_SHORT_LEN] + id[id.rfind('_'):]  # 'vout_TX_SHORT_LENchars_index'
                    all_data['coinjoins'][short_cjtx]['inputs'][index]['spending_tx'] = shorter

        for index in all_data['coinjoins'][short_cjtx]['outputs'].keys():
            # Remove
            #all_data['coinjoins'][short_cjtx]['outputs'][index].pop('script', None)
            all_data['coinjoins'][short_cjtx]['outputs'][index].pop('script_type', None)
            all_data['coinjoins'][short_cjtx]['outputs'][index].pop('wallet_name', None)
            # Shorten
            if compact_strong:
                if 'spend_by_tx' in all_data['coinjoins'][short_cjtx]['outputs'][index]:
                    id = all_data['coinjoins'][short_cjtx]['outputs'][index]['spend_by_tx']
                    shorter = id[0:4 + PERF_TX_SHORT_LEN] + id[id.rfind('_'):]  # 'vin_TX_SHORT_LENchars_index'
                    all_data['coinjoins'][short_cjtx]['outputs'][index]['spend_by_tx'] = shorter

        # Remove original long key
        if compact_strong and short_cjtx != cjtx:
            # Shorter cjtx id used, new record already created
            all_data['coinjoins'][cjtx] = None
            all_data['coinjoins'].pop(cjtx)

    return full_txid_mapping


def discover_coordinators(cjtxs: dict, sorted_cjtxs: list, coord_txs: dict, in_or_out: str,
                          min_coord_cjtxs: int, min_coord_fraction: float):
    """

    :param cjtxs:  All coinjoin transactions structure
    :param coord_txs: Mapping between cooridnator id and all its cjtxs
    :param sorted_cjtxs: Pre-sorted cjtxs (e.g., relative ordering based on transaction connections)
    :param in_or_out: if 'inputs', assignment wil be done based on cjtx inputs, if 'outputs' then on outputs
    :param min_coord_cjtxs minimum threshold number of coinjoins under coordinator to keep from filtering
    :param min_coord_fraction: minimum fraction of inputs/outputs to specific coordinator to assign
    :return: updated value of coord_txs and next_coord_index
    """
    print(f'\nFiltering small coordinators (min={min_coord_cjtxs})...')
    # Filter out coordinator ids with at least MIN_COORD_CJTXS transactions
    coord_txs_filtered = {coord_id: coord_txs[coord_id] for coord_id in coord_txs.keys()
                          if len(coord_txs[coord_id]) >= min_coord_cjtxs}
    print(f'  Total non-small coordinators: {len(coord_txs_filtered)}')
    # Reset coordinator ids for next iteration to start again from 0 to have unique counter again
    coord_ids = {}  # Speedup structure for fast cjtxs -> coordinator queries
    next_coord_index = -1
    coord_txs = {}  # Clear cjtx mapped to coordinator id for next iteration (will be re-created)
    for coord_id in coord_txs_filtered:  # All non-small coordinators
        next_coord_index = next_coord_index + 1
        for cjtx in coord_txs_filtered[coord_id]:
            coord_ids[cjtx] = next_coord_index
        coord_txs[next_coord_index] = coord_txs_filtered[coord_id]
        print(f'  coord. {next_coord_index}: {len(coord_txs_filtered[coord_id])} txs')
    print(f'Starting with next unused coordinator id: {next_coord_index + 1}\n')

    UNASSIGNED_COORD = -1
    for cjtx in sorted_cjtxs:
        if coord_ids.get(cjtx, UNASSIGNED_COORD) != UNASSIGNED_COORD:  # Check if already assigned
            continue
        if in_or_out == 'inputs':
            input_coords = [
                coord_ids.get(extract_txid_from_inout_string(cjtxs[cjtx]['inputs'][index]['spending_tx'])[0],
                              UNASSIGNED_COORD) for index in cjtxs[cjtx]['inputs'].keys()]
        elif in_or_out == 'outputs':
            input_coords = [
                coord_ids.get(extract_txid_from_inout_string(cjtxs[cjtx]['outputs'][index]['spend_by_tx'])[0],
                              UNASSIGNED_COORD) for index in cjtxs[cjtx]['outputs'].keys()
                                if 'spend_by_tx' in cjtxs[cjtx]['outputs'][index].keys()]
        else:
            assert False, f'Incorrect parameter in_or_out={in_or_out}'

        if len(input_coords) > 0:
            input_value_counts = Counter(input_coords)
            input_dominant_coord = input_value_counts.most_common()  # Take sorted list of the most common coordinators
            if input_dominant_coord[0][0] == UNASSIGNED_COORD:  # Dominant is not assigned
                if len(input_dominant_coord) > 1 and input_dominant_coord[1][1] / len(input_coords) >= min_coord_fraction:
                    # Take the second most dominant coordinator (after unassigned one which might be zksnacks)
                    coord_ids[cjtx] = input_dominant_coord[1][0]  # Mark this cjtx as belonging to the dominant coordinator
                    coord_txs[input_dominant_coord[1][0]].append(cjtx)  # Store cjtx for this coordinator
                else:
                    # Setup new coordinator
                    next_coord_index = next_coord_index + 1  # Assign unique new id (counter) for the coordinator
                    coord_ids[cjtx] = next_coord_index  # Assign coordinator id to this cjtx for future reference
                    coord_txs[next_coord_index] = [cjtx]  # Create new list for this coordinator, store current cjtx
            else:  # Dominant coordinator is already existing one
                coord_ids[cjtx] = input_dominant_coord[0][0]  # Mark this cjtx as belonging to the dominant coordinator
                coord_txs[input_dominant_coord[0][0]].append(cjtx)  # Store cjtx for this coordinator

    return coord_txs, next_coord_index


def wasabi_detect_coordinators_orig(mix_id: str, protocol: MIX_PROTOCOL, target_path):
    """
    Detect propagation of remix outputs to identify separate coordinators. Based on the assumption,
    that coinjoins under same coordinator will have majority of remixed inputs from the same coordinator.
    :param mix_id:
    :param protocol:
    :param target_path:
    :return:
    """
    # Read, filter and sort coinjoin transactions
    cjtxs = load_coinjoins_from_file(target_path, None, True)["coinjoins"]
    ordering = compute_cjtxs_relative_ordering(cjtxs)
    sorted_cjtxs = sorted(ordering, key=ordering.get)

    # Load known coordinators (will be used as starting set to expend to additional transactions)
    ground_truth_known_coord_txs = load_json_from_file(os.path.join(target_path, 'txid_coord.json'))  # Load known coordinators
    # Transform dictionary to {'coord': [cjtstxs]} format
    transformed_dict = defaultdict(list)
    for key, value in ground_truth_known_coord_txs.items():
        transformed_dict[value].append(key)
    initial_known_txs = dict(transformed_dict)
    save_json_to_file_pretty(os.path.join(target_path, 'txid_coord_t.json'), initial_known_txs)

    # Establish coordinator ids using two-pass process:
    # 1. First pass: Count dominant already existing coordinator for cjtx inputs.
    #    If not existing yet (-1), get new unique id (counter) and assign it for future processing
    # 2. Second pass: Perform second pass with coordinators with lower than MIN_COORD_CJTXS
    #    First pass may misclassify coordinators if transactions are out of order.
    MIN_COORD_CJTXS = 10
    MIN_COORD_FRACTION = 0.4

    coord_txs = initial_known_txs
    last_num_coordinators = -1
    pass_step = 0
    while last_num_coordinators != len(coord_txs):
        last_num_coordinators = len(coord_txs)
        print(f'\n# Current step {pass_step}')

        # Discover based on inputs
        coord_txs, next_coord_index = discover_coordinators(cjtxs, sorted_cjtxs, coord_txs, 'inputs', MIN_COORD_CJTXS, MIN_COORD_FRACTION)
        print_coordinators_counts(coord_txs, MIN_COORD_CJTXS)

        # Discover additionally based on outputs
        DISCOVER_ON_OUTPUTS = True
        if DISCOVER_ON_OUTPUTS:
            coord_txs, next_coord_index = discover_coordinators(cjtxs, sorted_cjtxs, coord_txs, 'outputs', MIN_COORD_CJTXS, MIN_COORD_FRACTION)
            print_coordinators_counts(coord_txs, MIN_COORD_CJTXS)

        pass_step = pass_step + 1

    # Print all coordinators and their txs
    print(f'\nTotal passes executed: {pass_step}')

    # TODO: Compute discovered stats to initial_known_txs

    # Try to merge coordinators
    # Idea: Almost all transactions are now assigned to perspective non-small coordinators
    #   Check again if coordinator infered from inputs and outputs match.
    #   If not, the is candidate for merging
    UNASSIGNED_COORD = -1
    coord_ids = {cjtx: coord_id for coord_id in coord_txs for cjtx in coord_txs[coord_id]}
    mergers = {coord_id: [] for coord_id in coord_txs.keys()}
    for cjtx in sorted_cjtxs:
        if cjtx not in coord_ids or coord_ids[cjtx] == UNASSIGNED_COORD:
            print(f'No coordinator set for {cjtx}')
    for cjtx in sorted_cjtxs:
        input_coords = [coord_ids.get(extract_txid_from_inout_string(cjtxs[cjtx]['inputs'][index]['spending_tx'])[0], UNASSIGNED_COORD) for index in cjtxs[cjtx]['inputs'].keys()]
        output_coords = [coord_ids.get(extract_txid_from_inout_string(cjtxs[cjtx]['outputs'][index]['spend_by_tx'])[0], UNASSIGNED_COORD)for index in cjtxs[cjtx]['outputs'].keys()
                         if 'spend_by_tx' in cjtxs[cjtx]['outputs'][index].keys()]
        input_value_counts = Counter(input_coords)
        output_value_counts = Counter(output_coords)
        if len(input_value_counts) > 0 and len(output_value_counts) > 0:
            input_dominant_coord = input_value_counts.most_common()[0]
            output_dominant_coord = output_value_counts.most_common()[0]
            if input_dominant_coord[0] != output_dominant_coord[0]:
                print(f'Dominant coordinator inconsistency detected for {cjtx}: {input_dominant_coord} vs. {output_dominant_coord}')
                print(f'  now set as {coord_ids[cjtx]}')
                if input_dominant_coord[0] != UNASSIGNED_COORD and output_dominant_coord[0] != UNASSIGNED_COORD:
                    print(f'  candidate for merger: {input_dominant_coord[0]} and {output_dominant_coord[0]}')
                    mergers[input_dominant_coord[0]].append(output_dominant_coord[0])


    print('Going to print detected candidates for merging. The merging shall be considered when multiple cases '
          'of same merge candidates are shown. '
          'E.g. {0: [1, 1], 1: [3, 3, 3, 3, 10], 2: [], 3: [1, 1, 1, 1], 4: [1], means that 1 and 3 shall be merged, while 1 and 4 likely not.')
    print(mergers)
    print_coordinators_counts(coord_txs, MIN_COORD_CJTXS)
    print_coordinators_counts(coord_txs, 2)

    DO_MERGING = False
    merged_coord_cjtxs_list = {}
    if DO_MERGING:
        def complete_bidirectional_closure(graph):
            # Function to perform DFS and return all reachable nodes from a given node
            def dfs(node, visited):
                if node not in visited:
                    visited.add(node)
                    for neighbor in graph.get(node, []):
                        dfs(neighbor, visited)
                return visited

            visited_global = set()

            # Process each key in the dictionary (each node)
            for key in graph.keys():
                if key not in visited_global:
                    # Find all nodes in the connected component of `key`
                    reachable = dfs(key, set())

                    # Mark all nodes in this component as visited globally
                    visited_global.update(reachable)

                    # Update all nodes in this component with the full list of reachable nodes
                    for node in reachable:
                        graph[node] = list(reachable)

            return graph

        # BUGBUG: this seems to merge too aggresively
        # mergers = complete_bidirectional_closure(mergers)

        # Manually filtered merge:
        #mergers = {0: [0], 1: [1, 3, 10], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7, 8, 9]}
        #print(f'Manual merges={mergers}')
        #mergers = {0: [0], 1: [1, 3, 10], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7, 8, 9]}
        mergers = {0: [0, 35], 1: [1, 30], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6, 39], 7: [7, 8, 9], 230: [230]}
        # wasabi2_opencoordinator is 2 and is good

        # Now merge
        merged_coord_cjtxs = {}
        for coord_id in sorted(mergers.keys()):
            if len(mergers[coord_id]) > 0:
                merged_coord_cjtxs[coord_id] = set()
                merged_coord_cjtxs[coord_id].update([tx for tx in coord_txs[coord_id]])
                for other_coord_id in mergers[coord_id]:
                    merged_coord_cjtxs[coord_id].update([tx for tx in coord_txs[other_coord_id]])
        print_coordinators_counts(merged_coord_cjtxs, MIN_COORD_CJTXS)

        # Turn from set to list
        for coord_id in merged_coord_cjtxs.keys():
            merged_coord_cjtxs_list[coord_id] = list(merged_coord_cjtxs[coord_id])

    # Detect coordinators
    # known_txs = {'kruw': ['0ec761ff2492659c86b416395d00bb7bd33d63ff0e9cbb896bf0acb3cf30456c',
    #                       'ca23ecbc3d5748d3655aa24b7a375378916a32b7480abce7ac3264f6c098efb9'],
    #              'gingerwallet': ['4a11b4e831db8dfd2a28428abd5f7d61d9df2390cdd48246919e954a357d29ae',
    #                               'eaec3b4e692d566dd4e0d3b76e4774eee15c7a07e933b2857a255f74c140e2e6',
    #                               '8205f43ab1f0ef4190c56bbc2633dda92c7837232ee537cb8771e9b98eae0314'],
    #              'opencoordinator': ['5097807006cb1b7d146263623c89e266cb0f7880b1566df6ec7bf1245bc72c15',
    #                                  '00eb9cbb7f93b72ad54d1825019b7c1a6c6730a03259aaeb95d51e4f22b16ad5'],
    #              'mega.cash': ['f16eac45453ba9614432de1507ec0783fe1e5144326a49ee32f73b006484857d',
    #                            '13d1681f239f185a4cdac4c403cd15952500f8576479aa0edaea60256af6ac4d']}

    pair_coords = {}
    for coord_name in initial_known_txs.keys():
        for coord_tx in initial_known_txs[coord_name][0:1]:
            if coord_tx in coord_ids.keys():
                print(f'coord_ids: {coord_ids[coord_tx]} paired to {coord_name} by {coord_tx}')
                pair_coords[coord_ids[coord_tx]] = coord_name
            else:
                print(f'Missing entry of Dumplings-based list for {coord_tx}')
            # for coord_id in sorted(merged_coord_cjtxs.keys()):
            #     if coord_tx in merged_coord_cjtxs[coord_id]:
            #         print(f'merged_coord_cjtxs: {coord_id} paired to {coord_name} by {coord_tx}')


    # # Sort coord txs based on its broadcast time
    # sorted_items = {}
    # for coord in coord_txs.keys():
    #     sorted_items[coord] = sorted(coord_txs[coord], key=lambda x: precomp_datetime.strptime(cjtxs[x]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f"))
    # coord_txs = sorted_items

    # Save discovered coordinators
    if DO_MERGING:
        coord_txs_to_save = merged_coord_cjtxs_list
    else:
        coord_txs_to_save = coord_txs

    save_json_to_file_pretty(os.path.join(target_path, 'txid_coord_discovered.json'), coord_txs_to_save)
    for coord_id in pair_coords.keys():
        if coord_id in coord_txs_to_save:
            coord_txs_to_save[pair_coords[coord_id]] = coord_txs_to_save.pop(coord_id)
    save_json_to_file_pretty(os.path.join(target_path, 'txid_coord_discovered_renamed.json'), coord_txs_to_save)

    PRINT_FINAL = False
    if PRINT_FINAL:
        print_coordinators_counts(coord_txs, 2)
        coord_txs_filtered = {coord_id: coord_txs[coord_id] for coord_id in coord_txs.keys() if
                              len(coord_txs[coord_id]) >= MIN_COORD_CJTXS}
        #print(coord_txs_filtered)
        print(f'# Total non-small coordinators (min={MIN_COORD_CJTXS}): {len(coord_txs_filtered)}')


def run_coordinator_detection(cjtxs: dict, sorted_cjtxs: list, ground_truth_known_coord_txs: dict, initial_known_txs: dict,
                              ASSERT_DUPLICATE_COORD_ASSIGNMENT = True, MIN_COORD_FRACTION: float = 0.4, MIN_COORD_CJTXS: int = 10):
    # Establish coordinator ids using two-pass process:
    # 1. First pass: Count dominant, already existing coordinator for cjtx inputs.
    #    If not existing yet (-1), get new unique id (counter) and assign it for future processing
    # 2. Second pass: Perform second pass with coordinators with lower than MIN_COORD_CJTXS
    # First pass may misclassify coordinators if transactions are out of order.

    # Store names of named coordinators from initial already known set (additional coordinator ids can be recovered later)
    named_coords = list(initial_known_txs.keys())

    coord_txs = initial_known_txs
    last_num_coordinators = -1
    last_coord_txs = {}
    pass_step = 0
    # while last_num_coordinators != len(coord_txs):
    #     last_num_coordinators = len(coord_txs)
    while last_coord_txs != coord_txs:
        last_coord_txs = copy.deepcopy(coord_txs)
        print(f'\n# Current step {pass_step}: {len(coord_txs)} coordinators')

        # Discover based on inputs
        coord_txs, next_coord_index = discover_coordinators(cjtxs, sorted_cjtxs, coord_txs, 'inputs', MIN_COORD_CJTXS, MIN_COORD_FRACTION)
        print_coordinators_counts(coord_txs, MIN_COORD_CJTXS)

        # Discover additionally based on outputs
        DISCOVER_ON_OUTPUTS = True
        if DISCOVER_ON_OUTPUTS:
            coord_txs, next_coord_index = discover_coordinators(cjtxs, sorted_cjtxs, coord_txs, 'outputs', MIN_COORD_CJTXS, MIN_COORD_FRACTION)
            print_coordinators_counts(coord_txs, MIN_COORD_CJTXS)

        pass_step = pass_step + 1

    print(f'\nTotal passes executed: {pass_step}')

    # Try to find candidates for merging (no actual merging performed)
    # Idea: Almost all transactions are now assigned to perspective non-small coordinators
    #   Check again if coordinator inferred from inputs and outputs match.
    #   If not, that is candidate for merging of clusters.
    #   'mergers' structure contains one record for each transaction, which has mismatch dominant coordinator based
    #      on inputs and outputs => mismatched coordinators might be actually same one
    UNASSIGNED_COORD = -1
    coord_ids = {cjtx: coord_id for coord_id in coord_txs for cjtx in coord_txs[coord_id]}
    merge_candidates = {coord_id: [] for coord_id in coord_txs.keys()}
    merge_candidates_dict = {}
    merge_candidates_dict["merge_candidates"] = {coord_id: [] for coord_id in coord_txs.keys()}
    merge_candidates_dict["all_cluster_links"] = {coord_id: [] for coord_id in coord_txs.keys()}
    merge_candidates_dict["all_cluster_links"][UNASSIGNED_COORD] = []
    for cjtx in sorted_cjtxs:
        if cjtx not in coord_ids or coord_ids[cjtx] == UNASSIGNED_COORD:
            print(f'No coordinator set for {cjtx}')
    for cjtx in sorted_cjtxs:
        input_coords = [coord_ids.get(extract_txid_from_inout_string(cjtxs[cjtx]['inputs'][index]['spending_tx'])[0], UNASSIGNED_COORD) for index in cjtxs[cjtx]['inputs'].keys()]
        output_coords = [coord_ids.get(extract_txid_from_inout_string(cjtxs[cjtx]['outputs'][index]['spend_by_tx'])[0], UNASSIGNED_COORD)for index in cjtxs[cjtx]['outputs'].keys()
                         if 'spend_by_tx' in cjtxs[cjtx]['outputs'][index].keys()]
        input_value_counts = Counter(input_coords)
        output_value_counts = Counter(output_coords)

        if cjtx in coord_ids:
            merge_candidates_dict["all_cluster_links"][coord_ids[cjtx]].append(
                {'txid': cjtx, 'input_coords': input_value_counts, 'output_coords': output_value_counts})
        else:
            merge_candidates_dict["all_cluster_links"][UNASSIGNED_COORD].append(
                {'txid': cjtx, 'input_coords': input_value_counts, 'output_coords': output_value_counts})

        if len(input_value_counts) > 0 and len(output_value_counts) > 0:
            input_dominant_coord = input_value_counts.most_common()[0]
            output_dominant_coord = output_value_counts.most_common()[0]

            if input_dominant_coord[0] != output_dominant_coord[0]:
                print(f'Dominant coordinator inconsistency detected for {cjtx}: coord={input_dominant_coord[0]}:{input_dominant_coord[1]}x vs. coord={output_dominant_coord[0]}:{output_dominant_coord[1]}x')
                print(f'  now set as {coord_ids[cjtx]}')
                if input_dominant_coord[0] != UNASSIGNED_COORD and output_dominant_coord[0] != UNASSIGNED_COORD:
                    print(f'  candidate for merger: {input_dominant_coord[0]} and {output_dominant_coord[0]}')
                    print(f'    input coordinators: {input_coords}')
                    print(f'    output coordinators: {output_coords}')
                    merge_candidates[input_dominant_coord[0]].append(output_dominant_coord[0])

                    merge_candidates_dict["merge_candidates"][input_dominant_coord[0]].append({'txid': cjtx, 'output_coord': output_dominant_coord[0], 'input_coords': input_coords, 'output_coords': output_coords})


    print('Going to print detected candidates for merging. The merging shall be considered when multiple cases '
          'of same merge candidates are shown. '
          'E.g. {0: [1, 1], 1: [3, 3, 3, 3, 10], 2: [], 3: [1, 1, 1, 1], 4: [1], means that 1 and 3 shall be merged, while 1 and 4 likely not.')
    print(merge_candidates)
    print_coordinators_counts(coord_txs, MIN_COORD_CJTXS)
    print_coordinators_counts(coord_txs, 2)

    #
    # Merging detected clusters to already known named coordinators
    #
    # Note: fully automated merging of complete whole clusters is NOT performed, consult wasabi_detect_coordinators_orig.complete_bidirectional_closure() for such option
    pair_cluster_index_2_coord_name = {}
    for cluster_index in coord_txs.keys():
        if cluster_index in pair_cluster_index_2_coord_name.keys():
            CHECK_COORDINATOR_CONSISTENCY = True
            if not CHECK_COORDINATOR_CONSISTENCY:
                continue  # If already paired, then we can speedup and do no checking
        for txid in coord_txs[cluster_index]:
            if txid in ground_truth_known_coord_txs:
                # Transaction is known to be paired to known coordinator => pair whole cluster
                if cluster_index not in pair_cluster_index_2_coord_name.keys():
                    # New pairing detected
                    print(f'coord_ids: {cluster_index} paired to {ground_truth_known_coord_txs[txid]} by {txid}')
                    pair_cluster_index_2_coord_name[cluster_index] = ground_truth_known_coord_txs[txid]
                else:
                    # For testing of robustness of coordinator detection, we need to disable this assert
                    if ASSERT_DUPLICATE_COORD_ASSIGNMENT:
                        assert pair_cluster_index_2_coord_name[cluster_index] == ground_truth_known_coord_txs[txid], \
                            f'Duplicate coordinator pairing detected for cluster {cluster_index}: {pair_cluster_index_2_coord_name[cluster_index]} vs. {ground_truth_known_coord_txs[txid]}'

    merge_candidates_dict["cluster_names"] = pair_cluster_index_2_coord_name
    for cluster_id in list(merge_candidates_dict["all_cluster_links"].keys()):
        if cluster_id in merge_candidates_dict["cluster_names"]:
            merge_candidates_dict["all_cluster_links"][f"{cluster_id}__{merge_candidates_dict['cluster_names'][cluster_id]}"] = merge_candidates_dict["all_cluster_links"].pop(cluster_id)

    coord_txs_named = copy.deepcopy(coord_txs)
    for coord_id in pair_cluster_index_2_coord_name.keys():
        if coord_id in coord_txs_named:
            coord_txs_named[pair_cluster_index_2_coord_name[coord_id]] = coord_txs_named.pop(coord_id)
    coord_txs_named_sorted = {}
    for coord_id in coord_txs_named.keys():
        without_date_sorted = [txid for txid in coord_txs_named[coord_id] if txid not in cjtxs.keys()]
        with_date = [txid for txid in coord_txs_named[coord_id] if txid in cjtxs.keys()]
        with_date_sorted = sorted(with_date, key=lambda x: cjtxs[x]['broadcast_time'])
        coord_txs_named_sorted[coord_id] = without_date_sorted + with_date_sorted

    # Add all not attributed transactions (coord not in named_coords)
    all_attributed = [txid for coord_id in coord_txs_named_sorted for txid in coord_txs_named_sorted[coord_id] if coord_id in named_coords]
    unattributed = [txid for txid in cjtxs.keys() if txid not in all_attributed]
    coord_txs_named_sorted['unattributed'] = sorted(unattributed, key=lambda x: cjtxs[x]['broadcast_time'])

    PRINT_FINAL = False
    if PRINT_FINAL:
        print_coordinators_counts(coord_txs_named, 2)
        coord_txs_filtered = {coord_id: coord_txs_named[coord_id] for coord_id in coord_txs_named.keys() if
                              len(coord_txs_named[coord_id]) >= MIN_COORD_CJTXS}
        #print(coord_txs_filtered)
        print(f'# Total non-small coordinators (min={MIN_COORD_CJTXS}): {len(coord_txs_filtered)}')

    return merge_candidates_dict, coord_txs, coord_txs_named, coord_txs_named_sorted


def wasabi_detect_coordinators(mix_id: str, protocol: MIX_PROTOCOL, target_path):
    """
    Detect propagation of remix outputs to identify separate coordinators. Is based on the assumption,
    that coinjoins under same coordinator will have majority of remixed inputs from/to the same coordinator.
    The method iteratively places coinjoin transaction into a cluster based on cluster of a majority of inputs/outputs,
    combined with list of know ground truth mappings between transactions and known coordinators ('txid_coord.json').
    The method is not fully automatic as outputs for a deceased coordinator typically join another coordinator eventually
    and naive application of heuristic would results in overly aggresive merge of clusters (typically to a dominant one
    at the times). Instead, candidate clusters are created and user analyst is expected to decide if ones not yet
    attributed to specific coordinator are separate from known ones or not (see txid_coord_merge_candidates.json).
    :param mix_id:
    :param protocol:
    :param target_path:
    :return:
    """
    # Read, filter and sort coinjoin transactions
    cjtxs = load_coinjoins_from_file(target_path, None, True)["coinjoins"]
    ordering = compute_cjtxs_relative_ordering(cjtxs)
    sorted_cjtxs = sorted(ordering, key=ordering.get)

    # Load known coordinators (will be used as starting set to expand to additional transactions)
    data = load_json_from_file(os.path.join(target_path, 'txid_coord.json'))  # Load known coordinators
    ground_truth_known_coord_txs = {key:data[sublist][key] for sublist in data.keys() for key in data[sublist].keys()}

    # Transform dictionary to {'coord': [cjtx]} format
    transformed_dict = defaultdict(list)
    for key, value in ground_truth_known_coord_txs.items():
        transformed_dict[value].append(key)
    initial_known_txs = dict(transformed_dict)
    save_json_to_file_pretty(os.path.join(target_path, 'txid_coord_t.json'), initial_known_txs)  # Save transformed version for easier human lookup

    #
    # Run core detection
    #
    merge_candidates_dict, coord_txs_unnamed, coord_txs_named, coord_txs_named_sorted = run_coordinator_detection(
        cjtxs, sorted_cjtxs, ground_truth_known_coord_txs, initial_known_txs
    )

    # Save results
    save_json_to_file_pretty(os.path.join(target_path, 'txid_coord_discovered.json'), coord_txs_unnamed)
    save_json_to_file_pretty(os.path.join(target_path, 'txid_coord_merge_candidates.json'), merge_candidates_dict)
    save_json_to_file_pretty(os.path.join(target_path, 'txid_coord_discovered_renamed.json'), coord_txs_named_sorted)

    tx_to_coord_map = {txid:coord for coord in coord_txs_named.keys() for txid in coord_txs_named[coord]}
    save_json_to_file_pretty(os.path.join(target_path, 'txid_to_coord_discovered_renamed.json'), tx_to_coord_map)


def get_missing_cjtxs(cjtxs: dict, mappings: dict, dataset_names: list, target_path: str | Path):
    # Extract and save txids for transactions included in mappings, but not in cjtxs
    crawl_coord_txs = {txid: None for dataset, txs in mappings.items() if dataset in dataset_names for txid in txs}
    missing_crawl = {txid: None for txid in cjtxs['coinjoins'] if
                     cjtxs['coinjoins'][txid]['broadcast_time'] > '2024-06-01' and txid not in crawl_coord_txs}
    missing_cjtxs = {txid: None for txid in crawl_coord_txs if txid not in cjtxs['coinjoins']}

    SM.print(f'Missing in crawl: {len(missing_crawl)}, in cjtxs: {len(missing_cjtxs)}')

    save_json_to_file(os.path.join(target_path, 'missing_cjtxs_from_crawl.json'),
                          list(missing_crawl.keys()))
    save_json_to_file(os.path.join(target_path, 'missing_cjtxs_from_dumplings.json'),
                          list(missing_cjtxs.keys()))

    return missing_cjtxs, missing_crawl


def split_coinjoins_per_interval(cjtxs: dict, mix_protocol):
    # Compute liquidity inflows (sum of days/weeks/months)
    days_dict = defaultdict(dict)
    weeks_dict = defaultdict(dict)
    months_dict = defaultdict(dict)
    # Split cjtxs into weeks, then compute sum of MIX_ENTER
    for key, record in cjtxs.items():
        # Parse the 'broadcast_time/virtual' string into a datetime object
        if mix_protocol == MIX_PROTOCOL.WASABI2:
            dt = precomp_datetime.strptime(record['broadcast_time_virtual'], '%Y-%m-%d %H:%M:%S.%f')
        else:
            dt = precomp_datetime.strptime(record['broadcast_time'], '%Y-%m-%d %H:%M:%S.%f')
        year, week_num, _ = dt.isocalendar()
        weeks_dict[(year, week_num)][key] = record
        day_key = (dt.year, dt.month, dt.day)
        days_dict[day_key][key] = record
        month_key = (dt.year, dt.month)
        months_dict[month_key][key] = record

    return days_dict, weeks_dict, months_dict


def compute_interval_aggregate_custom(interval_to_aggregate: dict, in_out_case: str, mix_events_types: list, limit_bounds: tuple[int, int]=None):
    """
    Computes aggregation of values of provide type(s) (mix_events_types) over provided interval
    :param interval_to_aggregate: conjoin records already separated per desired interval (e.g., daily/weekly/monthly)
    :param in_out_case: transaction 'inputs' or 'outputs' to consider
    :param mix_events_types: list of MIX_EVENT_TYPE types which shall be considered
    :param limit_bounds: upper and lower bound for value (in satoshis) size to consider
    :return: dictionary of stats for intervals
    """
    if limit_bounds is None:  # If not given, set bounds to extreme values
        limit_bounds = (-1, MAX_SATS)

    aggregated_vals = {}
    # Aggregate desired property over desired interval
    for interval in sorted(interval_to_aggregate.keys()):
        records = interval_to_aggregate[interval]  # Records from single interval to aggregate

        # If 'value' (in sats) is required (use_value_property == True), then read ['value'] property, otherwise just note occurence (=1)
        interval_items = [records[cjtx][in_out_case][index]['value'] for cjtx in records.keys()
                          for index in records[cjtx][in_out_case].keys()
                          if records[cjtx][in_out_case][index]['mix_event_type'] in mix_events_types and
                          limit_bounds[0] <= records[cjtx][in_out_case][index]['value'] <= limit_bounds[1]]
        aggregated_vals_sats = sum(interval_items)
        aggregated_vals_btc = sum(interval_items) / SATS_IN_BTC
        aggregated_vals_counts = len(interval_items)
        logging.debug(f"Interval {interval}: {aggregated_vals_btc} btc, {aggregated_vals_sats} sats, {aggregated_vals_counts} cases, num_cjtxs={len(records)}")

        aggregated_vals[interval] = {'interval': interval, 'values_sats': aggregated_vals_sats,
                                     'values_btc': aggregated_vals_btc, 'counts': aggregated_vals_counts}
        cfg_used = {'in_out_case': in_out_case, 'mix_events_types': mix_events_types,
                    'limit_value_min': limit_bounds[0], 'limit_value_max': limit_bounds[1]}

    return aggregated_vals


def compute_interval_aggregates(cjtxs: dict, mix_id):
    aggregates = {}

    # Split provided coinjoins per base intervals
    days_dict, weeks_dict, months_dict = split_coinjoins_per_interval(cjtxs, mix_id)

    # Compute select aggregated properties
    for interval_str, interval_data in [('day', days_dict), ('week', weeks_dict), ('month', months_dict)]:
        aggregated_vals_interval = {}
        # Aggregate desired property over desired interval
        for interval in sorted(interval_data.keys()):
            records = interval_data[interval]  # Records from single interval to aggregate
            lr = compute_liquidity_summary(records, False)

            interval_string = '_'.join(map(str, interval))
            aggregated_vals_interval[interval_string] = lr

        aggregates[f'{interval_str}'] = aggregated_vals_interval

    return aggregates