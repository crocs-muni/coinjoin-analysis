import logging
import copy
import math
import multiprocessing
import os
import pickle
import random
import shutil
from copy import deepcopy
from enum import Enum
from multiprocessing.pool import ThreadPool
from datetime import timedelta, UTC, datetime
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
import numpy as np
from cj_process.cj_analysis import get_output_name_string, get_input_name_string
from cj_process import cj_analysis as als
import argparse
import gc
import time
import ast
import requests
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
import tracemalloc


import cj_process.cj_visualize as cjvis
from cj_process.cj_consts import VerboseInOutInfoInLineSeparator, WHIRLPOOL_FUNDING_TXS, SATS_IN_BTC, \
    WHIRLPOOL_POOL_NAMES_ALL, WHIRLPOOL_POOL_SIZES
from cj_process.cj_structs import CJ_TX_CHECK, MIX_PROTOCOL, CLUSTER_INDEX, MIX_EVENT_TYPE, SM, \
    CJ_TX_CHECK_WHIRLPOOL_DEFAULTS, FILTER_REASON, CJ_TX_CHECK_WASABI1_DEFAULTS, CJ_TX_CHECK_WASABI2_DEFAULTS, \
    CJ_TX_CHECK_JOINMARKET_DEFAULTS, precomp_datetime, CoinMixInfo, CoinJoinStats

# If True, difference between assigned and existing cluster id is checked and failed upon if different
# If False, only warning is printed, but execution continues.
# TODO: Systematic solution requires merging and resolving different cluster ids
CLUSTER_ID_CHECK_HARD_ASSERT = False

op = None  # Global settings for the experiment


# Configure the logging module
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger_to_disable = logging.getLogger("mathplotlib")
logger_to_disable.setLevel(logging.WARNING)


# SM = als.SummaryMessages()
# als.SM = SM



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


def load_coinjoin_stats_from_dumplings(target_file, start_date: str = None, stop_date: str = None, multiple_files: bool = True):
    """
    Loads stored coinjoin records from files created by Dumplings project into internal structure.
    Multiple files can be loaded from all files with 'target_file' prefix (if exists) with results merged together.
    :param target_file: target file name
    :param start_date: start date to filter
    :param stop_date: end date to filter
    :param multiple_files: if false, hen only target_file is parsed. If true, target_file is treated as prefix with 'target_file' and any 'target_file.*' also loaded if present
    :return: loaded internal structure from one or more files merged together
    """
    search_target = Path(target_file)
    candidates = [search_target]
    if multiple_files:
        candidates = candidates + list(search_target.parent.glob(f"{search_target.name}.*"))  # Grab every candidate that starts with 'target_file'

    cj_stats = {}
    for file in candidates:
        cj_stats.update(load_coinjoin_stats_from_dumplings_file(str(file), start_date, stop_date))

    return cj_stats


def load_coinjoin_stats_from_dumplings_file(target_file: str, start_date: str = None, stop_date: str = None):
    cj_stats = {}
    logging.debug(f'Processing file {target_file}')
    json_file = target_file + '.json'
    if os.path.exists(json_file):
        with open(json_file, "rb") as file:
            cj_stats = pickle.load(file)
    else:
        with open(target_file, "r") as file:
            num_lines = 0
            for line in file.readlines():
                num_lines += 1
                # if num_lines % 10 == 0:
                #     print('.', end="")
                # if num_lines % 1000 == 0:
                #     print(f"{num_lines}")
                parts = line.split(als.VerboseTransactionInfoLineSeparator)
                record = {}

                # Be careful, broadcast time and blocktime can be significantly different
                block_time = None if parts[3] is None else datetime.fromtimestamp(int(parts[3]), tz=UTC)
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
                    this_input['script_type'] = segments[4].strip()
                    # TODO: generate proper address from script, now replaced by synthetic
                    # BUGBUG: if segments[3], segments[1] is used, then incorrect synthetic address is generated in case
                    # of address resuse (cj_analysis.py", line 910) : AssertionError: Inconsistent value found for
                    # 9be067b5311adb18a3458a6f9e164a25e0590ad8a8fc6907da0288f80bf25bc9/3/synbc1001407fb8593407d_1
                    #this_input['address'] = get_synthetic_address(segments[3], segments[1])

                    this_input['address'] = get_synthetic_address(segments[0], segments[1])
                    #this_input['address'], this_input['script_type'] = als.get_address(this_input['script'])

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
                    this_output['script_type'] = segments[2].strip()
                    this_output['address'] = get_synthetic_address(tx_id, index)  # TODO: Compute proper address from script
                    #this_output['address'], this_output['script_type'] = als.get_address(this_output['script'])

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
        coinjoin_stats[target_file]["coinjoins"] = load_coinjoin_stats_from_dumplings(target_file)

    return coinjoin_stats


def extract_rounds_info(data):
    rounds_info = {}
    txs_data = data["coinjoins"]
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
    update_spend_by_reference(data['postmix'], data["coinjoins"])

    if 'premix' in data.keys():
        # backward reference from coinjoin to premix is already set ('spending_tx')
        # now set also forward link ('spend_by_tx')
        update_spend_by_reference(data["coinjoins"], data['premix'])

    return data


def detect_false_coinjoins(data, mix_protocol, checks: CJ_TX_CHECK=CJ_TX_CHECK.BASIC, checks_params=None):
    checks_params = {} if checks_params is None else checks_params

    false_cjtxs = {}
    cjtxids = list(data["coinjoins"].keys())
    for cjtx in cjtxids:
        valid, reason = is_coinjoin_tx(data["coinjoins"][cjtx], mix_protocol, checks, checks_params)
        if not valid:
            false_cjtxs[cjtx] = deepcopy(data["coinjoins"][cjtx])
            false_cjtxs[cjtx]['is_cjtx'] = False
            false_cjtxs[cjtx]['fp_reason'] = reason.name

    return false_cjtxs


def filter_false_coinjoins(data, mix_protocol, checks: CJ_TX_CHECK=CJ_TX_CHECK.BASIC, checks_params=None):
    false_cjtxs = detect_false_coinjoins(data, mix_protocol, checks, checks_params)
    for cjtx in false_cjtxs:
        logging.debug(f'{cjtx} is not coinjoin, removing from coinjoin list')
        data["coinjoins"].pop(cjtx)

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


def filter_postmix_transactions(data):
    for txid in list(data['postmix'].keys()):
        is_postmix = False
        for index in data['postmix'][txid]['inputs'].keys():
            if 'spending_tx' in data['postmix'][txid]['inputs'][index]:
                tx, vout = als.extract_txid_from_inout_string(data['postmix'][txid]['inputs'][index]['spending_tx'])
                if tx in data['coinjoins'].keys():
                    is_postmix = True  # At least one direct postmix spend input found
                    break
            else:
                logging.warning(f'spending_tx not found in {txid} {index}')
        if not is_postmix:
            data['postmix'].pop(txid)
    return data


def detect_referenced_trasactions(txs_from, txs_to):
    """
    Detects list of transcations from txs_to which are referenced (spending_tx)
    by at least one transaction from txs_from.
    :param txs_from:
    :param txs_to:
    :return: referenced
    """
    referenced = {}
    for txid in txs_from.keys():
        for index in txs_from[txid]['inputs'].keys():
            if 'spending_tx' in txs_from[txid]['inputs'][index]:
                tx, vout = als.extract_txid_from_inout_string(txs_from[txid]['inputs'][index]['spending_tx'])
                if tx in txs_to.keys():
                    referenced[tx] = txs_to[tx]
    return referenced


def update_all_spend_by_reference(data: dict):
    # backward reference to spending transaction output is already set ('spending_tx'),
    # now set also forward link ('spend_by_tx')

    # Update 'premix' based on 'coinjoin' tx 'spending_tx'
    total_updated = update_spend_by_reference(data["coinjoins"], data['premix'])
    logging.debug(f'Update premix based on coinjoins: {total_updated}')
    # Update 'coinjoin' based on 'coinjoin'
    total_updated = update_spend_by_reference(data["coinjoins"], data["coinjoins"])
    logging.debug(f'Update coinjoins based on coinjoins: {total_updated}')
    # Update 'coinjoin' based on 'postmix'
    total_updated = update_spend_by_reference(data["coinjoins"], data['postmix'])
    logging.debug(f'Update coinjoins based on postmix: {total_updated}')

    return data


def analyze_false_positives(data, false_cjtxs, mix_protocol):
    """
    Analyze properties of detected false positives
    :param data:
    :param false_cjtxs:
    :param mix_protocol:
    :return:
    """
    # Detect transactions from false positives list which are referenced by real coinjoins and print it
    referenced = detect_referenced_trasactions(data['coinjoins'], false_cjtxs)
    if len(false_cjtxs) > 0:
        print(f'Number of false positives coinjoins referenced by real coinjoins: {len(referenced)} of {len(false_cjtxs.keys())} false positives ({round(len(referenced) / len(false_cjtxs.keys()), 2)}%)')
    else:
        print(f'Number of false positives coinjoins referenced by real coinjoins: {len(referenced)} of {len(false_cjtxs.keys())} false positives ({0}%)')

    for txid in referenced.keys():
        logging.debug(f"  {txid}: {false_cjtxs[txid]['fp_reason']}")

    # Detect and filter only transactions which has too small number of same value outputs
    for min_same_values_threshold in range(2, 10):
        # Use detection rule on for the MIN_SAME_VALUES_THRESHOLD
        detected = detect_false_coinjoins(data, mix_protocol, CJ_TX_CHECK.MIN_SAME_VALUES_THRESHOLD, {CJ_TX_CHECK.MIN_SAME_VALUES_THRESHOLD: min_same_values_threshold})
        print(f'Number of detected coinjoins with only {min_same_values_threshold - 1} : {len(detected)}')


def load_coinjoins(target_path: str, mix_protocol: MIX_PROTOCOL, mix_filename: str, postmix_filename: str, premix_filename: str,
                   start_date: str, stop_date: str) -> (dict, dict, dict):
    SM.print("### Load from Dumplings artifacts")
    # All mixes are having mixing coinjoins and postmix spends
    data = {'rounds': {}, 'filename': os.path.join(target_path, mix_filename),
            'coinjoins': load_coinjoin_stats_from_dumplings(os.path.join(target_path, mix_filename), start_date, stop_date),
            'postmix': load_coinjoin_stats_from_dumplings(os.path.join(target_path, postmix_filename), start_date, stop_date)}
    SM.print(f"  Number of raw loaded coinjoins: {len(data['coinjoins'].keys())}")
    # Only Samourai Whirlpool is having premix tx (TX0)
    cjtxs_fixed = 0
    if mix_protocol == MIX_PROTOCOL.WHIRLPOOL:
        data['premix'] = load_coinjoin_stats_from_dumplings(os.path.join(target_path, premix_filename), start_date, stop_date)
        for txid in list(data['premix'].keys()):
            valid, reason = is_coinjoin_tx(data['premix'][txid], mix_protocol)
            if valid:
                # Misclassified mix transaction, move between groups
                data["coinjoins"][txid] = data['premix'][txid]
                data['premix'].pop(txid)
                logging.info(f'{txid} is mix transaction, removing from premix and putting to mix')
                cjtxs_fixed += 1
    else:
        data['premix'] = {}
    SM.print(f'  {cjtxs_fixed} total premix txs moved into coinjoins')

    # Detect misclassified Whirlpool coinjoin transactions found in Dumpling's postmix txs
    cjtxs_fixed = 0
    if mix_protocol == MIX_PROTOCOL.WHIRLPOOL:
        for txid in list(data['postmix'].keys()):
            valid, reason = is_coinjoin_tx(data['postmix'][txid], mix_protocol)
            if valid:
                # Misclassified mix transaction, move between groups
                data["coinjoins"][txid] = data['postmix'][txid]
                data["coinjoins"][txid]['is_cjtx'] = True
                data['postmix'].pop(txid)
                logging.info(f'{txid} is mix transaction, removing from postmix and putting to mix')
                cjtxs_fixed += 1
    SM.print(f'  {cjtxs_fixed} total postmix txs moved into coinjoins')

    # Filter mistakes (false positives) in Dumplings analysis of coinjoins
    data, false_cjtxs = filter_false_coinjoins(data, mix_protocol, CJ_TX_CHECK.BASIC)
    SM.print(f'  Number of filtered false positives (CJ_TX_CHECK.BASIC): {len(false_cjtxs)}')
    # Analyze properties of detected false positives
    analyze_false_positives(data, false_cjtxs, mix_protocol)

    if mix_protocol == MIX_PROTOCOL.JOINMARKET:
        # Idea:
        # 0. Filter away transactions with only two equal outputs (even initial joinmarket had at least three)
        # 1. Filter transactions with less than X(=5) participants and remove from coinjoins.
        # 2. Recursively add back into coinjoin set all transactions with at least 3 participants, if referenced from coinjoins
        # 3. Stop when no new transaction is added back

        # Remove all transactions with only two equal outputs
        REMOVE_BELOW_THREE_PARTICIPANTS_ALWAYS = True
        if REMOVE_BELOW_THREE_PARTICIPANTS_ALWAYS:
            data, false_cjtxs_min3 = filter_false_coinjoins(data, mix_protocol, CJ_TX_CHECK.MIN_SAME_VALUES_THRESHOLD,
                                                      {CJ_TX_CHECK.MIN_SAME_VALUES_THRESHOLD: 3})
            SM.print(f'  Number of filtered false positives with only 2 participants: {len(false_cjtxs_min3)}')
            false_cjtxs.update(false_cjtxs_min3)

        REMOVE_MULTIPLE_DIFFERENT_EQUAL_OUTPUTS = True
        if REMOVE_MULTIPLE_DIFFERENT_EQUAL_OUTPUTS:
            data, false_cjtxs_multipleEqualOuts = filter_false_coinjoins(data, mix_protocol, CJ_TX_CHECK.MULTIPLE_ALLOWED_DIFFERENT_EQUAL_OUTPUTS,
                                                       {CJ_TX_CHECK.MULTIPLE_ALLOWED_DIFFERENT_EQUAL_OUTPUTS: 1})
            SM.print(
                f'  Number of filtered false positives (CJ_TX_CHECK.MULTIPLE_ALLOWED_DIFFERENT_EQUAL_OUTPUTS): {len(false_cjtxs_multipleEqualOuts)}')
            false_cjtxs.update(false_cjtxs_multipleEqualOuts)

        # Remove all bellow 5 participants (but possibly return some back based on references later)
        data, false_cjtxs_min5 = filter_false_coinjoins(data, mix_protocol, CJ_TX_CHECK.MIN_SAME_VALUES_THRESHOLD,
                                                  {CJ_TX_CHECK.MIN_SAME_VALUES_THRESHOLD: 5})
        # Return some back based on references from preserved coinjoins
        RETURN_BELOW_THRESHOLD_PARTICIPANTS_BUT_REFERENCED = True
        if RETURN_BELOW_THRESHOLD_PARTICIPANTS_BUT_REFERENCED:
            referenced = detect_referenced_trasactions(data['coinjoins'], false_cjtxs_min5)
            initial_candidate_false_positives = len(false_cjtxs_min5)
            SM.print(f'  Number of initially detected (potential) false positives with >2 & <5 participants: {len(false_cjtxs_min5)}')
            SM.print(f'    referenced by real coinjoin txs: {len(referenced)}')
            while len(referenced) > 0:
                # Return back to coinjoins, remove from false list
                for txid in list(referenced.keys()):
                    data['coinjoins'][txid] = referenced[txid]  # Is referenced from real coinjoins, put it back
                    data['coinjoins'][txid]['is_cjtx'] = True
                    false_cjtxs_min5.pop(txid) # Remove from false positives
                if len(false_cjtxs_min5) == 0:
                    break
                # Next iteration
                referenced = detect_referenced_trasactions(data['coinjoins'], false_cjtxs_min5)
                SM.print(f'    referenced by real coinjoin txs: {len(referenced)}')

        # Update complete list of filtered false positives using false positives left
        false_cjtxs.update(false_cjtxs_min5)
        SM.print(f'  Filtered false positives based on >2 & <5 & no_ref heuristic: {len(false_cjtxs_min5)}')
        SM.print(f'  Candidate false positives returned back to coinjoins based on reference heuristics: {initial_candidate_false_positives - len(false_cjtxs_min5)}')

    SM.print(f"  Total detected coinjoins: {len(data['coinjoins'])}")

    # Move false positives coinjoins into potential postmix
    data['postmix'].update(false_cjtxs)
    # Filter postmix spendings only to ones really spending from coinjoins (as we removed false positives)
    prev_postmix_len = len(data['postmix'])
    filter_postmix_transactions(data)
    SM.print(f"  Reducing postmix transactions from {prev_postmix_len} to {len(data['postmix'])} total postmix txs based on real coinjoins")
    # Update 'spend_by_tx' record
    data = update_all_spend_by_reference(data)

    # Set spending transactions also between mix and postmix
    data = compute_mix_postmix_link(data)

    data_extended = {}
    data_extended['wallets_info'], data_extended['wallets_coins'] = als.extract_wallets_info(data)
    data_extended['rounds'] = extract_rounds_info(data)

    return data, data_extended, false_cjtxs


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
                spending_tx = postmix_txs[txid]['inputs'][input]['spending_tx']
                logging.warning(f'Potentially suspicious link between two premixes (badbank/peelchain?) from {spending_tx} to {get_input_name_string(txid, input)}')
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
                spend_by_tx = premix_txs[txid]['outputs'][output]['spend_by_tx']
                logging.warning(f'Potentially suspicious link between two premixes (badbank/peelchain?) from {spend_by_tx} to {get_output_name_string(txid, output)}')
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
    mix_txs = tx_dict["coinjoins"]
    print(f'Analyzing analyze_postmix_spends for {len(postmix_txs)} postmixes and {len(mix_txs)} coinjoins')

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


def clear_clusters(tx_dict: dict) -> dict:
    for txid in tx_dict["coinjoins"].keys():
        for index in tx_dict["coinjoins"][txid]['outputs']:
            tx_dict["coinjoins"][txid]['outputs'][index].pop('cluster_id', None)
    return tx_dict


def assign_merge_cluster(tx_dict: dict) -> dict:
    """
    Simple chain analysis for outputs based on common input ownership
    If cjtx output(s) are used in non-coinjoin transaction, assign them same cluster id
    :param tx_dict: input dict with transactions
    :return: updated dict with transactions
    """
    mix_txs = tx_dict["coinjoins"]
    print(f'Analyzing assign_merge_cluster for {len(mix_txs)} coinjoins')

    offset = CLUSTER_INDEX.get_current_index()   # starting offset of cluster index used to compute number of assigned indexes
    total_outputs_merged = 0
    spent_txs = {}  # Construct all postmix spending trasaction with inputs used by it
    for txid in mix_txs.keys():
        for index in mix_txs[txid]['outputs'].keys():
            if mix_txs[txid]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_LEAVE.name:
                spent_txid, vin = als.extract_txid_from_inout_string(mix_txs[txid]['outputs'][index]['spend_by_tx'])
                spent_txs.setdefault(spent_txid, []).append(als.get_input_name_string(txid, index))

    for item in spent_txs.keys():
        cluster_name = f'c_{CLUSTER_INDEX.get_new_index()}'
        for output in spent_txs[item]:
            txid, index = als.extract_txid_from_inout_string(output)
            set_key_value_assert(mix_txs[txid]['outputs'][index], 'cluster_id', cluster_name, CLUSTER_ID_CHECK_HARD_ASSERT)
            total_outputs_merged += 1

    # Compute total number of inputs used in postmix spending
    total_outputs = sum([len(mix_txs[txid]['outputs']) for txid in mix_txs.keys()])
    SM.print(f'  {als.get_ratio_string(total_outputs_merged, total_outputs)} '
             f'N:k postmix merges detected (merged outputs / all outputs)')

    return tx_dict


def is_coinjoin_tx(test_tx: dict, mix_protocol: MIX_PROTOCOL, checks: CJ_TX_CHECK=CJ_TX_CHECK.BASIC, checks_config: dict=None):
    # BUGBUG: For SW, WW1 and WW2, it checks expected structure (whitelist). For JoinMarket, it only checks incorrect cases (blacklist)
    # This works well as Dumplings scanning first process SW, WW1 and WW2 and only then checks remaining coinjoin-like txs
    # But if this function is used separately, checking for is_coinjoin_tx(JoinMarket) will return True for a lot of non-coinjoin transactions

    if mix_protocol == MIX_PROTOCOL.WHIRLPOOL:  # whitelist
        checks_config = {**CJ_TX_CHECK_WHIRLPOOL_DEFAULTS, **(checks_config or {})}
        # The transaction is whirlpool coinjoin transaction if number of inputs is bigger than 4
        if len(test_tx['inputs']) >= 5:
            # ... number of inputs and outputs is the same
            if len(test_tx['inputs']) == len(test_tx['outputs']):
                # ... all outputs are the same value
                if all(test_tx['outputs'][vout]['value'] == test_tx['outputs']['0']['value']
                       for vout in test_tx['outputs'].keys()):
                    # ... and output sizes are one of the pool sizes [100k, 1M, 5M, 50M, 25M, 2.5M]
                    if all(test_tx['outputs'][vout]['value'] in WHIRLPOOL_FUNDING_TXS.keys()
                           for vout in test_tx['outputs'].keys()):
                        return True, FILTER_REASON.VALID
        return False, FILTER_REASON.INVALID_STRUCTURE

    if mix_protocol == MIX_PROTOCOL.WASABI1:  # whitelist
        checks_config = {**CJ_TX_CHECK_WASABI1_DEFAULTS, **(checks_config or {})}
        # The transaction is wasabi1 coinjoin transaction if number of inputs is at least MIN_WASABI1_INPUTS
        WASABI1_MIN_INPUTS = 5
        WASABI1_MAX_INPUTS = 5
        WASABI1_MIN_MIXED_OUTPUT = 3
        if len(test_tx['inputs']) >= WASABI1_MIN_INPUTS and len(test_tx['outputs']) >= WASABI1_MAX_INPUTS:
            output_values = Counter([test_tx['outputs'][index]['value'] for index in test_tx['outputs'].keys()])
            most_common_output_value, count = output_values.most_common(1)[0]
            most_common_output_value = most_common_output_value / SATS_IN_BTC
            # ... and the most common outputs size is around 0.1btc and
            # is at least WASABI1_MIN_MIXED_OUTPUTS
            if count >= WASABI1_MIN_MIXED_OUTPUT and 0.08 < most_common_output_value < 0.12:
                return True, FILTER_REASON.VALID
        return False, FILTER_REASON.INVALID_STRUCTURE

    if mix_protocol == MIX_PROTOCOL.WASABI2:  # whitelist
        checks_config = {**CJ_TX_CHECK_WASABI2_DEFAULTS, **(checks_config or {})}
        # Not implemented, return True always
        return True, FILTER_REASON.VALID
        #assert False, 'is_coinjoin_tx() not supported for WASABI2'

    if mix_protocol == MIX_PROTOCOL.JOINMARKET:  # blacklist, original transaction is expected to be pre-selected to be coinjoin-like
        checks_config = {**CJ_TX_CHECK_JOINMARKET_DEFAULTS, **(checks_config or {})}

        # Positive example: 7732a03c8cf133e0475ae37e4f2f49ba77beb631378216889e33e9847aa0049b

        # TODO: NO Multisig inputs : f73564ae9e7913303e9fc2a46c4e0f0d942378ce50fb65e40ebe0227adecc47d
        # TODO: only wrapped‑segwit P2SH or native segwit addresses are used (NOT true, earlier used 1x as well)

        # OP_RETURN (Runestone / Atom / Omni...)
        if checks & CJ_TX_CHECK.OP_RETURN:
            # TxNullData + data in OP_RETURN script
            for output in test_tx['outputs']:
                # Simplified rule = any OP_RETURN in output is not coinjoin
                if test_tx['outputs'][output]['script_type'] == 'TxNullData':
                    return False, FILTER_REASON.OP_RETURN
                # Complicated rule - search for specific op_return bytes
                # if test_tx['outputs'][output]['script_type'] == 'TxNullData' and (
                #         test_tx['outputs'][output]['script'].find('00c0a233') != -1 or
                #         test_tx['outputs'][output]['script'].find('0090e533') != -1 or
                #         test_tx['outputs'][output]['script'].find('00c4cf33') != -1 or
                #         test_tx['outputs'][output]['script'].find('0088a633') != -1 or
                #         test_tx['outputs'][output]['script'].find('00f7f334') != -1 or
                #         test_tx['outputs'][output]['script'].find('ff7f8192') != -1 or
                #         test_tx['outputs'][output]['script'].find('00c2f634') != -1 or
                #         test_tx['outputs'][output]['script'].find('00d6a233') != -1 or
                #         test_tx['outputs'][output]['script'].find('00b0d836') != -1 or
                #         test_tx['outputs'][output]['script'].find('00b5b633') != -1 or
                #         test_tx['outputs'][output]['script'].find('00a9e734') != -1 or
                #         test_tx['outputs'][output]['script'].find('00ecca33') != -1 or
                #         test_tx['outputs'][output]['script'].find('00fea333') != -1 or
                #         test_tx['outputs'][output]['script'].find('14011400') != -1 or
                #         test_tx['outputs'][output]['script'].find('00a4ed34') != -1 or
                #         test_tx['outputs'][output]['script'].find('00d5e636') != -1 or
                #         test_tx['outputs'][output]['script'].find('61746f6d') != -1 or  # Atom OP_RETURN : 4b6c2130baae7d38e5bfaebe39822bfb817717752c1a393285ad5ba9079286f5
                #         test_tx['outputs'][output]['script'].find('6f6d6e69') != -1     # Omni OP_RETURN : 3dc876a23165c8e6b9f81b4aeb9e0f1caaab77466ed20231901b2fe72f1c71d8
                #    return False

        # Minimal number of equal outputs
        if checks & CJ_TX_CHECK.MIN_SAME_VALUES_THRESHOLD:
            values = [test_tx['outputs'][index]['value'] for index in test_tx['outputs'].keys()]
            if Counter(values).most_common(1)[0][1] < checks_config[CJ_TX_CHECK.MIN_SAME_VALUES_THRESHOLD]:
                return False, FILTER_REASON.MIN_SAME_VALUES_THRESHOLD

        # Minimal number of equal outputs
        if checks & CJ_TX_CHECK.MULTIPLE_ALLOWED_DIFFERENT_EQUAL_OUTPUTS:
            values = [test_tx['outputs'][index]['value'] for index in test_tx['outputs'].keys()]
            counts = Counter(values)
            num_values_with_duplicates = sum(1 for count in counts.values() if count > 1)
            if num_values_with_duplicates > checks_config[CJ_TX_CHECK.MULTIPLE_ALLOWED_DIFFERENT_EQUAL_OUTPUTS]:
                return False, FILTER_REASON.MULTIPLE_ALLOWED_DIFFERENT_EQUAL_OUTPUTS

        # Significant address reuse a2bd490cc63c1b70bf7b328c20f69115a1b20cdd06cd035d6b200e17ee8d934e
        if checks & CJ_TX_CHECK.ADDRESS_REUSE_THRESHOLD:
            scripts = [test_tx['inputs'][index]['script'] for index in test_tx['inputs'].keys()]
            scripts.extend([test_tx['outputs'][index]['script'] for index in test_tx['outputs'].keys()])
            if Counter(scripts).most_common(1)[0][1] / len(scripts) > checks_config[CJ_TX_CHECK.ADDRESS_REUSE_THRESHOLD]:
                return False, FILTER_REASON.ADDRESS_REUSE_THRESHOLD

        # Too many inputs or too many outputs (coordination with more than tens of participants will likely fail)
        if checks & CJ_TX_CHECK.NUM_INOUT_THRESHOLD:
            num_inputs = len(test_tx['inputs'])
            num_outputs = len(test_tx['outputs'])
            if num_inputs > checks_config[CJ_TX_CHECK.NUM_INOUT_THRESHOLD] or num_outputs > checks_config[CJ_TX_CHECK.NUM_INOUT_THRESHOLD]:
                return False, FILTER_REASON.NUM_INOUT_THRESHOLD

        # Heavy disbalance between number of inputs and outputs: 8f7933a5d127b3bd8723ae9ea9eb7da96df1e33a7a9c1bcf9e68a2e6263d0640
        if checks & CJ_TX_CHECK.INOUTS_RATIO_THRESHOLD:
            # Compute ratio, prevent division by zero
            ratio = 100000 if num_inputs == 0 or num_outputs == 0 else max(num_inputs, num_outputs) / min(num_inputs, num_outputs)
            if ratio > checks_config[CJ_TX_CHECK.INOUTS_RATIO_THRESHOLD]:
                return False, FILTER_REASON.INOUTS_RATIO_THRESHOLD

        # If none of the check above fails, the transaction is assumed to be coinjoin
        return True, FILTER_REASON.VALID

    return False, FILTER_REASON.UNSPECIFIED


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
    mix_txs = tx_dict["coinjoins"]

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
    for txid in data["coinjoins"].keys():
        same_block_coinjoins[data["coinjoins"][txid]['block_hash']].append(txid)
    filtered_dict = {key: value for key, value in same_block_coinjoins.items() if len(value) > 1}
    SM.print(f'  {als.get_ratio_string(len(filtered_dict), len(data["coinjoins"]))} coinjoins in same block')






def compute_real_addresses(data: dict):
    # Extract all lock scripts, parallelize address computation, then collate back to main dictionary
    scripts = {data[cjtx]['inputs'][index]['script']: "" for cjtx in data.keys() for index in data[cjtx]['inputs'].keys()}
    scripts.update({data[cjtx]['outputs'][index]['script']: "" for cjtx in data.keys() for index in
               data[cjtx]['outputs'].keys()})
    scripts_only = list(scripts.keys())

    # Parallelize conversion
    def compute_address(script):
        return script, als.get_address(script)[0]

    logging.debug(f'Obtaining addresses from scripts, using {multiprocessing.cpu_count()} threads')
    results = {}
    with tqdm(total=len(scripts_only)) as progress:
        for result in ThreadPool(multiprocessing.cpu_count()).imap(compute_address, scripts_only):
            progress.update(1)
            results[result[0]] = result[1]

    logging.debug('Setting computed real addresses to coinjoin dict')
    for cjtx in data.keys():
        for index in data[cjtx]['inputs'].keys():
            data[cjtx]['inputs'][index]['address_real'] = results[data[cjtx]['inputs'][index]['script']]
    for cjtx in data.keys():
        for index in data[cjtx]['outputs'].keys():
            data[cjtx]['outputs'][index]['address_real'] = results[data[cjtx]['outputs'][index]['script']]
    logging.debug('  DONE: Finished assigning computed real addresses to coinjoin dict')


def process_coinjoins(target_path, mix_protocol: MIX_PROTOCOL, mix_filename, postmix_filename, premix_filename, start_date: str, stop_date: str):
    data, data_extended, false_cjtxs = load_coinjoins(target_path, mix_protocol, mix_filename, postmix_filename, premix_filename, start_date, stop_date)
    if len(data["coinjoins"]) == 0:
        return data

    # Store transactions filtered based on false positives rules
    false_cjtxs_file = os.path.join(target_path, f'{mix_protocol.name.lower()}_false_filtered_cjtxs.json')
    als.save_json_to_file_pretty(false_cjtxs_file, false_cjtxs)

    SM.print('*******************************************')
    SM.print(f'{mix_filename} coinjoins: {len(data["coinjoins"])}')
    min_date = min([data["coinjoins"][txid]['broadcast_time'] for txid in data["coinjoins"].keys()])
    max_date = max([data["coinjoins"][txid]['broadcast_time'] for txid in data["coinjoins"].keys()])
    SM.print(f'Dates from {min_date} to {max_date}')

    SM.print('### Simple chain analysis')
    cj_relative_order = als.analyze_input_out_liquidity(target_path, data["coinjoins"], data['postmix'], data.get('premix', {}), mix_protocol)

    analyze_postmix_spends(data)
    analyze_premix_spends(data)
    analyze_coinjoin_blocks(data)
    # Analysis temporarily disabled as mathplotlib will fail
    #analyze_coordinator_fees(mix_filename, data, mix_protocol)
    #analyze_mining_fees(mix_filename, data)

    return data, data_extended, cj_relative_order


def filter_liquidity_events(data):
    events = {}
    for txid in data["coinjoins"]:
        events[txid] = copy.deepcopy(data["coinjoins"][txid])
        events[txid].pop('block_hash', None)
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
                    if item not in ['value', 'wallet_name', 'mix_event_type', 'address_real']:
                        events[txid]['inputs'][input].pop(item)
        # Process outputs
        events[txid]['num_outputs'] = len(events[txid]['outputs'])
        for output in list(events[txid]['outputs'].keys()):
            if ('mix_event_type' not in events[txid]['outputs'][output]
                    or events[txid]['outputs'][output]['mix_event_type'] not in [MIX_EVENT_TYPE.MIX_ENTER.name, MIX_EVENT_TYPE.MIX_LEAVE.name]):
                # Remove whole given output (not mix enter/leave event)
                events[txid]['outputs'].pop(output)
            else:
                # Remove all unnecessary data
                for item in events[txid]['outputs'][output].copy():
                    if item not in ['value', 'wallet_name', 'mix_event_type', 'address_real']:
                        events[txid]['outputs'][output].pop(item)
    return events


def process_and_save_coinjoins(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: os.path, mix_filename: str, postmix_filename: str,
                               premix_filename: str, start_date: str, stop_date: str, target_save_path: os.path=None, save_base_files: bool=False):
    if not target_save_path:
        target_save_path = target_path
    # Process and save full conjoin information
    data, data_extended, cj_relative_order = process_coinjoins(target_path, mix_protocol, mix_filename, postmix_filename, premix_filename, start_date, stop_date)
    als.save_json_to_file_pretty(os.path.join(target_save_path, f'cj_relative_order.json'), cj_relative_order)

    # If found, enrich data with coinjoin-specific metadata
    metadata_file = os.path.join(target_path, f'{mix_id}_wallet_predictions.json')
    if os.path.exists(metadata_file):
        wallet_nums_predictions = als.load_json_from_file(metadata_file)
        for cjtx in data['coinjoins'].keys():
            data['coinjoins'][cjtx]['num_wallets_predicted'] = wallet_nums_predictions.get(cjtx, -100)

    # FIXME: Compute and update real addresses from lock scripts
    # Problems: 1) Time consuming (not a big problem), 2) address resuse will break analysis later (need fix)
    #compute_real_addresses(data['coinjoins'])

    if save_base_files:
        als.save_json_to_file(os.path.join(target_save_path, f'coinjoin_tx_info.json'), data)
        als.save_json_to_file(os.path.join(target_save_path, f'coinjoin_tx_info_extended.json'), data_extended)

    # Filter only liquidity-relevant events to maintain smaller file
    events = filter_liquidity_events(data)
    als.save_json_to_file_pretty(os.path.join(target_save_path, f'{mix_id}_events.json'), events)

    return data


def process_and_save_intervals_onload(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: os.path, start_date: str, stop_date: str, mix_filename: str,
                                      postmix_filename: str, premix_filename: str=None, save_base_files: bool=False):

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
        interval_path = os.path.join(target_path, f'{last_stop_date_str.replace(":", "-")}--{current_stop_date_str.replace(":", "-")}_unknown-static-100-1utxo')
        if not os.path.exists(interval_path):
            os.makedirs(interval_path.replace('\\', '/'))
            os.makedirs(os.path.join(interval_path, 'data').replace('\\', '/'))
        process_and_save_coinjoins(mix_id, mix_protocol, target_path, mix_filename, postmix_filename, premix_filename,
                                          last_stop_date_str, current_stop_date_str, interval_path, save_base_files)
        # Move to the next month
        last_stop_date_str = current_stop_date_str

        current_stop_date = current_stop_date + timedelta(days=32)
        current_stop_date = datetime(current_stop_date.year, current_stop_date.month, 1)
        current_stop_date_str = current_stop_date.strftime("%Y-%m-%d %H:%M:%S")


def process_interval(mix_id: str, data: dict, mix_filename: str, premix_filename: str, target_save_path: str, last_stop_date_str: str, current_stop_date_str: str):
    logging.info(f'Processing interval {last_stop_date_str} - {current_stop_date_str}')

    # Create folder structure compatible with ww2 coinjoin simulation for further processing
    interval_path = os.path.join(target_save_path, f'{last_stop_date_str.replace(":", "-")}--{current_stop_date_str.replace(":", "-")}_unknown-static-100-1utxo')
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
        extract_inputs_distribution(mix_id, interval_path, premix_filename, interval_data['premix'], True)
    else:
        # WW1, WW2
        extract_inputs_distribution(mix_id, interval_path, mix_filename, interval_data["coinjoins"], True)


def process_and_save_intervals_filter(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: os.path, start_date: str, stop_date: str, mix_filename: str,
                                      postmix_filename: str, premix_filename: str=None, save_base_files_json=True, load_base_files=False, preloaded_data: dict=None):
    # Create directory structure with files split per month (around 1000 subsequent coinjoins)
    # Load all coinjoins first, then filter based on intervals
    target_save_path = os.path.join(target_path, mix_id)
    if not os.path.exists(target_save_path):
        os.makedirs(target_save_path.replace('\\', '/'))

    if preloaded_data is None:
        if load_base_files:
            # Load base files from already stored json
            logging.info(f'Loading {target_save_path}/coinjoin_tx_info.json ...')

            data = als.load_coinjoins_from_file(target_save_path, None, False)

            # If found, enrich data with coinjoin-specific metadata
            metadata_file = os.path.join(target_path, f'{mix_id}_wallet_predictions.json')
            if os.path.exists(metadata_file):
                wallet_nums_predictions = als.load_json_from_file(metadata_file)
                for cjtx in data['coinjoins'].keys():
                    data['coinjoins'][cjtx]['num_wallets_predicted'] = wallet_nums_predictions.get(cjtx, -100)

            logging.info(f'{target_save_path}/coinjoin_tx_info.json loaded with {len(data["coinjoins"])} conjoins')
        else:
            #
            # Convert all Dumplings files into json (time intensive)
            data = process_and_save_coinjoins(mix_id, mix_protocol, target_path, mix_filename, postmix_filename, premix_filename, None, None, target_save_path, save_base_files_json)
    else:
        data = preloaded_data

    if mix_protocol == MIX_PROTOCOL.WHIRLPOOL:
        # Whirlpool
        extract_inputs_distribution(mix_id, target_save_path, premix_filename, data['premix'], True)
    else:
        # WW1, WW2
        extract_inputs_distribution(mix_id, target_save_path, mix_filename, data["coinjoins"], True)

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
    txs = load_coinjoin_stats_from_dumplings(os.path.join(target_path, premix_filename))
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
    txs = load_coinjoin_stats_from_dumplings(os.path.join(target_path, tx_filename))
    return find_address_reuse(mix_id, txs, target_path, [], save_outputs)


def find_address_reuse(mix_id: str, txs: dict, target_path: Path = None, ignore_denominations: list = [], save_outputs = False):
    """
    Detects all address reuse in given list of transactions
    :param mix_id:
    :param txs: dictionary of transactions
    :param target_path: path used for saving results
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
        als.save_json_to_file(os.path.join(target_path, f'{mix_id}_conjoins_interval_{start_date[:start_date.find(" ") - 1]}-{stop_date[:stop_date.find(" ") - 1]}.json'), interval_data)

    return interval_data


def print_interval_data_stats(pool_stats: dict, client_stats: CoinMixInfo, results: dict):
    num_inputs = [len(pool_stats[txid]['inputs']) for txid in pool_stats.keys()]
    num_freeremix_inputs = [1 for txid in pool_stats.keys() for index in pool_stats[txid]['inputs']
                            if math.isclose(pool_stats[txid]['inputs'][index]['value'], client_stats.pool_size, rel_tol=1e-9, abs_tol=0.0)]
    assert max(num_inputs) < 9, 'Whirpool shall not have more than 9 inputs in mix tx'
    #print(num_inputs)
    num_inputs_pool = sum(num_inputs)
    num_freeremix_inputs_pool = sum(num_freeremix_inputs)
    logging.info(
        f'  {round(client_stats.pool_size / SATS_IN_BTC, 3)} pool total inputs={num_inputs_pool}, pool free inputs={num_freeremix_inputs_pool}, client mixes={client_stats.num_mixes}, '
        f'client coins={client_stats.num_coins}')
    if client_stats.num_mixes > 0:
        #ratio_all_inputs = round((num_inputs_pool / client_stats.num_mixes) * client_stats.num_coins, 1)
        ratio_all_inputs = round(num_inputs_pool / client_stats.num_mixes, 1)
        ratio_freeremix_inputs = round(num_freeremix_inputs_pool / client_stats.num_mixes, 1)
        logging.info(f'    {round(client_stats.pool_size / SATS_IN_BTC, 3)} pool participation rate(based on all cjtxs)= 1:{ratio_all_inputs}')
        logging.info(f'    {round(client_stats.pool_size / SATS_IN_BTC, 3)} pool participation rate(based on free remix inputs)= 1:{ratio_freeremix_inputs}')
        #results[str(client_stats.pool_size)].append(ratio_all_inputs)
        results[str(client_stats.pool_size)].append(ratio_freeremix_inputs)
        present_probability = client_stats.num_mixes / len(pool_stats.keys()) * 100
        est_queue_len = round(100 / (present_probability / client_stats.num_coins))
        present_in_mixes_single_coin = round(present_probability / client_stats.num_coins, 2)
        logging.info(f'    present in % of mixes= {round(present_probability, 2)}%')
        logging.info(f'    estimated queue length = {round(100 / present_probability)} coins')
        logging.info(f'    present in % of mixes (per single coin)= {present_in_mixes_single_coin}%')
        logging.info(f'    estimated queue length (per single coin)= {est_queue_len} coins')

        #print('###################################')
        SM.print(f'{round(client_stats.pool_size / SATS_IN_BTC, 3)} (DATE) & {len(pool_stats)} / {num_freeremix_inputs_pool} & {client_stats.num_coins} / {client_stats.num_mixes} / {present_in_mixes_single_coin}\\% & {est_queue_len}')
        #print('###################################')

def analyze_interval_data(interval_data, stats: CoinJoinStats, results: dict):
    if stats.cj_type == MIX_PROTOCOL.WHIRLPOOL:
        # Count number of coinjoins in different pools
        pool_100k = {txid: interval_data["coinjoins"][txid] for txid in interval_data["coinjoins"].keys() if interval_data["coinjoins"][txid]['outputs']['0']['value'] == 100000}
        pool_1M = {txid: interval_data["coinjoins"][txid] for txid in interval_data["coinjoins"].keys() if interval_data["coinjoins"][txid]['outputs']['0']['value'] == 1000000}
        pool_5M = {txid: interval_data["coinjoins"][txid] for txid in interval_data["coinjoins"].keys() if interval_data["coinjoins"][txid]['outputs']['0']['value'] == 5000000}

        logging.info(f'  Total cjs={len(interval_data["coinjoins"].keys())}, 100k pool={len(pool_100k)}, 1M pool={len(pool_1M)}, 5M pool={len(pool_5M)}')
        logging.info(f'  {interval_data["start_date"]} - {interval_data["stop_date"]}')
        print_interval_data_stats(pool_100k, stats.pool_100k, results)
        print_interval_data_stats(pool_1M, stats.pool_1M, results)
        print_interval_data_stats(pool_5M, stats.pool_5M, results)

    if stats.cj_type == MIX_PROTOCOL.WASABI2:
        logging.info(f'  Total cjs in interval= {len(interval_data["coinjoins"].keys())}')
        logging.info(f'  {interval_data["start_date"]} - {interval_data["stop_date"]}')
        logging.info(f'  Used cjs= {stats.no_pool.num_mixes}, skipped= {len(interval_data["coinjoins"]) - stats.no_pool.num_mixes}')
        logging.info(f'  #cjs per one input coin= {round(stats.no_pool.num_mixes / stats.no_pool.num_coins, 2)}')


def process_inputs_distribution_whirlpool(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: Path, tx_filename: str, save_outputs: bool= False):
    logging.info(f'Processing {mix_id}')
    txs = load_coinjoin_stats_from_dumplings(os.path.join(target_path, tx_filename))

    # Process TX0 transactions, try to find ones with many pool outputs and long time to mix them (possible chain analysis input)
    tx0_by_outputs_dict = {}
    for txid in txs.keys():
        valid, reason = is_coinjoin_tx(txs[txid], mix_protocol)
        if not valid:
            num_outputs = len(txs[txid]['outputs'])
            tx0_by_outputs_dict.setdefault(num_outputs, []).append(txid)

    tx0_results = {}
    for num_outputs in sorted(tx0_by_outputs_dict.keys()):
        print(f'#outputs {num_outputs}: {len(tx0_by_outputs_dict[num_outputs])}x')
        for item in tx0_by_outputs_dict[num_outputs]:
            if num_outputs not in tx0_results.keys():
                tx0_results[num_outputs] = {}
            in_values = [txs[item]['inputs'][index]['value'] for index in txs[item]['inputs'].keys()]
            out_values = [txs[item]['outputs'][index]['value'] for index in txs[item]['outputs'].keys()]
            pool_size_out_sats = np.median(out_values)
            pool_size = round(pool_size_out_sats / SATS_IN_BTC, 3)
            pool_size_sats = pool_size * SATS_IN_BTC
            out_values_pool = [value for value in out_values if math.isclose(value, pool_size_sats, rel_tol=1e-1, abs_tol=0.0)]
            out_mfees = [value - pool_size_sats for value in out_values_pool]
            tx0_results[num_outputs][item] = {'pool': pool_size, 'pool_total_inflow': round(sum(out_values_pool) / SATS_IN_BTC, 2),
                                              'num_pool_inputs': len(out_values_pool), 'tx0_input_size': round(sum(in_values) / SATS_IN_BTC, 2),
                                              'sum_mfee': round(sum(out_mfees) / SATS_IN_BTC, 4)}
            if num_outputs > 50:
                print(f'{item}, pool: {pool_size}: pool_total_inflow: {round(sum(out_values_pool) / SATS_IN_BTC, 2)} btc in {len(out_values_pool)} inputs '
                      f'(sum TXO inputs: {round(sum(in_values) / SATS_IN_BTC, 2)} btc), sum mfee in post-txo outputs = {round(sum(out_mfees) / SATS_IN_BTC, 4)}')
    if save_outputs:
        als.save_json_to_file_pretty(os.path.join(target_path, mix_id, f'{mix_id}_tx0_analysis.json'), tx0_results)

    inputs = [txs[txid]['inputs'][index]['value'] for txid in txs.keys() for index in txs[txid]['inputs'].keys() if not (valid := is_coinjoin_tx(txs[txid], mix_protocol))[0]]
    inputs_distrib = Counter(inputs)
    inputs_distrib = dict(sorted(inputs_distrib.items(), key=lambda item: (-item[1], item[0])))
    inputs_info = {'mix_id': mix_id, 'path': tx_filename, 'distrib': inputs_distrib}
    logging.info(f'  Distribution extracted, total {len(inputs_info["distrib"])} different input values found')
    if save_outputs:
        als.save_json_to_file_pretty(os.path.join(target_path, mix_id, f'{mix_id}_inputs_distribution.json'), inputs_info)

    cjvis.plot_analyze_liquidity(mix_id, inputs)


def process_estimated_wallets_distribution(mix_id: str, target_path: Path, inputs_wallet_factor: list, save_outputs: bool= True):
    logging.info(f'Processing process_estimated_wallets_distribution({mix_id})')
    # Load txs for all pools
    target_load_path = os.path.join(target_path, mix_id)

    data = als.load_coinjoins_from_file(target_load_path, None, True)

    # For each cjtx compute rough number of wallets present based on the inputs_wallet_factor
    num_wallets = [len(data["coinjoins"][txid]['inputs'].keys()) for txid in data["coinjoins"].keys()]

    for factor in inputs_wallet_factor:
        logging.info(f' Processing factor={factor}')
        wallets_distrib = Counter([round(item / factor) for item in num_wallets])
        wallets_distrib = dict(sorted(wallets_distrib.items(), key=lambda item: (-item[1], item[0])))
        wallets_info = {'mix_id': mix_id, 'path': target_load_path, 'wallets_distrib': wallets_distrib, 'wallets_distrib_factor': factor}
        logging.info(f'  Distribution of walets extracted, total {len(wallets_info["wallets_distrib"])} different input values found')
        if save_outputs:
            als.save_json_to_file_pretty(os.path.join(target_path, mix_id, f'{mix_id}_wallets_distribution_factor{factor}.json'), wallets_info)

        cjvis.plot_wallets_distribution(target_path, mix_id, factor, wallets_distrib)


def process_inputs_distribution(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: Path, tx_filename: str, save_outputs: bool= True):
    logging.info(f'Processing {mix_id} process_inputs_distribution()')
    # Load txs for all pools
    target_load_path = os.path.join(target_path, mix_id)
    data = als.load_coinjoins_from_file(target_load_path, None, True)

    inputs_info, inputs = extract_inputs_distribution(mix_id, target_load_path, tx_filename, data["coinjoins"], save_outputs, '')
    cjvis.plot_inputs_distribution(mix_id, inputs)


def extract_inputs_distribution(mix_id: str, target_path: Path, tx_filename: str, txs: dict, save_outputs = False, file_spec: str = ''):
    inputs = [txs[txid]['inputs'][index]['value'] for txid in txs.keys() for index in txs[txid]['inputs'].keys()
              if 'mix_event_type' in txs[txid]['inputs'][index].keys() and
              txs[txid]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_ENTER.name]
    inputs_distrib = Counter(inputs)
    inputs_distrib = dict(sorted(inputs_distrib.items(), key=lambda item: (-item[1], item[0])))
    inputs_info = {'mix_id': mix_id, 'path': tx_filename, 'distrib': inputs_distrib}
    logging.info(f'  Distribution extracted, total {len(inputs_info["distrib"])} different input values found')
    if save_outputs:
        als.save_json_to_file_pretty(os.path.join(target_path, f'{mix_id}_inputs_distribution{file_spec}.json'), inputs_info)

    return inputs_info, inputs



def process_outputs_distribution(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: Path, tx_filename: str, save_outputs: bool= True):
    logging.info(f'Processing {mix_id} process_outputs_distribution()')
    # Load txs for all pools
    target_load_path = os.path.join(target_path, mix_id)
    data = als.load_coinjoins_from_file(target_load_path, None, True)

    #outputs_info, outputs_noremix_stddenom, outputs_noremix_all, outputs_all =
    extract_outputs_distribution(mix_id, target_path, tx_filename, data["coinjoins"], save_outputs, '')
    #plot_distribution(outputs_all)


def extract_outputs_distribution(mix_id: str, target_path: Path, tx_filename: str, txs: dict, save_outputs = False, file_spec: str = ''):
    outputs_noremix_stddenom = [txs[txid]['outputs'][index]['value'] for txid in txs.keys() for index in txs[txid]['outputs'].keys()
              if 'mix_event_type' in txs[txid]['outputs'][index].keys() and
              txs[txid]['outputs'][index]['mix_event_type'] in [MIX_EVENT_TYPE.MIX_LEAVE.name, MIX_EVENT_TYPE.MIX_STAY.name] and
               txs[txid]['outputs'][index]['is_standard_denom'] == True]
    outputs_noremix_all = [txs[txid]['outputs'][index]['value'] for txid in txs.keys() for index in txs[txid]['outputs'].keys()
              if 'mix_event_type' in txs[txid]['outputs'][index].keys() and
              txs[txid]['outputs'][index]['mix_event_type'] in [MIX_EVENT_TYPE.MIX_LEAVE.name, MIX_EVENT_TYPE.MIX_STAY.name]]
    outputs_all = [txs[txid]['outputs'][index]['value'] for txid in txs.keys() for index in txs[txid]['outputs'].keys()]

    outputs_noremix_stddenom_distrib = dict(sorted(Counter(outputs_noremix_stddenom).items(), key=lambda item: (-item[1], item[0])))
    outputs_noremix_all_distrib = dict(sorted(Counter(outputs_noremix_all).items(), key=lambda item: (-item[1], item[0])))
    outputs_all_distrib = dict(sorted(Counter(outputs_all).items(), key=lambda item: (-item[1], item[0])))
    outputs_info = {'mix_id': mix_id, 'path': tx_filename,
                    'outputs_noremix_stddenom_distrib': outputs_noremix_stddenom_distrib,
                    'outputs_noremix_all_distrib': outputs_noremix_all_distrib,
                    'outputs_all_distrib': outputs_all_distrib}

    logging.info(f'  Distribution extracted')
    logging.info(f'    total outputs_noremix_stddenom_distrib={len(outputs_info["outputs_noremix_stddenom_distrib"])} different output values found')
    logging.info(f'    total outputs_noremix_all_distrib={len(outputs_info["outputs_noremix_all_distrib"])} different output values found')
    logging.info(f'    total outputs_all_distrib={len(outputs_info["outputs_all_distrib"])} different output values found')
    if save_outputs:
        als.save_json_to_file_pretty(os.path.join(target_path, mix_id, f'{mix_id}_outputs_distribution{file_spec}.json'), outputs_info)

    return outputs_info, outputs_noremix_stddenom, outputs_noremix_all, outputs_all


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


def whirlpool_analyze_fees(mix_id: str, cjtxs):
    whirlpool_analyze_coordinator_fees(mix_id, cjtxs)
    cjvis.visualize_mining_fees(mix_id, cjtxs)


def wasabi2_analyze_fees(mix_id: str, cjtxs):
    wasabi_analyze_coordinator_fees(mix_id, cjtxs)
    cjvis.visualize_mining_fees(mix_id, cjtxs)


def wasabi1_analyze_fees(mix_id: str, cjtxs):
    wasabi_analyze_coordinator_fees(mix_id, cjtxs)
    cjvis.visualize_mining_fees(mix_id, cjtxs)



def analyze_coordinator_fees(mix_id: str, data, mix_protocol):
    if mix_protocol == MIX_PROTOCOL.WASABI1 or mix_protocol == MIX_PROTOCOL.WASABI2:
        return wasabi_analyze_coordinator_fees(mix_id, data)
    elif mix_protocol == MIX_PROTOCOL.WHIRLPOOL:
        return whirlpool_analyze_fees(mix_id, data)
    else:
        assert False, f'Unexpected value of mix_protocol provided: {mix_protocol.name}'


def wasabi_analyze_coordinator_fees(mix_id: str, cjtxs: dict):
    only_cjtxs = cjtxs["coinjoins"]
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

    print(f'Total coordination fee: {sum(cjtxs_coordinator_fee) / SATS_IN_BTC} btc ({sum(cjtxs_coordinator_fee)} sats)')

    cjvis.plot_wasabi_coordinator_fees(mix_id, cjtxs_coordinator_fee)

    return cjtxs_coordinator_fee


def whirlpool_analyze_coordinator_fees(mix_id: str, data: dict):
    tx0s = data['premix']
    cjtxs = data["coinjoins"]
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
        for out_index in tx0s[tx0]['outputs'].keys():
            if 'spend_by_tx' in tx0s[tx0]['outputs'][out_index]:
                txid, vin = als.extract_txid_from_inout_string(tx0s[tx0]['outputs'][out_index]['spend_by_tx'])
                if txid in cjtxs.keys():
                    for pool_size in WHIRLPOOL_POOLS:
                        if abs(tx0s[tx0]['outputs'][out_index]['value'] - pool_size) < pool_size * 0.1:
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
    print(f'Total coordination fee: {sum(cjtxs_coordinator_fees) / SATS_IN_BTC} btc ({sum(cjtxs_coordinator_fees)} sats)')

    cjvis.plot_whirlpool_coordinator_fees(mix_id, cjtxs_coordinator_fees)

    return cjtxs_coordinator_fees


def whirlpool_analyse_remixes(mix_id: str, target_path: str):
    data = als.load_coinjoins_from_file(os.path.join(target_path, mix_id), None, True)
    als.analyze_input_out_liquidity(target_path, data["coinjoins"], data['postmix'], data['premix'], MIX_PROTOCOL.WHIRLPOOL)
    whirlpool_analyze_fees(mix_id, data)
    cjvis.inputs_value_burntime_heatmap(mix_id, data)
    cjvis.burntime_histogram(mix_id, data)


def wasabi2_analyse_remixes(mix_id: str, target_path: str):
    data = als.load_coinjoins_from_file(os.path.join(target_path, mix_id), None, False)
    cj_relative_order = als.analyze_input_out_liquidity(target_path, data["coinjoins"], data['postmix'], [], MIX_PROTOCOL.WASABI2)
    als.save_json_to_file_pretty(os.path.join(target_path, mix_id, f'cj_relative_order.json'), cj_relative_order)

    wasabi2_analyze_fees(mix_id, data)
    cjvis.inputs_value_burntime_heatmap(mix_id, data)
    cjvis.burntime_histogram(mix_id, data)


def wasabi_detect_false(target_path: Path, tx_file: str):
    PROCESS_SUBFOLDERS = False
    if PROCESS_SUBFOLDERS:
        # Process all subfolders
        files = os.listdir(target_path) if os.path.exists(target_path) else print(
            f'Path {target_path} does not exist')
    else:
        # Process only single root directory
        files = [""] if os.path.exists(target_path) else print(
            f'Path {target_path} does not exist')

    REUSE_THRESHOLD = 0.7

    STRANGE_2025_CJ_DENOMS = [86093442, 43046721, 28697814, 14348907, 1062882]  # Multiplication of 1062882
    STRANGE_2025_CJ_DENOMS.extend([134217728, 67108864, 33554432, 16777216, 8388608])  # Multiplication of 8388608
    #STRANGE_2025_CJ_DENOMS.extend([50000000, 20000000, 10000000])  # Multiplication of 10000000 - present in 2025 strange txs, but too common elsewhere as well
    STRANGE_2025_CJ_DENOMS_MIN_OCCURENCE = 2     # Manual analysis of 25 coinjoins shown that at least some of the specific values are present at least several times
    STRANGE_2025_CJ_TIMES_LEAST_FREQUENT = 1     # Number of least common denomination

    print(f'Going to process the following subfolders of {target_path}: {files}')
    # Load false positives
    false_cjtxs = als.load_false_cjtxs_from_file(os.path.join(target_path, 'false_cjtxs.json'))

    no_remix_all = {'inputs_noremix': {}, 'outputs_noremix': {}, 'both_noremix': {},
                    'inputs_address_reuse': {}, 'outputs_address_reuse': {},
                    'both_reuse': {}, 'specific_denoms_noremix_in': {}, 'specific_denoms_noremix_out':{},
                    'specific_denoms_noremix_both':{}, 'specific_denoms_noremix_inorout': {}}
    for dir_name in files:
        target_base_path = os.path.join(target_path, dir_name)
        tx_json_file = os.path.join(target_base_path, f'{tx_file}')
        if os.path.isdir(target_base_path) and os.path.exists(tx_json_file):
            # Perform bare loading and filtering using pre-loaded false_cjtxs list
            data = als.load_json_from_file(tx_json_file)
            filtered_false_coinjoins = {}
            for false_tx in false_cjtxs:
                if false_tx in data["coinjoins"].keys():
                    filtered_false_coinjoins[false_tx] = data["coinjoins"].pop(false_tx)
            # Store transactions filtered based on false positives file
            false_cjtxs_file = os.path.join(target_base_path, f'false_filtered_cjtxs_manual.json')
            als.save_json_to_file_pretty(false_cjtxs_file, filtered_false_coinjoins)


            # Detect transactions with no remixes on input/out or both
            no_remix = als.detect_no_inout_remix_txs(data["coinjoins"])
            for key in no_remix.keys():
                no_remix_all[key].update(no_remix[key])

            # Detect transactions with too many address reuse
            address_reuse = als.detect_address_reuse_txs(data["coinjoins"], REUSE_THRESHOLD)
            for key in address_reuse.keys():
                no_remix_all[key].update(address_reuse[key])

            # Detect transactions with specific WW2-like input/output denominations and structure
            strange_2025_cj = als.detect_specific_cj_denoms(data["coinjoins"], STRANGE_2025_CJ_DENOMS, STRANGE_2025_CJ_DENOMS_MIN_OCCURENCE, STRANGE_2025_CJ_TIMES_LEAST_FREQUENT)
            print(f'Strange CJs: {len(strange_2025_cj["specific_denoms"].keys())}')
            # Keep only these which are also no_remix
            strange_2025_cj_noremix_in =  {'specific_denoms_noremix_in': {cjtx: strange_2025_cj['specific_denoms'][cjtx] for cjtx in strange_2025_cj['specific_denoms'].keys() if cjtx in no_remix['inputs_noremix'].keys()}}
            strange_2025_cj_noremix_out = {'specific_denoms_noremix_out': {cjtx: strange_2025_cj['specific_denoms'][cjtx] for cjtx in strange_2025_cj['specific_denoms'].keys() if cjtx in no_remix['outputs_noremix'].keys()}}
            strange_2025_cj_noremix_both = {'specific_denoms_noremix_both': {cjtx: strange_2025_cj['specific_denoms'][cjtx] for cjtx in strange_2025_cj['specific_denoms'].keys() if cjtx in no_remix['both_noremix'].keys()}}
            strange_2025_cj_noremix_inorout = {}
            strange_2025_cj_noremix_inorout['specific_denoms_noremix_inorout'] = copy.deepcopy(strange_2025_cj_noremix_in['specific_denoms_noremix_in'])
            strange_2025_cj_noremix_inorout['specific_denoms_noremix_inorout'].update(strange_2025_cj_noremix_out['specific_denoms_noremix_out'])
            print(f'Strange CJs noremix_in: {len(strange_2025_cj_noremix_in["specific_denoms_noremix_in"].keys())}')
            print(f'Strange CJs noremix_out: {len(strange_2025_cj_noremix_out["specific_denoms_noremix_out"].keys())}')
            print(f'Strange CJs noremix_both: {len(strange_2025_cj_noremix_both["specific_denoms_noremix_both"].keys())}')
            print(f'Strange CJs noremix_inorout: {len(strange_2025_cj_noremix_inorout["specific_denoms_noremix_inorout"].keys())}')
            for key in strange_2025_cj_noremix_in.keys():
                no_remix_all[key].update(strange_2025_cj_noremix_in[key])
            for key in strange_2025_cj_noremix_out.keys():
                no_remix_all[key].update(strange_2025_cj_noremix_out[key])
            for key in strange_2025_cj_noremix_both.keys():
                no_remix_all[key].update(strange_2025_cj_noremix_both[key])
            for key in strange_2025_cj_noremix_inorout.keys():
                no_remix_all[key].update(strange_2025_cj_noremix_inorout[key])

    # Add used threshold value into key value in dictionary
    reuse_threshold_string = f"{REUSE_THRESHOLD:.2f}".replace('.', '_')
    no_remix_all[f'inputs_address_reuse_{reuse_threshold_string}'] = no_remix_all.pop('inputs_address_reuse')
    no_remix_all[f'outputs_address_reuse_{reuse_threshold_string}'] = no_remix_all.pop('outputs_address_reuse')
    no_remix_all[f'both_reuse_{reuse_threshold_string}'] = no_remix_all.pop('both_reuse')

    # save detected no transactions with no remixes (potentially false positives)
    als.save_json_to_file_pretty(os.path.join(target_path, 'no_remix_txs.json'), no_remix_all)


def wasabi1_analyse_remixes(mix_id: str, target_path: str):
    data = als.load_coinjoins_from_file(os.path.join(target_path, mix_id), None, False)
    als.analyze_input_out_liquidity(target_path, data["coinjoins"], data['postmix'], [], MIX_PROTOCOL.WASABI1)

    wasabi1_analyze_fees(mix_id, data)
    cjvis.inputs_value_burntime_heatmap(mix_id, data)
    cjvis.burntime_histogram(mix_id, data)


def fix_ww2_for_fdnp_ww1(mix_id: str, target_path: str):
    """
    Detects and corrects all information of WW2 extracted from coinjoin_tx_info.json based on WW1 inflows.
    Process also subfolders with monthly intervals
    :param mix_id:
    :param target_path:
    :return:
    """
    logging.info(f'Going to fix_ww2_for_fdnp_ww1({mix_id})')

    #'wasabi2', target_path, os.path.join(target_path, 'wasabi1_burn', 'coinjoin_tx_info.json.full'))
    # Load Wasabi1 files, then update MIX_ENTER for Wasabi2 where friends-do-not-pay rule does not apply
    # We will need only WW1 txids, drop all other values to decrease peak memory requirements
    ww1_coinjoins = als.load_coinjoin_txids_from_file(os.path.join(target_path, 'WasabiCoinJoins.txt'))
    ww1_postmix_spend = als.load_coinjoin_txids_from_file(os.path.join(target_path, 'WasabiPostMixTxs.txt'))
    # ww1_coinjoins = load_coinjoin_stats_from_file(os.path.join(target_path, 'WasabiCoinJoins.txt'))
    # ww1_postmix_spend = load_coinjoin_stats_from_file(os.path.join(target_path, 'WasabiPostMixTxs.txt'))

    target_path = os.path.join(target_path, mix_id)  # Go into target ww2 folder

    paths_to_process = []
    # Add subpaths for months if present
    files = os.listdir(target_path)
    for file_name in files:
        target_base_path = os.path.join(target_path, file_name)
        tx_json_file = os.path.join(target_base_path, f'coinjoin_tx_info.json')
        if os.path.isdir(target_base_path) and os.path.exists(tx_json_file):
            paths_to_process.append(target_base_path)

    # Always process 'coinjoin_tx_info.json' with all transactions.
    paths_to_process.append(target_path)

    # Now fix all prepared paths
    for path in sorted(paths_to_process):
        logging.info(f'Processing {path}...')

        ww2_data = als.load_coinjoins_from_file(path, None, False)

        # For all values with mix_event_type equal to MIX_ENTER check if they are not from WW1
        # with friends-do-not-pay rule
        total_ww1_inputs = 0
        for cjtx in ww2_data["coinjoins"]:
            for input in ww2_data["coinjoins"][cjtx]['inputs']:
                if ww2_data["coinjoins"][cjtx]['inputs'][input]['mix_event_type'] == MIX_EVENT_TYPE.MIX_ENTER.name:
                    if 'spending_tx' in ww2_data["coinjoins"][cjtx]['inputs'][input].keys():
                        spending_tx, index = als.extract_txid_from_inout_string(ww2_data["coinjoins"][cjtx]['inputs'][input]['spending_tx'])
                        if spending_tx in ww1_coinjoins or spending_tx in ww1_postmix_spend:
                            # Friends do not pay rule tx - change to MIX_REMIX_FRIENDS_WW1
                            ww2_data["coinjoins"][cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name
                            total_ww1_inputs += 1

        logging.info(f'Total WW1 inputs with friends-do-not-pay rule: {total_ww1_inputs} for {path}')

        als.save_json_to_file(os.path.join(path, f'coinjoin_tx_info.json'), ww2_data)

        free_memory(ww2_data)


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
        cjvis.plot_flows_steamgraph(flows_in_time['broadcast_time_bridge'], 'BlockSci flows (1 hop), bridge tx time')
        cjvis.plot_flows_steamgraph(flows_in_time['broadcast_time_mix2'], 'BlockSci flows (1 hop), mix2 tx time')

    TWO_HOPS = False
    if TWO_HOPS:
        flows_file = os.path.join(target_path, 'two_hops_flows.json')
        if os.path.exists(flows_file):
            flows = als.load_json_from_file(flows_file)
            flows_in_time = extract_flows_blocksci(flows)
            cjvis.plot_flows_steamgraph(flows_in_time, 'BlockSci flows (2 hops)')

    # Visualization of results from Dumplings
    flows_file = os.path.join(target_path, 'mix_flows.json')
    if os.path.exists(flows_file):
        flows = als.load_json_from_file(flows_file)
        flows_in_time = extract_flows_dumplings(flows)
        cjvis.plot_flows_steamgraph(flows_in_time, 'Dumplings flows (1 hop)')
    else:
        whirlpool_postmix = load_coinjoin_stats_from_dumplings(os.path.join(target_path, 'SamouraiPostMixTxs.txt'))
        wasabi1_postmix = load_coinjoin_stats_from_dumplings(os.path.join(target_path, 'WasabiPostMixTxs.txt'))
        wasabi2_postmix = load_coinjoin_stats_from_dumplings(os.path.join(target_path, 'Wasabi2PostMixTxs.txt'))

        wasabi1_cj = load_coinjoin_stats_from_dumplings(os.path.join(target_path, 'WasabiCoinJoins.txt'))
        wasabi2_cj = load_coinjoin_stats_from_dumplings(os.path.join(target_path, 'Wasabi2CoinJoins.txt'))
        whirlpool_cj = load_coinjoin_stats_from_dumplings(os.path.join(target_path, 'SamouraiCoinJoins.txt'))

        whirlpool_premix = load_coinjoin_stats_from_dumplings(os.path.join(target_path, 'SamouraiTx0s.txt'))

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
        cjvis.plot_flows_steamgraph(flows_in_time, 'Dumplings flows')


def analyze_extramix_flows(experiment_id: str, target_path: Path, mix1_precomp_vout_txid_index: dict, mix1_postmix: dict, mix2_premix: dict, mix2_precomp_vin_txid_index: dict):
    """
    Analyze extramix coinjoin flows between two mixes (Mix1 and Mix2) in the context of a specific experiment.
    The function determines the transactions bridging Mix1 postmix outputs to Mix2 premix inputs, computes the
    flow values, and detects discrepancies in input-output differences that exceed specified thresholds.

    :param experiment_id: Identifier for the experiment being analyzed
    :type experiment_id: str
    :param target_path: Path to output the results or processed data
    :type target_path: Path
    :param mix1_precomp_vout_txid_index: A mapping of spent transaction IDs and indexes from Mix1
    :type mix1_precomp_vout_txid_index: dict
    :param mix1_postmix: Transaction data for postmix outputs of Mix1
    :type mix1_postmix: dict
    :param mix2_premix: Transaction data for premix inputs of Mix2
    :type mix2_premix: dict
    :param mix2_precomp_vin_txid_index: A mapping of vin transaction IDs and indexes associated with Mix2
    :type mix2_precomp_vin_txid_index: dict
    :return: A dictionary containing the bridging transactions and their computed flow sizes,
             including broadcast time and minimum input/output value
    :rtype: dict
    """
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
    all_cjtxs_keys = full_data["coinjoins"].keys()
    # Initial seeding for given pool size
    pool_txs = {cjtx: full_data["coinjoins"][cjtx] for cjtx in WHIRLPOOL_FUNDING_TXS[pool_size]['funding_txs']}
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
                        pool_txs[txid] = full_data["coinjoins"][txid]

                        # If Whirlpool, check all inputs for this tx if is premix
                        if 'premix' in full_data.keys():
                            for input in full_data["coinjoins"][txid]['inputs'].keys():
                                if 'spending_tx' in full_data["coinjoins"][txid]['inputs'][input].keys():
                                    txid_premix, index = als.extract_txid_from_inout_string(full_data["coinjoins"][txid]['inputs'][input]['spending_tx'])
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


def wasabi2_extract_pools_destroys_data(data: dict, target_path: str, interval_start_date: str,  interval_stop_date: str):
    """
    Takes dictionary with all coinjoins and split it to ones belonging to zksnacks coordinator and other coordinators.
    IMPORTANT: due to peak memory requirements of higher tens of GBs (03/2025), this function filters transactions inplace
    and as a result erases data from 'data' input argument - you need to load it again after calling this function.
    :param data: Dictionary will all coinjoins for all coordinators (IS erased afterwards)
    :param target_path: directory where to store jsons with separated coordinators
    :param interval_start_date: the first date to process (all coinjoins before it are ignored)
    :param interval_stop_date: the last date to process (all coinjoins after it are ignored)
    :return: dictionary with basic information regarding separated cooridnators
    """
    logging.debug('wasabi2_extract_pools_destroys_data() started')

    def save_split_coordinator(cjtx_coord: dict, target_path: str, coordinator_name: str, save_interval_start_date, save_interval_stop_date):
        target_save_path = os.path.join(target_path, coordinator_name)
        if not os.path.exists(target_save_path):
            os.makedirs(target_save_path.replace('\\', '/'))
        als.save_json_to_file(os.path.join(target_save_path, 'coinjoin_tx_info.json'), {'coinjoins': cjtx_coord})
        return {'pool_name': coordinator_name, 'start_date': save_interval_start_date,
                                              'stop_date': save_interval_stop_date,
                                              'num_cjtxs': len(cjtx_coord)}

    split_pools_info = {}
    # Extract post-zksnacks coordinator(s)
    # Rule: only after 2024-06-02, with few transactions from 2024-05-30 but with lower than 150 inputs (which is minimum for zkSNACKs)
    interval_start_date_others = '2024-05-01 00:00:00.000'
    cjtx_others = {cjtx: data["coinjoins"][cjtx] for cjtx in data["coinjoins"].keys() if data["coinjoins"][cjtx][
        'broadcast_time'] > "2024-06-02 00:00:00.000"}
    logging.debug(f'cjtx_others len={len(cjtx_others)}')
    cjtx_others_overlap = {cjtx:data["coinjoins"][cjtx] for cjtx in data["coinjoins"].keys() if data["coinjoins"][cjtx][
        'broadcast_time'] > interval_start_date_others and data["coinjoins"][cjtx][
        'broadcast_time'] < "2024-06-02 00:00:00.000" and len(data["coinjoins"][cjtx]['inputs']) < 150}
    logging.debug(f'cjtx_others_overlap len={len(cjtx_others_overlap)}')
    cjtx_others.update(cjtx_others_overlap)
    logging.debug(f'cjtx_others joined len={len(cjtx_others)}')
    split_pools_info['wasabi2_others'] = save_split_coordinator(cjtx_others, target_path,
                                                                'wasabi2_others', interval_start_date_others, interval_stop_date)
    SM.print(f'Total cjtxs extracted for pool WW2-others: {len(cjtx_others)}')

    # Extract zksnacks coordinator
    # Rule: All till 2024-06-02 00:00:00.000, in final 10 days must have >= 150 inputs
    interval_stop_date_zksnacks = "2024-06-03 00:00:00.000"
    cjtx_zksnacks_keys = {cjtx: None for cjtx in data["coinjoins"].keys() if data["coinjoins"][cjtx][
        'broadcast_time'] < "2024-05-20 00:00:00.000"}
    logging.debug(f'cjtx_zksnacks len={len(cjtx_zksnacks_keys)}')
    cjtx_zksnacks_overlap_keys = {cjtx: None for cjtx in data["coinjoins"].keys() if data["coinjoins"][cjtx][
        'broadcast_time'] > "2024-05-20 00:00:00.000" and data["coinjoins"][cjtx]['broadcast_time'] < interval_stop_date_zksnacks
                             and len(data["coinjoins"][cjtx]['inputs']) >= 150}
    logging.debug(f'cjtx_zksnacks_overlap len={len(cjtx_zksnacks_overlap_keys)}')
    cjtx_zksnacks_keys.update(cjtx_zksnacks_overlap_keys)
    logging.debug(f'cjtx_zksnacks joined len={len(cjtx_zksnacks_keys)}')

    # We have coinjoins to keep - delete all others. Use in place deletion not to cause high peak memory
    non_zksnacks_cjtxs = [cjtx for cjtx in data["coinjoins"].keys() if cjtx not in cjtx_zksnacks_keys]
    for cjtx in non_zksnacks_cjtxs:
        del data["coinjoins"][cjtx]

    target_save_path = os.path.join(target_path, 'wasabi2_zksnacks')
    split_pools_info['wasabi2_zksnacks'] = {'pool_name': 'wasabi2_zksnacks', 'start_date': '2022-06-01 00:00:07.000', 'stop_date': interval_stop_date_zksnacks,
                                           'num_cjtxs': len(cjtx_zksnacks_keys)}

    if not os.path.exists(target_save_path):
        os.makedirs(target_save_path.replace('\\', '/'))
    als.save_json_to_file(os.path.join(target_save_path, 'coinjoin_tx_info.json'), data)
    SM.print(f'Total cjtxs extracted for pool WW2-zkSNACKs: {len(data)}')
    # IMPORTANT Explicitly change data dictionary to empty one as we already modified it inplace for peak memory requirements
    data.clear()  # Clears the original dictionary
    data["deleted"] = "deleted"

    # Detect transactions which were not assigned to any pool (neither zksnacks, nor others)
    missed_cjtxs = list(set(non_zksnacks_cjtxs) - set(cjtx_others.keys()))
    als.save_json_to_file_pretty(os.path.join(target_path, f'coinjoin_tx_info__missed.json'), missed_cjtxs)
    SM.print(f'Total transactions not separated into pools: {len(missed_cjtxs)}')
    logging.debug(missed_cjtxs)

    als.save_json_to_file_pretty(os.path.join(target_path, f'split_pools_info.json'), split_pools_info)

    # Backup corresponding log file
    backup_log_files(target_path)

    return split_pools_info


def wasabi2_extract_other_pools(selected_coords: list, data: dict, target_path: str, interval_stop_date: str, txid_coord_discovered: dict):
    """
    Takes dictionary with all post-zksnacks coinjoins and split it to separate coordinators.
    :param selected_coords: list of coordinator names which shall be separated
    :param data: Dictionary will all coinjoins for all coordinators
    :param target_path: directory where to store jsons with separated coordinators
    :param interval_stop_date: the last date to process (all coinjoins after it are ignored)
    :param txid_coord_discovered: optional list with mapping between coordinators and their cjtxs
    :return: dictionary with basic information regarding separated cooridnators
    """
    logging.debug('wasabi2_extract_other_pools() started')
    interval_start_date_others = '2024-05-01 00:00:00.000'

    split_pools_info = {}
    # Extract selected post-zksnacks coordinators
    # Precompute transaction-to-entity mapping for faster lookup
    tx_to_entity = {tx_id: entity for entity, tx_ids in txid_coord_discovered.items() for tx_id in tx_ids}
    for coord_name in selected_coords:
        coord_full_name = f'wasabi2_{coord_name}'
        cjtx_coord = {cjtx: data["coinjoins"][cjtx] for cjtx in data["coinjoins"].keys()
                      if cjtx in tx_to_entity and tx_to_entity[cjtx] == coord_name}

        target_save_path = os.path.join(target_path, coord_full_name)
        if not os.path.exists(target_save_path):
            os.makedirs(target_save_path.replace('\\', '/'))
        als.save_json_to_file(os.path.join(target_save_path, 'coinjoin_tx_info.json'), {'coinjoins': cjtx_coord})
        split_pools_info[coord_full_name] = {'pool_name': coord_full_name, 'start_date': interval_start_date_others,
                'stop_date': interval_stop_date,
                'num_cjtxs': len(cjtx_coord)}

        logging.info(f'Total cjtxs extracted for pool {coord_name}: {len(cjtx_coord)}')

    return split_pools_info


def wasabi2_recompute_inputs_outputs_other_pools(selected_coords: list, target_path: str, mix_protocol: MIX_PROTOCOL, save_base_files_json: bool):
    """
    Takes list of coordinators and re-analyze liquidity inputs for each
    :param selected_coords: list of coordinator names which shall be separated
    :param target_path: directory where to store jsons with separated coordinators
    :return: dictionary with basic information regarding separated cooridnators
    """
    logging.debug('wasabi2_analyze_inputs_outputs_other_pools() started')

    # Process each coordinator
    for coord_name in selected_coords:
        coord_full_name = f'wasabi2_{coord_name}'

        target_save_path = os.path.join(target_path, coord_full_name)
        data = als.load_coinjoins_from_file(target_save_path, None, False)

        als.analyze_input_out_liquidity(target_path, data["coinjoins"], data.get('postmix', {}), data.get('premix', {}),
                                        mix_protocol, None, None, False)

        als.save_json_to_file(os.path.join(target_save_path, 'coinjoin_tx_info.json'), data)

        pool_data = process_and_save_intervals_filter(coord_full_name, MIX_PROTOCOL.WASABI2, target_path,
                                                      '2024-05-01 00:00:07.000', op.interval_stop_date,
                                                      'Wasabi2CoinJoins.txt', 'Wasabi2PostMixTxs.txt', None,
                                                      save_base_files_json, True, data)

        logging.info(f'Recomputed mix events for pool {coord_full_name}: {len(data["coinjoins"])}')

    return None


def save_coinjoins_create_folder(cjtx_coord: dict, target_path: str, coord_full_name: str):
    target_save_path = os.path.join(target_path, coord_full_name)
    if not os.path.exists(target_save_path):
        os.makedirs(target_save_path.replace('\\', '/'))
    als.save_json_to_file(os.path.join(target_save_path, 'coinjoin_tx_info.json'), {'coinjoins': cjtx_coord})


def wasabi1_extract_other_pools(selected_coords: list, data: dict, target_path: str, interval_start_date: str, interval_stop_date: str, txid_coord_discovered: dict):
    """
    Takes dictionary with all post-zksnacks WW1 coinjoins and split it to separate coordinators.
    :param selected_coords: list of coordinator names which shall be separated
    :param data: Dictionary will all coinjoins for all coordinators
    :param target_path: directory where to store jsons with separated coordinators
    :param interval_start_date: the first date to process (all coinjoins before it are ignored)
    :param interval_stop_date: the last date to process (all coinjoins after it are ignored)
    :param txid_coord_discovered: optional list with mapping between coordinators and their cjtxs
    :return: dictionary with basic information regarding separated cooridnators
    """
    logging.debug('wasabi1_extract_other_pools() started')

    # Splitting idea:
    # 1. WW1-zksnacks are coinjoins between 2018-07-19 18:09:16 and 2023-07-13 11:27:08
    #       AND having higher 'relative_order' AND having lower ratio of MIX_ENTER
    #       the exceptions are early WW1 coinjoins with naturally low 'relative_order' and having higher
    #       ratio of MIX_ENTER - these are filtered manually by false_cjtx.json
    # 2. WW1-others are all other coinjoins

    # WW1 starts 2018-07-19 18:09:16 f250e997dc1a2d68861e03689d1709973e1964a62f929ba5727fe8607dafb676
    # WW1 ends   2023-07-13 11:27:08 635fa30bfb56b6f24f6474142a57ee58306a98b9c2887ee8a799ccb4fea4a219
    interval_start_ww1_zksnacks = '2018-07-19 18:08:16.000'  # 1 minute before
    interval_stop_ww1_zksnacks = '2023-07-13 11:28:08.000'   # 1 minute after

    split_pools_info = {}
    # Extract selected post-zksnacks coordinators
    # Note: For now, we simply split based on date

    coord_full_name = f'wasabi1_zksnacks'
    # Basic filtering for WW1-zksnacks time interval
    cjtx_coord_zksnacks = {cjtx: data["coinjoins"][cjtx] for cjtx in data["coinjoins"].keys()
                          if interval_start_ww1_zksnacks < data["coinjoins"][cjtx]['broadcast_time'] < interval_stop_ww1_zksnacks}
    # Additional filtering based on relative_order and MIX_ENTER ratios
    # If 'broadcast_time' is over '2018-12-01 00:00:00.000' and 'relative_order' > 100 (after few initial WW1 transactions,
    # no stream of non-WW1 txs longer than 20 was detected)
    cjtx_coord_zknacks_filtered = {cjtx: cjtx_coord_zksnacks[cjtx] for cjtx in cjtx_coord_zksnacks.keys()
                                   if cjtx_coord_zksnacks[cjtx]['broadcast_time'] < '2018-12-01 00:00:00.000'
                                   or cjtx_coord_zksnacks[cjtx]['relative_order'] > 100}
    to_remove = {}
    # Additional filtering check - the ~0.1 output value shall be the most common one
    for cjtx in cjtx_coord_zknacks_filtered.keys():
        most_common_output_value = Counter([cjtx_coord_zknacks_filtered[cjtx]['outputs'][index]['value']
                                     for index in cjtx_coord_zknacks_filtered[cjtx]['outputs'].keys()]
                                    ).most_common(1)[0][0]
        most_common_output_value = most_common_output_value / SATS_IN_BTC
        if most_common_output_value < 0.08 or most_common_output_value > 0.12:
            print(f'{cjtx} ({data["coinjoins"][cjtx]["broadcast_time"]}) has suspicious most common output of {most_common_output_value}')
            to_remove[cjtx] = True
    # Remove found candidates for filtering
    cjtx_coord_zknacks_filtered2 = {cjtx: cjtx_coord_zknacks_filtered[cjtx] for cjtx in cjtx_coord_zknacks_filtered.keys()
                                   if cjtx not in to_remove.keys()}
    cjtx_coord_zknacks_filtered = cjtx_coord_zknacks_filtered2

    # Recompute liquidity events based on the current coinjoin set
    als.recompute_enter_remix_liquidity_after_removed_cjtxs(cjtx_coord_zknacks_filtered, MIX_PROTOCOL.WASABI1)
    #save_coinjoins_create_folder(cjtx_coord_zknacks_filtered, target_path, coord_full_name + '_after_sus_output')

    # Additional filtering check - too many fresh inputs are suspicious
    to_remove = {}
    SUS_MIX_ENTER_RATIO = 0.7
    for cjtx in cjtx_coord_zknacks_filtered.keys():
        num_inputs_enter = sum([1 for index in cjtx_coord_zknacks_filtered[cjtx]['inputs'].keys()
                          if cjtx_coord_zknacks_filtered[cjtx]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_ENTER.name])
        fresh_ratio = (num_inputs_enter / len(cjtx_coord_zknacks_filtered[cjtx]['inputs']))
        if fresh_ratio > SUS_MIX_ENTER_RATIO:
            print(f'{cjtx} ({data["coinjoins"][cjtx]["broadcast_time"]}) has suspiciously high fresh inputs of {fresh_ratio}')
            if cjtx != 'f250e997dc1a2d68861e03689d1709973e1964a62f929ba5727fe8607dafb676':  # Very first WW1 transaction, keep
                to_remove[cjtx] = True
    # Remove found candidates for filtering
    cjtx_coord_zknacks_filtered2 = {cjtx: cjtx_coord_zknacks_filtered[cjtx] for cjtx in cjtx_coord_zknacks_filtered.keys()
                                   if cjtx not in to_remove.keys()}
    cjtx_coord_zknacks_filtered = cjtx_coord_zknacks_filtered2

    # Recompute liquidity events based on the current coinjoin set
    als.recompute_enter_remix_liquidity_after_removed_cjtxs(cjtx_coord_zknacks_filtered, MIX_PROTOCOL.WASABI1)
    #save_coinjoins_create_folder(cjtx_coord_zknacks_filtered, target_path, coord_full_name + '_after_sus_fresh_rate')

    save_coinjoins_create_folder(cjtx_coord_zknacks_filtered, target_path, coord_full_name)
    logging.info(f'Total cjtxs extracted for pool {coord_full_name}: {len(cjtx_coord_zknacks_filtered)}')
    split_pools_info[coord_full_name] = {'pool_name': coord_full_name, 'start_date': interval_start_date,
            'stop_date': interval_stop_ww1_zksnacks,
            'num_cjtxs': len(cjtx_coord_zknacks_filtered)}

    # Early Wasabi1 mystery coordinator
    # First tx: 2018-08-02 15:57:32 38a83a9766357871a77992ecaead52f70c5f9f703769e6ebd4dcdb05172b28a9
    # Last tx: 2019-01-02 12:57:09 db73c667fd25aa6cf56a24cd4909d3d4b28479f79ba6ec86fe91125dc12e2022
    coord_full_name = f'wasabi1_mystery'
    cjtx_coord_mystery = {cjtx: data['coinjoins'][cjtx] for cjtx in data['coinjoins'].keys()
                         if cjtx not in cjtx_coord_zknacks_filtered.keys() and
                         '2018-08-02 15:57:00.000' < data['coinjoins'][cjtx]['broadcast_time'] < '2019-01-02 12:57:10.000'}
    save_coinjoins_create_folder(cjtx_coord_mystery, target_path, coord_full_name)
    logging.info(f'Total cjtxs extracted for pool {coord_full_name}: {len(cjtx_coord_mystery)}')
    split_pools_info[coord_full_name] = {'pool_name': coord_full_name, 'start_date': '2018-08-02 15:57:00.000',
            'stop_date': '2019-01-02 12:57:10.000',
            'num_cjtxs': len(cjtx_coord_mystery)}

    # All other cooridnators
    coord_full_name = f'wasabi1_others'
    cjtx_coord_others = {cjtx: data["coinjoins"][cjtx] for cjtx in data["coinjoins"].keys()
                         if cjtx not in cjtx_coord_zknacks_filtered.keys()}
    save_coinjoins_create_folder(cjtx_coord_others, target_path, coord_full_name)
    logging.info(f'Total cjtxs extracted for pool {coord_full_name}: {len(cjtx_coord_others)}')
    split_pools_info[coord_full_name] = {'pool_name': coord_full_name, 'start_date': interval_start_ww1_zksnacks,
            'stop_date': interval_stop_date,
            'num_cjtxs': len(cjtx_coord_others)}

    return split_pools_info


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

    sorted_cjtxs = als.sort_coinjoins(data["coinjoins"], True)
    num_cjtxs = [len(data["coinjoins"][cjtx['txid']]['inputs']) for cjtx in sorted_cjtxs]

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
        cjvis.plot_autocorrelation(autocorr)

    for i in range(1, 5):
        compute_corr(num_cjtxs, i)


# Initialize the counter
cluster_counter = 1

def analyze_zksnacks_output_clusters(mix_id, target_path):
    target_load_path = os.path.join(target_path, mix_id)
    # all_data = als.load_coinjoins_from_file(target_load_path, None, True)
    # all_data = clear_clusters(all_data)
    # all_data = assign_merge_cluster(all_data)
    # als.save_json_to_file(os.path.join(target_load_path, 'coinjoin_tx_info_clusters.json'), {'postmix': all_data['postmix'], 'coinjoins': all_data["coinjoins"]})
    data = als.load_json_from_file(os.path.join(target_load_path, 'coinjoin_tx_info_clusters.json'))

    def get_counter():
        global cluster_counter
        value = cluster_counter
        cluster_counter += 1
        return f'u_{value}'

    ONLY_ZKSNACKS = True
    if ONLY_ZKSNACKS:
        cjtx_zksnacks = [cjtx for cjtx in data["coinjoins"].keys() if data["coinjoins"][cjtx][
            'broadcast_time'] < "2024-05-27 00:00:00.000"]  # Get only cjtx till May
        # if len(cjtx_zksnacks) > 5000:
        #     cjtx_zksnacks = cjtx_zksnacks[5000:]  # Drop initial 5000 coinjoins which may be
        # cjtx_range = data["coinjoins"].keys()  # All coinjoins in interval
        cjtx_all = [cjtx for cjtx in data["coinjoins"].keys()]
        cjtx_range = cjtx_zksnacks
    else:
        cjtx_range = list(data["coinjoins"].keys())

    # Compute distribution fo different clusters of outputs
    number_output_clusters = [len(set(
        [data["coinjoins"][cjtx]['outputs'][index].get('cluster_id', get_counter()) for index in
         data["coinjoins"][cjtx]['outputs'].keys()])) for cjtx in cjtx_range]
    number_input_clusters = [len(set(
        [data["coinjoins"][cjtx]['inputs'][index].get('cluster_id', get_counter()) for index in
         data["coinjoins"][cjtx]['inputs'].keys()])) for cjtx in cjtx_range]

    number_of_outputs = [len(data["coinjoins"][cjtx]['outputs']) for cjtx in cjtx_range]
    cluster_ratio = [number_output_clusters[index] / number_of_outputs[index] for index in
                     range(0, len(number_of_outputs))]
    CUTOFF_RATIO = 0.8
    CUTOFF_RATIO = 1.1
    indexes = [index for index, value in enumerate(cluster_ratio) if value < CUTOFF_RATIO]
    high_merge_txids = {cjtx_range[index]: number_output_clusters[index] for index in indexes}
    print(
        f'txids with high merge ratio under {CUTOFF_RATIO}, total {len(high_merge_txids)}: {high_merge_txids}')

    cjtx_range = high_merge_txids
    # Compute distribution fo different clusters of outputs
    number_output_clusters = [len(set(
        [data["coinjoins"][cjtx]['outputs'][index].get('cluster_id', get_counter()) for index in
         data["coinjoins"][cjtx]['outputs'].keys()])) for cjtx in cjtx_range]
    number_input_clusters = [len(set(
        [data["coinjoins"][cjtx]['inputs'][index].get('cluster_id', get_counter()) for index in
         data["coinjoins"][cjtx]['inputs'].keys()])) for cjtx in cjtx_range]

    input_clusters_distrib = Counter(number_input_clusters)
    sorted_input_distrib = dict(sorted(input_clusters_distrib.items(), reverse=False))
    print(f'Input distribution: {sorted_input_distrib}')
    output_clusters_distrib = Counter(number_output_clusters)
    sorted_output_distrib = dict(sorted(output_clusters_distrib.items(), reverse=False))
    print(f'Output distribution: {sorted_output_distrib}')

    sorted_input_nums = dict(
        sorted(Counter([len(data["coinjoins"][cjtx]['inputs']) for cjtx in cjtx_range]).items(), reverse=False))
    sorted_output_nums = dict(
        sorted(Counter([len(data["coinjoins"][cjtx]['outputs']) for cjtx in cjtx_range]).items(),
               reverse=False))

    cjvis.plot_zksnacks_output_clusters(target_path, mix_id, sorted_output_nums, sorted_output_distrib)


def visualize_interval(mix_id: str, target_save_path: str, last_stop_date_str: str, current_stop_date_str: str):
    logging.info(f'Processing interval {last_stop_date_str} - {current_stop_date_str}')

    false_cjtxs = als.load_false_cjtxs_from_file(os.path.join(target_save_path, 'false_cjtxs.json'))

    interval_path = os.path.join(target_save_path, f'{last_stop_date_str.replace(":", "-")}--{current_stop_date_str.replace(":", "-")}_unknown-static-100-1utxo')
    assert os.path.exists(interval_path), f'{interval_path} does not exist'
    interval_data = als.load_coinjoins_from_file(interval_path, false_cjtxs, True)
    events = filter_liquidity_events(interval_data)

    # Visualize coinjoins
    if len(interval_data["coinjoins"]) > 0:
        cjvis.visualize_coinjoins(mix_id, interval_data, events, interval_path, os.path.basename(interval_path))


def visualize_intervals(mix_id: str, target_path: os.path, start_date: str, stop_date: str):
    # Process all intervals and visualize coinjoin statistics
    # TODO: This code makes own separation and does not respect existing folders with intervals
    target_save_path = os.path.join(target_path, mix_id)
    if not os.path.exists(target_save_path):
        os.makedirs(target_save_path.replace('\\', '/'))

    false_cjtxs = als.load_false_cjtxs_from_file(os.path.join(target_save_path, 'false_cjtxs.json'))

    # Visualize all data
    interval_data = als.load_coinjoins_from_file(target_save_path, false_cjtxs, True)
    if len(interval_data["coinjoins"]) > 0:
        events = filter_liquidity_events(interval_data)
        cjvis.visualize_coinjoins(mix_id, interval_data, events, target_save_path, os.path.basename(target_save_path))

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
        visualize_interval(mix_id, target_save_path, last_stop_date_str, current_stop_date_str)

        # Move to the next month
        last_stop_date_str = current_stop_date_str

        current_stop_date = current_stop_date + timedelta(days=32)
        current_stop_date = datetime(current_stop_date.year, current_stop_date.month, 1)
        current_stop_date_str = current_stop_date.strftime("%Y-%m-%d %H:%M:%S")


def print_remix_stats(target_base_path):
    def print_base_remix_info(mix_id: str, remix_stats: dict):
        SM.print(f'Remix {mix_id}')
        SM.print(f'  remix_ratios_all remix ratio (num inputs)')
        SM.print(f'    median={np.median(remix_stats["remix_ratios_all"])}')
        SM.print(f'    average={np.average(remix_stats["remix_ratios_all"])}')
        SM.print(f'    min={min(remix_stats["remix_ratios_all"])}')
        SM.print(f'    max={max(remix_stats["remix_ratios_all"])}')
        SM.print(f'  remix_ratios_std remix ratio (num inputs)')
        SM.print(f'    median={np.median(remix_stats["remix_ratios_std"])}')
        SM.print(f'    average={np.average(remix_stats["remix_ratios_std"])}')
        SM.print(f'    min={min(remix_stats["remix_ratios_std"])}')
        SM.print(f'    max={max(remix_stats["remix_ratios_std"])}')

    cfg_options = ['nums_norm', 'nums_notnorm', 'values_norm', 'values_notnorm']
    for option in cfg_options:
        SM.print(f'## Processing option {option}')
        try:
            remix_ww1 = als.load_json_from_file(os.path.join(target_base_path, f'wasabi1_remixrate_{option}.json'))
            print_base_remix_info('WW1', remix_ww1)
        except FileNotFoundError as e:
            print(e)

        try:
            remix_ww2 = als.load_json_from_file(os.path.join(target_base_path, f'wasabi2_remixrate_{option}.json'))
            remix_ww2_zksnacks = als.load_json_from_file(os.path.join(target_base_path, f'wasabi2_zksnacks_remixrate_{option}.json'))
            remix_ww2_others = als.load_json_from_file(os.path.join(target_base_path, f'wasabi2_others_remixrate_{option}.json'))
            print_base_remix_info('WW2 all', remix_ww2)
            print_base_remix_info('WW2 zksnacks', remix_ww2_zksnacks)
            print_base_remix_info('WW2 others', remix_ww2_others)
        except FileNotFoundError as e:
            print(e)

        try:
            remix_whirlpool = als.load_json_from_file(os.path.join(target_base_path, f'whirlpool_remixrate_{option}.json'))
            print_base_remix_info('Whirlpool all', remix_whirlpool)

            remix_whirlpool_100k = als.load_json_from_file(os.path.join(target_base_path, f'whirlpool_100k_remixrate_{option}.json'))
            print_base_remix_info('Whirlpool 100k', remix_whirlpool_100k)

            remix_whirlpool_1M = als.load_json_from_file(os.path.join(target_base_path, f'whirlpool_1M_remixrate_{option}.json'))
            print_base_remix_info('Whirlpool 1M', remix_whirlpool_1M)

            remix_whirlpool_5M = als.load_json_from_file(os.path.join(target_base_path, f'whirlpool_5M_remixrate_{option}.json'))
            print_base_remix_info('Whirlpool 5M', remix_whirlpool_5M)

            remix_whirlpool_50M = als.load_json_from_file(os.path.join(target_base_path, f'whirlpool_50M_remixrate_{option}.json'))
            print_base_remix_info('Whirlpool 50M', remix_whirlpool_50M)

        except FileNotFoundError as e:
            print(e)


def analyze_liquidity_summary(mix_protocol, target_path: str):
    if mix_protocol == CoinjoinType.SW:
        pools_default = WHIRLPOOL_POOL_NAMES_ALL
        # Force MIX_IDS subset if required
        pools = pools_default if op.MIX_IDS == "" else op.MIX_IDS
        for mix_id in pools:
            data = als.load_coinjoins_from_file(os.path.join(target_path, mix_id), None, True)
            SM.print(f'{mix_id}')
            liq_sum = als.print_liquidity_summary(data["coinjoins"], mix_id)
            als.save_json_to_file_pretty(os.path.join(target_path, f'liquidity_summary_{mix_id}.json'), liq_sum)
    else:
        coords = []
        if op.CJ_TYPE == CoinjoinType.WW2:
            coords = [('wasabi2', 'zksnacks'), ('wasabi2', 'others'), ('wasabi2', 'kruw'), ('wasabi2', 'gingerwallet'),
                      ('wasabi2', 'opencoordinator'), ('wasabi2', 'coinjoin_nl'), ('wasabi2', 'wasabicoordinator'),
                      ('wasabi2', 'mega'), ('wasabi2', 'btip')]
        if op.CJ_TYPE == CoinjoinType.WW1:
            coords = [('wasabi1', 'zksnacks'), ('wasabi1', 'others')]
        if op.CJ_TYPE == CoinjoinType.JM:
            coords = [('joinmarket', 'all')]
        for coord in coords:
            cjtx_coord = als.load_coinjoins_from_file(os.path.join(target_path, f'{coord[0]}_{coord[1]}'), None, True)
            SM.print(f'{coord[0]}_{coord[1]}')
            liq_sum = als.print_liquidity_summary(cjtx_coord["coinjoins"], f'{coord[0]}_{coord[1]}')
            als.save_json_to_file_pretty(os.path.join(target_path, f'liquidity_summary_{coord[0]}_{coord[1]}.json'), liq_sum)


def discover_coordinators(cjtxs: dict, sorted_cjtxs: list, coord_txs: dict, in_or_out: str,
                          min_coord_cjtxs: int, min_coord_fraction: float):
    """

    :param cjtxs:  All coinjoin transactions structure
    :param coord_txs: Mapping between cooridnator id and all its cjtxs
    :param sorted_cjtxs: Pre-sorted cjtxs (e.g., relative ordering based on transaction connections)
    :param in_or_out: if 'inputs', assignment wil be done based on cjtx inputs, if 'outputs' then on outputs
    :param min_coord_fraction: minimum fraction of inputs/outputs to specific coordinator to assign
    :param next_coord_index incremental index for coordinators
    :return: updated value of coord_txs and next_coord_index
    """
    print(f'\nFiltering small coordinators (min={min_coord_cjtxs})...')
    # Filter out coordinator ids with less than MIN_COORD_CJTXS transactions
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
                coord_ids.get(als.extract_txid_from_inout_string(cjtxs[cjtx]['inputs'][input]['spending_tx'])[0],
                              UNASSIGNED_COORD) for input in cjtxs[cjtx]['inputs'].keys()]
        elif in_or_out == 'outputs':
            input_coords = [
                coord_ids.get(als.extract_txid_from_inout_string(cjtxs[cjtx]['outputs'][input]['spend_by_tx'])[0],
                              UNASSIGNED_COORD) for input in cjtxs[cjtx]['outputs'].keys()
                                if 'spend_by_tx' in cjtxs[cjtx]['outputs'][input].keys()]
        else:
            assert False, f'Incorrect parameter in_or_out={in_or_out}'

        if len(input_coords) > 0:
            input_value_counts = Counter(input_coords)
            input_dominant_coord = input_value_counts.most_common()  # Take sorted list of the most common cooridnators
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


def wasabi_detect_coordinators(mix_id: str, protocol: MIX_PROTOCOL, target_path):
    """
    Detect propagation of remix outputs to identify separate coordinators. Based on the assumption,
    that coinjoins under same coordinator will have majority of remixed inputs from the same coordinator.
    :param mix_id:
    :param protocol:
    :param target_path:
    :return:
    """
    # Read, filter and sort coinjoin transactions
    cjtxs = als.load_coinjoins_from_file(target_path, None, True)["coinjoins"]
    ordering = als.compute_cjtxs_relative_ordering(cjtxs)
    sorted_cjtxs = sorted(ordering, key=ordering.get)

    # Load known coordinators (will be used as starting set to expand to additional transactions)
    if os.path.exists(os.path.join(target_path, 'txid_coord_t.json')):
        initial_known_txs = als.load_json_from_file(os.path.join(target_path, 'txid_coord_t.json'))  # Load known coordinators
    else:
        ground_truth_known_coord_txs = als.load_json_from_file(os.path.join(target_path, 'txid_coord.json'))  # Load known coordinators
        # Transform dictionary to {'coord': [cjtstxs]} format
        transformed_dict = defaultdict(list)
        for key, value in ground_truth_known_coord_txs.items():
            transformed_dict[value].append(key)
        initial_known_txs = dict(transformed_dict)
        als.save_json_to_file_pretty(os.path.join(target_path, 'txid_coord_t.json'), initial_known_txs)

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
        als.print_coordinators_counts(coord_txs, MIN_COORD_CJTXS)

        # Discover additionally based on outputs
        DISCOVER_ON_OUTPUTS = True
        if DISCOVER_ON_OUTPUTS:
            coord_txs, next_coord_index = discover_coordinators(cjtxs, sorted_cjtxs, coord_txs, 'outputs', MIN_COORD_CJTXS, MIN_COORD_FRACTION)
            als.print_coordinators_counts(coord_txs, MIN_COORD_CJTXS)

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
        input_coords = [coord_ids.get(als.extract_txid_from_inout_string(cjtxs[cjtx]['inputs'][index]['spending_tx'])[0], UNASSIGNED_COORD) for index in cjtxs[cjtx]['inputs'].keys()]
        output_coords = [coord_ids.get(als.extract_txid_from_inout_string(cjtxs[cjtx]['outputs'][index]['spend_by_tx'])[0], UNASSIGNED_COORD)for index in cjtxs[cjtx]['outputs'].keys()
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
    als.print_coordinators_counts(coord_txs, MIN_COORD_CJTXS)
    als.print_coordinators_counts(coord_txs, 2)

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
        als.print_coordinators_counts(merged_coord_cjtxs, MIN_COORD_CJTXS)

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

    als.save_json_to_file_pretty(os.path.join(target_path, 'txid_coord_discovered.json'), coord_txs_to_save)
    for coord_id in pair_coords.keys():
        if coord_id in coord_txs_to_save:
            coord_txs_to_save[pair_coords[coord_id]] = coord_txs_to_save.pop(coord_id)
    coord_txs_to_save_sorted = {}
    for coord_id in coord_txs_to_save.keys():
        without_date_sorted = [txid for txid in coord_txs_to_save[coord_id] if txid not in cjtxs.keys()]
        with_date = [txid for txid in coord_txs_to_save[coord_id] if txid in cjtxs.keys()]
        with_date_sorted = sorted(with_date, key=lambda x: cjtxs[x]['broadcast_time'])
        coord_txs_to_save_sorted[coord_id] = without_date_sorted + with_date_sorted

    als.save_json_to_file_pretty(os.path.join(target_path, 'txid_coord_discovered_renamed.json'), coord_txs_to_save_sorted)

    PRINT_FINAL = False
    if PRINT_FINAL:
        als.print_coordinators_counts(coord_txs, 2)
        coord_txs_filtered = {coord_id: coord_txs[coord_id] for coord_id in coord_txs.keys() if
                              len(coord_txs[coord_id]) >= MIN_COORD_CJTXS}
        #print(coord_txs_filtered)
        print(f'# Total non-small coordinators (min={MIN_COORD_CJTXS}): {len(coord_txs_filtered)}')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # --cjtype ww2 --action process_dumplings --action detect_false_positives --target-path c:\!blockchains\CoinJoin\Dumplings_Stats_20241225\
    parser.add_argument("-t", "--cjtype",
                        help="Type of coinjoin. 'ww1'...Wasabi 1.x; 'ww2'...Wasabi 2.x; 'sw'...Samourai Whirlpool; 'jm'...JoinMarket ",
                        choices=["ww1", "ww2", "sw", "jm"],
                        action="store", metavar="TYPE",
                        required=False)
    parser.add_argument("-a", "--action",
                        help="Action to performed. Can be multiple. 'process_dumplings'...extract data from Dumpling files; "
                             "'detect_false_positives'...heuristic detection of false cjtxs; "
                             "'detect_coordinators' ...heuristic detection of coordinators for cjtxs; "
                             "'split_coordinators' ...separate data files for different cooridnators; 'plot_remixes'...plot coinjoins",
                        choices=["process_dumplings", "detect_false_positives", "detect_coordinators", "split_coordinators", "plot_coinjoins"],
                        action="append", metavar="ACTION",
                        required=False)
    parser.add_argument("-tp", "--target-path",
                        help="Target path with experiment(s) to be processed. Can be multiple.",
                        action="store", metavar="PATH",
                        required=False)
    parser.add_argument("-lc", "--load-config",
                        help="Load all configuration from file",
                        action="store", metavar="FILE",
                        required=False)
    parser.add_argument("-ev", "--env_vars",
                        help="Allows to set internal variable and switches. Use with maximal care.",
                        action="store", metavar="ENV_VARS",
                        required=False)

    parser.print_help()

    return parser.parse_args(argv)


class CoinjoinType(Enum):
    WW1 = 1         # Wasabi 1.x
    WW2 = 2         # Wasabi 2.x
    SW = 3          # Samourai Whirlpool
    JM = 4          # Samourai Whirlpool


class DumplingsParseOptions:
    DEBUG = False
    # Limit analysis only to specific coinjoin type
    CJ_TYPE = CoinjoinType.WW2
    MIX_IDS = ""
    SORT_COINJOINS_BY_RELATIVE_ORDER = True
    SAVE_BASE_FILES_JSON = True
    USE_COMPACT_MEMORY_STRUCTURE = True

    ANALYSIS_PROCESS_ALL_COINJOINS_INTERVALS = False
    DETECT_FALSE_POSITIVES = False
    PLOT_REMIXES = False
    PLOT_REMIXES_SINGLE_INTERVAL = False
    PLOT_REMIXES_MULTIGRAPH = True
    PROCESS_NOTABLE_INTERVALS = False
    SPLIT_WHIRLPOOL_POOLS = False
    DETECT_COORDINATORS = False
    SPLIT_COORDINATORS = False
    PLOT_REMIXES_FLOWS = False
    ANALYSIS_ADDRESS_REUSE = False
    ANALYSIS_PROCESS_ALL_COINJOINS = False
    ANALYSIS_PROCESS_ALL_COINJOINS_DEBUG = False
    ANALYSIS_PROCESS_ALL_COINJOINS_INTERVALS_DEBUG = False
    ANALYSIS_INPUTS_DISTRIBUTION = False
    ANALYSIS_BURN_TIME = False
    ANALYSIS_CLUSTERS = False
    PLOT_INTERMIX_FLOWS = False
    VISUALIZE_ALL_COINJOINS_INTERVALS = False
    ANALYSIS_REMIXRATE = True
    ANALYSIS_LIQUIDITY = False
    ANALYSIS_BYBIT_HACK = False
    ANALYSIS_OUTPUT_CLUSTERS = False
    ANALYSIS_WALLET_PREDICTION = False

    target_base_path = ''
    #interval_stop_date = '2024-10-10 00:00:07.000'  # Last date to be analyzed, e.g., 2024-10-10 00:00:07.000
    now = datetime.now()
    interval_stop_date = now.strftime('%Y-%m-%d %H:%M:%S.') + f'{int(now.microsecond / 1000):03d}'
    interval_start_date = ""

    def __init__(self):
        self.default_values()

    def set_args(self, a):
        if a.cjtype is not None:
            if a.cjtype == 'ww1':
                self.CJ_TYPE = CoinjoinType.WW1
            if a.cjtype == 'ww2':
                self.CJ_TYPE = CoinjoinType.WW2
            if a.cjtype == 'sw':
                self.CJ_TYPE = CoinjoinType.SW
            if a.cjtype == 'jm':
                self.CJ_TYPE = CoinjoinType.JM

            if self.CJ_TYPE == CoinjoinType.WW2:
                self.SORT_COINJOINS_BY_RELATIVE_ORDER = True
            else:
                self.SORT_COINJOINS_BY_RELATIVE_ORDER = False

        if a.action is not None:
            for act in a.action:
                if act == 'process_dumplings':
                    self.ANALYSIS_PROCESS_ALL_COINJOINS_INTERVALS = True
                if act == 'detect_false_positives':
                    self.DETECT_FALSE_POSITIVES = True
                if act == 'detect_coordinators':
                    self.DETECT_COORDINATORS = True
                if act == 'split_coordinators':
                    self.SPLIT_COORDINATORS = True
                if act == 'plot_coinjoins':
                    self.PLOT_REMIXES = True

        if a.target_path is not None:
            self.target_base_path = a.target_path

        if a.env_vars is not None:
            for item in a.env_vars.split(";"):
                item = item.strip()  # Remove extra spaces
                if "=" in item:
                    key, value = map(str.strip, item.split("=", 1))  # Split and strip spaces

                    try:
                        value = ast.literal_eval(value)  # Try to evaluate the value (e.g., bool, list, int)
                    except (ValueError, SyntaxError):
                        logging.warning(f"Unable to parse value '{value}' for key '{key}', using raw string.")
                        value = value  # Fallback: use the original string as-is

                    if hasattr(self, key):  # Only set existing attributes
                        setattr(self, key, value)
                    else:
                        logging.warning(f"'{item}' command line is not a recognized attribute and will be ignored.")
    def default_values(self):
        self.DEBUG = False
        self.CJ_TYPE = CoinjoinType.WW2
        # Sorting strategy for coinjoins in time.
        # If False, coinjoins are sorted using 'broadcast_time'
        #    (which is equal to mining_time for on-chain cjtxs where we lack real broadcast time)
        # If True, then relative ordering based on connections in graph formed by remix inputs/outputs is used
        if self.CJ_TYPE == CoinjoinType.WW2:
            self.SORT_COINJOINS_BY_RELATIVE_ORDER = True
        else:
            self.SORT_COINJOINS_BY_RELATIVE_ORDER = False
        als.SORT_COINJOINS_BY_RELATIVE_ORDER = self.SORT_COINJOINS_BY_RELATIVE_ORDER
        self.MIX_IDS = ""

        self.SAVE_BASE_FILES_JSON = True
        self.ANALYSIS_PROCESS_ALL_COINJOINS_INTERVALS = False
        self.DETECT_FALSE_POSITIVES = False
        self.PLOT_REMIXES = False
        self.PLOT_REMIXES_SINGLE_INTERVAL = False  # If True, separate standalone graph is generated for each interval
        self.PLOT_REMIXES_MULTIGRAPH = True     # If True, all intervals are plotted together in single graph
        self.PROCESS_NOTABLE_INTERVALS = False
        self.SPLIT_WHIRLPOOL_POOLS = False
        self.DETECT_COORDINATORS = False
        self.SPLIT_COORDINATORS = False

        self.ANALYSIS_ADDRESS_REUSE = False
        self.ANALYSIS_PROCESS_ALL_COINJOINS = False
        self.ANALYSIS_PROCESS_ALL_COINJOINS_DEBUG = False
        self.ANALYSIS_PROCESS_ALL_COINJOINS_INTERVALS_DEBUG = False
        self.ANALYSIS_INPUTS_DISTRIBUTION = False
        self.ANALYSIS_BURN_TIME = False
        self.ANALYSIS_CLUSTERS = False
        self.ANALYSIS_REMIXRATE = False
        self.ANALYSIS_LIQUIDITY = False
        self.ANALYSIS_BYBIT_HACK = False
        self.ANALYSIS_OUTPUT_CLUSTERS = False
        self.ANALYSIS_WALLET_PREDICTION = False

        self.PLOT_REMIXES_FLOWS = False
        self.PLOT_INTERMIX_FLOWS = False
        self.VISUALIZE_ALL_COINJOINS_INTERVALS = False

        self.target_base_path = ""
        # If not set, then use current date => take all coinjoins, no limit
        self.interval_stop_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        self.interval_start_date = ""

    def print_attributes(self):
        print('*******************************************')
        print('DumplingsParseOptions parameters:')
        for attr, value in vars(self).items():
            print(f'  {attr}={value}')
        print('*******************************************')


def free_memory(data_to_free):
    del data_to_free
    data_to_free = None
    gc.collect()
    time.sleep(3)


def generate_normalized_json(base_path: str, base_txs: list):
    # 1. Generate base download script for provided base transactions
    download_base_file = os.path.join(base_path, 'download_base_txs.sh')
    als.generate_tx_download_script(base_txs, download_base_file)

    # 2. Load base_txs from hex (after downloading) and generate download script for all input transactions
    raw_txs = {}
    for txid in base_txs:
        raw_txs[txid] = als.load_json_from_file(os.path.join(base_path, f'{txid}.json'))['result']
    txids = set(base_txs)
    for txid in raw_txs:
        for tx in raw_txs[txid]['vin']:
            txids.add(tx['txid'])
    download_base_file = os.path.join(base_path, 'download_all_txs.sh')
    als.generate_tx_download_script(list(txids), download_base_file)

    # 3. Load all txs downloaded in folder and create normalized coinjoin_tx_info.json
    json_files = [f for f in os.listdir(base_path) if f.endswith('.json')]
    raw_txs = {}
    for filename in json_files:
        if filename == 'coinjoin_tx_info.json':
            continue
        txid, extension = os.path.splitext(filename)
        tx = als.load_json_from_file(os.path.join(base_path, filename))
        if 'result' in tx:
            raw_txs[txid] = tx['result']
        else:
            print(f'Skipping {filename}')

    cjtxs = {'coinjoins': {}}
    for txid in base_txs:
        cjtxs['coinjoins'][txid] = als.extract_tx_info(txid, raw_txs)
    als.save_json_to_file_pretty(os.path.join(base_path, f'coinjoin_tx_info.json'), cjtxs)


def wasabi_plot_remixes(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: Path, tx_file: str,
                        analyze_values: bool = True, normalize_values: bool = True,
                        restrict_to_out_size = None, restrict_to_in_size = None,
                        plot_multigraph: bool = True, plot_only_intervals: bool = False):
    PARALLELIZE = True
    if PARALLELIZE:
        wasabi_plot_remixes_parallel(mix_id, mix_protocol, target_path, tx_file, analyze_values, normalize_values,
                      restrict_to_out_size, restrict_to_in_size, plot_multigraph, plot_only_intervals)
    else:
        wasabi_plot_remixes_serial(mix_id, mix_protocol, target_path, tx_file, analyze_values, normalize_values,
                      restrict_to_out_size, restrict_to_in_size, plot_multigraph, plot_only_intervals)


def wasabi_plot_remixes_parallel(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: Path, tx_file: str,
                        analyze_values: bool = True, normalize_values: bool = True,
                        restrict_to_out_size = None, restrict_to_in_size = None,
                        plot_multigraph: bool = True, plot_only_intervals: bool = False):
    max_processes = multiprocessing.cpu_count()
    if plot_only_intervals:
        #
        # Plot only single intervals
        #
        # Get all paths, prepare separate task for each
        files = os.listdir(target_path) if os.path.exists(target_path) else print(
            f'Path {target_path} does not exist')
        only_dirs = [file for file in files if os.path.isdir(os.path.join(target_path, file))]
        files = only_dirs

        ### Version with futures - works!
        results: List[dict] = []
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            futures = {
                executor.submit(
                    cjvis.wasabi_plot_remixes_worker, mix_id, mix_protocol, target_path, tx_file, op.SORT_COINJOINS_BY_RELATIVE_ORDER,
                    analyze_values, normalize_values, restrict_to_out_size, restrict_to_in_size, plot_multigraph, plot_only_intervals, [file]
                ): file for file in files
            }
            with tqdm(total=len(files)) as progress:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        #results.append(result)
                        progress.update(1)
                    except Exception as e:
                        results.append({
                            "mix_id": futures[future],
                            "status": "error",
                            "error": str(e)
                        })
    else:
        #
        # Plot all graphs together
        #
        cjvis.wasabi_plot_remixes_worker(mix_id, mix_protocol, target_path, tx_file, op.SORT_COINJOINS_BY_RELATIVE_ORDER, analyze_values, normalize_values,
                            restrict_to_out_size, restrict_to_in_size, plot_multigraph, False, None)


def wasabi_plot_remixes_serial(mix_id: str, mix_protocol: MIX_PROTOCOL, target_path: Path, tx_file: str,
                        analyze_values: bool = True, normalize_values: bool = True,
                        restrict_to_out_size = None, restrict_to_in_size = None,
                        plot_multigraph: bool = True, plot_only_intervals: bool = False):

    if plot_only_intervals:
        #
        # Plot only single intervals
        #
        #fig_single, ax_single = plt.subplots()
        cjvis.wasabi_plot_remixes_worker(mix_id, mix_protocol, target_path, tx_file, op.SORT_COINJOINS_BY_RELATIVE_ORDER, analyze_values, normalize_values,
                            restrict_to_out_size, restrict_to_in_size, plot_multigraph, True)
    else:
        #
        # Plot all graphs together
        #
        cjvis.wasabi_plot_remixes_worker(mix_id, mix_protocol, target_path, tx_file, op.SORT_COINJOINS_BY_RELATIVE_ORDER, analyze_values, normalize_values,
                            restrict_to_out_size, restrict_to_in_size, plot_multigraph, False)



# # noinspection PyUnboundLocalVariable
# def detect_additional_cjtxs(mix_id: str, mix_protocol: MIX_PROTOCOL, base_path: Path):
#     """
#     Use existing set of coinjoin transactions to detect additional ones missed during previous analysis.
#     Iteratively query bitcoin fullnode until coinjoin tx set stabilizes
#     :param mix_id: id of mix
#     :param base_path: basic path to coordinator
#     :return: updated dict
#     """
#
#     # Idea: Explore txids for inputs/outputs of existing coinjoins transactions. If more than defined X of existing coinjoins
#     # accepts outputs from a given txid, assume it might be coinjoin => fetch and analyze
#
#     data = als.load_coinjoins_from_file(base_path, None, True)
#
#     change_detected = True
#     new_coinjoins = {}
#     while change_detected:
#         change_detected = False
#         txid_dict = {}
#         for cjtx in data['coinjoins'].keys():
#             for input in data['coinjoins'][cjtx]['inputs'].keys():
#                 if 'spending_tx' in data['coinjoins'][cjtx]['inputs'][input].keys():
#                     prev_tx, prev_tx_index = als.extract_txid_from_inout_string(data['coinjoins'][cjtx]['inputs'][input]['spending_tx'])
#                     if prev_tx not in data['coinjoins']:
#                         if prev_tx not in txid_dict.keys():
#                             txid_dict[prev_tx] = 1
#                         else:
#                             txid_dict[prev_tx] = txid_dict[prev_tx] + 1
#         for cjtx in data['coinjoins'].keys():
#             for output in data['coinjoins'][cjtx]['outputs'].keys():
#                 if 'spend_by_tx' in data['coinjoins'][cjtx]['outputs'][output].keys():
#                     prev_tx, prev_tx_index = als.extract_txid_from_inout_string(data['coinjoins'][cjtx]['outputs'][output]['spend_by_tx'])
#                     if prev_tx not in data['coinjoins']:
#                         if prev_tx not in txid_dict.keys():
#                             txid_dict[prev_tx] = 1
#                         else:
#                             txid_dict[prev_tx] = txid_dict[prev_tx] + 1
#         # Now check transactions which has at least X hits
#         MIN_HITS_THRESHOLD = 5  # Number of times given transaction must be referenced by known coinjoins
#         frequent_txids = [txid for txid in txid_dict.keys() if txid_dict[txid] > MIN_HITS_THRESHOLD]
#         print(frequent_txids)
#
#
#         raw_txs = {}
#         to_fetch_txs = set()
#         to_fetch_txs.update(frequent_txids)
#         # Fetch tx info from fullnode and check if it looks like coinjoins
#         for txid in frequent_txids:
#             if txid not in raw_txs:
#                 txid_raw_file = f'{txid}.json'
#                 curl_str = "curl --user user:password --data-binary \'{\"jsonrpc\": \"1.0\", \"id\": \"curltest\", \"method\": \"getrawtransaction\", \"params\": [\"" + txid + "\", true]}\' -H \'Content-Type: application/json\' http://127.0.0.1:8332/" + f" > {txid_raw_file}\n"
#                 result = als.run_command(curl_str, True)
#                 if result.returncode != 0:
#                     print(f'Cannot retrieve tx info for {txid} with {result.stderr} error')
#                 else:
#                     tx_info = json.loads(result.stdout)
#                     raw_txs[txid] = tx_info['result']
#                     inputs = tx_info['vin']
#                     index = 0
#                     for input in inputs:
#                         to_fetch_txs.update(inputs['txid'])
#
#         txinfo = als.extract_tx_info(txid, raw_txs)
#         if is_coinjoin_tx(txinfo, mix_protocol):
#             new_coinjoins[txid] = txinfo
#             #change_detected = True
#
#     print(new_coinjoins)
#     return data


def main(argv=None):
    try:
        multiprocessing.set_start_method("spawn") # Set safer process spawning variant for multiprocessing
    except RuntimeError as e:
        if "context has already been set" not in str(e):
            raise

    global op
    op = DumplingsParseOptions()
    # parse arguments, overwrite default settings if required
    args = parse_arguments(argv)

    op.set_args(args)

    # Propagate control settings to cj_analysis
    als.SORT_COINJOINS_BY_RELATIVE_ORDER = op.SORT_COINJOINS_BY_RELATIVE_ORDER
    als.PERF_USE_COMPACT_CJTX_STRUCTURE = op.USE_COMPACT_MEMORY_STRUCTURE

    target_path = os.path.join(op.target_base_path, 'Scanner')
    SM.print(f'Starting analysis of {target_path}')
    op.print_attributes()

    # WARNING: SW 100k pool does not match exactly mix_stay and active liqudity at the end - likely reason are neglected mining fees

    #op.DEBUG = True
    if op.DEBUG:
        print('DEBUGING TIME!!!')

        wasabi_detect_false(os.path.join(target_path, 'wasabi2_kruw'), 'coinjoin_tx_info.json')
        exit(42)


    if op.ANALYSIS_BYBIT_HACK:
        url = "https://hackscan.hackbounty.io/public/hack-address.json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            bybit_hack = response.json()
        except requests.exceptions.RequestException as e:
            logging.error("bybit hack-adressess download error:", e)

        # Save for later use
        json_path = os.path.join(target_path, 'bybit_hack-address.json')
        print(f'Path to save {json_path}')
        als.save_json_to_file_pretty(json_path, bybit_hack)
        bybit_hack = als.load_json_from_file(json_path)
        bybit_hack_addresses = {addr: 1 for addr in bybit_hack['0221']['btc']}

        # Detect bybit addresses coordinator
        detected_addressed = {}
        for month in range(2, 10):
            interval_name = f'wasabi2_kruw/2025-0{month}-01 00-00-00--2025-0{month+1}-01 00-00-00_unknown-static-100-1utxo'
            if os.path.exists(os.path.join(target_path, interval_name)):
                bybit_interval = als.detect_bybit_hack(target_path, interval_name, bybit_hack_addresses)
                als.merge_dicts(bybit_interval, detected_addressed)

        als.save_json_to_file_pretty(os.path.join(target_path, 'bybit_hack-txs.json'), detected_addressed)

        total_btc_mixed = 0
        total_hits = 0
        for address in detected_addressed['hits'].keys():
            total_hits += len(detected_addressed['hits'][address])
            for item in detected_addressed['hits'][address]:
                total_btc_mixed += item['value']

        detected_addressed['_summary'] = {'hits_detected': total_hits, 'total_btc_mixed': total_btc_mixed, 'total_btc_mixed_str': f'{round(total_btc_mixed / SATS_IN_BTC, 2)} btc'}
        SM.print(f"Bybit hack detection:")
        SM.print(f"  Total address entering coinjoins: {total_hits}")
        SM.print(f"  Total detected mixed: {round(total_btc_mixed / SATS_IN_BTC, 2)} btc")

        als.save_json_to_file_pretty(os.path.join(target_path, 'bybit_hack-txs.json'), detected_addressed, True)

    if op.PROCESS_NOTABLE_INTERVALS:
        def process_joint_interval(mix_origin_name, interval_name, all_data, mix_type, target_path, start_date: str,
                                   end_date: str):
            process_and_save_single_interval(interval_name, all_data, mix_type, target_path, start_date, end_date)
            shutil.copyfile(os.path.join(target_path, mix_origin_name, 'fee_rates.json'),
                            os.path.join(target_path, interval_name, 'fee_rates.json'))
            shutil.copyfile(os.path.join(target_path, mix_origin_name, 'false_cjtxs.json'),
                            os.path.join(target_path, interval_name, 'false_cjtxs.json'))
            wasabi_plot_remixes(interval_name, MIX_PROTOCOL.WASABI1, os.path.join(target_path, interval_name),
                                'coinjoin_tx_info.json', True, False, None, None, op.PLOT_REMIXES_MULTIGRAPH, op.PLOT_REMIXES_SINGLE_INTERVAL)

        if op.CJ_TYPE == CoinjoinType.WW1:
            target_load_path = os.path.join(target_path, 'wasabi1')
            all_data = als.load_coinjoins_from_file(target_load_path, None, True)

            # Large inflows into WW1 in 2019-08-09, mixed and the all taken out
            process_joint_interval('wasabi1', 'wasabi1__2019_08-09', all_data, MIX_PROTOCOL.WASABI1, target_path, '2019-08-01 00:00:07.000', '2019-09-30 23:59:59.000')

            # Large single inflow with long remixing continously taken out
            process_joint_interval('wasabi1', 'wasabi1__2020_03-04', all_data, MIX_PROTOCOL.WASABI1, target_path, '2020-03-26 00:00:07.000','2020-04-20 23:59:59.000')

            # Two inflows, subsequent remixing
            process_joint_interval('wasabi1', 'wasabi1__2022_04-05', all_data, MIX_PROTOCOL.WASABI1, target_path, '2022-04-23 00:00:07.000', '2022-05-06 23:59:59.000')

        if op.CJ_TYPE == CoinjoinType.WW2:
            target_load_path = os.path.join(target_path, 'wasabi2')
            all_data = als.load_coinjoins_from_file(os.path.join(target_load_path), None, True)

            # Large inflow, in 2023-12, slightly mixed, send out, received as friend, then remixed
            process_joint_interval('wasabi2', 'wasabi2__2023_12-01', all_data, MIX_PROTOCOL.WASABI2, target_path, '2023-12-20 00:00:07.000', '2024-01-30 23:59:59.000')

        if op.CJ_TYPE == CoinjoinType.SW:
            logging.warning('No notable intervals for Whirlpool')

        if op.CJ_TYPE == CoinjoinType.JM:
            logging.warning('No notable intervals for JoinMarket')
    #
    #
    #
    if op.ANALYSIS_PROCESS_ALL_COINJOINS_INTERVALS:
        if op.CJ_TYPE == CoinjoinType.JM:
            interval_start_date = '2015-01-01 00:00:00.000' if op.interval_start_date == "" else op.interval_start_date
            all_data = process_and_save_intervals_filter('joinmarket_all', MIX_PROTOCOL.JOINMARKET, target_path, interval_start_date, op.interval_stop_date,
                                       'OtherCoinJoins.txt', 'OtherCoinJoinPostMixTxs.txt', None,
                                                op.SAVE_BASE_FILES_JSON, False)

        if op.CJ_TYPE == CoinjoinType.SW:
            interval_start_date = '2019-04-17 01:38:07.000' if op.interval_start_date == "" else op.interval_start_date
            all_data = process_and_save_intervals_filter('whirlpool', MIX_PROTOCOL.WHIRLPOOL, target_path, interval_start_date, op.interval_stop_date,
                                       'SamouraiCoinJoins.txt', 'SamouraiPostMixTxs.txt', 'SamouraiTx0s.txt',
                                                op.SAVE_BASE_FILES_JSON, False)

            mix_ids_default = WHIRLPOOL_POOL_NAMES_ALL
            # Force MIX_IDS subset if required
            mix_ids = mix_ids_default if op.MIX_IDS == "" else op.MIX_IDS
            # Split and process Whirlpool-based on pools
            for mix_id in mix_ids:
                whirlpool_extract_pool(all_data, 'whirlpool', target_path, mix_id, WHIRLPOOL_POOL_SIZES[mix_id])
                process_and_save_intervals_filter(mix_id, MIX_PROTOCOL.WHIRLPOOL, target_path,
                                                  WHIRLPOOL_FUNDING_TXS[WHIRLPOOL_POOL_SIZES[mix_id]]["start_date"], op.interval_stop_date,
                                                  None, None, None, op.SAVE_BASE_FILES_JSON, True)

        if op.CJ_TYPE == CoinjoinType.WW1:
            interval_start_date = '2018-07-19 01:38:07.000' if op.interval_start_date == "" else op.interval_start_date
            process_and_save_intervals_filter('wasabi1', MIX_PROTOCOL.WASABI1, target_path, interval_start_date, op.interval_stop_date,
                                       'WasabiCoinJoins.txt', 'WasabiPostMixTxs.txt', None, op.SAVE_BASE_FILES_JSON, False)


        if op.CJ_TYPE == CoinjoinType.WW2:
            interval_start_date = '2022-06-01 00:00:07.000' if op.interval_start_date == "" else op.interval_start_date
            data = process_and_save_intervals_filter('wasabi2', MIX_PROTOCOL.WASABI2, target_path, interval_start_date, op.interval_stop_date,
                    'Wasabi2CoinJoins.txt', 'Wasabi2PostMixTxs.txt', None, op.SAVE_BASE_FILES_JSON, False)

            # Split zkSNACKs (-> wasabi2_zksnacks) and post-zkSNACKs (-> wasabi2_others) pools
            # This splitting will allow to analyze separate pools, but also to make data files smaller and easier to process later
            logging.info('Going to wasabi2_extract_pools() *****************************')
            split_pool_info = wasabi2_extract_pools_destroys_data(data, target_path, op.interval_start_date, op.interval_stop_date)
            logging.info('done wasabi2_extract_pools() *****************************')
            free_memory(data)

            # Force MIX_IDS subset if required
            mix_ids = split_pool_info.keys() if op.MIX_IDS == "" else op.MIX_IDS

            # WW2 needs additional treatment - detect and fix origin of WW1 inflows as friends
            # Do first separated pools, then the original (large) unseparated one
            for pool_name in mix_ids:
                fix_ww2_for_fdnp_ww1(pool_name, target_path)

            for pool_name in mix_ids:
                logging.info(f'Going to process_and_save_intervals_filter({pool_name}) *****************************')
                pool_interval_start_date = split_pool_info[pool_name]['start_date']
                if op.interval_start_date != "" and pool_interval_start_date < op.interval_start_date:
                    pool_interval_start_date = op.interval_start_date
                pool_interval_stop_date = split_pool_info[pool_name]['stop_date']
                if op.interval_stop_date != "" and pool_interval_stop_date > op.interval_stop_date:
                    pool_interval_stop_date = op.interval_stop_date
                pool_data = process_and_save_intervals_filter(pool_name, MIX_PROTOCOL.WASABI2, target_path,
                                                         interval_start_date, pool_interval_stop_date,
                                                         'Wasabi2CoinJoins.txt', 'Wasabi2PostMixTxs.txt', None,
                                                         op.SAVE_BASE_FILES_JSON, True)
                free_memory(pool_data)
                logging.info(f'done for {pool_name}) *****************************')

            # Fix the large aggregate file (may crash due to huge memory requirements)
            # Precaution: Let's streamline large dictionary first and save
            ww2_data = als.load_coinjoins_from_file(os.path.join(target_path, 'wasabi2'), None, False)
            als.streamline_coinjoins_structure(ww2_data)
            als.save_json_to_file(os.path.join(target_path, 'wasabi2', 'coinjoin_tx_info.json'), ww2_data)
            del ww2_data

            logging.info(f'Going to fix_ww2_for_fdnp_ww1(wasabi2) *****************************')
            fix_ww2_for_fdnp_ww1('wasabi2', target_path)
            logging.info(f'done fix_ww2_for_fdnp_ww1(wasabi2) *****************************')
            logging.info(f'Going to process_and_save_intervals_filter(wasabi2) *****************************')
            interval_start_date = '2022-06-01 00:00:07.000' if op.interval_start_date == "" else op.interval_start_date
            process_and_save_intervals_filter('wasabi2', MIX_PROTOCOL.WASABI2, target_path, interval_start_date, op.interval_stop_date,
                                       'Wasabi2CoinJoins.txt', 'Wasabi2PostMixTxs.txt', None, op.SAVE_BASE_FILES_JSON, True)
            logging.info(f'done process_and_save_intervals_filter(wasabi2) *****************************')

    if op.VISUALIZE_ALL_COINJOINS_INTERVALS:
        if op.CJ_TYPE == CoinjoinType.SW:
            interval_start_date = '2019-04-17 01:38:07.000' if op.interval_start_date == "" else op.interval_start_date
            mix_ids = ['whirlpool'] if op.MIX_IDS == "" else op.MIX_IDS
            for mix_id in mix_ids:
                visualize_intervals(mix_id, target_path, interval_start_date, op.interval_stop_date)

        if op.CJ_TYPE == CoinjoinType.WW1:
            interval_start_date = '2018-07-19 01:38:07.000' if op.interval_start_date == "" else op.interval_start_date
            mix_ids = ['wasabi1'] if op.MIX_IDS == "" else op.MIX_IDS
            for mix_id in mix_ids:
                visualize_intervals(mix_id, target_path, interval_start_date, op.interval_stop_date)

        if op.CJ_TYPE == CoinjoinType.WW2:
            interval_start_date = '2022-06-01 00:00:07.000' if op.interval_start_date == "" else op.interval_start_date
            mix_ids = ['wasabi2', 'wasabi2_zksnacks'] if op.MIX_IDS == "" else op.MIX_IDS
            for mix_id in mix_ids:
                visualize_intervals(mix_id, target_path, interval_start_date, op.interval_stop_date)

        if op.CJ_TYPE == CoinjoinType.JM:
            interval_start_date = '2015-01-01 00:00:00.000' if op.interval_start_date == "" else op.interval_start_date
            mix_ids = ['joinmarket_all'] if op.MIX_IDS == "" else op.MIX_IDS
            for mix_id in mix_ids:
                visualize_intervals(mix_id, target_path, interval_start_date, op.interval_stop_date)

    if op.DETECT_FALSE_POSITIVES:
        if op.CJ_TYPE == CoinjoinType.WW1:
            wasabi_detect_false(os.path.join(target_path, 'wasabi1'), 'coinjoin_tx_info.json')
        if op.CJ_TYPE == CoinjoinType.WW2:
            wasabi_detect_false(os.path.join(target_path, 'wasabi2'), 'coinjoin_tx_info.json')
        if op.CJ_TYPE == CoinjoinType.JM:
            wasabi_detect_false(os.path.join(target_path, 'joinmarket_all'), 'coinjoin_tx_info.json')
        if op.CJ_TYPE == CoinjoinType.SW:
            mix_ids_default = WHIRLPOOL_POOL_NAMES_ALL
            # Force MIX_IDS subset if required
            mix_ids = mix_ids_default if op.MIX_IDS == "" else op.MIX_IDS
            for mix_id in mix_ids:
                wasabi_detect_false(os.path.join(target_path, mix_id), 'coinjoin_tx_info.json')

    if op.DETECT_COORDINATORS:
        if op.CJ_TYPE == CoinjoinType.WW2:
            # Detect coordinators for others (wasabi2_others)
            wasabi_detect_coordinators('wasabi2_others', MIX_PROTOCOL.WASABI2, os.path.join(target_path, 'wasabi2_others'))
        else:
            logging.error('Unsupported CJ_TYPE for DETECT_COORDINATORS')
            exit(-1)

    if op.SPLIT_COORDINATORS:
        if op.CJ_TYPE == CoinjoinType.WW2:
            data = als.load_coinjoins_from_file(os.path.join(target_path, 'wasabi2'), None, False)

            coord_tx_mapping = als.load_json_from_file(os.path.join(target_path, 'wasabi2_others', 'txid_coord_discovered_renamed.json'))
            selected_coords_default = ["kruw", "mega", "btip", "gingerwallet", "wasabicoordinator", "coinjoin_nl",
                               "opencoordinator", "dragonordnance", "wasabist", "strange_2025"]
            # Force MIX_IDS subset if required
            selected_coords = selected_coords_default if op.MIX_IDS == "" else op.MIX_IDS

            split_pool_info = wasabi2_extract_other_pools(selected_coords, data, target_path, op.interval_stop_date, coord_tx_mapping)
            # Perform splitting into month intervals for all processed coordinators
            for pool_name in split_pool_info.keys():
                logging.info(f'Going to process_and_save_intervals_filter({pool_name}) *****************************')
                pool_data = process_and_save_intervals_filter(pool_name, MIX_PROTOCOL.WASABI2, target_path,
                                                              split_pool_info[pool_name]['start_date'],
                                                              split_pool_info[pool_name]['stop_date'],
                                                              'Wasabi2CoinJoins.txt', 'Wasabi2PostMixTxs.txt', None,
                                                              op.SAVE_BASE_FILES_JSON, True)
                logging.info(f'done for {pool_name}) *****************************')

        if op.CJ_TYPE == CoinjoinType.WW1:
            data = als.load_coinjoins_from_file(os.path.join(target_path, 'wasabi1'), None, False)
            coord_tx_mapping = None
            selected_coords_default = ['zksnacks', 'mystery', 'others']
            # Force MIX_IDS subset if required
            selected_coords = selected_coords_default if op.MIX_IDS == "" else op.MIX_IDS

            interval_start_date = '2018-07-19 01:38:07.000' if op.interval_start_date == "" else op.interval_start_date
            split_pool_info = wasabi1_extract_other_pools(selected_coords, data, target_path, interval_start_date, op.interval_stop_date, coord_tx_mapping)
            # Perform splitting into month intervals for all processed coordinators
            for pool_name in split_pool_info.keys():
                logging.info(f'Going to process_and_save_intervals_filter({pool_name}) *****************************')
                pool_data = process_and_save_intervals_filter(pool_name, MIX_PROTOCOL.WASABI1, target_path,
                                                              split_pool_info[pool_name]['start_date'],
                                                              split_pool_info[pool_name]['stop_date'],
                                                              'WasabiCoinJoins.txt', 'WasabiPostMixTxs.txt', None,
                                                              op.SAVE_BASE_FILES_JSON, True)
                logging.info(f'done for {pool_name}) *****************************')

        if op.CJ_TYPE == CoinjoinType.SW:
            # Load txs for all pools
            target_load_path = os.path.join(target_path, 'whirlpool')
            logging.info(f'Loading {target_load_path}/coinjoin_tx_info.json ...')
            data = als.load_coinjoins_from_file(target_load_path, None, False)

            # Separate per pool
            pool_100k = whirlpool_extract_pool(data, 'whirlpool', target_path, 'whirlpool_100k', 100000)
            pool_1M = whirlpool_extract_pool(data, 'whirlpool', target_path, 'whirlpool_1M', 1000000)
            pool_5M = whirlpool_extract_pool(data, 'whirlpool', target_path, 'whirlpool_5M', 5000000)
            pool_50M = whirlpool_extract_pool(data, 'whirlpool', target_path, 'whirlpool_50M', 50000000)
            pool_25M = whirlpool_extract_pool(data, 'whirlpool', target_path, 'whirlpool_ashigaru_25M', 25000000)
            pool_2_5M = whirlpool_extract_pool(data, 'whirlpool', target_path, 'whirlpool_ashigaru_2_5M', 2500000)

            # Detect transactions which were not assigned to any pool
            missed_cjtxs = list(set(data["coinjoins"].keys()) - set(pool_100k["coinjoins"].keys()) - set(pool_1M["coinjoins"].keys())
                                - set(pool_5M["coinjoins"].keys()) - set(pool_50M["coinjoins"].keys())
                                - set(pool_25M["coinjoins"].keys()) - set(pool_2_5M["coinjoins"].keys()))
            als.save_json_to_file_pretty(os.path.join(target_load_path, f'coinjoin_tx_info__missed.json'), missed_cjtxs)
            print(f'Total transactions not separated into pools: {len(missed_cjtxs)}')
            print(missed_cjtxs)

        if op.CJ_TYPE == CoinjoinType.JM:
            logging.error('Unsupported CJ_TYPE for DETECT_COORDINATORS')
            exit(-1)

    if op.PLOT_INTERMIX_FLOWS:
        analyze_mixes_flows(target_path)

    if op.PLOT_REMIXES:
        def ww_plot_remixes_helper(mix_ids_default: list, mix_protocol):
            # Force MIX_IDS subset if required
            mix_ids = mix_ids_default if op.MIX_IDS == "" else op.MIX_IDS
            logging.info(f'Going to process following mixes: {mix_ids}')
            for mix_id in mix_ids:
                target_base_path = os.path.join(target_path, mix_id)
                if os.path.exists(target_base_path):
                    wasabi_plot_remixes(mix_id, mix_protocol, os.path.join(target_path, mix_id), 'coinjoin_tx_info.json', False, False, None, None, op.PLOT_REMIXES_MULTIGRAPH, op.PLOT_REMIXES_SINGLE_INTERVAL)
                    wasabi_plot_remixes(mix_id, mix_protocol, os.path.join(target_path, mix_id), 'coinjoin_tx_info.json', False, True, None, None, op.PLOT_REMIXES_MULTIGRAPH, op.PLOT_REMIXES_SINGLE_INTERVAL)
                    wasabi_plot_remixes(mix_id, mix_protocol, os.path.join(target_path, mix_id), 'coinjoin_tx_info.json', True, False, None, None, op.PLOT_REMIXES_MULTIGRAPH, op.PLOT_REMIXES_SINGLE_INTERVAL)
                    wasabi_plot_remixes(mix_id, mix_protocol, os.path.join(target_path, mix_id), 'coinjoin_tx_info.json', True, True, None, None, op.PLOT_REMIXES_MULTIGRAPH, op.PLOT_REMIXES_SINGLE_INTERVAL)
                else:
                    logging.warning(f'Path {target_base_path} does not exists.')

        if op.CJ_TYPE == CoinjoinType.WW1:
            ww_plot_remixes_helper(['wasabi1_mystery', 'wasabi1_zksnacks', 'wasabi1_others', 'wasabi1'], MIX_PROTOCOL.WASABI1)

        if op.CJ_TYPE == CoinjoinType.WW2:
            ww_plot_remixes_helper(['wasabi2_kruw', 'wasabi2_gingerwallet', 'wasabi2_opencoordinator',
                                    'wasabi2_coinjoin_nl', 'wasabi2_wasabicoordinator', 'wasabi2_wasabist',
                                    'wasabi2_dragonordnance', 'wasabi2_mega', 'wasabi2_btip', 'wasabi2_strange_2025', 'wasabi2_others',
                                    'wasabi2_zksnacks', 'wasabi2'], MIX_PROTOCOL.WASABI2)

        if op.CJ_TYPE == CoinjoinType.SW:
            ww_plot_remixes_helper(WHIRLPOOL_POOL_NAMES_ALL, MIX_PROTOCOL.WHIRLPOOL)

        if op.CJ_TYPE == CoinjoinType.JM:
            ww_plot_remixes_helper(['joinmarket_all'], MIX_PROTOCOL.JOINMARKET)

    # if op.PLOT_REMIXES_FLOWS:
    #     wasabi_plot_remixes_flows('wasabi2_select',
    #                          os.path.join(target_path, 'wasabi2_select'),
    #                          'coinjoin_tx_info.json', False, True)

    if op.ANALYSIS_CLUSTERS:
        if op.CJ_TYPE == CoinjoinType.WW1:
            target_load_path = os.path.join(target_path, 'wasabi1')
        if op.CJ_TYPE == CoinjoinType.WW2:
            target_load_path = os.path.join(target_path, 'wasabi2')
        if op.CJ_TYPE == CoinjoinType.SW:
            target_load_path = os.path.join(target_path, 'whirlpool')
        if op.CJ_TYPE == CoinjoinType.JM:
            target_load_path = os.path.join(target_path, 'joinmarket_all')

        all_data = als.load_coinjoins_from_file(target_load_path, None, True)
        all_data = analyze_postmix_spends(all_data)
        als.save_json_to_file(os.path.join(target_load_path, 'coinjoin_tx_info_clusters.json'), {'postmix': all_data['postmix'], 'coinjoins': all_data["coinjoins"]})

    if op.ANALYSIS_BURN_TIME:
        if op.CJ_TYPE == CoinjoinType.WW1:
            wasabi1_analyse_remixes('Wasabi1', target_path)
        if op.CJ_TYPE == CoinjoinType.WW2:
            wasabi2_analyse_remixes('Wasabi2', target_path)
        if op.CJ_TYPE == CoinjoinType.SW:
            whirlpool_analyse_remixes('Whirlpool', target_path)
        if op.CJ_TYPE == CoinjoinType.JM:
            logging.error('Unsupported CJ_TYPE for ANALYSIS_BURN_TIME')
            exit(-1)

    # Extract distribution of mix fresh input sizes
    if op.ANALYSIS_INPUTS_DISTRIBUTION:
        # Produce figure with distribution of diffferent pools merged
        if op.CJ_TYPE == CoinjoinType.WW1:
            process_inputs_distribution('wasabi1', MIX_PROTOCOL.WASABI1,  target_path, 'WasabiCoinJoins.txt', True)
            process_outputs_distribution('wasabi1', MIX_PROTOCOL.WASABI1,  target_path, 'WasabiCoinJoins.txt', True)

        if op.CJ_TYPE == CoinjoinType.SW:
            process_inputs_distribution_whirlpool('whirlpool', MIX_PROTOCOL.WHIRLPOOL,  target_path, 'SamouraiTx0s.txt', True)
            process_outputs_distribution('whirlpool', MIX_PROTOCOL.WHIRLPOOL, target_path, 'SamouraiTx0s.txt', True)
            process_inputs_distribution_whirlpool('whirlpool_ashigaru_2_5M', MIX_PROTOCOL.WHIRLPOOL,  target_path, 'SamouraiTx0s.txt', True)
            process_outputs_distribution('whirlpool_ashigaru_2_5M', MIX_PROTOCOL.WHIRLPOOL, target_path, 'SamouraiTx0s.txt', True)

        if op.CJ_TYPE == CoinjoinType.WW2:
            for pool in ['wasabi2_zksnacks', 'wasabi2_others']:
                process_inputs_distribution(pool, MIX_PROTOCOL.WASABI2,  target_path, 'Wasabi2CoinJoins.txt', True)
                process_outputs_distribution(pool, MIX_PROTOCOL.WASABI2,  target_path, 'Wasabi2CoinJoins.txt', True)

        if op.CJ_TYPE == CoinjoinType.JM:
            process_inputs_distribution('joinmarket_all', MIX_PROTOCOL.JOINMARKET,  target_path, 'OtherCoinJoins.txt', True)
            process_outputs_distribution('joinmarket_all', MIX_PROTOCOL.JOINMARKET,  target_path, 'OtherCoinJoins.txt', True)



    #
    # Analyze address reuse in all mixes
    #
    if op.ANALYSIS_ADDRESS_REUSE:
        analyze_address_reuse(target_path)

    if op.ANALYSIS_REMIXRATE:
        print_remix_stats(op.target_base_path)

    if op.ANALYSIS_LIQUIDITY:
        analyze_liquidity_summary(op.CJ_TYPE, target_path)

    if op.ANALYSIS_OUTPUT_CLUSTERS:
        analyze_zksnacks_output_clusters('wasabi2', target_path)

    if op.ANALYSIS_WALLET_PREDICTION:
        cjvis.estimate_wallet_prediction_factor(target_path, 'wasabi2_gingerwallet')
        cjvis.estimate_wallet_prediction_factor(target_path, 'wasabi2_opencoordinator')
        cjvis.estimate_wallet_prediction_factor(target_path, 'wasabi2_wasabicoordinator')
        cjvis.estimate_wallet_prediction_factor(target_path, 'wasabi2_kruw')
        cjvis.estimate_wallet_prediction_factor(target_path, 'wasabi2_zksnacks')
        cjvis.estimate_wallet_prediction_factor(target_path, 'wasabi1')


    print('### SUMMARY #############################')
    SM.print_summary()
    print('### END SUMMARY #########################')

    return 0

if __name__ == "__main__":
    main()


    # TODO: Set x labels for histogram of frequencies to rounded denominations
    # TODO: Detect likely cases of WW2 round split due to more than 400 inputs registered
    #   (two coinjoins broadcasted close to each other, sum of inputs is close or higher than 400)
    # TODO: Detect if multiple rounds were happening in parallel (coinjoin time close to each other)

    # TODO: Huge consolidation of Whirlpool coins: https://mempool.space/tx/d463b35b3d18dda4e59f432728c7a365eaefd50b24a6596ab42a077868e9d7e5
    #  (>60btc total, payjoin (possibly fake) attempted, 140+ inputs from various )
    # https://mempool.space/tx/8f59577b2dfa88e7d7fdd206a17618893db7559007a15658872b665bc16417c5
    # https://mempool.space/tx/d463b35b3d18dda4e59f432728c7a365eaefd50b24a6596ab42a077868e9d7e5
    # https://mempool.space/tx/57a8ea3ba1568fed4d9f7d7b3b84cdec552d9c49d4849bebf77a1053c180d0d1
    #

    # TODO: Analyze difference of unmoved and dynamic liquidity for Whirlpool between 2024-04-24 and 2024-08-24 (impact of knowledge of whirlpool seizure). Show last 1 year.

    # Analyze dominance cost:
    # 1. Coordinator fee to maintain X% pool liquidity at the time (put new input in if current liquidity below X%)
    # 2. Mining fees to maintain X% control of all inputs / outputs of each coinjoin. Disregard outliers with large sudden
    # incoming liquidity which will not be completely mixed anyway
    # - Stay in pool if already there (not to pay coordination fee again)
    # - Maximize impact of X% presence (WW2 outputs computation deviation)
    # Have X% control of all standard output denominations
    # (=> for whirlpool, have X% of all active remixing liquidity => will be selected )

    # TODO: Plot graph of remix rates (values, num_inputs) as line plot for all months into single graph

    # TODO: Recompute fresh inflows for post-zksnacks coordinators


# b71981909c440bcb29e9a2d1cde9992cc97d3ca338c925c4b0547566bdc62f4d
# ec9d5c2c678a70e304fa6e06a6430c9aff49e75107ac33f10165b14f0fa9a1f4
# f872a419a48578389994323e6ee565ba15f4b9f71e72fceabc6a866505d13a6f

# Initial transaction for some new wasabi2 pool (inputs are non-cjtx): cdb245e4981d140f0a3a56431c374f593782aa3bef0cfb3abe733cbc5849a243
# Search for previous cjtxs inputs for small pools:
#   db65f85f4ddb2feb4ffaa1d8eb1485b46329bdc291bc965b5c6b3e4ab5edf2ff
#   d6b7798869f4eb147e524d75d204a9476576465695bdca070711f47ebe838c82
#   3106e3766f95cb4964c36bdf3802dbd68bdc3fe82851ccd8f1a273db2f7fa84d

# Search for subsequent cjtxs for small pools:
#   607bc2b8e8cf3498885d0e908e134f3900d49e97efb96ea2ef65b5c676b6d49a
#   7f31565b9da80406d9994d9b35e71d921d19d3d5bebb9f0802d00908b9620408
#   b9857ec5dc86ed867f0329fd6982767fdc0f5d188df896c85ec9dcf2e3202952
#   ...
#   3106e3766f95cb4964c36bdf3802dbd68bdc3fe82851ccd8f1a273db2f7fa84d

# Strange false positives?
# 3106e3766f95cb4964c36bdf3802dbd68bdc3fe82851ccd8f1a273db2f7fa84d


#   Clever consolidation: 349f27c3104984f2668f981283695b81ce96a4ee5d984f8df26ee92c52dc6fe4

# cjtx with no output remixes (possibly end of coordinator): https://mempool.space/tx/22f64af816772533696b15677b00b780acff6fe39cd09b98d84ab95bb3c46c3a
#

# WW1 last cjtx?
# 2023-07-13 11:27:08 635fa30bfb56b6f24f6474142a57ee58306a98b9c2887ee8a799ccb4fea4a219 0.10143340


# WW1 paralell early coinjoin coordinator :
# start 2018-08-02 15:57:32   38a83a9766357871a77992ecaead52f70c5f9f703769e6ebd4dcdb05172b28a9
# end 2019-01-02 12:57:09 db73c667fd25aa6cf56a24cd4909d3d4b28479f79ba6ec86fe91125dc12e2022
# Then large consolidations

# Coinjoin-looking transaction, but having just 0.5 outputs eventually which are then spread again
# a35b759d3cc0ebda98b4be110498d50cb0a270a1053fe5ab910e5b350950255c

# Strange coinjoin-like transactions with consolidation of several subsequent outputs (=> same user might have own outputs sorted together?)
# 70e7cbbe816aa538b801600681e6213260eaf2849e111da82dbe98c303d14bc3