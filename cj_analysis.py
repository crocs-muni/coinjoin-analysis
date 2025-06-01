import logging
import os
import subprocess
from collections import Counter
import sqlite3, pathlib

import msgpack
import orjson
import json
import time
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
import re
import math
from  txstore import TxStore, TxStoreMsgPack

from bitcoin.core import CTransaction, CMutableTransaction, CTxWitness
# from bitcoin.core import CScript, x
# from bitcoin import SelectParams
# from bitcoin.core.script import OP_HASH160, OP_EQUAL
# from bitcoin.wallet import P2WPKHBitcoinAddress, CBitcoinAddressError, P2SHBitcoinAddress, P2WSHBitcoinAddress

from scipy.optimize import minimize

from bitcoinlib.transactions import Output


SATS_IN_BTC = 100000000

SORT_COINJOINS_BY_RELATIVE_ORDER = True  # If True then relative ordering of transactions based on remix connections

PERF_USE_COMPACT_CJTX_STRUCTURE = False
PERF_USE_SHORT_TXID = False
PERF_TX_SHORT_LEN = 16

avg_input_ratio = {'all': [], 'per_interval': {}}

class CJ_LOG_TYPES(Enum):
    ROUND_STARTED = 'ROUND_STARTED'
    BLAME_ROUND_STARTED = 'BLAME_ROUND_STARTED'
    COINJOIN_BROADCASTED = 'COINJOIN_BROADCASTED'
    INPUT_BANNED = 'INPUT_BANNED'
    NOT_ENOUGH_FUNDS = 'NOT_ENOUGH_FUNDS'
    NOT_ENOUGH_PARTICIPANTS = 'NOT_ENOUGH_PARTICIPANTS'
    WRONG_PHASE = 'WRONG_PHASE'
    MISSING_PHASE_BY_TIME = 'MISSING_PHASE_BY_TIME'
    SIGNING_PHASE_TIMEOUT = 'SIGNING_PHASE_TIMEOUT'
    ALICE_REMOVED = 'ALICE_REMOVED'
    FILLED_SOME_ADDITIONAL_INPUTS = 'FILLED_SOME_ADDITIONAL_INPUTS'
    UTXO_IN_PRISON = 'UTXO_IN_PRISON'


class CJ_ALICE_TYPES(Enum):
    ALICE_REGISTERED = 'ALICE_REGISTERED'
    ALICE_CONNECTION_CONFIRMED = 'ALICE_CONNECTION_CONFIRMED'
    ALICE_READY_TO_SIGN = 'ALICE_READY_TO_SIGN'
    ALICE_POSTED_SIGNATURE = 'ALICE_POSTED_SIGNATURE'


class PRECOMP_STRPTIME:
    precomp_strptime = {}
    precomp_strftime = {}

    def strptime(self, datestr: str, datestr_format: str) -> datetime:
        if datestr not in self.precomp_strptime:
            self.precomp_strptime[datestr] = datetime.strptime(datestr, datestr_format)
        return self.precomp_strptime[datestr]

    def fromisoformat(self, datestr: str) -> datetime:
        if datestr not in self.precomp_strptime:
            self.precomp_strptime[datestr] = datetime.fromisoformat(datestr)
        return self.precomp_strptime[datestr]

    def strftime(self, dt: datetime) -> str:
        if dt not in self.precomp_strftime:
            self.precomp_strftime[dt] = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        return self.precomp_strftime[dt]


precomp_datetime = PRECOMP_STRPTIME()


def load_json_from_file(file_path: str) -> dict:
    with open(file_path, "rb") as file:
        return orjson.loads(file.read())

    # with open(file_path, "r") as file:
    #     return json.load(file)


def save_json_to_file(file_path: str, data: dict):
    with open(file_path, "wb") as file:
        file.write(orjson.dumps(data))

    # with open(file_path, "w") as file:
    #     file.write(json.dumps(dict(sorted(data.items())), indent=4))


def save_json_to_file_pretty(file_path: str, data: dict, sort: bool = False):
    with open(file_path, "w") as file:
        if sort:
            file.write(json.dumps(dict(sorted(data.items())), indent=4))
        else:
            file.write(json.dumps(data, indent=4))


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


class MIX_EVENT_TYPE(Enum):
    MIX_ENTER = 'MIX_ENTER'  # New liquidity coming to mix
    MIX_LEAVE = 'MIX_LEAVE'  # Liquidity leaving mix (postmix spend)
    MIX_REMIX = 'MIX_REMIX'  # Remixed value within mix
    MIX_REMIX_FRIENDS = 'MIX_REMIX_FRIENDS'  # Remixed value within mix, but not directly, but one hop friends (WW2)
    MIX_REMIX_FRIENDS_WW1 = 'MIX_REMIX_FRIENDS_WW1'  # Remixed value from WW1 mix (only for WW2)
    MIX_STAY = 'MIX_STAY'    # Mix output not yet spend (may be remixed or leave mix later)


class MIX_PROTOCOL(Enum):
    UNSET = 'UNSET'  # not set yet
    WASABI1 = 'WASABI1'  # Wasabi 1.0
    WASABI2 = 'WASABI2'  # Wasabi 2.0
    WHIRLPOOL = 'WHIRLPOOL'  # Whirlpool


class SummaryMessages:
    summary_messages = []

    def print(self, message: str):
        logging.info(message)
        self.summary_messages.append(message)

    def print_summary(self):
        print(f'Total log messages: {len(self.summary_messages)}')
        for message in self.summary_messages:
            print(message)


SM = SummaryMessages()


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
        return f'{numerator}/{denominator} ({get_ratio(numerator, denominator)}%)'
    else:
        return f'{numerator}/{0} (0%)'


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


def plot_inputs_type_ratio(mix_id: str, data: dict, initial_cj_index: int, ax, analyze_values: bool, normalize_values: bool, restrict_to_in_size: (int, int) = None):
    """
    Ratio between various types of inputs (fresh, remixed, remixed_friends)
    :param mix_id:
    :param data:
    :param ax:
    :param analyze_values if true, then size of inputs is analyzed, otherwise only numbers
    :return:
    """
    SM.print(f'plot_inputs_type_ratio(mix_id={mix_id}, analyze_values={analyze_values}, normalize_values={normalize_values})')

    coinjoins = data['coinjoins']
    sorted_cj_time = sort_coinjoins(coinjoins, SORT_COINJOINS_BY_RELATIVE_ORDER)
    #sorted_cj_time = sorted_cj_time[0:500]

    if restrict_to_in_size is None:
        restrict_to_in_size = (0, 1000000000000)
        print(f'No limits for inputs value')
    else:
        print(f'Limits for inputs value is {restrict_to_in_size[0]} - {restrict_to_in_size[1]}')

    input_types_nums = {}
    for event_type in MIX_EVENT_TYPE:
        if analyze_values:
            # Sum of values of inputs is taken
            input_types_nums[event_type.name] = [sum([coinjoins[cjtx['txid']]['inputs'][index]['value'] for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                        if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == event_type.name and
                                                      restrict_to_in_size[0] <= coinjoins[cjtx['txid']]['inputs'][index]['value'] <= restrict_to_in_size[1]])
                                            for cjtx in sorted_cj_time]
        else:
            # Only number of inputs is taken
            input_types_nums[event_type.name] = [sum([1 for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                        if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == event_type.name and
                                                      restrict_to_in_size[0] <= coinjoins[cjtx['txid']]['inputs'][index]['value'] <= restrict_to_in_size[1]])
                                   for cjtx in sorted_cj_time]

    # Obtain vector number of inputs/values for each remix, based on burn time
    # First take remixes with standard denominations
    event_type = MIX_EVENT_TYPE.MIX_REMIX
    BURN_TIME_RANGES = [('1', 1, 1), ('2', 2, 2), ('3-5', 3, 5), ('6-19', 6, 19), ('20+', 20, 999), ('1000-1999', 1000, 1999), ('2000+', 2000, 1000000)]
    for range_val in BURN_TIME_RANGES:
        input_types_nums[f'{event_type.name}_{range_val[0]}'] = get_inputs_type_list(coinjoins, sorted_cj_time, event_type, 'inputs', range_val[1], range_val[2], analyze_values, restrict_to_in_size, True)
    # Add remixes of non-standard denominations ("change" outputs)
    input_types_nums['MIX_REMIX_nonstd'] = get_inputs_type_list(coinjoins, sorted_cj_time, MIX_EVENT_TYPE.MIX_REMIX, 'inputs', 1, 10000000, analyze_values, restrict_to_in_size, False)

    short_exp_name = mix_id

    # Normalize all values into range 0-1 (only MIX_ENTER, MIX_REMIX and MIX_REMIX_FRIENDS are considered for base total)
    input_types_nums_normalized = {}
    total_values = (np.array(input_types_nums[MIX_EVENT_TYPE.MIX_ENTER.name]) + np.array(input_types_nums[MIX_EVENT_TYPE.MIX_REMIX.name]) +
                    np.array(input_types_nums[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name]) + np.array(input_types_nums[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name]))
    # Normalize all values including 'MIX_REMIX_1-2' etc.
    for item in input_types_nums.keys():
        input_types_nums_normalized[item] = np.array(input_types_nums[item]) / total_values

    def print_inputs_stats(input_types: dict, start_offset: int = 0, end_offset: int = -1):
        SM.print(f'  MIX_ENTER median ratio: {round(np.median(input_types[MIX_EVENT_TYPE.MIX_ENTER.name][start_offset: end_offset]) * 100, 2)}%')
        SM.print(f'  MIX_REMIX_nonstd median ratio: {round(np.median(input_types["MIX_REMIX_nonstd"][start_offset: end_offset]) * 100, 2)}%')
        SM.print(f'  MIX_REMIX median ratio: {round(np.median(input_types[MIX_EVENT_TYPE.MIX_REMIX.name][start_offset: end_offset]) * 100, 2)}%')
        for range_val in BURN_TIME_RANGES:
            remix_name = f'{event_type.name}_{range_val[0]}'
            print(f'  {remix_name} median ratio: {round(np.median(input_types[remix_name][start_offset: end_offset]) * 100, 2)}%')
        SM.print(f'  MIX_REMIX_FRIENDS median ratio: {round(np.median(input_types[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name][start_offset: end_offset]) * 100, 2)}%')
        SM.print(f'  MIX_REMIX_FRIENDS_WW1 median ratio: {round(np.median(input_types[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name][start_offset: end_offset]) * 100, 2)}%')

    SM.print(f' Inputs ratios [all]')
    print_inputs_stats(input_types_nums_normalized)
    SM.print(f' Inputs ratios [skip first two]')
    print_inputs_stats(input_types_nums_normalized, 2, -1)
    SM.print(f' Inputs ratios [skip first five]')
    print_inputs_stats(input_types_nums_normalized, 5, -1)

    # Convert non-normalized values from sats to btc (for sats values only)
    if analyze_values:
        for item in input_types_nums.keys():
            input_types_nums[item] = np.array(input_types_nums[item]) / SATS_IN_BTC

    # Set normalized or non-normalized version to use
    input_types = input_types_nums_normalized if normalize_values else input_types_nums

    bar_width = 0.3
    categories = range(0, len(sorted_cj_time))

    # New version with separated remixes
    bars = []
    bars.append((input_types[MIX_EVENT_TYPE.MIX_ENTER.name], 'MIX_ENTER', 'blue', 0.9))
    bars.append((input_types['MIX_REMIX_nonstd'], 'MIX_REMIX_nonstd', 'blue', 0.3))
    bars.append((input_types['MIX_REMIX_1'], 'MIX_REMIX_1', 'gold', 0.8))
    bars.append((input_types['MIX_REMIX_2'], 'MIX_REMIX_2', 'orange', 0.4))
    bars.append((input_types['MIX_REMIX_3-5'], 'MIX_REMIX_3-5', 'orange', 0.8))
    bars.append((input_types['MIX_REMIX_6-19'], 'MIX_REMIX_6-19', 'moccasin', 0.5))
    bars.append((input_types['MIX_REMIX_20+'], 'MIX_REMIX_20+', 'lightcoral', 0.7))
    bars.append((input_types['MIX_REMIX_1000-1999'], 'MIX_REMIX_1000-1999', 'sienna', 0.7))
    bars.append((input_types['MIX_REMIX_2000+'], 'MIX_REMIX_2000+', 'sienna', 1))
    bars.append((input_types[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name], 'MIX_REMIX_FRIENDS', 'green', 0.5))
    bars.append((input_types[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name], 'MIX_REMIX_FRIENDS_WW1', 'green', 0.9))

    # Draw all inserted bars atop each other
    if ax:
        bar_bottom = None
        for bar_item in bars:
            if bar_bottom is None:
                ax.bar(categories, bar_item[0], bar_width, label=f'{bar_item[1]} {short_exp_name}', alpha=bar_item[3],
                       color=bar_item[2], linewidth=0)
                bar_bottom = np.array(bar_item[0])
            else:
                ax.bar(categories, bar_item[0], bar_width, label=f'{bar_item[1]} {short_exp_name}', alpha=bar_item[3], color=bar_item[2],
                        bottom=bar_bottom, linewidth=0)
                bar_bottom = bar_bottom + np.array(bar_item[0])

        ax.set_title(f'Type of inputs for given cjtx ({"values" if analyze_values else "number"})\n{short_exp_name}')
        ax.set_xlabel('Coinjoin in time')
        if analyze_values and normalize_values:
            ax.set_ylabel('Fraction of inputs values')
        if analyze_values and not normalize_values:
            ax.set_ylabel('Inputs values (btc)')
        if not analyze_values and normalize_values:
            ax.set_ylabel('Fraction of input numbers')
        if not analyze_values and not normalize_values:
            ax.set_ylabel('Number of inputs')

    PLOT_REMIX_RATIO = False
    if PLOT_REMIX_RATIO:
        WINDOWS_SIZE = 1
        remix_ratios_all = [input_types[MIX_EVENT_TYPE.MIX_REMIX.name][i] * 100 for i in range(len(input_types[MIX_EVENT_TYPE.MIX_REMIX.name]))]  # All remix including nonstandard
        remix_ratios_nonstd = [input_types['MIX_REMIX_nonstd'][i] * 100 for i in range(len(input_types['MIX_REMIX_nonstd']))]  # Nonstd remixes
        remix_ratios_std = [remix_ratios_all[i] - remix_ratios_nonstd[i] for i in range(len(remix_ratios_all))]  # Only standard remixes
        remix_ratios_avg = [np.average(remix_ratios_std[i:i+WINDOWS_SIZE]) for i in range(0, len(remix_ratios_std), WINDOWS_SIZE)]
        if ax:
            ax2 = ax.twinx()
            ax2.plot(range(0, len(remix_ratios_avg), WINDOWS_SIZE), remix_ratios_avg, label=f'MIX_REMIX avg({WINDOWS_SIZE})', color='brown', linewidth=1, linestyle='--', alpha=0.4)
            ax2.set_ylim(0, 100)  # Force whole range of yaxis
            ax2.tick_params(axis='y', colors='brown', labelsize=6)
            ax2.set_ylabel('Average remix rate %', color='brown', fontsize='6')

    return input_types


def plot_mix_liquidity(mix_id: str, data: dict, initial_liquidity, time_liquidity: dict, initial_cj_index: int, ax):
    coinjoins = data['coinjoins']
    sorted_cj_time = sort_coinjoins(coinjoins, SORT_COINJOINS_BY_RELATIVE_ORDER)

    # New fresh liquidity
    mix_enter = [sum([coinjoins[cjtx['txid']]['inputs'][index]['value'] for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_ENTER.name])
                           for cjtx in sorted_cj_time]
    # Input liquidity from friends (one hop remix)
    mix_remixfriend = [sum([coinjoins[cjtx['txid']]['inputs'][index]['value'] for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name])
                           for cjtx in sorted_cj_time]
    # Input liquidity from ww1
    mix_remixfriend_ww1 = [sum([coinjoins[cjtx['txid']]['inputs'][index]['value'] for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name])
                           for cjtx in sorted_cj_time]
    # Output spent outside mix
    mix_leave = [sum([coinjoins[cjtx['txid']]['outputs'][index]['value'] for index in coinjoins[cjtx['txid']]['outputs'].keys()
                                    if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_LEAVE.name])
                               for cjtx in sorted_cj_time]

    # Output staying in mix
    mix_stay = [sum([coinjoins[cjtx['txid']]['outputs'][index]['value'] for index in coinjoins[cjtx['txid']]['outputs'].keys()
                                    if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_STAY.name])
                               for cjtx in sorted_cj_time]

    INTERVAL_LENGTH = 3 * 30 * 24 * 3600  # 3 months == 3 * 30 * 24 * 3600
    INTERVAL_LENGTH = 30 * 24 * 3600  # 1 month == * 30 * 24 * 3600
    # Outputs leaving mix `fast` after its mixing (within 0-INTERVAL_LENGTH seconds)
    mix_leave_timecutoff_before = [sum([coinjoins[cjtx['txid']]['outputs'][index]['value'] for index in coinjoins[cjtx['txid']]['outputs'].keys()
                                    if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_LEAVE.name and
                                      coinjoins[cjtx['txid']]['outputs'][index]['burn_time'] < INTERVAL_LENGTH])
                               for cjtx in sorted_cj_time]
    COMPUTE_UNUSED = False
    if COMPUTE_UNUSED:
        # Outputs leaving mix `slow` after its mixing (at least after INTERVAL_LENGTH seconds)
        mix_leave_timecutoff_after = [sum([coinjoins[cjtx['txid']]['outputs'][index]['value'] for index in coinjoins[cjtx['txid']]['outputs'].keys()
                                        if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_LEAVE.name and
                                          coinjoins[cjtx['txid']]['outputs'][index]['burn_time'] >= INTERVAL_LENGTH])
                                   for cjtx in sorted_cj_time]

        # Output staying in mix MIX_EVENT_TYPE.MIX_REMIX
        mix_remix = [sum([coinjoins[cjtx['txid']]['outputs'][index]['value'] for index in coinjoins[cjtx['txid']]['outputs'].keys()
                                        if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name])
                                   for cjtx in sorted_cj_time]
        # Output staying in mix MIX_EVENT_TYPE.MIX_REMIX with non-standard values
        mix_remix_nonstandard = [sum([coinjoins[cjtx['txid']]['outputs'][index]['value'] for index in coinjoins[cjtx['txid']]['outputs'].keys()
                                        if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name and
                                           coinjoins[cjtx['txid']]['outputs'][index].get('is_standard_denom') == False])
                                   for cjtx in sorted_cj_time]


    cjtx_cummulative_liquidity = []
    cjtx_cummulative_liquidity_timecutoff = []
    curr_liquidity = initial_liquidity[0]  # Take last cummulative liquidity (MIX_ENTERxxx - MIX_LEAVE) from previous interval
    curr_liquidity_timecutoff = initial_liquidity[3]
    assert len(mix_enter) == len(mix_leave) == len(mix_remixfriend) == len(mix_remixfriend_ww1) == len(mix_stay), logging.error(f'Mismatch in length of input/out sum arrays: {len(mix_enter)} vs. {len(mix_leave)}')
    # Change in liquidity as observed by each coinjoin (increase directly when mix_enter, decrease directly even when mix_leave happens later in wall time)
    for index in range(0, len(mix_enter)):
        liquidity_step = mix_enter[index] + mix_remixfriend[index] + mix_remixfriend_ww1[index] - mix_leave[index]
        # Print significant changes in liquidity for easier debugging
        if mix_enter[index] > 100 * SATS_IN_BTC:
            print(f'Fresh input jump    of {round(mix_enter[index] / SATS_IN_BTC, 1)} at {index}: {sorted_cj_time[index]}')
        if liquidity_step > 100 * SATS_IN_BTC:
            print(f'Pool liquidity jump of {round(liquidity_step / SATS_IN_BTC, 1)} at {index}: {sorted_cj_time[index]}')
        curr_liquidity = curr_liquidity + liquidity_step
        cjtx_cummulative_liquidity.append(curr_liquidity)

        # Same computation, but assume as leaving only mix_leave_timecutoff value
        # time-limited value (< INTERVAL_LENGTH)
        curr_liquidity_timecutoff = curr_liquidity_timecutoff + liquidity_step + mix_leave[index] - mix_leave_timecutoff_before[index]
        cjtx_cummulative_liquidity_timecutoff.append(curr_liquidity_timecutoff)

    # Cumulative liquidity never remixed or leaving mix (MIX_STAY coins)
    stay_liquidity = []
    stay_liquidity_timecutoff = []
    curr_stay_liquidity = initial_liquidity[1]  # Take last cumulative liquidity (MIX_STAY) from previous interval
    curr_stay_liquidity_timecutoff = initial_liquidity[4]  # Take last cumulative liquidity (MIX_STAY) from previous interval
    for index in range(0, len(mix_stay)):
        curr_stay_liquidity = curr_stay_liquidity + mix_stay[index]
        stay_liquidity.append(curr_stay_liquidity)

        # time-limited value (=> INTERVAL_LENGTH)
        #curr_stay_liquidity_timecutoff = curr_stay_liquidity_timecutoff + mix_stay_timecutoff_before[index]
        #stay_liquidity_timecutoff.append(curr_stay_liquidity_timecutoff)

    # Remixed liquidity levels
    remix_liquidity = []
    curr_remix_liquidity = initial_liquidity[2]  # Take last remix liquidity from previous interval
    for index in range(0, len(mix_stay)):
        remix_liquidity_step = mix_enter[index] + mix_remixfriend[index] + mix_remixfriend_ww1[index] - mix_leave[index] - stay_liquidity[index]  # prev state + new input liquidity - output liqudity
        # BUGBUG: We must also consider exact evaluation of mining fee payed to get perfect match for the assert below
        #assert mix_remix[index] == remix_liquidity_step, f'Inconsistent remix liquidity estimation for {index}th coinjoin ({sorted_cj_time[index]['txid']}); Expected {mix_remix[index]} got {remix_liquidity_step[index]}'
        curr_remix_liquidity = curr_remix_liquidity + remix_liquidity_step
        remix_liquidity.append(curr_remix_liquidity)

    # Plot in btc
    liquidity_btc = [item / SATS_IN_BTC for item in cjtx_cummulative_liquidity]
    liquidity_timecutoff_btc = [item / SATS_IN_BTC for item in cjtx_cummulative_liquidity_timecutoff]
    stay_liquidity_btc = [item / SATS_IN_BTC for item in stay_liquidity]
    remix_liquidity_btc = [item / SATS_IN_BTC for item in remix_liquidity]
    if ax:
        #x_ticks = range(initial_cj_index, initial_cj_index + len(liquidity_btc))
        ax.plot(liquidity_btc, color='royalblue', alpha=0.6, linewidth=3)
        #ax.plot(stay_liquidity_btc, color='royalblue', alpha=0.6, linestyle='--')
        #ax.plot(remix_liquidity_btc, color='black', alpha=0.6, linestyle='--')
        PLOT_LEAVE_TIMECUTOFF = False
        if PLOT_LEAVE_TIMECUTOFF:
            ax.plot(liquidity_timecutoff_btc, color='blue', alpha=0.6, linestyle='--')

        ax.set_ylabel('btc in mix', color='royalblue')
        ax.tick_params(axis='y', colors='royalblue')

    return cjtx_cummulative_liquidity, stay_liquidity, remix_liquidity, cjtx_cummulative_liquidity_timecutoff, stay_liquidity_timecutoff


def plot_mining_fee_rates(mix_id: str, data: dict, mining_fees: dict, ax):
    coinjoins = data['coinjoins']
    # Take real mining time as mining fee are more relevant to it, but adapt to relative ordering used for plotting
    sorted_cj_fee_time = sort_coinjoins(coinjoins, False)  # Real time of mining => time of minig fee rate application
    sorted_cj_fee_time_dict = {cj['txid']: cj for cj in sorted_cj_fee_time}  # Turn list into dict for faster lookups
    sorted_cj_time = sort_coinjoins(coinjoins, False)  # Take relative ordering of cjtxs

    # For each coinjoin find the closest fee rate record and plot it
    fee_rates = []
    fee_start_index = 0
    for cj in sorted_cj_time:
        timestamp = sorted_cj_fee_time_dict[cj['txid']]['broadcast_time'].timestamp()
        while timestamp > mining_fees[fee_start_index]['timestamp']:
            fee_start_index = fee_start_index + 1
            if fee_start_index >= len(mining_fees):
                logging.error(f'Missing mining_fees entry for timestamp {sorted_cj_fee_time_dict[cj["txid"]]["broadcast_time"]} if {cj["txid"]}.')
                # Use the latest one and stop searching
                fee_start_index = fee_start_index - 1
                break

        closest_fee = mining_fees[fee_start_index - 1]['avgFee_90']
        fee_rates.append(closest_fee)

    if ax:
        ax.plot(fee_rates, color='gray', alpha=0.4, linewidth=1, linestyle='--')
        ax.tick_params(axis='y', colors='gray', labelsize=6)
        ax.set_ylabel('Mining fee rate sats/vB (90th percentil)', color='gray', fontsize='6', labelpad=-2)

    return fee_rates


def  get_wallets_prediction_ratios(mix_id: str):
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

    # zksnacks
    if 'zksnacks' in mix_id:
        AVG_NUM_OUTPUTS = 4.17 # real value taken from zksnacks as25 experiment (use for zksnacks)
        AVG_NUM_INPUTS = 2.72  # real value taken from zksnacks as25 experiment (use for zksnacks)
#        AVG_NUM_OUTPUTS = 2.91 # synthetic median value minimizing euclidean distance between output and input factors for zksnacks if AVG_NUM_INPUTS = 2.72

    # Wasabi 1.x
    if 'wasabi1' in mix_id:
        AVG_NUM_OUTPUTS = 2.0 # real value taken from wasabi1 experiments (one standard denomination, one change output)
        AVG_NUM_INPUTS = 1.15  # synthetic value

    # Whirlpool
    if 'whirlpool' in mix_id:
        AVG_NUM_OUTPUTS = 1 # real value taken from implementation of Whirlpool clients
        AVG_NUM_INPUTS = 1  # real value taken from implementation of Whirlpool clients


    return AVG_NUM_INPUTS, AVG_NUM_OUTPUTS


def plot_num_wallets(mix_id: str, data: dict, ax):
    coinjoins = data['coinjoins']
    sorted_cj_time = sort_coinjoins(coinjoins, SORT_COINJOINS_BY_RELATIVE_ORDER)

    if mix_id not in avg_input_ratio:
        avg_input_ratio['per_interval'][mix_id] = {}

    # Naive approach: For each coinjoin, compute as number of inputs divided by average inputs per wallet
    AVG_NUM_INPUTS, AVG_NUM_OUTPUTS = get_wallets_prediction_ratios(mix_id)

    # Find heuristically the AVG_NUM_INPUTS and AVG_NUM_OUTPUTS to minimize difference between computed number of inputs and outputs
    FIND_SYNTHETIC_RATIO = True
    if FIND_SYNTHETIC_RATIO:
        avg_input_ratio['factor_inputs_wallets'] = AVG_NUM_INPUTS
        avg_input_ratio['factor_outputs_wallets'] = AVG_NUM_OUTPUTS

        X = np.array([len(coinjoins[cj['txid']]['inputs']) for cj in sorted_cj_time])
        Y = np.array([len(coinjoins[cj['txid']]['outputs']) for cj in sorted_cj_time])

        # Objective function to minimize
        def objective_euclidean(params):
            x1, y1 = params
            return np.sum((X / x1 - Y / y1) ** 2)

        def objective_linear(params):
            x1, y1 = params
            return np.sum(np.abs(X / x1 - Y / y1))

        # Initial guess for x1 and y1
        initial_guess = [1, 1]
        # Minimize the objective function
        result = minimize(objective_linear, initial_guess, method='Nelder-Mead')
        # Optimal values
        x1_opt, y1_opt = result.x
        #AVG_NUM_OUTPUTS = AVG_NUM_INPUTS * (y1_opt / x1_opt)
        AVG_NUM_INPUTS = AVG_NUM_OUTPUTS * (x1_opt / y1_opt)
        print(f"Ratio y1/x1: {y1_opt / x1_opt}, Optimal x1: {x1_opt}, Optimal y1: {y1_opt}")
        print(f"AVG_NUM_OUTPUTS = {AVG_NUM_OUTPUTS} factor => AVG_NUM_INPUTS = {AVG_NUM_INPUTS} factor after scaling")
        avg_input_ratio['per_interval'][mix_id]['ratio_factor_outputs_inputs_wallets'] = y1_opt / x1_opt
        avg_input_ratio['per_interval'][mix_id]['factor_inputs_wallets'] = AVG_NUM_INPUTS
        avg_input_ratio['per_interval'][mix_id]['factor_outputs_wallets'] = AVG_NUM_OUTPUTS
        avg_input_ratio['all'].extend([AVG_NUM_INPUTS] * len(sorted_cj_time))

    num_wallets_naive_inputs = [len(coinjoins[cj['txid']]['inputs']) / AVG_NUM_INPUTS for cj in sorted_cj_time]
    num_wallets_naive_outputs = [len(coinjoins[cj['txid']]['outputs']) / AVG_NUM_OUTPUTS for cj in sorted_cj_time]

    # Load from other computed option
    num_wallets_predicted = [coinjoins[cj['txid']].get('num_wallets_predicted', -100) for cj in sorted_cj_time]
    # Set value for missing ones to nearby value
    last_val = 0
    for index in range(0, len(num_wallets_predicted)):
        if num_wallets_predicted[index] == -100:
            num_wallets_predicted[index] = last_val
        else:
            last_val = num_wallets_predicted[index]

    if ax:
        AVG_WINDOWS = 10
        #AVG_WINDOWS = 5
        COLOR_WALLETS_INPUTS = 'red'
        #COLOR_WALLETS_OUTPUTS = 'magenta'
        COLOR_WALLETS_OUTPUTS = 'green'
        num_wallets_avg_inputs = compute_averages(num_wallets_naive_inputs, AVG_WINDOWS)
        num_wallets_avg_inputs = np.array(num_wallets_avg_inputs)
        x = range(AVG_WINDOWS // 2, len(num_wallets_avg_inputs) + AVG_WINDOWS // 2)
        #x = range(AVG_WINDOWS, len(num_wallets_avg_inputs) + AVG_WINDOWS)
        #x = range(0, len(num_wallets_avg_inputs))
        ax.plot(x, num_wallets_avg_inputs, color=COLOR_WALLETS_INPUTS, alpha=0.4, linewidth=2, linestyle='-',
                label=f'Predicted wallets (inputs, avg={AVG_WINDOWS}, factor={round(AVG_NUM_INPUTS, 2)})')
        num_wallets_avg_outputs = compute_averages(num_wallets_naive_outputs, AVG_WINDOWS)
        num_wallets_avg_outputs = np.array(num_wallets_avg_outputs)
        ax.plot(x, num_wallets_avg_outputs, color=COLOR_WALLETS_OUTPUTS, alpha=0.4, linewidth=2, linestyle='-',
                label=f'Predicted wallets (outputs, avg={AVG_WINDOWS}, factor={round(AVG_NUM_OUTPUTS, 2)})')
        ax.tick_params(axis='y', colors=COLOR_WALLETS_INPUTS, labelsize=6)
        ax.fill_between(x, num_wallets_avg_inputs, num_wallets_avg_outputs, where=num_wallets_avg_inputs>num_wallets_avg_outputs, interpolate=True, color=COLOR_WALLETS_INPUTS, alpha=0.3)
        ax.fill_between(x, num_wallets_avg_inputs, num_wallets_avg_outputs, where=num_wallets_avg_outputs>num_wallets_avg_inputs, interpolate=True, color=COLOR_WALLETS_OUTPUTS, alpha=0.3)
        max_wallets_y = max(max(num_wallets_avg_outputs), max(num_wallets_avg_inputs))
        ax.set_yticks(np.arange(0, max_wallets_y + round(max_wallets_y * 0.1), step=10))

        #ax.set_ylabel('Estimated number of active wallets (naive)', color='red', fontsize='6')
        PLOT_WALLETS_PREDICTED = False
        if PLOT_WALLETS_PREDICTED:
            num_wallets_avg_predicted = compute_averages(num_wallets_predicted, AVG_WINDOWS)
            ax.plot(num_wallets_avg_predicted, color='green', alpha=0.4, linewidth=1, linestyle='-', label='Predicted wallets (model)')
            ax.tick_params(axis='y', colors='green', labelsize=6)
            ax.set_ylabel('Estimated number of active wallets (model)', color='green', fontsize='6')

        ax.legend()

    return num_wallets_predicted


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


def print_liquidity_summary(coinjoins: dict, mix_id: str):
    total_inputs_len = [len(coinjoins[cjtx]['inputs']) for cjtx in coinjoins.keys()]
    total_inputs = [coinjoins[cjtx]['inputs'][input]['value'] for cjtx in coinjoins.keys() for input in coinjoins[cjtx]['inputs']]
    total_outputs_len = [len(coinjoins[cjtx]['outputs']) for cjtx in coinjoins.keys()]
    total_inputs_number = len(total_inputs)
    total_inputs_value = sum(total_inputs)

    total_outputs = [coinjoins[cjtx]['outputs'][output]['value'] for cjtx in coinjoins.keys() for output in coinjoins[cjtx]['outputs']]
    total_outputs_number = len(total_outputs)
    total_outputs_value = sum(total_outputs)

    total_mix_entering = [coinjoins[cjtx]['inputs'][input]['value'] for cjtx in coinjoins.keys() for input in coinjoins[cjtx]['inputs']
                             if coinjoins[cjtx]['inputs'][input]['mix_event_type'] == MIX_EVENT_TYPE.MIX_ENTER.name]
    total_mix_entering_number = len(total_mix_entering)
    total_mix_entering_value = sum(total_mix_entering)

    total_mix_friends = [coinjoins[cjtx]['inputs'][input]['value'] for cjtx in coinjoins.keys() for input in coinjoins[cjtx]['inputs']
                             if coinjoins[cjtx]['inputs'][input]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name]
    total_mix_friends_ww1 = [coinjoins[cjtx]['inputs'][input]['value'] for cjtx in coinjoins.keys() for input in coinjoins[cjtx]['inputs']
                             if coinjoins[cjtx]['inputs'][input]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name]
    total_mix_friends_number = len(total_mix_friends) + len(total_mix_friends_ww1)
    total_mix_friends_value = sum(total_mix_friends) + sum(total_mix_friends_ww1)

    total_mix_remix = [coinjoins[cjtx]['inputs'][input]['value'] for cjtx in coinjoins.keys() for input in coinjoins[cjtx]['inputs']
                             if coinjoins[cjtx]['inputs'][input]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name]
    total_mix_remix_number = len(total_mix_remix)
    total_mix_remix_value = sum(total_mix_remix)

    total_mix_remix_out = [coinjoins[cjtx]['outputs'][input]['value'] for cjtx in coinjoins.keys() for input in coinjoins[cjtx]['outputs']
                             if coinjoins[cjtx]['outputs'][input]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name]
    total_mix_remix_out_number = len(total_mix_remix_out)
    total_mix_remix_out_value = sum(total_mix_remix_out)

    total_mix_leaving = [coinjoins[cjtx]['outputs'][output]['value'] for cjtx in coinjoins.keys() for output in coinjoins[cjtx]['outputs']
                             if coinjoins[cjtx]['outputs'][output]['mix_event_type'] == MIX_EVENT_TYPE.MIX_LEAVE.name]
    total_mix_leaving_number = len(total_mix_leaving)
    total_mix_leaving_value = sum(total_mix_leaving)

    total_mix_leaving_nonstd = [coinjoins[cjtx]['outputs'][output]['value'] for cjtx in coinjoins.keys() for output in coinjoins[cjtx]['outputs']
                             if coinjoins[cjtx]['outputs'][output]['mix_event_type'] == MIX_EVENT_TYPE.MIX_LEAVE.name and coinjoins[cjtx]['outputs'][output]['is_standard_denom'] == False]
    total_mix_leaving_nonstd_number = len(total_mix_leaving_nonstd)
    total_mix_leaving_nonstd_value = sum(total_mix_leaving_nonstd)


    total_mix_staying = [coinjoins[cjtx]['outputs'][output]['value'] for cjtx in coinjoins.keys() for output in coinjoins[cjtx]['outputs']
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

    # Print summary results
    SM.print(f"  Earliest broadcast: {earliest_time} from {earliest_cjtx}")
    SM.print(f"  Latest broadcast: {latest_time} from {latest_cjtx}")
    SM.print(f'  Total coinjoin transactions: {len(coinjoins.keys())}')
    SM.print(f'  Number of inputs: min={min(total_inputs_len)}, max={max(total_inputs_len)}, avg={np.average(total_inputs_len)}, median={np.median(total_inputs_len)}')
    SM.print(f'  Number of outputs: min={min(total_outputs_len)}, max={max(total_outputs_len)}, avg={np.average(total_outputs_len)}, median={np.median(total_outputs_len)}')

    SM.print(f'  {get_ratio_string(total_mix_entering_number, total_inputs_number)} Inputs entering mix / total inputs used by mix transactions')
    SM.print(f'  {get_ratio_string(total_mix_friends_number, total_inputs_number)} Friends inputs re-entering mix / total inputs used by mix transactions')
    SM.print(f'  {get_ratio_string(total_mix_leaving_number, total_outputs_number)} Outputs leaving mix / total outputs by mix transactions')
    SM.print(f'  {get_ratio_string(total_mix_staying_number, total_outputs_number)} Outputs staying in mix / total outputs by mix transactions')
    SM.print(f'  {get_ratio_string(total_mix_staying_number, total_outputs_number - total_mix_remix_out_number)} Outputs staying in mix / non-remix outputs')
    SM.print(f'  {get_ratio_string(total_mix_remix_number, total_inputs_number)} Inputs remixed / total inputs based on number of inputs')
    SM.print(f'  {get_ratio_string(total_mix_remix_value, total_inputs_value)} Inputs remixed / total inputs based on value of inputs')
    SM.print(f'  {total_mix_entering_value / SATS_IN_BTC} btc, total fresh entering mix')
    SM.print(f'  {total_mix_friends_value / SATS_IN_BTC} btc, total friends entering mix')
    SM.print(f'  {total_mix_staying_value / SATS_IN_BTC} btc, total value staying unmoved in mix')
    SM.print(f'  {total_mix_leaving_value / SATS_IN_BTC} btc, total value leaving mix')
    SM.print(f'  {total_mix_leaving_nonstd_value / SATS_IN_BTC} btc, total non-standard value leaving mix (not mixed)')
    SM.print(f'  {(total_mix_entering_value - total_mix_leaving_nonstd_value) / SATS_IN_BTC} btc, total fresh entering mix without non-standard leaving')

    mix_id_latex = mix_id.replace('_', '\\_' )
    SM.print(f'  \\hline   '
             + f'{mix_id_latex} & {earliest_time}--{latest_time} & '
             + f'{len(coinjoins.keys())} & {total_mix_entering_number} / {round(total_mix_entering_value / SATS_IN_BTC, 1)}~\\bitcoinSymbol' + '{} & '
             + f'{get_ratio(total_mix_remix_value, total_inputs_value)}\\% & '
             + f'{get_ratio(total_mix_staying_number, total_outputs_number - total_mix_remix_out_number)}\\%, {round(total_mix_staying_value / SATS_IN_BTC, 1)}~\\bitcoinSymbol' + '{} & '
             + f'{min(total_inputs_len)} / {round(np.average(total_inputs_len), 1)} / {max(total_inputs_len)} \\\\')


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


def analyze_input_out_liquidity(coinjoins, postmix_spend, premix_spend, mix_protocol: MIX_PROTOCOL, ww1_coinjoins:dict = None, ww1_postmix_spend:dict = None, warn_if_not_found_in_postmix:bool = True):
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
                    assert coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs'] >= 0, f'Invalid burn time computed for {cjtx}:{input}'
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
                    assert coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs'] >= 0, f'Invalid burn time computed for {cjtx}:{output}'

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
    save_json_to_file('tx_reordering_stats.json', difference_counts_str)

    # Print summary results
    print_liquidity_summary(coinjoins, '')
    SM.print(f'  {get_ratio_string(total_mix_entering, total_inputs)} Inputs entering mix / total inputs used by mix transactions')
    SM.print(f'  {get_ratio_string(total_mix_friends, total_inputs)} Friends inputs re-entering mix / total inputs used by mix transactions')
    SM.print(f'  {get_ratio_string(total_mix_leaving, total_outputs)} Outputs leaving mix / total outputs by mix transactions')
    SM.print(f'  {get_ratio_string(len(total_mix_staying), total_outputs)} Outputs staying in mix / total outputs by mix transactions')
    SM.print(f'  {sum(total_mix_staying) / SATS_IN_BTC} btc, total value staying in mix')

    logging.debug('analyze_input_out_liquidity() finished')

    return coinjoins_relative_order


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


def get_output_name_string(txid, index):
    return f'vout_{txid}_{index}'


def get_input_name_string(txid, index):
    return f'vin_{txid}_{index}'


def extract_interval(data: dict, start_date: str, end_date: str):
    interval_data = {}
    if SORT_COINJOINS_BY_RELATIVE_ORDER:
        interval_data['coinjoins'] = {txid: data['coinjoins'][txid] for txid in data['coinjoins'].keys()
                                      if start_date < data['coinjoins'][txid][
                                          'broadcast_time_virtual'] < end_date}
    else:
        interval_data['coinjoins'] = {txid: data['coinjoins'][txid] for txid in data['coinjoins'].keys()
                                      if start_date < data['coinjoins'][txid][
                                          'broadcast_time'] < end_date}
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
                #regex_pattern = "(?P<timestamp>.*) \[INFO\]  obtained tx"
                regex_pattern = r"(?P<timestamp>.*) [INFO]  obtained tx"
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
            #batch.append((txid, orjson.dumps(tx)))
            batch.append((txid, msgpack.packb(tx, use_bin_type=True)))

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
    #data['coinjoins'] = TxStore(db_path)
    data['coinjoins'] = TxStoreMsgPack(db_path)

    # # Filter false positives if required
    # if filter_false_positives:
    #     if false_cjtxs is None:
    #         fp_file = os.path.join(target_load_path, 'false_cjtxs.json')
    #         false_cjtxs = load_json_from_file(fp_file)
    #     for false_tx in false_cjtxs:
    #         if false_tx in data['coinjoins'].keys():
    #             data['coinjoins'].pop(false_tx)

    return data


def load_false_cjtxs_from_file(fp_file):
    false_cjtxs = load_json_from_file(fp_file)
    if PERF_USE_SHORT_TXID:
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
                if als.PERF_USE_SHORT_TXID:
                    tx_id = tx_id[0:als.PERF_TX_SHORT_LEN]
                cjtxs[tx_id] = None

    return cjtxs


def load_coinjoins_from_file(target_load_path: str, false_cjtxs: dict, filter_false_positives: bool) -> dict:
    logging.info(f'Loading {target_load_path}/coinjoin_tx_info.json ...')
    data = load_json_from_file(os.path.join(target_load_path, f'coinjoin_tx_info.json'))

    if PERF_USE_COMPACT_CJTX_STRUCTURE:
        logging.warning(f'IMPORTANT: PERF_USE_COMPACT_CJTX_STRUCTURE==True => compacting in-memory data structure')
        streamline_coinjoins_structure(data)

    # Filter false positives if required
    if filter_false_positives:
        if false_cjtxs is None:
            false_cjtxs = load_false_cjtxs_from_file(os.path.join(target_load_path, 'false_cjtxs.json'))
        for false_tx in false_cjtxs:
            if false_tx in data['coinjoins'].keys():
                data['coinjoins'].pop(false_tx)

    return data


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
    filtered_weight = original_tx.calc_weight() - filtered_tx.calc_weight()
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

# WORKS, but only for limited script types
# def get_address_legacy(script: str, script_type: str):
#     try:
#         SelectParams('mainnet')
#         if script_type.strip().lower() == 'unknown':
#             return None
#
#         scriptPubKey = CScript(x(script))
#
#         if script_type == 'TxWitnessV0Keyhash':
#             return str(P2WPKHBitcoinAddress.from_scriptPubKey(scriptPubKey))
#
#         if script_type == 'Unknown':
#             return str(P2WSHBitcoinAddress.from_scriptPubKey(scriptPubKey))
#
#         if script_type == 'TxScripthash':
#             if (len(scriptPubKey) == 3 and
#                     scriptPubKey[0] == OP_HASH160 and
#                     scriptPubKey[2] == OP_EQUAL):
#                 hash160 = scriptPubKey[1]
#                 return str(P2SHBitcoinAddress.from_scriptPubKey(hash160))
#
#         # If no previous types were hit, return default type
#         return str(P2WPKHBitcoinAddress.from_scriptPubKey(scriptPubKey))
#
#     except CBitcoinAddressError as e:
#         logging.error(f'{script_type}: {e}')
#         return None


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


def generate_tx_download_script(txids: list, file_name):
    curl_lines = []
    for cjtx in txids:
        curl_str = "curl --user user:password --data-binary \'{\"jsonrpc\": \"1.0\", \"id\": \"curltest\", \"method\": \"getrawtransaction\", \"params\": [\"" + cjtx + "\", true]}\' -H \'Content-Type: application/json\' http://127.0.0.1:8332/" + f" > {cjtx}.json\n"
        curl_lines.append(curl_str)
    with open(file_name, 'w') as f:
        f.writelines(curl_lines)


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

    # Use pre-loaded transactions if available
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

        inputs = parsed_data['vin']
        index = 0
        for input in inputs:
            # we need to read and parse previous transaction to obtain address and other information
            in_address, in_full_info = get_input_address(input['txid'], input['vout'], raw_txs)

            tx_record['inputs'][index] = {}
            tx_record['inputs'][index]['address'] = in_address
            tx_record['inputs'][index]['txid'] = input['txid']
            tx_record['inputs'][index]['value'] = int(in_full_info['vout'][input['vout']]['value'] * SATS_IN_BTC)
            tx_record['inputs'][index]['spending_tx'] = get_output_name_string(input['txid'], input['vout'])
            tx_record['inputs'][index]['wallet_name'] = 'real_unknown'

            input_addresses[index] = in_address  # store address to index of the input
            index = index + 1

        outputs = parsed_data['vout']
        for output in outputs:
            index = output['n']
            output_addresses[index] = output['scriptPubKey']['address']
            tx_record['outputs'][index] = {}
            tx_record['outputs'][index]['address'] = output['scriptPubKey']['address']
            tx_record['outputs'][index]['value'] = int(output['value'] * SATS_IN_BTC)
            # tx_record['outputs'][index]['spend_by_tx'] = get_input_name_string(output['txid'], output['vout'])
            tx_record['outputs'][index]['wallet_name'] = 'real_unknown'

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

