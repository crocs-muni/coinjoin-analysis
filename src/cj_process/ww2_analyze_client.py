import copy
import logging
import math
import os
from copy import deepcopy
from enum import Enum
from itertools import chain
from pathlib import Path
import ast
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import CubicSpline
from typing import Iterable, Dict, Tuple, List

import cj_process.cj_analysis as als
from collections import defaultdict, Counter
import argparse

# Suppress DEBUG logs from a specific library
logging.getLogger("matplotlib").setLevel(logging.WARNING)

SATS_IN_BTC = 100000000

FAIL_ON_CHECKS = False
LONG_LEGEND = False

class Multifig:
    num_rows = 1
    num_columns = 1
    ax_index = 1
    fig = None
    plt = None
    axes = []

    def __init__(self, plt, fig, num_rows, num_columns):
        self.ax_index = 1
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.fig = fig
        self.plt = plt
        self.axes = []

    def add_subplot(self):
        ax = self.fig.add_subplot(self.num_rows, self.num_columns, self.ax_index)
        self.axes.append(ax)
        self.ax_index += 1
        return ax

    def add_multiple_subplots(self, num_subplots: int):
        for i in range(0, num_subplots):
            self.add_subplot()

    def get(self, index: int):
        return self.axes[index]


def plot_cj_anonscores(mfig: Multifig, data: dict, title: str, total_sessions: int, anon_score: str, y_label: str, color: str, avg_color: str='red'):
    plot_cj_anonscores_ax(mfig.add_subplot(), data, title, total_sessions, anon_score, y_label, color, avg_color)


def plot_cj_anonscores_ax(ax, data: dict, title: str, total_sessions: int, anon_score: str, y_label: str, line_color: str, avg_color: str='red'):
    size_01_used = False
    size_02_used = False
    size_emul_used = False
    max_y = -1
    for cj_session in data.keys():
        line_style = '-.'
        if cj_session.find('0.1btc') != -1:
            line_style = 'solid'
            size_01_used = True
        elif cj_session.find('0.2btc') != -1:
            line_style = ':'
            size_02_used = True
        elif cj_session.find('emulation') != -1:
            line_style = 'solid'
            size_emul_used = True
        else:
            assert False, f'Unexpected session type found for {cj_session}'
        # cj_label = cj_session
        # if not show_txid and cj_session.find('txid:'):
        #     cj_label = cj_label[0:cj_session.find('txid:')]
        x_range = range(1, len(data[cj_session]) + 1)
        ax.plot(x_range, data[cj_session], color=line_color, linestyle=line_style, alpha=0.15)
        max_y = max(max(data[cj_session]), max_y) if len(data[cj_session]) > 0 else max_y  # Keep maximum y observed

    if size_01_used:
        label = f'Input size 0.1 btc (as={anon_score})' if LONG_LEGEND else f'0.1 btc (as={anon_score})'
        ax.plot([1], [1], color=line_color, label=label, linestyle='solid', alpha=0.5)
    if size_02_used:
        label = f'Input size 0.2 btc (as={anon_score})' if LONG_LEGEND else f'0.2 btc (as={anon_score})'
        ax.plot([1], [1], color=line_color, label=label, linestyle=':', alpha=0.5)
    if size_emul_used:
        label = f'Input size emul btc (as={anon_score})' if LONG_LEGEND else f'separate wallets'
        ax.plot([1], [1], color=line_color, label=label, linestyle='solid', alpha=0.5)
    ax.set_xticks(np.arange(1, 20, 2))
    ax.set_yticks(np.linspace(0, max_y, num=5))

    def compute_average_at_index(lists, index):
        """
        Returns average value at specific index and number of valyes present at that index
        :param lists:
        :param index:
        :return:
        """
        values = [lists[lst][index] for lst in lists.keys() if index < len(lists[lst])]
        if not values:
            return 0
        return sum(values) / len(values), len(values)

    max_index = max([len(data[cj_session]) for cj_session in data.keys()])
    avg_data = [compute_average_at_index(data, index)[0] for index in range(0, max_index)]
    num_at_index_data = [compute_average_at_index(data, index)[1] for index in range(0, max_index)]

    PLOT_BASIC_AVERAGE = True
    if PLOT_BASIC_AVERAGE:
        label = f'Average (as={anon_score})' if LONG_LEGEND else f'AVG (as={anon_score})'
        x_vals = range(1, len(avg_data) + 1)
        ax.plot(x_vals, avg_data, label=label, linestyle='solid',
                linewidth=7, alpha=0.7, color=avg_color)
    else:  # Plotting variable thickness average
        values_at_index = {}
        # Populate dictionary with values at each index
        for cj_session in data.keys():
            for i, value in enumerate(data[cj_session]):
                if i not in values_at_index:
                    values_at_index[i] = []
                values_at_index[i].append(value)
        x_vals = np.array(sorted(values_at_index.keys()))
        # Interpolate the average line for smoothing
        x_smooth = np.linspace(x_vals.min(), x_vals.max(), 1000)  # More points for smoothness
        cs = CubicSpline(x_vals, avg_data)  # Cubic interpolation function
        y_smooth = cs(x_smooth)  # Get smooth values

        # Interpolate line thickness based on data points at each index
        smoothed_widths = np.interp(x_smooth, x_vals, num_at_index_data)  # No normalization

        # Plot smoothed averaged line with varying thickness
        for i in range(len(x_smooth) - 1):
            ax.plot(
                [x_smooth[i], x_smooth[i + 1]],  # X range
                [y_smooth[i], y_smooth[i + 1]],  # Y range
                color=line_color,
                alpha=0.5,
                linewidth=float(smoothed_widths[i])  # Use raw number of lists for thickness
            )

    FONT_SIZE = '14'
    FONT_SIZE_SMALLER = '11'
    #ax.legend(loc="upper left", fontsize=FONT_SIZE)
    ax.legend(loc="best", fontsize='10')
    #ax.set_title(f'{title}; total_sessions={total_sessions}')
    ax.set_title(f'{title}', fontsize=FONT_SIZE_SMALLER)
    ax.set_xlabel('Number of coinjoins executed', fontsize=FONT_SIZE_SMALLER)
    ax.set_ylabel(y_label, fontsize=FONT_SIZE_SMALLER)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    #plt.show()

    PLOT_BOXPLOT = False
    if PLOT_BOXPLOT:
        # Same data, but boxplot
        max_index = max([len(data[cj_session]) for cj_session in data.keys()])
        data_cj = [[] for index in range(0, max_index)]
        for cj_session in data.keys():
            for index in range(0, max_index):
                if index < len(data[cj_session]):
                    data_cj[index].append(data[cj_session][index])
        #fig, ax_boxplot = plt.subplots(figsize=(10, 5))
        # ax_boxplot = mfig.add_subplot()  # Get next subplot
        # ax_boxplot.boxplot(data_cj)
        # ax_boxplot.set_title(title)
        # ax_boxplot.set_xlabel('Number of coinjoins executed')
        # ax_boxplot.set_ylabel(y_label)
        #plt.show()


def get_session_label(mix_name: str, session_size_inputs: int, segment: dict, session_funding_tx: dict) -> str:
    # Two options for session label
    cjsession_label_short_date = f"{mix_name} {round(session_size_inputs / SATS_IN_BTC, 1)}btc | {len(segment)} cjs | " + \
                                 session_funding_tx['broadcast_time'] + " " + session_funding_tx['txid'][0:8]
    cjsession_label_short_txid = f"{mix_name} {round(session_size_inputs / SATS_IN_BTC, 1)}btc | {len(segment)} cjs | txid: {session_funding_tx['txid']} "
    cjsession_label_short = cjsession_label_short_date
    cjsession_label_short = cjsession_label_short_txid
    return cjsession_label_short


def find_highest_scores(root_folder, mix_name: str):
    highest_scores = defaultdict(int)  # Default score is 0

    # Traverse all subfolders
    for subdir, _, files in os.walk(root_folder):
        if f'{mix_name}_coins.json' in files:
            file_path = os.path.join(subdir, f'{mix_name}_coins.json')

            try:
                # Read JSON file
                with open(file_path, "r", encoding="utf-8") as f:
                    #logging.debug(f'Reading from {file_path}')
                    coin_data = als.load_json_from_file(file_path)['result']

                # Update highest scores for each coin address
                for coin in coin_data:
                    #for coin_address, score in coin_data.items():
                    highest_scores[coin['address']] = max(highest_scores[coin['address']], coin['anonymityScore'])

            except (FileNotFoundError, PermissionError) as e:
                logging.error(f"Error reading {file_path}: {e}")

    return dict(highest_scores)  # Convert back to regular dictionary


def find_input_index_for_output(coinjoins: dict, prev_txid: str, prev_vout_index: str, prev_value: int, next_txid: str):
    # NOTE: tx['inputs'][index] refer to index within outputs from funding transaction (vout),
    # not vin index of this transaction
    spending_index = None
    if prev_txid in coinjoins['coinjoins'].keys():  # Find in coinjoin txs, extract from 'spend_by_tx'
        assert 'spend_by_tx' in coinjoins['coinjoins'][prev_txid]['outputs'][prev_vout_index], f'Missing newest coinjoin_tx_info.json (parse_dumplings.py --action process_dumplings ; then folder with the target month for experiment and coordinator)). The problematic transaction is {prev_txid}'
        # if 'spend_by_tx' not in coinjoins['coinjoins'][prev_txid]['outputs'][prev_vout_index]:
        #     logging.debug(f'{coinjoins['coinjoins'][prev_txid]['outputs'][prev_vout_index]} does not have spend_by_tx, setting output to 0')
        #     spending_index = 0
        # else:
        spending_txid, spending_index = als.extract_txid_from_inout_string(
            coinjoins['coinjoins'][prev_txid]['outputs'][prev_vout_index]['spend_by_tx'])
    elif 'premix' in coinjoins and prev_txid in coinjoins[
        'premix']:  # Find in remix txs, extract from 'spend_by_tx'
        spending_txid, spending_index = als.extract_txid_from_inout_string(
            coinjoins['premix'][prev_txid]['outputs'][prev_vout_index]['spend_by_tx'])
    else:  # Dirty heuristics - pick first input which has same 'value' in sats
        if next_txid in coinjoins['coinjoins']:
            for in_index in coinjoins['coinjoins'][next_txid]['inputs']:
                if prev_value == coinjoins['coinjoins'][next_txid]['inputs'][in_index]['value']:
                    logging.debug(f'Dirty heuristics: Input {in_index} established for {next_txid}')
                    spending_index = in_index
                    break
        else:
            logging.debug(f'{next_txid} not in coinjoins, setting output to 0')
            spending_index = 0
    assert spending_index is not None, f'Spending index for {prev_txid} not found'

    return spending_index



def compute_multisession_statistics(cjtxs: dict, coinjoins: dict, mix_name: str, target_as: int):
    # Compute basic statistics
    stats = {}
    stats['all_cjs_weight_anonscore'] = {}
    stats['anon_percentage_status'] = {}
    stats['anon_gain_weighted'] = {}
    stats['observed_mix_liquidity'] = {}
    stats['observed_remix_liquidity_ratio'] = {}
    stats['observed_remix_liquidity_ratio_cumul'] = {}  # Remix liquidity based on value of inputs
    stats['observed_remix_inputs_ratio_cumul'] = {}     # unused now, remix liquidity based on number of inputs
    stats['anonscore_coins_distribution'] = {}
    stats['num_wallet_coins'] = {}
    stats['wallet_coins_values'] = {}
    for session_label in cjtxs['sessions'].keys():
        session_coins = {}
        anon_percentage_status_list = []
        anon_gain_weighted_list = []
        observed_remix_liquidity_ratio_list = []
        observed_mix_liquidity_list = []
        observed_remix_liquidity_ratio_cumul_list = []
        observed_remix_inputs_ratio_cumul_list = []
        anonscore_coins_distribution_list = []
        num_wallet_coins_list = []
        wallet_coins_values_list = []
        session_size_inputs = cjtxs['sessions'][session_label]['funding_tx']['value']
        assert session_size_inputs > 0, f'Unexpected negative funding tx size of {session_size_inputs}'
        cj_index = 0
        for cjtxid in cjtxs['sessions'][session_label]['coinjoins'].keys():
            cj_index = cj_index + 1
            cjtx = cjtxs['sessions'][session_label]['coinjoins'][cjtxid]
            print(f'#', end='')
            assert len(cjtx['outputs']) != 0, f"No coins assigned to {cjtx['txid']}"

            # Print all output coins (at given state of time) based on their anonscore
            # red ... anonscore target not reached yet, green ... already reached
            for index in cjtx['outputs']:
                if cjtx['outputs'][index]['anon_score'] < target_as:
                    # Print in red - target as not yet reached
                    print("\033[31m" + f" {round(cjtx['outputs'][index]['anon_score'], 1)}" + "\033[0m", end='')
                    # if cjtx['outputs'][index]['anonymityScore'] == 1:
                    #     print(f' {cjtx['outputs'][index]['address']}', end='')
                else:
                    # Print in green - target as reached
                    print("\033[32m" + f" {round(cjtx['outputs'][index]['anon_score'], 1)}" + "\033[0m", end='')

            # Compute privacy progress
            # 1. Update pool of coins in the wallet by removal of input coins and addition of newly created output coins
            # 2. Compute percentage progress status as weighted fraction of coins anonscore wrt desired target onescore
            #    (if coin's current anonscore is bigger that target anonscore, target anonscore is used as maximum => effective_as)
            # 3. Check if result is not above 1 (100%), if yes then warn and limit to 100%
            # Update pool (step 1.)
            for index in cjtx['inputs']:  # Remove coins from session_coins spend by this cjtx
                session_coins.pop(cjtx['inputs'][index]['address'], None)
            for index in cjtx['outputs']:  # Add coins to session_coins created by this cjtx
                session_coins[cjtx['outputs'][index]['address']] = cjtx['outputs'][index]
            # Compute percentage progress status (step 2.)
            anon_percentage_status = 0
            anon_gain_weighted = 0  # Tae real achieved anonscore for given coin, weighted by its size to whole session
            for address in session_coins.keys():
                if session_coins[address]['anon_score'] > 1:
                    effective_as = min(session_coins[address]['anon_score'], target_as)
                    # Weighted percentage contribution of this specific coin to progress status
                    anon_percentage_status += (effective_as / target_as) * (
                            session_coins[address]['value'] / session_size_inputs)
                    anon_gain_weighted += session_coins[address]['anon_score'] * (
                            session_coins[address]['value'] / session_size_inputs)
            # Privacy progress can be sometimes slightly bigger than 100% for sessions where some previously prisoned
            # coins were included into mix during experiment (happened rarely and for very small coins)
            WARN_TOO_HIGH_PRIVACY_PROGRESS = True
            if WARN_TOO_HIGH_PRIVACY_PROGRESS and anon_percentage_status > 1.01:
                print(f'\nToo large anon_percentage_status {round(anon_percentage_status * 100, 1)}%: {cjtxid}')
                anon_percentage_status = 1
            print(f' {round(anon_percentage_status * 100, 1)}%', end='')
            anon_percentage_status_list.append(anon_percentage_status * 100)
            anon_gain_weighted_list.append(anon_gain_weighted)

            # Print number of all coins in wallet for given coinjoin and session
            print(f' [{cj_index}.s:{len(session_coins)}c]', end='')
            num_wallet_coins_list.append(len(session_coins))

            # Compute observed liquidity ratio for wallet's coins
            # This value enumerates multiplier of initial fresh liquidity over multiple cjtxs
            # (If all coins are fully mixed in the first coinjoin, then observed_remix_liquidity_ratio is 1,
            # every additional mix is adding additional input liquidity (remixed))
            # 1. Sum values of all wallet's input coins (to this cjtx), divided by fresh liquidity (of this session)
            # 2. Compute cummulative liquidity for each subsequent coinjoin (observed_remix_liquidity_ratio_cumul_list)
            observed_mix_liquidity = sum([cjtx['inputs'][index]['value'] for index in cjtx['inputs']])
            observed_mix_liquidity_list.append(observed_mix_liquidity)
            observed_remix_liquidity_ratio = observed_mix_liquidity / session_size_inputs
            observed_remix_liquidity_ratio_list.append(observed_remix_liquidity_ratio)
            if len(observed_remix_liquidity_ratio_cumul_list) == 0:
                # We do expect mix liquidity ratio to be close to 1.0 (minus fees paid) in case when single utxo (session_size_inputs) was started with
                if not math.isclose(observed_remix_liquidity_ratio, 1.0, rel_tol=1e-9):
                    print(f'\nWarning: Unexpected observed_remix_liquidity_ratio of {observed_remix_liquidity_ratio} instead 1.0')
                observed_remix_liquidity_ratio_cumul_list.append(0)  # The first value is fresh input, not remix
            else:
                observed_remix_liquidity_ratio_cumul_list.append(observed_remix_liquidity_ratio_cumul_list[-1] + observed_remix_liquidity_ratio)

            #
            # Compute list of anonscores for coinjoin round in a session
            #
            anonscore_coins_distribution_list.append([session_coins[address]['anon_score'] for address in session_coins.keys()])

            for coin in session_coins:
                wallet_coins_values_list.append(session_coins[coin]['value'])

        # Store computed data
        if len(anon_percentage_status_list) > 0:
            assert session_label not in stats['anon_percentage_status'], f'Duplicate session label {session_label}'
            stats['anon_percentage_status'][session_label] = anon_percentage_status_list
        if len(anon_gain_weighted_list) > 0:
            assert session_label not in stats['anon_gain_weighted'], f'Duplicate session label {session_label}'
            stats['anon_gain_weighted'][session_label] = anon_gain_weighted_list
        if len(observed_remix_liquidity_ratio_list) > 0:
            assert session_label not in stats['observed_remix_liquidity_ratio'], f'Duplicate session label {session_label}'
            stats['observed_mix_liquidity'][session_label] = observed_mix_liquidity_list
            stats['observed_remix_liquidity_ratio'][session_label] = observed_remix_liquidity_ratio_list
            stats['observed_remix_liquidity_ratio_cumul'][session_label] = observed_remix_liquidity_ratio_cumul_list
        if len(anonscore_coins_distribution_list) > 0:
            assert session_label not in stats['anonscore_coins_distribution'], f'Duplicate session label {session_label}'
            stats['anonscore_coins_distribution'][session_label] = anonscore_coins_distribution_list
        if len(num_wallet_coins_list) > 0:
            assert session_label not in stats['num_wallet_coins'], f'Duplicate session label {session_label}'
            stats['num_wallet_coins'][session_label] = num_wallet_coins_list
        if len(wallet_coins_values_list) > 0:
            assert session_label not in stats['wallet_coins_values'], f'Duplicate session label {session_label}'
            stats['wallet_coins_values'][session_label] = wallet_coins_values_list
        # Print finalized info
        session = cjtxs['sessions'][session_label]
        session_end_merge_tx = f"{len(session['coinjoins'].keys())} cjs | " + session['funding_tx']['label'] + " " + session['funding_tx']['broadcast_time'] + " " + \
                               session['funding_tx']['txid']
        print("\033[34m" + f' * ' + session_end_merge_tx + "\033[0m", end='')
        cjsession_label_short = get_session_label(mix_name, session_size_inputs, cjtxs['sessions'][session_label]['coinjoins'].keys(), session['funding_tx'])
        print(f' |--> \"{cjsession_label_short}\"', end='')
        print()

    # Number of completely skipped coinjoin transactions (no wallet's coin is participating in coinjoin executed   )
    sorted_cj_times = als.sort_coinjoins(coinjoins, als.SORT_COINJOINS_BY_RELATIVE_ORDER)
    coinjoins_index = {sorted_cj_times[i]['txid']: i for i in range(0, len(sorted_cj_times))}  # Precomputed mapping of txid to index for fast burntime computation
    # coord_logs_sanitized = [{**item, 'mp_first_seen': item['mp_first_seen'] if item['mp_first_seen'] is not None else item['cj_last_seen']} for item in coord_logs]
    # coord_logs_sorted = sorted(coord_logs_sanitized, key=lambda x: x['mp_first_seen'])
    # coinjoins_index = {coord_logs_sorted[i]['id']: i for i in range(0, len(coord_logs_sorted))}
    stats['skipped_cjtxs'] = {}
    for session_label in cjtxs['sessions'].keys():
        prev_cjtxid = None
        skipped_cjtxs_list = []
        for cjtxid in cjtxs['sessions'][session_label]['coinjoins'].keys():
            if cjtxid not in coinjoins_index.keys():
                print(f'{cjtxid} missing from coord_logs')
                continue
            skipped = 0 if prev_cjtxid is None else coinjoins_index[cjtxid] - coinjoins_index[prev_cjtxid] - 1

            # Compute minimum burn_time for remixed inputs
            burn_times = []
            cj_struct = cjtxs['sessions'][session_label]['coinjoins'][cjtxid]['inputs']
            for input in cj_struct:
                if 'spending_tx' in cj_struct[input].keys():
                    spending_tx, index = als.extract_txid_from_inout_string(cj_struct[input]['spending_tx'])
                    if spending_tx in coinjoins.keys():
                        burn_times.append(coinjoins_index[cjtxid] - coinjoins_index[spending_tx])
            min_burn_time = min(burn_times) - 1 if len(burn_times) > 0 else 0
            if skipped < 0:
                print(f'Inconsistent skipped coinjoins of {skipped} for {cjtxid} - {prev_cjtxid}')
            skipped_cjtxs_list.append(skipped)
            prev_cjtxid = cjtxid
        stats['skipped_cjtxs'][session_label] = skipped_cjtxs_list

    # Number of inputs and outputs
    stats['num_inputs'] = {}
    stats['num_outputs'] = {}
    for session_label in cjtxs['sessions'].keys():
        num_inputs_list = []
        num_outputs_list = []
        for cjtxid in cjtxs['sessions'][session_label]['coinjoins'].keys():
            num_inputs_list.append(len(cjtxs['sessions'][session_label]['coinjoins'][cjtxid]['inputs']))
            num_outputs_list.append(len(cjtxs['sessions'][session_label]['coinjoins'][cjtxid]['outputs']))
        stats['num_inputs'][session_label] = num_inputs_list
        stats['num_outputs'][session_label] = num_outputs_list

    # Anonscore gain achieved by given coinjoin (weighted by in/out size)
    stats['anon_gain'] = {}
    stats['anon_gain_ratio'] = {}
    for session_label in cjtxs['sessions'].keys():
        anon_gain_list = []
        anon_gain_ratio_list = []
        input_coins = {}
        output_coins = {}
        for cjtxid in cjtxs['sessions'][session_label]['coinjoins'].keys():
            cjtx = cjtxs['sessions'][session_label]['coinjoins'][cjtxid]
            for index in cjtx['inputs']:  # Compute anonscore for all inputs
                input_coins[cjtx['inputs'][index]['address']] = cjtx['inputs'][index]
            for index in cjtx['outputs']:  # Compute anonscore for all outputs
                output_coins[cjtx['outputs'][index]['address']] = cjtx['outputs'][index]

            inputs_size_inputs = sum([input_coins[address]['value'] for address in input_coins.keys()])
            outputs_size_inputs = sum([output_coins[address]['value'] for address in output_coins.keys()])
            input_anonscore = sum([input_coins[address]['anon_score'] * input_coins[address]['value'] / inputs_size_inputs for address in input_coins.keys()])
            output_anonscore = sum([output_coins[address]['anon_score'] * output_coins[address]['value'] / outputs_size_inputs for address in output_coins.keys()])

            anonscore_gain = output_anonscore - input_anonscore
            anon_gain_list.append(anonscore_gain)
            anon_gain_ratio_list.append(output_anonscore / input_anonscore)

        stats['anon_gain'][session_label] = anon_gain_list
        stats['anon_gain_ratio'][session_label] = anon_gain_ratio_list

    # Compute total number of output utxos created
    stats['num_coins'] = {}
    for session_label in cjtxs['sessions'].keys():
        stats['num_coins'][session_label] = sum([len(cjtxs['sessions'][session_label]['coinjoins'][txid]['outputs']) for txid in cjtxs['sessions'][session_label]['coinjoins'].keys()])

    # Compute total number of inputs used which already reached target anonscore level (aka overmixed coins)
    stats['num_overmixed_coins'] = {}
    for session_label in cjtxs['sessions'].keys():
        num_overmixed = [cjtxs['sessions'][session_label]['coinjoins'][txid]['inputs'][index]['value'] for txid in cjtxs['sessions'][session_label]['coinjoins'].keys()
                         for index in cjtxs['sessions'][session_label]['coinjoins'][txid]['inputs'].keys()
                         if cjtxs['sessions'][session_label]['coinjoins'][txid]['inputs'][index]['anon_score'] >= target_as]

        stats['num_overmixed_coins'][session_label] = len(num_overmixed)


    # Compute fees paid
    stats['experiment_cost'] = {'wallet_fair_mfee': 0, 'wallet_all_fee': 0}
    for session_label in cjtxs['sessions'].keys():
        wallet_fee_paid = [cjtxs['sessions'][session_label]['coinjoins'][txid].get('wallet_fee_paid', 0)
                   for txid in cjtxs['sessions'][session_label]['coinjoins'].keys()]
        wallet_fairmfee_paid = [cjtxs['sessions'][session_label]['coinjoins'][txid].get('wallet_fair_mfee', 0)
                   for txid in cjtxs['sessions'][session_label]['coinjoins'].keys()]
        assert all(v > 0 for v in wallet_fee_paid), f'Unexpected fees: {wallet_fee_paid}'
        assert all(v > 0 for v in wallet_fairmfee_paid), f'Unexpected fees: {wallet_fairmfee_paid}'
        stats['experiment_cost']['wallet_fair_mfee'] += sum(wallet_fairmfee_paid)
        stats['experiment_cost']['wallet_all_fee'] += sum(wallet_fee_paid)

    print(f"\n{mix_name}: Total experiments: {len(cjtxs['sessions'])}, "
          f"total coins: {sum([stats['num_coins'][session_label] for session_label in stats['num_coins'].keys()])}, "
          f"total overmixed coins: {sum([len([stats['num_overmixed_coins'][session_label] for session_label in stats['num_overmixed_coins'].keys()])])},"
          f"mining fee cost: {stats['experiment_cost']['wallet_fair_mfee']}, "
          f"coord. fee cost: {stats['experiment_cost']['wallet_all_fee'] - stats['experiment_cost']['wallet_fair_mfee']}, "
          f"total fee cost: {stats['experiment_cost']['wallet_all_fee']}")

    print("##################################################")

    return stats


def fill_cj_participation_info(record: dict, target_base_path: str, tx: dict, coinjoins: dict, txid: str):
    tx_file_path = os.path.join(target_base_path, 'data', f'{txid}.json')
    if os.path.exists(tx_file_path):
        tx_hex = als.load_json_from_file(tx_file_path)['result']
        # Compute total mining fee paid (sum(inputs) - sum(outputs))
        inputs_sum = sum([coinjoins[txid]['inputs'][index]['value'] for index in coinjoins[txid]['inputs'].keys()])
        outputs_sum = sum([coinjoins[txid]['outputs'][index]['value'] for index in coinjoins[txid]['outputs'].keys()])
        total_mining_fee = inputs_sum - outputs_sum
        # Compute vsize for "our" inputs and outputs out of whole transaction => our share of mining fees
        # wallet_inputs = [int(record['inputs'][item]['index']) for item in record['inputs'].keys()]
        # wallet_outputs = [int(record['outputs'][item]['index']) for item in record['outputs'].keys()]
        wallet_inputs = [int(item) for item in record['inputs'].keys()]
        wallet_outputs = [int(item) for item in record['outputs'].keys()]
        wallet_vsize, total_vsize = als.compute_partial_vsize(tx_hex['hex'], wallet_inputs, wallet_outputs)
        # Fee rate paid for whole transaction
        fee_rate = total_mining_fee / total_vsize
        # Mining fee rate to pay fair share for our inputs and outputs
        wallet_fair_mfee_sats = math.ceil(wallet_vsize * fee_rate)
        wallet_inputs_sum = sum([coinjoins[txid]['inputs'][index]['value'] for index in record['inputs'].keys()])
        wallet_outputs_sum = sum([coinjoins[txid]['outputs'][index]['value'] for index in record['outputs'].keys()])
        wallet_fee_paid_sats = wallet_inputs_sum - wallet_outputs_sum
        # assert tx['amount'] == -wallet_fee_paid_sats, f"Incorrect wallet fee computed {wallet_fee_paid_sats} sats vs. {tx['amount']} sats for {txid}"
        if tx and tx['amount'] != -wallet_fee_paid_sats:
            logging.error(
                f"Incorrect wallet fee computed {wallet_fee_paid_sats} sats vs. {tx['amount']} sats for {txid}")
            logging.debug(f"Inputs: ")
            for index in record['inputs'].keys():
                logging.debug(f"  [{index}]: {coinjoins[txid]['inputs'][index]['value']} sats")
            logging.debug(f"Outputs: ")
            for index in record['outputs'].keys():
                logging.debug(f"  [{index}]: {coinjoins[txid]['outputs'][index]['value']} sats")
        hidden_ctip = wallet_fee_paid_sats - wallet_fair_mfee_sats
        if hidden_ctip < -10:
            logging.debug(f"Sligthly smaller hidden tip than expected: {hidden_ctip} sats")
        if hidden_ctip >= -100:
            wallet_name = list(record['inputs'].values())[0]['wallet_name']
            logging.debug(f"Possibly incorrect hidden tip of {hidden_ctip} sats, {txid}, {wallet_name}")

        record['total_mining_fee'] = total_mining_fee
        record['mining_fee_rate'] = fee_rate
        record['total_vsize'] = total_vsize
        record['wallet_vsize'] = wallet_vsize
        record['wallet_fair_mfee'] = wallet_fair_mfee_sats
        record['wallet_fee_paid'] = wallet_fee_paid_sats
        record['wallet_hidden_ctip_paid'] = hidden_ctip
    else:
        logging.warning(f'{tx_file_path} is missing')

    return record


def parse_sessions(target_base_path: str, mix_name: str, experiment_start_date: str, coinjoins_all: dict):
    coinjoins = coinjoins_all['coinjoins']
    target_path = os.path.join(target_base_path, f'{mix_name}_history.json')
    history_all = als.load_json_from_file(target_path)['result']
    target_path = os.path.join(target_base_path, f'{mix_name}_coins.json')
    coins = als.load_json_from_file(target_path)['result']

    # FIX required for missing anonymity scores in final files for as38 (were merged before RPC call to obtain coins stats):
    # After each merge, anonymity score for merge transaction is set to 1 for all inputs.
    # Search older *_coins.json files and try to find one before experiment coins merge
    intermediate_coins_max_score = find_highest_scores(target_base_path, mix_name)
    for coin in coins:
        if coin['anonymityScore'] == 1:
            coin['anonymityScore'] = intermediate_coins_max_score[coin['address']] if coin['address'] in intermediate_coins_max_score else 1

    # target_path = os.path.join(target_base_path, f'logww2.json')
    # coord_logs = als.load_json_from_file(target_path)

    # Filter all items from history older than experiment start date
    history = [tx for tx in history_all if tx['datetime'] >= experiment_start_date]

    # Pair wallet coins to transactions from wallet history
    for cjtx in history:
        if 'outputs' not in cjtx.keys():
            cjtx['outputs'] = {}
        if 'inputs' not in cjtx.keys():
            cjtx['inputs'] = {}
        for coin in coins:
            if coin['txid'] == cjtx['tx']:
                cjtx['outputs'][str(coin['index'])] = coin
            if coin['spentBy'] == cjtx['tx']:
                # We do not know correct vin index - need to search for in subsequent transaction
                input_index = find_input_index_for_output(coinjoins_all, coin['txid'], str(coin['index']), coin['amount'], coin['spentBy'])
                cjtx['inputs'][str(input_index)] = coin

    # If last tx is coinjoin, add one artificial non-coinjoin one
    if history[-1]['islikelycoinjoin'] is True:
        artificial_end = copy.deepcopy(history[-1])
        artificial_end['islikelycoinjoin'] = False
        artificial_end['tx'] = '0000000000000000000000000000000000000000000000000000000000000000'
        artificial_end['label'] = 'artificial end merge'
        history.append(artificial_end)

    #
    # Detect separate coinjoin sessions and split based on them.
    # Assumption: 1 non-coinjoin tx followed by one or more coinjoin session, finished again with non-coinjoin tx
    #
    cjtxs = {'sessions': {}}
    session_cjtxs = {}
    session_size_inputs = 0
    for index in range(0, len(history)):
        tx = history[index]
        if tx['islikelycoinjoin'] is True:
            txid = tx['tx']
            # Inside coinjoin session, append
            record = {'txid': tx['tx'], 'inputs': {}, 'outputs': {}, 'round_id': tx['tx'], 'is_blame_round': False}
            record['round_start_time'] = als.precomp_datetime.fromisoformat(tx['datetime']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            record['broadcast_time'] = als.precomp_datetime.fromisoformat(tx['datetime']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            record['inputs'] = {}
            for index in tx['inputs']:
                record['inputs'][index] = {}
                record['inputs'][index]['index'] = index
                record['inputs'][index]['address'] = tx['inputs'][index]['address']
                record['inputs'][index]['value'] = tx['inputs'][index]['amount']
                record['inputs'][index]['wallet_name'] = mix_name
                record['inputs'][index]['anon_score'] = tx['inputs'][index]['anonymityScore']

            record['outputs'] = {}
            for index in tx['outputs']:  # For outputs, index is correct value in this coinjoin cjtx
                record['outputs'][index] = {}
                record['outputs'][index]['index'] = index
                record['outputs'][index]['address'] = tx['outputs'][index]['address']
                record['outputs'][index]['value'] = tx['outputs'][index]['amount']
                record['outputs'][index]['wallet_name'] = mix_name
                record['outputs'][index]['anon_score'] = tx['outputs'][index]['anonymityScore']

            # Try to load full serialized tx (if available) and extract additional info
            fill_cj_participation_info(record, target_base_path, tx, coinjoins, txid)
            session_cjtxs[txid] = record
        else:
            # Non-coinjoin transaction detected (either initial funding one at the start of session, or start of of next session )
            if len(session_cjtxs) > 0:
                # We hit first non-coinjoin transaction after session => end of session
                assert len(session_funding_tx[
                               'outputs'].keys()) == 1, f"Funding tx {session_funding_tx['tx']} has unexpected number of outputs of {len(session_funding_tx['outputs'].keys())}"
                norm_tx = {'txid': session_funding_tx['tx'], 'label': session_funding_tx['label'],
                           'broadcast_time': session_funding_tx['datetime'],
                           'value': session_funding_tx['outputs']['0']['amount']}
                session_label = get_session_label(mix_name, session_size_inputs, session_cjtxs, norm_tx)
                print(f'{session_label}: {session_size_inputs}')
                als.remove_link_between_inputs_and_outputs(session_cjtxs)
                als.compute_link_between_inputs_and_outputs(session_cjtxs, [cjtxid for cjtxid in session_cjtxs.keys()])

                cjtxs['sessions'][session_label] = {'coinjoins': session_cjtxs, 'funding_tx': norm_tx}
                session_cjtxs = {}
                session_size_inputs = 0
                session_funding_tx = None

            # Non-coinjoin trasaction, potentially initial funding tx, then extract input liquidity into session_size_inputs
            if len(tx['outputs']) == 1 and tx['outputs'][list(tx['outputs'].keys())[0]]['amount'] > 0:
                #if tx['outputs'][list(tx['outputs'].keys())[0]]['amount'] > session_size_inputs:
                session_size_inputs = tx['outputs'][list(tx['outputs'].keys())[0]]['amount']
                session_funding_tx = tx

    return cjtxs


def analyze_multisession_mix_experiments(target_base_path: str, mix_name: str, target_as: int,
                                         experiment_start_date: str):
    # Load all real coinjoins executed
    target_path = os.path.join(target_base_path, f'coinjoin_tx_info.json')
    coinjoins_all = als.load_json_from_file(target_path)
    # Parse and extract sessions from provided files
    cjtxs = parse_sessions(target_base_path, mix_name, experiment_start_date, coinjoins_all)
    # Compute statistics
    stats = compute_multisession_statistics(cjtxs, coinjoins_all['coinjoins'], mix_name, target_as)

    return cjtxs, stats


def merge_coins_files(base_path: str, file1: str, file2: str):
    coins1 = als.load_json_from_file(os.path.join(base_path, file1))['result']
    coins2 = als.load_json_from_file(os.path.join(base_path, file2))['result']

    for coin1 in coins1:
        for coin2 in coins2:
            if (coin1['txid'] == coin2['txid'] and coin1['index'] == coin2['index']
                    and coin1['amount'] == coin2['amount']):
                if coin1['spentBy'] is None and coin2['spentBy'] is not None:
                    coin1['spentBy'] = coin2['spentBy']
                if coin1['anonymityScore'] == 1 and coin2['anonymityScore'] > 1:
                    coin1['anonymityScore'] = coin2['anonymityScore']
                coin2['used'] = True

    for coin2 in coins2:
        if 'used' not in coin2.keys():
            coins1.append(coin2)

    return {'result': coins1}


def parse_outpoint(hex_outpoint: str):
    # Ensure the input is a valid length
    if len(hex_outpoint) != 72:  # 64 characters for TXID + 8 characters for index
        raise ValueError("Invalid outpoint length. Must be 72 hex characters (36 bytes).")
    # Extract the TXID and the index from the hex_outpoint
    txid_hex_little_endian = hex_outpoint[:64]
    index_hex_little_endian = hex_outpoint[64:]
    # Convert TXID from little-endian to big-endian (human-readable format)
    txid_hex = ''.join([txid_hex_little_endian[i:i + 2] for i in range(0, len(txid_hex_little_endian), 2)][::-1])
    # Convert index from little-endian to integer
    index = int(''.join([index_hex_little_endian[i:i + 2] for i in range(0, len(index_hex_little_endian), 2)][::-1]),
                16)
    return txid_hex, index


def analyse_prison_logs(target_path: str):
    """
    Reads all zip files from target_path, extract PrisonedCoins.json and time of capture.
    Merge all information, extract prisoned coins info
    :param target_path:
    :return:
    """


# # Load all prison coin files, merge and compute statistics
    # hex_outpoint = "82A23500AD90C8C42F00F2DA0A4C265C0D0A91543C5D3A037F44436F14B8D9039A000000"
    # txid, index = parse_outpoint(hex_outpoint)
    # print(f"TXID: {txid}")
    # print(f"Index: {index}")


def plot_scatter(mfig: Multifig, x, y, x_label, y_label, title, color: str):
    ax = mfig.add_subplot()
    ax.scatter(x, y, color=color, s=5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


def plot_cj_heatmap(mfig: Multifig, x, y, x_label, y_label, title, annotate_cells: bool, normalize_y: bool=False):
    ax = mfig.add_subplot()

    x_edges = np.arange(min(x), max(x) + 2)
    y_edges = np.arange(min(y), max(y) + 2)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[x_edges, y_edges])

    if normalize_y:
        # normalize so for each x-bin, sum over y is 1
        row_sums = heatmap.sum(axis=1, keepdims=True)  # per-x totals
        heatmap = np.divide(heatmap, row_sums, out=np.zeros_like(heatmap), where=row_sums > 0)
        heatmap_percentage = heatmap * 100
        sns.heatmap(heatmap_percentage.T, cmap='coolwarm', annot=annotate_cells,
                    annot_kws={"size": 6}, fmt='.1f', cbar=True, ax=ax, linecolor='white')
    else:
        # Standard heatmap scaled to 100%
        heatmap_percentage = (heatmap / np.sum(heatmap)) * 100
        sns.set_style("white")
        sns.heatmap(heatmap_percentage.T, cmap='coolwarm', annot=annotate_cells, annot_kws={"size": 6}, fmt='.1f',
                    cbar=True, ax=ax, linecolor='white')

    ax.invert_yaxis()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    max_ticks = 10
    n_x = len(xedges) - 1
    n_y = len(yedges) - 1
    step_x = max(1, math.ceil(n_x / max_ticks))
    step_y = max(1, math.ceil(n_y / max_ticks))
    # Tick positions (centered)
    x_ticks = np.arange(0, n_x, step_x) + 0.5
    y_ticks = np.arange(0, n_y, step_y) + 0.5
    # Tick labels (1-based indexing)
    x_labels = np.arange(1, n_x + 1, step_x)
    y_labels = np.arange(1, n_y + 1, step_y)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # ax.set_xticks(np.arange(len(xedges) - 1) + 0.5)
    # ax.set_yticks(np.arange(len(yedges) - 1) + 0.5)
    # ax.set_xticklabels(np.arange(1, len(xedges)))
    # ax.set_yticklabels(np.arange(1, len(yedges)))
    ax.set_title(title)
    #plt.show()


def recreate_cjtxs(base_path: str):
    # 2. Re-create (missing) history.json for each wallet (if required)
    # For each wallet, load coins.json, create history.json from each coin in coins.json
    # 2. Create directly xxx_coinjoin_tx_info.json with each wallet being one 'session'
    cjtxs = {'sessions': {}}
    wallets_coins = als.load_json_from_file(os.path.join(base_path, 'wallets_coins.json'))
    for wallet_name in wallets_coins:
        cjtxs['sessions'][wallet_name] = {}
        cjtxs['sessions'][wallet_name]['coinjoins'] = {}

        for coin in wallets_coins[wallet_name]:
            coin['anon_score'] = coin['anonymityScore']
            coin['wallet_name'] = wallet_name
            if 'spentBy' in coin:
                coin['spend_by_txid'][0] = coin['spentBy']
                coin['spend_by_txid'][1] = 1000
            if coin['txid'] not in cjtxs['sessions'][wallet_name]['coinjoins']:
                cjtxs['sessions'][wallet_name]['coinjoins'][coin['txid']] = {'txid': coin['txid'], 'inputs': {}, 'outputs': {}}
            cjtxs['sessions'][wallet_name]['coinjoins'][coin['txid']]['broadcast_time'] = coin['create_time']


def parse_sessions_from_wallets(base_path: str, coinjoins_all: dict):
    cjtxs = {'sessions': {}}
    wallets_coins = als.load_json_from_file(os.path.join(base_path, 'wallets_coins.json'))

    for wallet_name in wallets_coins:
        logging.info(f"{wallet_name} wallet: ")
        session_name = f'emulation {wallet_name}'

        cjtxs['sessions'][session_name] = {}

        # Funding tx for emulations can be spread. Create virtual one computed over all coins with anon_score=1.0
        # BUGBUG: This can create temporary incorrect result for privacy progress
        cjtxs['sessions'][session_name]['funding_tx'] = {'label': session_name}
        fund_value = 0
        for coin in wallets_coins[wallet_name]:
            if math.isclose(coin['anonymityScore'], 1.0, rel_tol=1e-9):
                fund_value += coin['amount']
                if 'txid' in cjtxs['sessions'][session_name]['funding_tx'].keys():
                    logging.info(f"  More than one funding transaction detected: {cjtxs['sessions'][session_name]['funding_tx']['txid']}")
                cjtxs['sessions'][session_name]['funding_tx']['txid'] = coin['txid']
                cjtxs['sessions'][session_name]['funding_tx']['broadcast_time'] = coin['create_time']
        cjtxs['sessions'][session_name]['funding_tx']['value'] = fund_value
        print(f'{wallet_name}: {fund_value/SATS_IN_BTC} funding value')

        # Get data from existing structure, but filter out all !wallet_name items
        cjtxs['sessions'][session_name]['coinjoins'] = deepcopy(coinjoins_all['coinjoins'])  # As we will be removing some items, create deep copy
        for cjtx in coinjoins_all['coinjoins'].keys():
            record = cjtxs['sessions'][session_name]['coinjoins'][cjtx]
            for index in list(record['inputs'].keys()):  # Remove all inputs of other wallets
                if record['inputs'][index]['wallet_name'] != wallet_name:
                    record['inputs'].pop(index)
            for index in list(record['outputs'].keys()):  # Remove all outputs of other wallets
                if record['outputs'][index]['wallet_name'] != wallet_name:
                    record['outputs'].pop(index)

            if len(record['inputs']) == 0 and len(record['outputs']) == 0:
                # No participation in this coinjoin
                cjtxs['sessions'][session_name]['coinjoins'].pop(cjtx)
            else:
                # Compute additional transaction metadata
                fill_cj_participation_info(cjtxs['sessions'][session_name]['coinjoins'][cjtx], base_path, None,
                                           coinjoins_all['coinjoins'], cjtx)

        SANITIZE_ANON_SCORES = False  # If True, setting of anon_score values is performed (was not saved properly for older emulation processing)
        if SANITIZE_ANON_SCORES:
            # Fill all anon_score records for inputs where missing (use anon_score of output if filled, otherwise set to 1)
            # Start with outputs - if spent, then fill anon_score
            for cjtx in cjtxs['sessions'][session_name]['coinjoins'].keys():
                record = cjtxs['sessions'][session_name]['coinjoins'][cjtx]
                for index in record['outputs'].keys():
                    if 'spend_by_txid' in record['outputs'][index].keys():
                        txid, tx_index = record['outputs'][index]['spend_by_txid']
                        tx_index = str(tx_index)
                        if txid in cjtxs['sessions'][session_name]['coinjoins'].keys():
                            if (txid in cjtxs['sessions'][session_name]['coinjoins'].keys() and
                                    'anon_score' in cjtxs['sessions'][session_name]['coinjoins'][txid]['inputs'][tx_index].keys()):
                                assert math.isclose(cjtxs['sessions'][session_name]['coinjoins'][txid]['inputs'][tx_index]['anon_score'], record['outputs'][index]['anon_score'], rel_tol=1e-9)
                            else:
                                cjtxs['sessions'][session_name]['coinjoins'][txid]['inputs'][tx_index]['anon_score'] = record['outputs'][index]['anon_score']

            # Fill all non-set inputs to anonscore 1.0
            for cjtx in cjtxs['sessions'][session_name]['coinjoins'].keys():
                record = cjtxs['sessions'][session_name]['coinjoins'][cjtx]
                for index in record['inputs'].keys():
                    if 'anon_score' not in record['inputs'][index].keys():
                        record['inputs'][index]['anon_score'] = 1.0

        # Inserted coinjoins may not be sorted in dict based on broadcast_time - sort and re-assign
        session_cjs = cjtxs['sessions'][session_name]['coinjoins']
        sorted_session_cjs = dict(sorted(session_cjs.items(), key=lambda item: item[1]['broadcast_time']))
        cjtxs['sessions'][session_name]['coinjoins'] = sorted_session_cjs

        for cjtx in cjtxs['sessions'][session_name]['coinjoins'].keys():
            record = cjtxs['sessions'][session_name]['coinjoins'][cjtx]
            for index in record['inputs'].keys():
                if 'anon_score' not in record['inputs'][index].keys():
                    record['inputs'][index]['anon_score'] = 1.0

        # Sanity check - all inputs and outputs coins shall have assigned anonymity score and it shall match wallets_coins.json
        for coin in wallets_coins[wallet_name]:
            if coin['txid'] not in cjtxs['sessions'][session_name]['coinjoins'].keys():
                # Likely funding transaction
                if not math.isclose(coin['anonymityScore'], 1.0, rel_tol=1e-9):
                    logging.warning(f"{wallet_name} coin from {coin['txid']} ({coin['address']}) missing from coinjoins")
            else:
                # Check all outputs
                assert math.isclose(coin['anonymityScore'], cjtxs['sessions'][session_name]['coinjoins'][coin['txid']]['outputs'][str(coin['index'])]['anon_score'], rel_tol=1e-9), f"Mismatch in anonscore for {coin['txid']}:{coin['index']}"

        # Sanity check - at least one coinjoin shall be present for a given wallet. Drop inactive wallets
        if len(cjtxs['sessions'][session_name]['coinjoins']) == 0:
            logging.warning(f'No coinjoin participation detected for wallet {wallet_name} with {len(wallets_coins[wallet_name])} initial coins, dropping from dict')
            cjtxs['sessions'].pop(session_name)

    return cjtxs


def extract_tx_info_from_blocks(target_path: str):
    """
    Extract all transactions from 'block_*.json' files in target directory target_path/btc-node
    and save these as 'txid.json'. The saved files are in target_path/txid.json
    :param target_path: path with sub-folder 'btc-node' containing 'block_*.json' files
    :return:
    """
    block_files = list(Path(os.path.join(target_path, 'btc-node')).glob('block_*.json'))
    for block_file in block_files:
        block = als.load_json_from_file(block_file)
        for tx in block['tx']:
            out = {'result': tx, 'error': None, 'id': 'curltest'}
            als.save_json_to_file(os.path.join(target_path, f"{tx['txid']}.json"), out)


def full_analyze_emulations(base_path: str, target_as: int, exp_name_label: str):
    """
    Process data files from coinjoin emulations and produce graphs. Each participating wallet is treated as separate
    entity, resulting in one "session" for subsequent analysis.
    Expects downloaded blocks and files 'coinjoin_tx_info.json' and 'wallets_coins.json' extracted by parse_cj_logs.py
    :param base_path: root directory with emulation files
    :param target_as: anon score set for clients in this experiment
    :param exp_name_label: custom label for this experiment
    :return: processed coinjoins, statistics and generated graphs
    """
    # Extract transactions info from block_xxx.json to (many) txid.json files
    extract_tx_info_from_blocks(os.path.join(base_path, 'data'))

    # Create directly xxx_coinjoin_tx_info.json with each wallet being one 'session'.
    # Use pre-created coinjoin_tx_info.json, filter out all transactions and ins/outs not belonging
    # to a specific wallet
    coinjoins_all = als.load_json_from_file(os.path.join(base_path, 'coinjoin_tx_info.json'))
    cjtxs = parse_sessions_from_wallets(base_path, coinjoins_all)
    als.save_json_to_file_pretty(os.path.join(base_path, f'{exp_name_label}_coinjoin_tx_info.json'), cjtxs)

    # Compute statistics
    stats = compute_multisession_statistics(cjtxs, coinjoins_all['coinjoins'], '', target_as)
    als.save_json_to_file_pretty(os.path.join(base_path, f'{exp_name_label}_stats.json'), stats)

    # Plot results
    plot_client_experiments_graphs(cjtxs, stats, 'emul', target_as, base_path, True)

    return cjtxs, stats


def full_analyze_as25_202405(base_path: str):
    """
    Analyze and plot results for real client-side experiment for zksnacks coordinator in 05/2024,
     with wallet's anonscore target set to 25.
    :param base_path: base directory with client-side files
    :return:
    """
    # Experiment configuration
    target_path = os.path.join(base_path, 'as25\\')
    experiment_start_cut_date = '2024-05-14T19:02:49+00:00'  # AS=25 experiment start time
    experiment_target_anonscore = 25
    problematic_sessions = ['mix1 0.1btc | 12 cjs | txid: 34']  # Failed experiments to be removed from processing
    wallets_names = ['mix1', 'mix2', 'mix3']

    # Generate download scripts for wallet transactions
    create_download_script(wallets_names, target_path, 'download_as25.sh')

    return analyze_ww2_artifacts(target_path, 'all', experiment_start_cut_date, experiment_target_anonscore,
                          wallets_names, problematic_sessions, 23)


def full_analyze_as25_202405_only1m(base_path: str):
    # Experiment configuration
    target_path = os.path.join(base_path, 'as25\\')
    experiment_start_cut_date = '2024-05-14T19:02:49+00:00'  # AS=25 experiment start time
    experiment_target_anonscore = 25
    problematic_sessions = ['mix1 0.1btc | 12 cjs | txid: 34', 'mix2 0.2btc']  # Remove all 0.2 sessions + one problematic 0.1
    wallets_names = ['mix1', 'mix2', 'mix3']
    return analyze_ww2_artifacts(target_path, 'only0.1btc', experiment_start_cut_date, experiment_target_anonscore,
                          wallets_names, problematic_sessions, 16, '_0.1btc_')


def full_analyze_as25_202405_only2m(base_path: str):
    # Experiment configuration
    target_path = os.path.join(base_path, 'as25\\')
    experiment_start_cut_date = '2024-05-14T19:02:49+00:00'  # AS=25 experiment start time
    experiment_target_anonscore = 25
    problematic_sessions = ['mix1 0.1', 'mix2 0.1', 'mix3 0.1']  # remove all 0.1 sessions
    wallets_names = ['mix1', 'mix2', 'mix3']
    return analyze_ww2_artifacts(target_path, 'only0.2btc', experiment_start_cut_date, experiment_target_anonscore,
                          wallets_names, problematic_sessions, 7, '_0.2btc_')


def full_analyze_as38_202503(base_path: str):
    """
    Analyze and plot results for real client-side experiment for kruw.io coordinator in 03/2025,
     with wallet's anonscore target set to 38.
    :param base_path: base directory with client-side files
    :return:
    """
    # Experiment configuration
    target_path = os.path.join(base_path, 'as38\\')
    experiment_start_cut_date = '2025-03-09T00:02:49+00:00'  # AS=38 experiment start time
    experiment_target_anonscore = 38
    #experiment_target_anonscore = 10
    problematic_sessions = ['mix7 0.1btc | 3 cjs | txid: 3493c971d']  # Failed experiments to be removed from processing
    wallets_names = ['mix6', 'mix7']

    # Generate download scripts for wallet transactions
    create_download_script(wallets_names, target_path, 'download_as38.sh')

    return analyze_ww2_artifacts(target_path, 'all', experiment_start_cut_date, experiment_target_anonscore,
                          wallets_names, problematic_sessions, 23)


def plot_client_experiments_graphs(all_cjs: dict, all_stats: dict, exp_label: str, experiment_target_anonscore: int,
                                   target_path: str, wide_fig: bool):
    if wide_fig:
        NUM_COLUMNS = 4
        NUM_ROWS = 5
        fig = plt.figure(figsize=(20, NUM_ROWS * 2.5))
    else:
        NUM_COLUMNS = 3
        NUM_ROWS = 6
        fig = plt.figure(figsize=(10, NUM_ROWS * 2.5))

    mfig = Multifig(plt, fig, NUM_ROWS, NUM_COLUMNS)

    # Plot graphs
    plot_cj_anonscores(mfig, all_stats['anon_percentage_status'],
                       f'Progress towards fully anonymized liquidity (AS={experiment_target_anonscore})',
                       len(all_stats['anon_percentage_status']),
                       f'{experiment_target_anonscore}', 'Privacy progress (%)', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['anon_gain'],
                       f'Change in anonscore weighted (AS={experiment_target_anonscore})',
                       len(all_stats['anon_gain']),
                       f'{experiment_target_anonscore}', 'Anonscore gain', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['observed_mix_liquidity'],
                       f'Mix liquidity (AS={experiment_target_anonscore})',
                       len(all_stats['observed_mix_liquidity']),
                       f'{experiment_target_anonscore}', 'Cummulative mixed value (sats)', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['skipped_cjtxs'],
                       f'Skipped cjtxs', len(all_stats['skipped_cjtxs']),
                       f'{experiment_target_anonscore}', 'num cjtxs skipped', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['num_inputs'],
                       f'Number of inputs', len(all_stats['num_inputs']),
                       f'{experiment_target_anonscore}', 'number of inputs', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['num_outputs'],
                       f'Number of outputs', len(all_stats['num_outputs']),
                       f'{experiment_target_anonscore}', 'number of outputs', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['anon_gain_ratio'],
                       f'Change in anonscore weighted ratio out/in (AS={experiment_target_anonscore})',
                       len(all_stats['anon_gain']),
                       f'{experiment_target_anonscore}', 'Anonscore gain (weighted, ratio)', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['observed_remix_liquidity_ratio_cumul'],
                       f'Cumulative remix liquidity ratio (AS={experiment_target_anonscore})',
                       len(all_stats['observed_remix_liquidity_ratio_cumul']),
                       f'{experiment_target_anonscore}', 'Cumulative remix ratio', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['anon_gain_weighted'],
                       f'Nocap progress towards fully anonymized liquidity (AS={experiment_target_anonscore})',
                       len(all_stats['anon_gain_weighted']),
                       f'{experiment_target_anonscore}', 'Privacy progress (%)', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['num_wallet_coins'],
                       f'Number of wallet coins (AS={experiment_target_anonscore})',
                       len(all_stats['num_wallet_coins']),
                       f'{experiment_target_anonscore}', '# coins', 'royalblue')

    # Plot heatmap of relation between number of registered inputs and outputs
    x, y = [], []
    for session in all_stats['num_inputs'].keys():
        x.extend(all_stats['num_inputs'][session])
        y.extend(all_stats['num_outputs'][session])
    title = 'Frequency of inputs to outputs pairs' if LONG_LEGEND else 'Freq. of input/output nums'
    plot_cj_heatmap(mfig, x, y, 'number of inputs', 'number of outputs', title, True)

    # Plot correlation of total coins in wallet and number of registered inputs/outputs
    x, y_ins, y_outs = [], [], []
    for session in all_stats['num_wallet_coins'].keys():
        x.extend(all_stats['num_wallet_coins'][session])
        y_ins.extend(all_stats['num_inputs'][session])
        y_outs.extend(all_stats['num_outputs'][session])
    title = 'Number of inputs to coins in wallet' if LONG_LEGEND else 'Num. inputs to total coins'
    plot_cj_heatmap(mfig, x, y_ins, 'number of coins', 'number of inputs', title, False)
    title = 'Number of outputs to coins in wallet' if LONG_LEGEND else 'Num. outputs to total coins'
    plot_cj_heatmap(mfig, x, y_outs, 'number of coins', 'number of outputs', title, False)
    # Plot the same, but normalized
    title = 'Number of inputs to coins in wallet (norm)' if LONG_LEGEND else 'Num. inputs to total coins (norm)'
    plot_cj_heatmap(mfig, x, y_ins, 'number of coins', 'number of inputs', title, False, True)
    title = 'Number of outputs to coins in wallet (norm)' if LONG_LEGEND else 'Num. outputs to total coins (norm)'
    plot_cj_heatmap(mfig, x, y_outs, 'number of coins', 'number of outputs', title, False, True)

    # # Plot the same as colored scatterplot
    # ax = mfig.add_subplot()
    # ax.scatter(x, y_ins, color='red', s=1, alpha=0.3)
    # ax.scatter(x, y_outs, color='green', s=1, alpha=0.3)
    # ax.set_xlabel('number of coins')
    # ax.set_ylabel('number of inputs/outputs')
    # ax.set_title(title)

    # Plot histogram of hidden coordination fees (cfee)
    ax = mfig.add_subplot()
    data_mfee = [all_cjs['sessions'][session_label]['coinjoins'][cjtxid]['wallet_fair_mfee'] for session_label in
                 all_cjs['sessions'].keys() for cjtxid in all_cjs['sessions'][session_label]['coinjoins'].keys()]
    data_ctips = [all_cjs['sessions'][session_label]['coinjoins'][cjtxid]['wallet_hidden_ctip_paid'] for
                  session_label in all_cjs['sessions'].keys() for cjtxid in
                  all_cjs['sessions'][session_label]['coinjoins'].keys()]
    data_ctips_small = [value for value in data_ctips if value < 10000]
    print(f'Mining fee sum={sum(data_mfee)}')
    print(f'Hidden ctip (sum={sum(data_ctips)}): {sorted(data_ctips)}')
    label = f'Fair mining fee: {sum(data_mfee)} sats' if LONG_LEGEND else f'Mining fee'
    ax.hist(data_mfee, bins=30, color='green', edgecolor='black', alpha=0.5, label=label)
    label = f'Hidden coord. tips: {sum(data_ctips)} sats' if LONG_LEGEND else f'Hidden coord. tips'
    ax.hist(data_ctips_small, bins=100, color='red', edgecolor='black', alpha=0.5, label=label)
    ax.set_xlabel('fee (sats)')
    ax.set_ylabel('# occurences')
    max_x = ax.get_xlim()[1]
    if max_x < 9000:  # Unify axis for real as25 and as38 experiments
        ax.set_xlim(0, 9000)
    title = 'Distribution of mining and hidden coordination tips' if LONG_LEGEND else 'Mining & hidden coord. tips'
    ax.set_title(title)
    ax.legend()

    # Plot histogram of anonscores at the last round of session (when all coins are mixed)
    ax = mfig.add_subplot()
    data_anonscores = []
    for session_label in all_stats['anonscore_coins_distribution'].keys():
        data_anonscores.extend(all_stats['anonscore_coins_distribution'][session_label][-1])
    # num_bins = math.ceil(max(data_anonscores)) - math.floor(min(data_anonscores))
    bins = np.arange(min(data_anonscores), max(data_anonscores) + 1, 1)
    ax.hist(data_anonscores, bins=bins, color='green', edgecolor='black', alpha=0.5, label=f'Anonscore frequency')
    ax.axvline(experiment_target_anonscore, color='r', linestyle='--',
               label=f"Target anonscore={experiment_target_anonscore}")
    ax.set_xlabel('Anonscore')
    ax.set_ylabel('# occurences')
    title = 'Distribution of anonscores at the end of mixing sessions' if LONG_LEGEND else 'Final anonscores distribution'
    ax.set_title(title)
    ax.legend()

    # Plot frequency of values in coins
    ax = mfig.add_subplot()
    data_coinvalues = []
    for session_label in all_stats['wallet_coins_values'].keys():
        data_coinvalues.extend(all_stats['wallet_coins_values'][session_label])
    counts = Counter(data_coinvalues)
    items = sorted(counts.items(), key=lambda kv: kv[0])
    labels = [k for k, v in items]  # x values
    values = [v for k, v in items]  # counts
    x = np.arange(len(labels))
    ax.bar(x, values, width=0.9)
    n_ticks = 10
    tick_idx = np.linspace(0, len(labels) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx, [labels[i] for i in tick_idx], rotation=90)
    ax.set_xlabel('values (sats)')
    ax.set_ylabel('# occurences')
    ax.set_title('Distribution of output values' if LONG_LEGEND else 'Output values distribution')


    sessions_lengths = [len(all_cjs['sessions'][session]['coinjoins']) for session in all_cjs['sessions'].keys()]
    print(f"Total sessions={len(all_cjs['sessions'].keys())}, total coinjoin txs={sum(sessions_lengths)}")
    print(
        f'Session lengths (#cjtxs): median={round(np.median(sessions_lengths), 2)}, average={round(np.average(sessions_lengths), 2)}, min={min(sessions_lengths)}, max={max(sessions_lengths)}')

    total_output_coins = [all_stats['num_coins'][session] for session in all_stats['num_coins']]
    print(f'Total output coins: {sum(total_output_coins)}')

    total_overmixed_coins = [all_stats['num_overmixed_coins'][session] for session in
                             all_stats['num_overmixed_coins']]
    print(f'Total overmixed input coins: {sum(total_overmixed_coins)}')

    # num_skipped = list(chain.from_iterable(all_stats['skipped_cjtxs'][session] for session in all_stats['skipped_cjtxs']))
    # print(f'Skipped txs stats: median={np.median(num_skipped)}, average={round(np.average(num_skipped), 2)}, min={min(num_skipped)}, max={max(num_skipped)}')

    remix_ratios = [max(all_stats['observed_remix_liquidity_ratio_cumul'][session]) for session in
                    all_stats['observed_remix_liquidity_ratio_cumul'].keys()]
    print(
        f'Remix ratios: median={round(np.median(remix_ratios), 2)}, average={round(np.average(remix_ratios), 2)}, min={round(min(remix_ratios), 2)}, max={round(max(remix_ratios), 2)}')

    expected_remix_fraction = round((np.average(remix_ratios) / (np.average(remix_ratios) + 1)) * 100, 2)
    print(f'Expected remix fraction: {expected_remix_fraction}%')

    num_inputs = list(chain.from_iterable(all_stats['num_inputs'][session] for session in all_stats['num_inputs']))
    print(
        f'Input stats: median={np.median(num_inputs)}, average={round(np.average(num_inputs), 2)}, min={min(num_inputs)}, max={max(num_inputs)}')

    num_outputs = list(
        chain.from_iterable(all_stats['num_outputs'][session] for session in all_stats['num_outputs']))
    print(
        f'Output stats: median={np.median(num_outputs)}, average={round(np.average(num_outputs), 2)}, min={min(num_outputs)}, max={max(num_outputs)}')

    progress_100 = len(
        [all_stats['anon_percentage_status'][session][0] for session in all_stats['anon_percentage_status'] if
         all_stats['anon_percentage_status'][session][0] > 99])
    print(
        f"Anonscore target of {experiment_target_anonscore} hit already during first coinjoin for {progress_100} of {len(all_stats['anon_percentage_status'])} sessions {round(progress_100 / len(all_stats['anon_percentage_status']) * 100, 2)}%")

    anonscore_gains = list(
        chain.from_iterable(all_stats['anon_gain'][session] for session in all_stats['anon_gain']))
    geometric_mean = np.exp(np.mean(np.log(anonscore_gains)))
    print(
        f'Anonscore (weighted) gain per one coinjoin: median={round(np.median(anonscore_gains), 2)}, average={round(np.average(anonscore_gains), 2)}, geometric average={round(geometric_mean, 2)}, min={round(min(anonscore_gains), 2)}, max={round(max(anonscore_gains), 2)}')

    anonscore_gains = list(
        chain.from_iterable(all_stats['anon_gain_ratio'][session] for session in all_stats['anon_gain']))
    geometric_mean = np.exp(np.mean(np.log(anonscore_gains)))
    print(
        f'Anonscore (weighted) ratio gain per one coinjoin: median={round(np.median(anonscore_gains), 2)}, average={round(np.average(anonscore_gains), 2)}, geometric average={round(geometric_mean, 2)}, min={round(min(anonscore_gains), 2)}, max={round(max(anonscore_gains), 2)}')

    # TODO: Probability of coin selection based on its current anonymity score

    # save graph
    mfig.plt.suptitle(f'{Path(target_path).name}, as{experiment_target_anonscore}_{exp_label}', fontsize=16)
    mfig.plt.subplots_adjust(bottom=0.1, wspace=0.5, hspace=0.5)
    save_file = os.path.join(target_path, f'as{experiment_target_anonscore}_{exp_label}_coinjoin_stats')
    mfig.plt.savefig(f'{save_file}.png', dpi=300)
    mfig.plt.savefig(f'{save_file}.pdf', dpi=300)
    mfig.plt.close()


def analyze_ww2_artifacts(target_path: str, exp_label: str, experiment_start_cut_date: str, experiment_target_anonscore: int,
                          wallets_names: list, problematic_sessions: list, assert_num_expected_sessions: int, addon_label: str=""):
    all_cjs = {}
    all_stats = {}

    def filter_sessions(data: dict, remove_sessions: list):
        """
        Filter sessions listed in remove_sessions from the results collected
        :param data: results collected (to be filtered)
        :param remove_sessions: session prefixes to be removed
        :return: filtered results
        """
        for remove_session in remove_sessions:
            for session in list(data['anon_percentage_status'].keys()):
                if session.find(remove_session) != -1:
                    for stat_name in data.keys():
                        if session in data[stat_name].keys():
                            data[stat_name].pop(session)
        return data

    def analyze_mix(target_path, mix_name, experiment_target_anonscore, experiment_start_cut_date, problematic_sessions):
        cjs, wallet_stats = analyze_multisession_mix_experiments(target_path, mix_name, experiment_target_anonscore, experiment_start_cut_date)
        wallet_stats = filter_sessions(wallet_stats, problematic_sessions)
        for to_remove in problematic_sessions:
            if len(to_remove) > 0:
                for session in list(cjs['sessions'].keys()):
                    if session.find(to_remove) != -1:
                        cjs['sessions'].pop(session)
        return cjs, wallet_stats


    for wallet_name in wallets_names:
        cjs, wallet_stats = analyze_mix(target_path, wallet_name, experiment_target_anonscore, experiment_start_cut_date, problematic_sessions)
        als.merge_dicts(cjs, all_cjs)
        als.merge_dicts(wallet_stats, all_stats)
    if assert_num_expected_sessions > -1:
        assert len(all_stats['anon_percentage_status']) == assert_num_expected_sessions, f"Unexpected number of coinjoin sessions {len(all_stats['anon_percentage_status'])}"

    # Save extracted information
    save_path = os.path.join(target_path, f'as{experiment_target_anonscore}{addon_label}coinjoin_tx_info.json')
    als.save_json_to_file_pretty(save_path, all_cjs)
    save_path = os.path.join(target_path, f'as{experiment_target_anonscore}{addon_label}stats.json')
    als.save_json_to_file_pretty(save_path, all_stats)

    plot_client_experiments_graphs(all_cjs, all_stats, exp_label, experiment_target_anonscore, target_path, True)

    return all_stats, all_cjs


def plot_ww2mix_stats(mfig, all_stats: dict, experiment_label: str, experiment_target_anonscore: str, color: str, avg_color: str):
    # Plot graphs
    index = 0
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['anon_percentage_status'], f'Progress towards fully anonymized liquidity (as={experiment_label})', len(all_stats['anon_percentage_status']),
                       experiment_target_anonscore, 'Privacy progress (%)', color, avg_color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['anon_gain'], f'Change in anonscore weighted (as={experiment_label})', len(all_stats['anon_gain']),
                       experiment_target_anonscore, 'Anonscore gain', color, avg_color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['anon_gain_ratio'], f'Change in anonscore weighted ratio out/in (as={experiment_label})', len(all_stats['anon_gain']),
                       experiment_target_anonscore, 'Anonscore gain (weighted, ratio)', color, avg_color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['observed_remix_liquidity_ratio_cumul'], f'Cumullative remix liquidity ratio (as={experiment_label})', len(all_stats['observed_remix_liquidity_ratio_cumul']),
                       experiment_target_anonscore, 'Cummulative remix ratio', color, avg_color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['num_inputs'],
                       f'Number of inputs', len(all_stats['num_inputs']),
                       experiment_target_anonscore, 'number of inputs', color, avg_color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['num_outputs'],
                       f'Number of outputs', len(all_stats['num_outputs']),
                       experiment_target_anonscore, 'number of outputs', color, avg_color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['num_wallet_coins'], f'Number of wallet coins', len(all_stats['num_wallet_coins']),
                       f'{experiment_target_anonscore}', '# coins', color, avg_color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['observed_mix_liquidity'],
                       f'Mixed liquidity per coinjoin', len(all_stats['observed_mix_liquidity']),
                       experiment_target_anonscore, 'sum of mixed inputs', color, avg_color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['skipped_cjtxs'],
                       f'Skipped cjtxs', len(all_stats['skipped_cjtxs']),
                       experiment_target_anonscore, 'num cjtxs skipped', color, avg_color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['anon_gain_weighted'], f'Privacy gain sum weighted', len(all_stats['anon_gain_weighted']),
                       experiment_target_anonscore, 'Privacy gain', color, avg_color)


def create_download_script(wallets_names: list, target_path: str, file_name: str):
    """
    Generate download script for hex versions of provided transactions.
    :param all_cjtxs: list of cjtxs to download
    :param file_name: output name of download sript with all generated commands
    :return:
    """
    history_all = []
    for wallet_name in wallets_names:
        file_path = os.path.join(target_path, f'{wallet_name}_history.json')
        history_all.extend(als.load_json_from_file(file_path)['result'])
    cjtxs = [item['tx'] for item in history_all]

    curl_lines = []
    for cjtx in cjtxs:
        curl_str = "curl --user user:password --data-binary \'{\"jsonrpc\": \"1.0\", \"id\": \"curltest\", \"method\": \"getrawtransaction\", \"params\": [\"" + cjtx + "\", true]}\' -H \'Content-Type: application/json\' http://127.0.0.1:8332/" + f" > {cjtx}.json\n"
        curl_lines.append(curl_str)
    with open(file_name, 'w') as f:
        f.writelines(curl_lines)



def _z_from_alpha(alpha: float) -> float:
    table = {
        0.10: 1.6448536269514722,
        0.05: 1.959963984540054,
        0.02: 2.3263478740408408,
        0.01: 2.5758293035489004,
        0.001: 3.2905267314919255,
    }
    if alpha in table:
        return table[alpha]
    raise ValueError(
        f"alpha={alpha} not in supported set {sorted(table.keys())}. "
        "Add a numeric inverse-CDF if you need arbitrary alpha."
    )

def estimate_wallets_from_inputs(
    num_inputs: Iterable[float],
    Y: float,
    alpha: float = 0.05,
    clip_min: float | None = None,
) -> Dict[str, float | int | Tuple[float, float]]:
    """
    N_hat = Y / mu_hat
    SE_hat = sqrt( (N_hat * sigma2_hat) / mu_hat^2  +  (Y^2 / mu_hat^4) * (sigma2_hat / m) )
    CI = N_hat ± z_{1-alpha/2} * SE_hat
    Also returns mean(K)=mu_hat and ratio Y / mean(K).
    """
    arr = np.asarray(list(num_inputs), dtype=float)
    m = int(arr.size)
    if m < 2:
        raise ValueError("Need at least 2 observations in num_inputs to estimate variance.")
    mu_hat = float(arr.mean())
    if mu_hat <= 0:
        raise ValueError(f"Mean of num_inputs must be > 0; got {mu_hat}.")
    sigma2_hat = float(arr.var(ddof=1))  # unbiased sample variance

    N_hat = float(Y) / mu_hat  # == Y / mean(K)

    SE_hat = math.sqrt(
        (N_hat * sigma2_hat) / (mu_hat ** 2) +
        ((Y ** 2) / (mu_hat ** 4)) * (sigma2_hat / m)
    )

    z = _z_from_alpha(alpha)
    ci_lo = N_hat - z * SE_hat
    ci_hi = N_hat + z * SE_hat

    result: Dict[str, float | int | Tuple[float, float]] = {
        "m": m,
        "mu_hat": mu_hat,           # mean of K
        "sigma2_hat": sigma2_hat,
        "Y_over_meanK": N_hat,      # explicitly expose Y / mean(K)
        "N_hat": N_hat,
        "SE_hat": SE_hat,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "alpha": alpha,
        "confidence": 1 - alpha,
    }

    if clip_min is not None:
        result["N_hat_clipped"] = max(N_hat, float(clip_min))
        result["ci_lo_clipped"] = max(ci_lo, float(clip_min))

    return result


def precompute_wallet_nums_stats(cj_stats: dict, distrib_key: str, values_range: list, drop_front_offset: int=0,
                                 alpha: float=0.05, clip_min: float=1.0):
    """
    Precomputes predictions for wallet numbers for the given observed distribution of number of inputs/outputs
    during client-side experiments.
    :param cj_stats: all data collected during client-side experiments
    :param distrib_key: string 'number_inputs' or 'number_outputs' identifying which distribution we want to process
    :param values_range: range of values we want to precompute wallet prediction for
    :param drop_front_offset: if required, first drop_front_offset coinjoins are dropped from distribution
            of each session (rationale: if we start with single utxo, then wallet cannot register more. Dropping will
            remove coinjoins till wallet has enough utxos to select from)
    :param alpha: confidence interval level
    :param clip_min: clipping minimum level
    :return: directory with precomputed values
    """
    results = {}
    num_inputs = []
    for session in cj_stats[distrib_key]:
        num_inputs.extend(cj_stats[distrib_key][session][drop_front_offset:])
    print(f'precompute_wallet_nums_stats() computed from {len(num_inputs)} dataset size')
    for Y in values_range:
        results[Y] = estimate_wallets_from_inputs(num_inputs, int(Y), alpha=alpha, clip_min=clip_min)
        #print(f"  {Y}:\t N={results[Y]['N_hat']:.1f} wallets, 95% CI [{results[Y]['ci_lo']:.1f}, {results[Y]['ci_hi']:.1f}] (Wald CI via delta method using our measured K)")

    return results


def create_wallet_estimation_matrix(cj_stats: dict, values_range: list):
    full_matrix = {}
    clip_min = 1.0
    for alpha in [0.1, 0.05, 0.01]:
        full_matrix[alpha] = {}
        DROP_INITIAL_COINJOINS_INPUTS = 5  # Drop first X coinjoins from each session from inputs till wallet does not have enough utxos
        DROP_INITIAL_COINJOINS_OUTPUTS = 0  # No dropping from outputs which is more stable from first coinjoin
        full_matrix[alpha]['inputs'] = precompute_wallet_nums_stats(cj_stats, 'num_inputs', values_range,
                                                                    DROP_INITIAL_COINJOINS_INPUTS, alpha, clip_min)
        full_matrix[alpha]['outputs'] = precompute_wallet_nums_stats(cj_stats, 'num_outputs', values_range,
                                                                     DROP_INITIAL_COINJOINS_OUTPUTS, alpha, clip_min)
    return full_matrix

def visualize_estimate_wallet_bounds(base_path: str, cj_stats1: dict, prefix1: str, color1: str, cj_stats2: dict, prefix2: str, color2: str, alpha: float=0.05, clip_min: float=1.0):
    # Estimate wallets estimation bounds
    plt.figure(figsize=(10, 6))

    def plot_wallets_stats(values_range, N_hat_list, ci_hi_list, ci_lo_list, line_color: str, prefix: str):
        plt.plot(values_range, N_hat_list, label=f"{prefix}: num. wallets point estimate", color=line_color, alpha=0.7, linewidth=3)
        plt.plot(values_range, ci_hi_list, label=f"{prefix}: upper bound (CI=95%)", linestyle='-.', color=line_color, alpha=0.7)
        plt.plot(values_range, ci_lo_list, label=f"{prefix}: lower bound (CI=95%)", linestyle='--', color=line_color, alpha=0.7)

    def compute_and_plot(cj_stats: dict, distrib_key: str, line_color: str, prefix: str):
        # Compute across Y = 10..700
        values_range = list(range(10, 701, 1))
        precomputed = precompute_wallet_nums_stats(cj_stats, distrib_key, values_range, 0, alpha, clip_min)
        Ys = list(precomputed.keys())
        N_hat_list: List[float] = []
        ci_lo_list: List[float] = []
        ci_hi_list: List[float] = []
        for Y in Ys:
            N_hat_list.append(precomputed[Y]["N_hat"])
            ci_lo_list.append(precomputed[Y]["ci_lo"])
            ci_hi_list.append(precomputed[Y]["ci_hi"])
        plot_wallets_stats(values_range, N_hat_list, ci_lo_list, ci_hi_list, line_color, prefix)
        return precomputed

    if cj_stats2:
        compute_and_plot(cj_stats2, 'num_outputs', 'green', f'{prefix2} [outputs]')
    if cj_stats1:
        compute_and_plot(cj_stats1, 'num_outputs', 'blue', f'{prefix1} [outputs]')
    if cj_stats2:
        compute_and_plot(cj_stats2, 'num_inputs', 'magenta', f'{prefix2} [inputs]')
    if cj_stats1:
        compute_and_plot(cj_stats1, 'num_inputs', 'red', f'{prefix1} [inputs]')
    # compute_and_plot(cj_stats1, 'num_inputs', color1, f'{prefix1}_in')
    # compute_and_plot(cj_stats1, 'num_outputs', 'coral', f'{prefix1}_out')
    # compute_and_plot(cj_stats2, 'num_inputs', color2, f'{prefix2}_in')
    # compute_and_plot(cj_stats2, 'num_outputs', 'darkblue', f'{prefix2}_out')

    plt.xlabel("Number of coinjoin inputs/outputs")
    plt.ylabel("Estimated number of wallets")
    plt.title("Wallet count estimate and 95% Wald CI vs # coinjoin inputs/outputs")
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    save_file = os.path.join(base_path, f'{prefix1}_{prefix2}_wallet_predict_confidence')
    plt.savefig(f'{save_file}.png', dpi=300)
    plt.savefig(f'{save_file}.pdf', dpi=300)
    plt.close()



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # --cjtype ww2 --action process_dumplings --action detect_false_positives --target-path c:\!blockchains\CoinJoin\Dumplings_Stats_20241225\
    parser.add_argument("-t", "--cjtype",
                        help="Type of coinjoin. 'ww1'...Wasabi 1.x; 'ww2'...Wasabi 2.x; 'sw'...Samourai Whirlpool; 'jm'...JoinMarket ",
                        choices=["ww1", "ww2", "sw", "jm"],
                        action="store", metavar="TYPE",
                        required=False)
    parser.add_argument("-a", "--action",
                        help="Action to performed. Can be multiple. 'process_as25as38'...process data from real client-side experiments; "
                             "'process_emulations'...process data from coinjoin emulations;"
                             "'compute_num_predict_matrix'...compute wallet number prediction matrix",
                        choices=["process_as25as38", "process_emulations", "compute_num_predict_matrix"],
                        action="append", metavar="ACTION",
                        required=False)
    parser.add_argument("-tp", "--target-path",
                        help="Target path with experiment(s) to be processed. Can be multiple.",
                        action="store", metavar="PATH",
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

class ClientParseOptions:
    DEBUG = False
    # Limit analysis only to specific coinjoin type
    CJ_TYPE = CoinjoinType.WW2
    SORT_COINJOINS_BY_RELATIVE_ORDER = True

    PROCESS_AS25AS38 = False
    PROCESS_EMULATIONS = False

    COMPUTE_NUMWALLETS_PREDICT_MATRIX = False

    TARGET_AS = 5

    target_base_path = ''

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
                if act == 'process_as25as38':
                    self.PROCESS_AS25AS38 = True
                    self.COMPUTE_NUMWALLETS_PREDICT_MATRIX = True
                if act == 'process_emulations':
                    self.PROCESS_EMULATIONS = True
                if act == 'compute_num_predict_matrix':
                    self.COMPUTE_NUMWALLETS_PREDICT_MATRIX = True

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
        self.PROCESS_AS25AS38 = False
        self.PROCESS_EMULATIONS = False

        self.COMPUTE_NUMWALLETS_PREDICT_MATRIX = False

        self.TARGET_AS = 5


    def print_attributes(self):
        print('*******************************************')
        print('ClientExpParseOptions parameters:')
        for attr, value in vars(self).items():
            print(f'  {attr}={value}')
        print('*******************************************')


def main(argv=None):
    global op
    op = ClientParseOptions()
    args = parse_arguments(argv)  # parse arguments, overwrite default settings if required
    op.set_args(args)

    als.SORT_COINJOINS_BY_RELATIVE_ORDER = False

    if op.PROCESS_EMULATIONS:
        als.SORT_COINJOINS_BY_RELATIVE_ORDER = False

        # Analyze produced emulations and plot graphs
        all_emu, all_emu_stats = full_analyze_emulations(op.target_base_path, op.TARGET_AS, 'emu')
        # Plot estimated prediction error bounds
        visualize_estimate_wallet_bounds(op.target_base_path, all_emu_stats, 'emu', 'lightcoral', None, '', 'royalblue')
        # Precompute and save wallet prediction matrix
        values_range = list(range(1, 701, 1))
        full_matrix_emu = create_wallet_estimation_matrix(all_emu_stats, values_range)
        als.save_json_to_file_pretty(os.path.join(op.target_base_path, 'wallet_estimation_matrix_emul.json'), full_matrix_emu)


    if op.PROCESS_AS25AS38:
        #op.target_base_path = 'c:/!blockchains/CoinJoin/WasabiWallet_experiments/mn1_temp/'
        # Full analysis of as25 and as38 client side real experiments
        all38_stats, all38 = full_analyze_as38_202503(op.target_base_path)
        all25_stats, all25 = full_analyze_as25_202405(op.target_base_path)
        als.save_json_to_file_pretty(os.path.join(op.target_base_path, 'as38', 'as38_all_stats.json'), all38_stats)
        als.save_json_to_file_pretty(os.path.join(op.target_base_path, 'as25', 'as25_all_stats.json'), all25_stats)
        all25_1m_stats, all25_1m = full_analyze_as25_202405_only1m(op.target_base_path)
        all25_2m_stats, all25_2m = full_analyze_as25_202405_only2m(op.target_base_path)

        NARROW_FIGURES = True
        NUM_COLUMNS = 3 if NARROW_FIGURES else 3
        NUM_ROWS = 6     # 5
        fig = plt.figure(figsize=(20, NUM_ROWS * 4))
        mfig = Multifig(plt, fig, NUM_ROWS, NUM_COLUMNS)
        mfig.add_multiple_subplots(10)

        # Plot both experiments into single image
    #    plot_ww2mix_stats(mfig, all25_stats, '25&38', '25', 'royalblue')
        plot_ww2mix_stats(mfig, all25_1m_stats, '25&38', '25', 'royalblue', 'royalblue')
        plot_ww2mix_stats(mfig, all38_stats, '25&38', '38', 'lightcoral', 'lightcoral')
        plot_ww2mix_stats(mfig, all25_2m_stats, '25&38', '25', 'darkblue', 'darkblue')

        # save graph
        mfig.plt.suptitle(f'Combined plots as25 and as38', fontsize=16)  # Adjust the fontsize and y position as needed
        mfig.plt.subplots_adjust(bottom=0.1, wspace=0.5, hspace=0.5)
        save_file = os.path.join(op.target_base_path, 'as25_38_coinjoin_stats')
        mfig.plt.savefig(f'{save_file}.png', dpi=300)
        mfig.plt.savefig(f'{save_file}.pdf', dpi=300)
        mfig.plt.close()


    if op.COMPUTE_NUMWALLETS_PREDICT_MATRIX:
        #op.target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\mn1_temp\\'

        all38_stats = als.load_json_from_file(os.path.join(op.target_base_path, 'as38', 'as38_all_stats.json'))
        all25_stats = als.load_json_from_file(os.path.join(op.target_base_path, 'as25', 'as25_all_stats.json'))
        visualize_estimate_wallet_bounds(op.target_base_path, all25_stats, 'as25', 'lightcoral', all38_stats, 'as38', 'royalblue')

        # Precompute and save wallet prediction matrix
        values_range = list(range(1, 701, 1))
        full_matrix_as25 = create_wallet_estimation_matrix(all25_stats, values_range)
        als.save_json_to_file_pretty(os.path.join(op.target_base_path, 'wallet_estimation_matrix_ww2zksnacks.json'), full_matrix_as25)
        full_matrix_as38 = create_wallet_estimation_matrix(all38_stats, values_range)
        als.save_json_to_file_pretty(os.path.join(op.target_base_path, 'wallet_estimation_matrix_ww2kruw.json'), full_matrix_as38)

    #round_logs = als.parse_client_coinjoin_logs(target_path)
    #exit(42)

    # prison_logs = analyse_prison_logs(target_path)
    # exit(42)

    # TODO: limits stats
    # TODO: Prison time distribution
    # TODO: Add wallclock time for coinjoin


if __name__ == "__main__":
    main()
