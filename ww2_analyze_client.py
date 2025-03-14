import copy
import logging

# Suppress DEBUG logs from a specific library
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import math
import os
from itertools import chain

import orjson
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

import parse_dumplings as dmp
import cj_analysis as als

from collections import defaultdict

SATS_IN_BTC = 100000000


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


def plot_cj_anonscores(mfig: Multifig, data: dict, title: str, y_label: str, color: str, show_txid: bool = False):
    plot_cj_anonscores_ax(mfig.add_subplot(), data, title, y_label, color, show_txid)


def plot_cj_anonscores_ax(ax, data: dict, title: str, y_label: str, line_color: str, show_txid: bool = False):
    #fig, ax = plt.subplots(figsize=(10, 5))
    size_01_used = False
    size_02_used = False
    for cj_session in data.keys():
        line_style = ':'
        if cj_session.find('0.1btc') != -1:
            line_style = 'solid'
            size_01_used = True
        if cj_session.find('0.2btc') != -1:
            line_style = ':'
            size_02_used = True
        cj_label = cj_session
        if not show_txid and cj_session.find('txid:'):
            cj_label = cj_label[0:cj_session.find('txid:')]
        x_range = range(1, len(data[cj_session]) + 1)
        ax.plot(x_range, data[cj_session], color=line_color, linestyle=line_style, alpha=0.15)
    if size_01_used:
        ax.plot([1], [1], color=line_color, label='Input size 0.1 btc', linestyle='solid', alpha=0.5)
    if size_02_used:
        ax.plot([1], [1], color=line_color, label='Input size 0.2 btc', linestyle=':', alpha=0.5)
    ax.set_xticks(np.arange(1, 20, 2))

    def compute_average_at_index(lists, index):
        values = [lists[lst][index] for lst in lists.keys() if index < len(lists[lst])]
        if not values:
            return 0
        return sum(values) / len(values)

    max_index = max([len(data[cj_session]) for cj_session in data.keys()])
    avg_data = [compute_average_at_index(data, index) for index in range(0, max_index)]
    ax.plot(range(1, len(avg_data) + 1), avg_data, label='Average', linestyle='solid',
            linewidth=7, alpha=0.7, color=line_color)

    ax.legend(loc="best", fontsize='8')
    ax.set_title(title)
    ax.set_xlabel('Number of coinjoins executed')
    ax.set_ylabel(y_label)
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
        ax_boxplot = mfig.add_subplot()  # Get next subplot
        ax_boxplot.boxplot(data_cj)
        ax_boxplot.set_title(title)
        ax_boxplot.set_xlabel('Number of coinjoins executed')
        ax_boxplot.set_ylabel(y_label)
        #plt.show()


def get_session_label(mix_name: str, session_size_inputs: int, segment: list, session_funding_tx: dict) -> str:
    # Two options for session label
    cjsession_label_short_date = f'{mix_name} {round(session_size_inputs / SATS_IN_BTC, 1)}btc | {len(segment)} cjs | ' + \
                                 session_funding_tx['broadcast_time'] + ' ' + session_funding_tx['txid'][0:8]
    cjsession_label_short_txid = f'{mix_name} {round(session_size_inputs / SATS_IN_BTC, 1)}btc | {len(segment)} cjs | txid: {session_funding_tx['txid']} '
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
                    coin_data = als.load_json_from_file(file_path)['result']

                # Update highest scores for each coin address
                for coin in coin_data:
                    #for coin_address, score in coin_data.items():
                    highest_scores[coin['address']] = max(highest_scores[coin['address']], coin['anonymityScore'])

            except (FileNotFoundError, PermissionError) as e:
                logging.error(f"Error reading {file_path}: {e}")

    return dict(highest_scores)  # Convert back to regular dictionary


def analyze_as25(target_base_path: str, mix_name: str, target_as: int, experiment_start_date: str):
    target_path = os.path.join(target_base_path, f'{mix_name}_history.json')
    history_all = als.load_json_from_file(target_path)['result']
    target_path = os.path.join(target_base_path, f'{mix_name}_coins.json')
    coins = als.load_json_from_file(target_path)['result']

    # After each merge, anonymity score for merge transaction is set to 1 for all inputs.
    # Search older *_coins.json files and try to find one before experiment coins merge
    intermediate_coins_max_score = find_highest_scores(target_base_path, mix_name)
    for coin in coins:
        if coin['anonymityScore'] == 1:
            coin['anonymityScore'] = intermediate_coins_max_score[coin['address']] if coin['address'] in intermediate_coins_max_score else 1

    target_path = os.path.join(target_base_path, f'coinjoin_tx_info.json')
    coinjoins = als.load_json_from_file(target_path)['coinjoins']
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
                cjtx['inputs'][str(coin['index'])] = coin  # BUGBUG: We do not know correct vin index

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
    for tx in history:
        if tx['islikelycoinjoin'] is True:
            # Inside coinjoin session, append
            record = {'txid': tx['tx'], 'inputs': {}, 'outputs': {}, 'round_id': tx['tx'], 'is_blame_round': False}
            record['round_start_time'] = als.precomp_datetime.fromisoformat(tx['datetime']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            record['broadcast_time'] = als.precomp_datetime.fromisoformat(tx['datetime']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            record['inputs'] = {}
            for index in tx['inputs']:
                record['inputs'][index] = {}
                record['inputs'][index]['address'] = tx['inputs'][index]['address']
                record['inputs'][index]['value'] = tx['inputs'][index]['amount']
                record['inputs'][index]['wallet_name'] = mix_name
                record['inputs'][index]['anon_score'] = tx['inputs'][index]['anonymityScore']

            record['outputs'] = {}
            for index in tx['outputs']:
                record['outputs'][index] = {}
                record['outputs'][index]['address'] = tx['outputs'][index]['address']
                record['outputs'][index]['value'] = tx['outputs'][index]['amount']
                record['outputs'][index]['wallet_name'] = mix_name
                record['outputs'][index]['anon_score'] = tx['outputs'][index]['anonymityScore']

            session_cjtxs[tx['tx']] = record
        else:
            # Non-coinjoin transaction detected (assume initial or final merge)

            # If initial funding tx, then extract input liquidity into session_size_inputs
            if len(tx['outputs']) == 1 and tx['outputs'][list(tx['outputs'].keys())[0]]['amount'] > 0:
                session_size_inputs = tx['outputs'][list(tx['outputs'].keys())[0]]['amount']

            # If final merge transaction detected (some coinjoin txs already detected, use it)
            if len(session_cjtxs) > 0:
                assert len(session_funding_tx['outputs'].keys()) == 1, f'Funding tx has unexpected number of outputs of {len(session_funding_tx['outputs'].keys())}'
                norm_tx = {'txid': session_funding_tx['tx'], 'label': session_funding_tx['label'], 'broadcast_time': session_funding_tx['datetime'], 'value': session_funding_tx['outputs']['0']['amount']}
                session_label = get_session_label(mix_name, session_size_inputs, session_cjtxs, norm_tx)

                als.remove_link_between_inputs_and_outputs(session_cjtxs)
                als.compute_link_between_inputs_and_outputs(session_cjtxs, [cjtxid for cjtxid in session_cjtxs.keys()])

                cjtxs['sessions'][session_label] = {'coinjoins': session_cjtxs, 'funding_tx': norm_tx}
                session_cjtxs = {}


            session_funding_tx = tx

    # Compute basic statistics
    stats = {}
    stats['all_cjs_weight_anonscore'] = {}
    stats['anon_percentage_status'] = {}
    stats['observed_remix_liquidity_ratio'] = {}
    stats['observed_remix_liquidity_ratio_cumul'] = {}  # Remix liquidity based on value of inputs
    stats['observed_remix_inputs_ratio_cumul'] = {}     # unused now, remix liquidity based on number of inputs
    for session_label in cjtxs['sessions'].keys():
        session_coins = {}
        anon_percentage_status_list = []
        observed_remix_liquidity_ratio_list = []
        observed_remix_liquidity_ratio_cumul_list = []
        observed_remix_inputs_ratio_cumul_list = []
        session_size_inputs = cjtxs['sessions'][session_label]['funding_tx']['value']
        assert session_size_inputs > 0, f'Unexpected negative funding tx size of {session_size_inputs}'
        for cjtxid in cjtxs['sessions'][session_label]['coinjoins'].keys():
            cjtx = cjtxs['sessions'][session_label]['coinjoins'][cjtxid]
            print(f'#', end='')
            assert len(cjtx['outputs']) != 0, f'No coins assigned to {cjtx['txid']}'

            for index in cjtx['outputs']:
                if cjtx['outputs'][index]['anon_score'] < target_as:
                    # Print in red - target as not yet reached
                    print("\033[31m" + f' {round(cjtx['outputs'][index]['anon_score'], 1)}' + "\033[0m", end='')
                    # if cjtx['outputs'][index]['anonymityScore'] == 1:
                    #     print(f' {cjtx['outputs'][index]['address']}', end='')
                else:
                    # Print in green - target as reached
                    print("\033[32m" + f' {round(cjtx['outputs'][index]['anon_score'], 1)}' + "\033[0m", end='')

            # Compute privacy progress
            for index in cjtx['inputs']:  # Remove coins from session_coins spend by this cj
                session_coins.pop(cjtx['inputs'][index]['address'], None)
            for index in cjtx['outputs']:  # Add coins to session_coins created by this cj
                session_coins[cjtx['outputs'][index]['address']] = cjtx['outputs'][index]
            anon_percentage_status = 0
            for address in session_coins.keys():
                if session_coins[address]['anon_score'] > 1:
                    effective_as = min(session_coins[address]['anon_score'], target_as)
                    anon_percentage_status += (effective_as / target_as) * (
                            session_coins[address]['value'] / session_size_inputs)
            WARN_TOO_HIGH_PRIVACY_PROGRESS = True
            if WARN_TOO_HIGH_PRIVACY_PROGRESS and anon_percentage_status > 1.01:
                print(f'\nToo large anon_percentage_status {round(anon_percentage_status * 100, 1)}%: {cjtxid}')
                anon_percentage_status = 1
            print(f' {round(anon_percentage_status * 100, 1)}%', end='')
            anon_percentage_status_list.append(anon_percentage_status * 100)

            observed_remix_liquidity_ratio = 0
            for index in cjtx['inputs']:
                observed_remix_liquidity_ratio += cjtx['inputs'][index]['value'] / session_size_inputs
            observed_remix_liquidity_ratio_list.append(observed_remix_liquidity_ratio)
            if len(observed_remix_liquidity_ratio_cumul_list) == 0:
                if not math.isclose(observed_remix_liquidity_ratio, 1.0, rel_tol=1e-9):
                    print(f'\nWarning: Unexpected observed_remix_liquidity_ratio of {observed_remix_liquidity_ratio} instead 1.0')
                observed_remix_liquidity_ratio_cumul_list.append(0)  # The first value is fresh input, not remix
            else:
                observed_remix_liquidity_ratio_cumul_list.append(observed_remix_liquidity_ratio_cumul_list[-1] + observed_remix_liquidity_ratio)

        # Store computed data
        if len(anon_percentage_status_list) > 0:
            assert session_label not in stats['anon_percentage_status'], f'Duplicate session label {session_label}'
            stats['anon_percentage_status'][session_label] = anon_percentage_status_list
        if len(observed_remix_liquidity_ratio_list) > 0:
            assert session_label not in stats['observed_remix_liquidity_ratio'], f'Duplicate session label {session_label}'
            stats['observed_remix_liquidity_ratio'][session_label] = observed_remix_liquidity_ratio_list
            stats['observed_remix_liquidity_ratio_cumul'][session_label] = observed_remix_liquidity_ratio_cumul_list

        # Print finalized info
        session = cjtxs['sessions'][session_label]
        session_end_merge_tx = f'{len(session['coinjoins'].keys())} cjs | ' + session['funding_tx']['label'] + ' ' + session['funding_tx']['broadcast_time'] + ' ' + \
                               session['funding_tx']['txid']
        print("\033[34m" + f' * ' + session_end_merge_tx + "\033[0m", end='')
        cjsession_label_short = get_session_label(mix_name, session_size_inputs, cjtxs['sessions'][session_label]['coinjoins'].keys(), session['funding_tx'])
        print(f' |--> \"{cjsession_label_short}\"', end='')
        print()

    # Number of skipped coinjoins
    sorted_cj_times = als.sort_coinjoins(coinjoins, als.SORT_COINJOINS_BY_RELATIVE_ORDER)
    coinjoins_index = {sorted_cj_times[i]['txid']: i for i in range(0, len(sorted_cj_times))}  # Precomputed mapping of txid to index for fast burntime computation
    # coord_logs_sanitized = [{**item, 'mp_first_seen': item['mp_first_seen'] if item['mp_first_seen'] is not None else item['cj_last_seen']} for item in coord_logs]
    # coord_logs_sorted = sorted(coord_logs_sanitized, key=lambda x: x['mp_first_seen'])
    # coinjoins_index = {coord_logs_sorted[i]['id']: i for i in range(0, len(coord_logs_sorted))}
    stats['skipped_cjtxs'] = {}
    stats['skipped_cjtxs_corrected'] = {}  # TODO: Remove skipped_cjtxs_corrected, is now fixed by relative ordering
    for session_label in cjtxs['sessions'].keys():
        prev_cjtxid = None
        skipped_cjtxs_list = []
        skipped_cjtxs_corrected_list = []
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
            skipped_cjtxs_corrected_list.append(skipped - min_burn_time)
            prev_cjtxid = cjtxid
        stats['skipped_cjtxs'][session_label] = skipped_cjtxs_list
        stats['skipped_cjtxs_corrected'][session_label] = skipped_cjtxs_corrected_list

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

    print(f'\n{mix_name}: Total experiments: {len(cjtxs['sessions'])}, total txs={len(history)}, '
          f'total coins: {sum([stats['num_coins'][session_label] for session_label in stats['num_coins'].keys()])}, '
          f'total overmixed coins: {sum([len([stats['num_overmixed_coins'][session_label] for session_label in stats['num_overmixed_coins'].keys()])])}')

    print("##################################################")

    return cjtxs, stats


def merge_coins_files(base_path: str, file1: str, file2: str):
    coins1 = dmp.load_json_from_file(os.path.join(base_path, file1))['result']
    coins2 = dmp.load_json_from_file(os.path.join(base_path, file2))['result']

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


def plot_cj_heatmap(mfig: Multifig, x, y, x_label, y_label, title):
    heatmap_size = (max(x), max(y))
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=heatmap_size)
    #plt.figure(figsize=(8, 6))
    ax = mfig.add_subplot()
    sns.heatmap(heatmap.T, cmap='viridis', annot=True, fmt='.0f', cbar=True, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(len(xedges) - 1) + 0.5)
    ax.set_yticks(np.arange(len(yedges) - 1) + 0.5)
    ax.set_xticklabels(np.arange(1, len(xedges)))
    ax.set_yticklabels(np.arange(1, len(yedges)))
    ax.set_title(title)
    #plt.show()


def full_analyze_as25():
    # Experiment configuration
    target_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\mn1\\as25\\'
    experiment_start_cut_date = '2024-05-14T19:02:49+00:00'  # AS=25 experiment start time
    experiment_target_anonscore = 25
    problematic_sessions = ['mix1 0.1btc | 12 cjs | txid: 34']  # Failed experiments to be removed from processing

    return analyze_ww2_artifacts(target_path, experiment_start_cut_date, experiment_target_anonscore,
                          ['mix1', 'mix2', 'mix3'], problematic_sessions, 23)


def full_analyze_as38():
    # Experiment configuration
    target_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\mn1\\as38\\'
    experiment_start_cut_date = '2025-03-09T00:02:49+00:00'  # AS=38 experiment start time
    experiment_target_anonscore = 38
    problematic_sessions = []  # Failed experiments to be removed from processing

    return analyze_ww2_artifacts(target_path, experiment_start_cut_date, experiment_target_anonscore,
                          ['mix6'], problematic_sessions, -1)


def analyze_ww2_artifacts(target_path: str, experiment_start_cut_date: str, experiment_target_anonscore: int,
                          wallets_names: list, problematic_sessions: list, assert_num_expected_sessions: int):
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
        cjs, wallet_stats = analyze_as25(target_path, mix_name, experiment_target_anonscore, experiment_start_cut_date)
        wallet_stats = filter_sessions(wallet_stats, problematic_sessions)
        for to_remove in problematic_sessions:
            if len(to_remove) > 0:
                for session in list(cjs['sessions'].keys()):
                    if session.find(to_remove) != -1:
                        cjs['sessions'].pop(session)
        PLOT_FOR_WALLETS = False
        if PLOT_FOR_WALLETS:
            plot_cj_anonscores(wallet_stats['anon_percentage_status'],
                               f'Wallet {mix_name}, progress towards fully anonymized liquidity (anonscore threshold);total sessions={len(wallet_stats['anon_percentage_status'])}',
                               'privacy progress (%)')
            plot_cj_anonscores(wallet_stats['observed_remix_liquidity_ratio_cumul'],
                               f'Wallet {mix_name}, cumullative remix liquidity ratio;total sessions={len(wallet_stats['observed_remix_liquidity_ratio_cumul'])}',
                               'cummulative remix ratio')
            plot_cj_anonscores(wallet_stats['skipped_cjtxs'],
                               f'Wallet {mix_name}, skipped cjtxs;total sessions={len(wallet_stats['skipped_cjtxs'])}',
                               'num cjtxs skipped')
        return cjs, wallet_stats


    NUM_COLUMNS = 2  # 4
    NUM_ROWS = 6     # 5
    fig = plt.figure(figsize=(20, NUM_ROWS * 2.5))
    mfig = Multifig(plt, fig, NUM_ROWS, NUM_COLUMNS)

    for wallet_name in wallets_names:
        cjs, wallet_stats = analyze_mix(target_path, wallet_name, experiment_target_anonscore, experiment_start_cut_date, problematic_sessions)
        als.merge_dicts(cjs, all_cjs)
        als.merge_dicts(wallet_stats, all_stats)
    if assert_num_expected_sessions > -1:
        assert len(all_stats['anon_percentage_status']) == assert_num_expected_sessions, f'Unexpected number of coinjoin sessions {len(all_stats['anon_percentage_status'])}'

    # Save extracted information
    save_path = os.path.join(target_path, f'as{experiment_target_anonscore}_coinjoin_tx_info.json')
    als.save_json_to_file_pretty(save_path, all_cjs)
    save_path = os.path.join(target_path, f'as{experiment_target_anonscore}_stats.json')
    als.save_json_to_file_pretty(save_path, all_stats)

    # Plot graphs
    plot_cj_anonscores(mfig, all_stats['anon_percentage_status'], f'Progress towards fully anonymized liquidity (AS={experiment_target_anonscore}); total sessions={len(all_stats['anon_percentage_status'])}',
                       'Privacy progress (%)', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['anon_gain'], f'All wallets, change in anonscore weighted (AS={experiment_target_anonscore}); total sessions={len(all_stats['anon_gain'])}',
                       'Anonscore gain', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['anon_gain_ratio'], f'All wallets, change in anonscore weighted ratio out/in (AS={experiment_target_anonscore}); total sessions={len(all_stats['anon_gain'])}',
                       'Anonscore gain (weighted, ratio)', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['observed_remix_liquidity_ratio_cumul'], f'All wallets, cumullative remix liquidity ratio (AS={experiment_target_anonscore}); total sessions={len(all_stats['observed_remix_liquidity_ratio_cumul'])}',
                       'Cummulative remix ratio', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['skipped_cjtxs'],
                       f'All wallets, skipped cjtxs;total sessions={len(all_stats['skipped_cjtxs'])}',
                       'num cjtxs skipped', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['skipped_cjtxs_corrected'],
                       f'All wallets, skipped cjtxs corrected by smallest burntime;total sessions={len(all_stats['skipped_cjtxs_corrected'])}',
                       'num cjtxs skipped', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['num_inputs'],
                       f'All wallets, number of inputs;total sessions={len(all_stats['num_inputs'])}',
                       'number of inputs', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['num_outputs'],
                       f'All wallets, number of outputs;total sessions={len(all_stats['num_outputs'])}',
                       'number of outputs', 'royalblue')
    x, y = [], []
    for session in all_stats['num_inputs'].keys():
        x.extend(all_stats['num_inputs'][session])
        y.extend(all_stats['num_outputs'][session])
    plot_cj_heatmap(mfig, x, y, 'number of inputs', 'number of outputs','Occurence frequency of inputs to outputs pairs')

    sessions_lengths = [len(all_cjs['sessions'][session]['coinjoins']) for session in all_cjs['sessions'].keys()]
    print(f'Total sessions={len(all_cjs['sessions'].keys())}, total coinjoin txs={sum(sessions_lengths)}')
    print(f'Session lengths (#cjtxs): median={round(np.median(sessions_lengths), 2)}, average={round(np.average(sessions_lengths), 2)}, min={min(sessions_lengths)}, max={max(sessions_lengths)}')

    total_output_coins = [all_stats['num_coins'][session] for session in all_stats['num_coins']]
    print(f'Total output coins: {sum(total_output_coins)}')

    total_overmixed_coins = [all_stats['num_overmixed_coins'][session] for session in all_stats['num_overmixed_coins']]
    print(f'Total overmixed input coins: {sum(total_overmixed_coins)}')


    # num_skipped = list(chain.from_iterable(all_stats['skipped_cjtxs'][session] for session in all_stats['skipped_cjtxs']))
    # print(f'Skipped txs stats: median={np.median(num_skipped)}, average={round(np.average(num_skipped), 2)}, min={min(num_skipped)}, max={max(num_skipped)}')
    #
    # num_skipped_corrected = list(chain.from_iterable(all_stats['skipped_cjtxs_corrected'][session] for session in all_stats['skipped_cjtxs_corrected']))
    # print(f'Skipped corrected txs stats: median={np.median(num_skipped_corrected)}, average={round(np.average(num_skipped_corrected), 2)}, min={min(num_skipped_corrected)}, max={max(num_skipped_corrected)}')

    remix_ratios = [max(all_stats['observed_remix_liquidity_ratio_cumul'][session]) for session in all_stats['observed_remix_liquidity_ratio_cumul'].keys()]
    print(f'Remix ratios: median={round(np.median(remix_ratios), 2)}, average={round(np.average(remix_ratios), 2)}, min={round(min(remix_ratios), 2)}, max={round(max(remix_ratios), 2)}')

    expected_remix_fraction = round((np.average(remix_ratios) / (np.average(remix_ratios) + 1)) * 100, 2)
    print(f'Expected remix fraction: {expected_remix_fraction}%')

    num_inputs = list(chain.from_iterable(all_stats['num_inputs'][session] for session in all_stats['num_inputs']))
    print(f'Input stats: median={np.median(num_inputs)}, average={round(np.average(num_inputs), 2)}, min={min(num_inputs)}, max={max(num_inputs)}')

    num_outputs = list(chain.from_iterable(all_stats['num_outputs'][session] for session in all_stats['num_outputs']))
    print(f'Output stats: median={np.median(num_outputs)}, average={round(np.average(num_outputs), 2)}, min={min(num_outputs)}, max={max(num_outputs)}')

    progress_100 = len([all_stats['anon_percentage_status'][session][0] for session in all_stats['anon_percentage_status'] if all_stats['anon_percentage_status'][session][0] > 99])
    print(f'Anonscore target of {experiment_target_anonscore} hit already during first coinjoin for {progress_100} of {len(all_stats['anon_percentage_status'])} sessions {round(progress_100 / len(all_stats['anon_percentage_status']) * 100, 2)}%')

    anonscore_gains = list(chain.from_iterable(all_stats['anon_gain'][session] for session in all_stats['anon_gain']))
    geometric_mean = np.exp(np.mean(np.log(anonscore_gains)))
    print(f'Anonscore (weighted) gain per one coinjoin: median={round(np.median(anonscore_gains), 2)}, average={round(np.average(anonscore_gains), 2)}, geometric average={round(geometric_mean, 2)}, min={round(min(anonscore_gains), 2)}, max={round(max(anonscore_gains), 2)}')

    anonscore_gains = list(chain.from_iterable(all_stats['anon_gain_ratio'][session] for session in all_stats['anon_gain']))
    geometric_mean = np.exp(np.mean(np.log(anonscore_gains)))
    print(f'Anonscore (weighted) ratio gain per one coinjoin: median={round(np.median(anonscore_gains), 2)}, average={round(np.average(anonscore_gains), 2)}, geometric average={round(geometric_mean, 2)}, min={round(min(anonscore_gains), 2)}, max={round(max(anonscore_gains), 2)}')

    # save graph
    mfig.plt.suptitle(f'as{experiment_target_anonscore}', fontsize=16)  # Adjust the fontsize and y position as needed
    mfig.plt.subplots_adjust(bottom=0.1, wspace=0.5, hspace=0.5)
    save_file = os.path.join(target_path, f'as{experiment_target_anonscore}_coinjoin_stats')
    mfig.plt.savefig(f'{save_file}.png', dpi=300)
    mfig.plt.savefig(f'{save_file}.pdf', dpi=300)
    mfig.plt.close()

    return all_stats


def plot_ww2mix_stats(mfig, all_stats: dict, experiment_target_anonscore: int, color: str):
    # Plot graphs
    index = 0
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['anon_percentage_status'], f'Progress towards fully anonymized liquidity (AS={experiment_target_anonscore}); total sessions={len(all_stats['anon_percentage_status'])}',
                       'Privacy progress (%)', color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['anon_gain'], f'All wallets, change in anonscore weighted (AS={experiment_target_anonscore}); total sessions={len(all_stats['anon_gain'])}',
                       'Anonscore gain', color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['anon_gain_ratio'], f'All wallets, change in anonscore weighted ratio out/in (AS={experiment_target_anonscore}); total sessions={len(all_stats['anon_gain'])}',
                       'Anonscore gain (weighted, ratio)', color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['observed_remix_liquidity_ratio_cumul'], f'All wallets, cumullative remix liquidity ratio (AS={experiment_target_anonscore}); total sessions={len(all_stats['observed_remix_liquidity_ratio_cumul'])}',
                       'Cummulative remix ratio', color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['skipped_cjtxs'],
                       f'All wallets, skipped cjtxs;total sessions={len(all_stats['skipped_cjtxs'])}',
                       'num cjtxs skipped', color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['skipped_cjtxs_corrected'],
                       f'All wallets, skipped cjtxs corrected by smallest burntime;total sessions={len(all_stats['skipped_cjtxs_corrected'])}',
                       'num cjtxs skipped', color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['num_inputs'],
                       f'All wallets, number of inputs;total sessions={len(all_stats['num_inputs'])}',
                       'number of inputs', color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['num_outputs'],
                       f'All wallets, number of outputs;total sessions={len(all_stats['num_outputs'])}',
                       'number of outputs', color)


if __name__ == "__main__":
    als.SORT_COINJOINS_BY_RELATIVE_ORDER = False
    # round_logs = als.parse_client_coinjoin_logs(target_path)
    # exit(42)

    # prison_logs = analyse_prison_logs(target_path)
    # exit(42)
    all25 = full_analyze_as25()
    all38 = full_analyze_as38()

    NUM_COLUMNS = 2  # 4
    NUM_ROWS = 6     # 5
    fig = plt.figure(figsize=(20, NUM_ROWS * 2.5))
    mfig = Multifig(plt, fig, NUM_ROWS, NUM_COLUMNS)
    mfig.add_multiple_subplots(8)

    # Plot both experiments into single image
    plot_ww2mix_stats(mfig, all25, 25, 'royalblue')
    plot_ww2mix_stats(mfig, all38, 38, 'lightcoral')

    # save graph
    mfig.plt.suptitle(f'Combined plots as25 and as38', fontsize=16)  # Adjust the fontsize and y position as needed
    mfig.plt.subplots_adjust(bottom=0.1, wspace=0.5, hspace=0.5)
    save_file = os.path.join(f'c:/!blockchains/CoinJoin/WasabiWallet_experiments/mn1/as25_38_coinjoin_stats')
    mfig.plt.savefig(f'{save_file}.png', dpi=300)
    mfig.plt.savefig(f'{save_file}.pdf', dpi=300)
    mfig.plt.close()

    # base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\mn1\\tmp\\'
    # merged = merge_coins_files(base_path, 'mix2_coins.json', 'mix2_coins_20240528.json')
    # dmp.save_json_to_file_pretty(os.path.join(base_path, 'mix2_coins_merged.json'), merged)
    # merged = merge_coins_files(base_path, 'mix1_coins.json', 'mix1_coins_20240528.json')
    # dmp.save_json_to_file_pretty(os.path.join(base_path, 'mix1_coins_merged.json'), merged)
    # exit(42)

    # TODO: Fraction of coins already above AS=25, yet remixed again
    # TODO: num inputs / outputs
    # TODO: Prison time distribution

    # TODO: Compute remixed liquidity when as limited to 5


