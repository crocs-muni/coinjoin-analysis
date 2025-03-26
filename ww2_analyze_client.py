import copy
import logging
import math
import os
from itertools import chain
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import parse_dumplings as dmp
import cj_analysis as als
from collections import defaultdict


# Suppress DEBUG logs from a specific library
logging.getLogger("matplotlib").setLevel(logging.WARNING)

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


def plot_cj_anonscores(mfig: Multifig, data: dict, title: str, anon_score: str, y_label: str, color: str, show_txid: bool = False):
    plot_cj_anonscores_ax(mfig.add_subplot(), data, title, anon_score, y_label, color, show_txid)


def plot_cj_anonscores_ax(ax, data: dict, title: str, anon_score: str, y_label: str, line_color: str, show_txid: bool = False):
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
        ax.plot([1], [1], color=line_color, label=f'Input size 0.1 btc (as={anon_score})', linestyle='solid', alpha=0.5)
    if size_02_used:
        ax.plot([1], [1], color=line_color, label=f'Input size 0.2 btc (as={anon_score})', linestyle=':', alpha=0.5)
    ax.set_xticks(np.arange(1, 20, 2))

    def compute_average_at_index(lists, index):
        values = [lists[lst][index] for lst in lists.keys() if index < len(lists[lst])]
        if not values:
            return 0
        return sum(values) / len(values)

    max_index = max([len(data[cj_session]) for cj_session in data.keys()])
    avg_data = [compute_average_at_index(data, index) for index in range(0, max_index)]
    ax.plot(range(1, len(avg_data) + 1), avg_data, label=f'Average (as={anon_score})', linestyle='solid',
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


def find_input_index_for_output(coinjoins: dict, prev_txid: str, prev_vout_index: str, prev_value: int, next_txid: str):
    # NOTE: tx['inputs'][index] refer to index within outputs from funding transaction (vout),
    # not vin index of this transaction
    spending_index = None
    if prev_txid in coinjoins['coinjoins'].keys():  # Find in coinjoin txs, extract from 'spend_by_tx'
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
            logging.debug(f'{next_txid} not in coinjoins, settings output to 0')
            spending_index = 0
    assert spending_index is not None, f'Spending index for {prev_txid}:{index} not found'

    return spending_index


def analyze_multisession_mix_experiments(target_base_path: str, mix_name: str, target_as: int, experiment_start_date: str):
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
    coinjoins_all = als.load_json_from_file(target_path)
    coinjoins = coinjoins_all['coinjoins']
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
    for tx in history:
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
            tx_file_path = os.path.join(target_base_path, 'data', f'{tx['tx']}.json')
            if os.path.exists(tx_file_path):
                tx_hex = als.load_json_from_file(tx_file_path)['result']
                # Compute total mining fee paid (sum(inputs) - sum(outputs))
                inputs_sum = sum([coinjoins[txid]['inputs'][index]['value'] for index in coinjoins[txid]['inputs'].keys()])
                outputs_sum = sum([coinjoins[txid]['outputs'][index]['value'] for index in coinjoins[txid]['outputs'].keys()])
                total_mining_fee = inputs_sum - outputs_sum
                # Compute vsize for "our" inputs and outputs out of whole transaction => our share of mining fees
                wallet_inputs = [int(record['inputs'][item]['index']) for item in record['inputs'].keys()]
                wallet_outputs = [int(record['outputs'][item]['index']) for item in record['outputs'].keys()]
                wallet_vsize, total_vsize = als.compute_partial_vsize(tx_hex['hex'], wallet_inputs, wallet_outputs)
                # Fee rate paid for whole transaction
                fee_rate = total_mining_fee / total_vsize
                # Mining fee rate to pay fair share for our inputs and outputs
                wallet_fair_mfee_sats = math.ceil(wallet_vsize * fee_rate)
                wallet_inputs_sum = sum([coinjoins[txid]['inputs'][index]['value'] for index in record['inputs'].keys()])
                wallet_outputs_sum = sum([coinjoins[txid]['outputs'][index]['value'] for index in record['outputs'].keys()])
                wallet_fee_paid_sats = wallet_inputs_sum - wallet_outputs_sum
                #assert tx['amount'] == -wallet_fee_paid_sats, f"Incorrect wallet fee computed {wallet_fee_paid_sats} sats vs. {tx['amount']} sats for {txid}"
                if tx['amount'] != -wallet_fee_paid_sats:
                    logging.error(f"Incorrect wallet fee computed {wallet_fee_paid_sats} sats vs. {tx['amount']} sats for {txid}")
                    logging.debug(f"Inputs: ")
                    for index in record['inputs'].keys():
                        logging.debug(f"  [{index}]: {coinjoins[txid]['inputs'][index]['value']} sats")
                    logging.debug(f"Outputs: ")
                    for index in record['outputs'].keys():
                        logging.debug(f"  [{index}]: {coinjoins[txid]['outputs'][index]['value']} sats")
                hidden_cfee = -tx['amount'] - wallet_fair_mfee_sats
                if hidden_cfee < -10:
                    logging.debug(f"Sligthly smaller hidden fee than expected: {hidden_cfee} sats")
                assert hidden_cfee >= -100, f"Incorrect hidden fee of {hidden_cfee} sats"

                record['total_mining_fee'] = total_mining_fee
                record['mining_fee_rate'] = fee_rate
                record['total_vsize'] = total_vsize
                record['wallet_vsize'] = wallet_vsize
                record['wallet_fair_mfee'] = wallet_fair_mfee_sats
                record['wallet_fee_paid'] = -tx['amount']
                record['wallet_hidden_cfee_paid'] = hidden_cfee

            session_cjtxs[txid] = record
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

    #

    # Compute basic statistics
    stats = {}
    stats['all_cjs_weight_anonscore'] = {}
    stats['anon_percentage_status'] = {}
    stats['anon_gain_weighted'] = {}
    stats['observed_remix_liquidity_ratio'] = {}
    stats['observed_remix_liquidity_ratio_cumul'] = {}  # Remix liquidity based on value of inputs
    stats['observed_remix_inputs_ratio_cumul'] = {}     # unused now, remix liquidity based on number of inputs
    for session_label in cjtxs['sessions'].keys():
        session_coins = {}
        anon_percentage_status_list = []
        anon_gain_weighted_list = []
        observed_remix_liquidity_ratio_list = []
        observed_remix_liquidity_ratio_cumul_list = []
        observed_remix_inputs_ratio_cumul_list = []
        session_size_inputs = cjtxs['sessions'][session_label]['funding_tx']['value']
        assert session_size_inputs > 0, f'Unexpected negative funding tx size of {session_size_inputs}'
        for cjtxid in cjtxs['sessions'][session_label]['coinjoins'].keys():
            cjtx = cjtxs['sessions'][session_label]['coinjoins'][cjtxid]
            print(f'#', end='')
            assert len(cjtx['outputs']) != 0, f'No coins assigned to {cjtx['txid']}'

            # Print all output coins (at given state of time) based on their anonscore
            # red ... anonscore target not reached yet, green ... already reached
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

            # Compute observed liquidity ratio for wallet's coins
            # This value enumerates multiplier of initial fresh liquidity over multiple cjtxs
            # (If all coins are fully mixed in the first coinjoin, then observed_remix_liquidity_ratio is 1,
            # every additional mix is adding additional input liquidity (remixed))
            # 1. Sum values of all wallet's input coins (to this cjtx), divided by fresh liquidity (of this session)
            # 2. Compute cummulative liquidity for each subsequent coinjoin (observed_remix_liquidity_ratio_cumul_list)
            observed_remix_liquidity_ratio = sum([cjtx['inputs'][index]['value'] for index in cjtx['inputs']]) / session_size_inputs
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
        if len(anon_gain_weighted_list) > 0:
            assert session_label not in stats['anon_gain_weighted'], f'Duplicate session label {session_label}'
            stats['anon_gain_weighted'][session_label] = anon_gain_weighted_list
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

    ax = mfig.add_subplot()
    #sns.heatmap(heatmap.T, cmap='viridis', annot=True, fmt='.0f', cbar=True, ax=ax)
    heatmap_percentage = (heatmap / np.sum(heatmap)) * 100
    sns.heatmap(heatmap_percentage.T, cmap='viridis', annot=True, fmt='.1f', cbar=True, ax=ax)


    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(len(xedges) - 1) + 0.5)
    ax.set_yticks(np.arange(len(yedges) - 1) + 0.5)
    ax.set_xticklabels(np.arange(1, len(xedges)))
    ax.set_yticklabels(np.arange(1, len(yedges)))
    ax.set_title(title)
    #plt.show()


def full_analyze_as25_202405(base_path: str):
    # Experiment configuration
    target_path = os.path.join(base_path, 'as25\\')
    experiment_start_cut_date = '2024-05-14T19:02:49+00:00'  # AS=25 experiment start time
    experiment_target_anonscore = 25
    problematic_sessions = ['mix1 0.1btc | 12 cjs | txid: 34']  # Failed experiments to be removed from processing

    return analyze_ww2_artifacts(target_path, experiment_start_cut_date, experiment_target_anonscore,
                          ['mix1', 'mix2', 'mix3'], problematic_sessions, 23)


def full_analyze_as38_202503(base_path: str):
    # Experiment configuration
    target_path = os.path.join(base_path, 'as38\\')
    experiment_start_cut_date = '2025-03-09T00:02:49+00:00'  # AS=38 experiment start time
    experiment_target_anonscore = 38
    problematic_sessions = ['mix7 0.1btc | 3 cjs | txid: 3493c971d']  # Failed experiments to be removed from processing

    return analyze_ww2_artifacts(target_path, experiment_start_cut_date, experiment_target_anonscore,
                          ['mix6', 'mix7'], problematic_sessions, -1)  # TODO: once as38 experimen is finisihed, set number of expected sessions


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
        cjs, wallet_stats = analyze_multisession_mix_experiments(target_path, mix_name, experiment_target_anonscore, experiment_start_cut_date)
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
                               'privacy progress (%)', f'{experiment_target_anonscore}')
            plot_cj_anonscores(wallet_stats['observed_remix_liquidity_ratio_cumul'],
                               f'Wallet {mix_name}, cumullative remix liquidity ratio;total sessions={len(wallet_stats['observed_remix_liquidity_ratio_cumul'])}',
                               'cummulative remix ratio', f'{experiment_target_anonscore}')
            plot_cj_anonscores(wallet_stats['skipped_cjtxs'],
                               f'Wallet {mix_name}, skipped cjtxs;total sessions={len(wallet_stats['skipped_cjtxs'])}',
                               'num cjtxs skipped', f'{experiment_target_anonscore}')
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
                       f'{experiment_target_anonscore}', 'Privacy progress (%)', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['anon_gain'], f'All wallets, change in anonscore weighted (AS={experiment_target_anonscore}); total sessions={len(all_stats['anon_gain'])}',
                       f'{experiment_target_anonscore}','Anonscore gain', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['anon_gain_ratio'], f'All wallets, change in anonscore weighted ratio out/in (AS={experiment_target_anonscore}); total sessions={len(all_stats['anon_gain'])}',
                       f'{experiment_target_anonscore}','Anonscore gain (weighted, ratio)', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['observed_remix_liquidity_ratio_cumul'], f'All wallets, cumullative remix liquidity ratio (AS={experiment_target_anonscore}); total sessions={len(all_stats['observed_remix_liquidity_ratio_cumul'])}',
                       f'{experiment_target_anonscore}','Cummulative remix ratio', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['num_inputs'],
                       f'All wallets, number of inputs;total sessions={len(all_stats['num_inputs'])}',
                       f'{experiment_target_anonscore}','number of inputs', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['num_outputs'],
                       f'All wallets, number of outputs;total sessions={len(all_stats['num_outputs'])}',
                       f'{experiment_target_anonscore}','number of outputs', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['skipped_cjtxs'],
                       f'All wallets, skipped cjtxs;total sessions={len(all_stats['skipped_cjtxs'])}',
                       f'{experiment_target_anonscore}','num cjtxs skipped', 'royalblue')
    plot_cj_anonscores(mfig, all_stats['anon_gain_weighted'], f'Nocap progress towards fully anonymized liquidity (AS={experiment_target_anonscore}); total sessions={len(all_stats['anon_gain_weighted'])}',
                       f'{experiment_target_anonscore}', 'Privacy progress (%)', 'royalblue')

    x, y = [], []
    for session in all_stats['num_inputs'].keys():
        x.extend(all_stats['num_inputs'][session])
        y.extend(all_stats['num_outputs'][session])
    plot_cj_heatmap(mfig, x, y, 'number of inputs', 'number of outputs','Occurence frequency of inputs to outputs pairs')

    # Plot histogram of hidden coordination fees (cfee)
    ax = mfig.add_subplot()
    data_mfee = [all_cjs['sessions'][session_label]['coinjoins'][cjtxid]['wallet_fair_mfee'] for session_label in all_cjs['sessions'].keys() for cjtxid in all_cjs['sessions'][session_label]['coinjoins'].keys()]
    data_cfee = [all_cjs['sessions'][session_label]['coinjoins'][cjtxid]['wallet_hidden_cfee_paid'] for session_label in all_cjs['sessions'].keys() for cjtxid in all_cjs['sessions'][session_label]['coinjoins'].keys()]
    data_cfee_small = [value for value in data_cfee if value < 10000]
    data_cfee
    print(f'Mining fee sum={sum(data_mfee)}')
    print(f'Hidden cfee (sum={sum(data_cfee)}): {sorted(data_cfee)}')
    ax.hist(data_mfee, bins=30, color='green', edgecolor='black', alpha=0.5, label=f'Fair mining fee: {sum(data_mfee)} sats')
    ax.hist(data_cfee_small, bins=30, color='red', edgecolor='black', alpha=0.5, label=f'Hidden coord. fee: {sum(data_cfee)} sats')
    ax.set_xlabel('Hidden cfee (sats)')
    ax.set_ylabel('Occurence')
    ax.set_title('Distribution of hidden coordination fee')
    ax.legend()
    #ax.set_title(title)

    sessions_lengths = [len(all_cjs['sessions'][session]['coinjoins']) for session in all_cjs['sessions'].keys()]
    print(f'Total sessions={len(all_cjs['sessions'].keys())}, total coinjoin txs={sum(sessions_lengths)}')
    print(f'Session lengths (#cjtxs): median={round(np.median(sessions_lengths), 2)}, average={round(np.average(sessions_lengths), 2)}, min={min(sessions_lengths)}, max={max(sessions_lengths)}')

    total_output_coins = [all_stats['num_coins'][session] for session in all_stats['num_coins']]
    print(f'Total output coins: {sum(total_output_coins)}')

    total_overmixed_coins = [all_stats['num_overmixed_coins'][session] for session in all_stats['num_overmixed_coins']]
    print(f'Total overmixed input coins: {sum(total_overmixed_coins)}')

    # num_skipped = list(chain.from_iterable(all_stats['skipped_cjtxs'][session] for session in all_stats['skipped_cjtxs']))
    # print(f'Skipped txs stats: median={np.median(num_skipped)}, average={round(np.average(num_skipped), 2)}, min={min(num_skipped)}, max={max(num_skipped)}')

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

    return all_stats, all_cjs


def plot_ww2mix_stats(mfig, all_stats: dict, experiment_label: str, experiment_target_anonscore: str, color: str):
    # Plot graphs
    index = 0
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['anon_percentage_status'], f'Progress towards fully anonymized liquidity (as={experiment_label}); total sessions={len(all_stats['anon_percentage_status'])}',
                       experiment_target_anonscore, 'Privacy progress (%)', color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['anon_gain'], f'All wallets, change in anonscore weighted (as={experiment_label}); total sessions={len(all_stats['anon_gain'])}',
                       experiment_target_anonscore, 'Anonscore gain', color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['anon_gain_ratio'], f'All wallets, change in anonscore weighted ratio out/in (as={experiment_label}); total sessions={len(all_stats['anon_gain'])}',
                       experiment_target_anonscore, 'Anonscore gain (weighted, ratio)', color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['observed_remix_liquidity_ratio_cumul'], f'All wallets, cumullative remix liquidity ratio (as={experiment_label}); total sessions={len(all_stats['observed_remix_liquidity_ratio_cumul'])}',
                       experiment_target_anonscore, 'Cummulative remix ratio', color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['num_inputs'],
                       f'All wallets, number of inputs;total sessions={len(all_stats['num_inputs'])}',
                       experiment_target_anonscore, 'number of inputs', color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['num_outputs'],
                       f'All wallets, number of outputs;total sessions={len(all_stats['num_outputs'])}',
                       experiment_target_anonscore, 'number of outputs', color)
    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['skipped_cjtxs'],
                       f'All wallets, skipped cjtxs;total sessions={len(all_stats['skipped_cjtxs'])}',
                       experiment_target_anonscore, 'num cjtxs skipped', color)

    index += 1
    ax = mfig.get(index)
    plot_cj_anonscores_ax(ax, all_stats['anon_gain_weighted'], f'Privacy gain sum weighted; total sessions={len(all_stats['anon_gain_weighted'])}',
                       experiment_target_anonscore, 'Privacy gain', color)


def create_download_script(cjtxs: dict, file_name: str):
    """
    Generate download script for hex versions of provided transactions.
    :param all_cjtxs: list of cjtxs to download
    :param file_name: output name of download sript with all generated commands
    :return:
    """
    curl_lines = []
    for cjtx in cjtxs:
        curl_str = "curl --user user:password --data-binary \'{\"jsonrpc\": \"1.0\", \"id\": \"curltest\", \"method\": \"getrawtransaction\", \"params\": [\"" + cjtx + "\", true]}\' -H \'Content-Type: application/json\' http://127.0.0.1:8332/" + f" > {cjtx}.json\n"
        curl_lines.append(curl_str)
    with open(file_name, 'w') as f:
        f.writelines(curl_lines)


if __name__ == "__main__":
    als.SORT_COINJOINS_BY_RELATIVE_ORDER = False
    # round_logs = als.parse_client_coinjoin_logs(target_path)
    # exit(42)

    # prison_logs = analyse_prison_logs(target_path)
    # exit(42)
    base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\mn1\\'
    all38_stats, all38 = full_analyze_as38_202503(base_path)
    all25_stats, all25 = full_analyze_as25_202405(base_path)

    # Create download script for full transactions download
    cjtxs = [cjtx for session in all25['sessions'].keys() for cjtx in all25['sessions'][session]['coinjoins'].keys()]
    create_download_script(cjtxs, 'download_as25.sh')
    cjtxs = [cjtx for session in all38['sessions'].keys() for cjtx in all38['sessions'][session]['coinjoins'].keys()]
    create_download_script(cjtxs, 'download_as38.sh')

    NUM_COLUMNS = 2  # 4
    NUM_ROWS = 6     # 5
    fig = plt.figure(figsize=(20, NUM_ROWS * 4))
    mfig = Multifig(plt, fig, NUM_ROWS, NUM_COLUMNS)
    mfig.add_multiple_subplots(8)

    # Plot both experiments into single image
    plot_ww2mix_stats(mfig, all25_stats, '25&38', '25', 'royalblue')
    plot_ww2mix_stats(mfig, all38_stats, '25&38', '38', 'lightcoral')
    #plot_ww2mix_stats(mfig, all38, 38, 'lightcoral')

    # save graph
    mfig.plt.suptitle(f'Combined plots as25 and as38', fontsize=16)  # Adjust the fontsize and y position as needed
    mfig.plt.subplots_adjust(bottom=0.1, wspace=0.5, hspace=0.5)
    save_file = os.path.join(base_path, 'as25_38_coinjoin_stats')
    mfig.plt.savefig(f'{save_file}.png', dpi=300)
    mfig.plt.savefig(f'{save_file}.pdf', dpi=300)
    mfig.plt.close()

    # base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\mn1\\tmp\\'
    # merged = merge_coins_files(base_path, 'mix2_coins.json', 'mix2_coins_20240528.json')
    # dmp.save_json_to_file_pretty(os.path.join(base_path, 'mix2_coins_merged.json'), merged)
    # merged = merge_coins_files(base_path, 'mix1_coins.json', 'mix1_coins_20240528.json')
    # dmp.save_json_to_file_pretty(os.path.join(base_path, 'mix1_coins_merged.json'), merged)
    # exit(42)

    # TODO: limits stats
    # TODO: Prison time distribution
    # TODO: Compute cost of mixing including hidden coordination fee
    # TODO: Compute remixed liquidity when as limited to 5
    # TODO: plot how long it takes (#coinjoins, walltime) to achieve desired anonscore target
    #


