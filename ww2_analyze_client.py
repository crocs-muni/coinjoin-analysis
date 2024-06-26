import copy
import math
import os
from datetime import datetime
from itertools import chain

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

import parse_dumplings as dmp
import cj_analysis as als


SATS_IN_BTC = 100000000


def plot_cj_anonscores(data: dict, title: str, y_label: str, show_txid: bool = False):
    fig, ax = plt.subplots(figsize=(10, 5))
    for cj_session in data.keys():
        line_style = ':'
        if cj_session.find('0.1btc') != -1:
            line_style = 'solid'
        if cj_session.find('0.2btc') != -1:
            line_style = '-.'
        cj_label = cj_session
        if not show_txid and cj_session.find('txid:'):
            cj_label = cj_label[0:cj_session.find('txid:')]
        ax.plot(range(1, len(data[cj_session]) + 1), data[cj_session], label=cj_label, linestyle=line_style)

    def compute_average_at_index(lists, index):
        values = [lists[lst][index] for lst in lists.keys() if index < len(lists[lst])]
        if not values:
            return 0
        return sum(values) / len(values)
    max_index = max([len(data[cj_session]) for cj_session in data.keys()])
    avg_data = [compute_average_at_index(data, index) for index in range(0, max_index)]
    ax.plot(range(1, len(avg_data) + 1), avg_data, label='Average', linestyle='solid',
            linewidth=5, alpha=0.5, color='gray')

    ax.legend(loc="best", fontsize='6')
    ax.set_title(title)
    ax.set_xlabel('Number of coinjoins executed')
    ax.set_ylabel(y_label)
    plt.show()

    # Same data, but boxplot
    max_index = max([len(data[cj_session]) for cj_session in data.keys()])
    data_cj = [[] for index in range(0, max_index)]
    for cj_session in data.keys():
        for index in range(0, max_index):
            if index < len(data[cj_session]):
                data_cj[index].append(data[cj_session][index])
    fig, ax_boxplot = plt.subplots(figsize=(10, 5))
    ax_boxplot.boxplot(data_cj)
    ax_boxplot.set_title(title)
    ax_boxplot.set_xlabel('Number of coinjoins executed')
    ax_boxplot.set_ylabel(y_label)
    plt.show()


def get_session_label(mix_name: str, session_size_inputs: int, segment: list, session_funding_tx: dict) -> str:
    # Two options for session label
    cjsession_label_short_date = f'{mix_name} {round(session_size_inputs / SATS_IN_BTC, 1)}btc | {len(segment)} cjs | ' + \
                                 session_funding_tx['broadcast_time'] + ' ' + session_funding_tx['txid'][0:8]
    cjsession_label_short_txid = f'{mix_name} {round(session_size_inputs / SATS_IN_BTC, 1)}btc | {len(segment)} cjs | txid: {session_funding_tx['txid']} '
    cjsession_label_short = cjsession_label_short_date
    cjsession_label_short = cjsession_label_short_txid
    return cjsession_label_short


def analyze_as25(target_base_path: str, mix_name: str, target_as: int, experiment_start_date: str):
    target_path = os.path.join(target_base_path, f'{mix_name}_history.json')
    history_all = dmp.load_json_from_file(target_path)['result']
    target_path = os.path.join(target_base_path, f'{mix_name}_coins.json')
    coins = dmp.load_json_from_file(target_path)['result']
    target_path = os.path.join(target_base_path, f'coinjoin_tx_info.json')
    coinjoins = dmp.load_json_from_file(target_path)['coinjoins']
    target_path = os.path.join(target_base_path, f'logww2.json')
    coord_logs = dmp.load_json_from_file(target_path)

    # Filter all items from history older than experiment start date
    history = [tx for tx in history_all if tx['datetime'] >= experiment_start_cut_date]

    # Pair wallet coins to transactions from wallet history
    for cjtx in history:
        if 'outputs' not in cjtx.keys():
            cjtx['outputs'] = {}
        if 'inputs' not in cjtx.keys():
            cjtx['inputs'] = {}
        input_index = 0
        for coin in coins:
            if coin['txid'] == cjtx['tx']:
                cjtx['outputs'][str(coin['index'])] = coin
            if coin['spentBy'] == cjtx['tx']:
                cjtx['inputs'][str(input_index)] = coin  # BUGBUG: We do not know correct vin index
                input_index += 1

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
                assert len(session_funding_tx['outputs'].keys()) == 1, f'Funding tx has unexpecetd numeber of outputs of {len(session_funding_tx['outputs'].keys())}'
                norm_tx = {'txid': session_funding_tx['tx'], 'label': session_funding_tx['label'], 'broadcast_time': session_funding_tx['datetime'], 'value': session_funding_tx['outputs']['0']['amount']}
                session_label = get_session_label(mix_name, session_size_inputs, session_cjtxs, norm_tx)
                cjtxs['sessions'][session_label] = {'coinjoins': session_cjtxs, 'funding_tx': norm_tx}
                session_cjtxs = {}

            session_funding_tx = tx

    # Compute basic statistics
    stats = {}
    stats['all_cjs_weight_anonscore'] = {}
    stats['anon_percentage_status'] = {}
    stats['observed_remix_liquidity_ratio'] = {}
    stats['observed_remix_liquidity_ratio_cumul'] = {}
    for session_label in cjtxs['sessions'].keys():
        session_coins = {}
        anon_percentage_status_list = []
        observed_remix_liquidity_ratio_list = []
        observed_remix_liquidity_ratio_cumul_list = []
        session_size_inputs = cjtxs['sessions'][session_label]['funding_tx']['value']
        assert session_size_inputs > 0, f'Unexpected negative funding tx size of {session_size_inputs}'
        for cjtxid in cjtxs['sessions'][session_label]['coinjoins'].keys():
            cjtx = cjtxs['sessions'][session_label]['coinjoins'][cjtxid]
            print(f'#', end='')
            assert len(cjtx['outputs']) != 0, f'No coins assigned to {cjtx['txid']}'

            # # Get statistics about number of inputs and outputs
            # num_inputs_list.append(len(cjtx['inputs']))
            # num_outputs_list.append(len(cjtx['outputs']))

            for index in cjtx['outputs']:
                if cjtx['outputs'][index]['anon_score'] < target_as:
                    print("\033[31m" + f' {round(cjtx['outputs'][index]['anon_score'], 1)}' + "\033[0m", end='')
                    # if cjtx['outputs'][index]['anonymityScore'] == 1:
                    #     print(f' {cjtx['outputs'][index]['address']}', end='')
                else:
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
            WARN_TOO_HIGH_PRIVACY_PROGRESS = False
            if WARN_TOO_HIGH_PRIVACY_PROGRESS and anon_percentage_status > 1:
                print(f'Too large anon_percentage_status: {cjtx['tx']}')
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
    cj_time = [{'txid':cjtxid, 'broadcast_time': datetime.strptime(coinjoins[cjtxid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")} for cjtxid in coinjoins.keys()]
    sorted_cj_times = sorted(cj_time,  key=lambda x: x['broadcast_time'])
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
            #assert skipped >= 0, f'Inconsistent skipped coinjoins of {skipped} for {cjtxid} - {prev_cjtxid}'
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

    print(f'\n{mix_name}: Total experiments: {len(cjtxs['sessions'][session_label]['coinjoins'].keys())}, total txs={len(history)}, total coins: {len(coins)}')

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

def plot_cj_heatmap(x, y, x_label, y_label, title):
    heatmap_size = (max(x), max(y))
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=heatmap_size)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(heatmap.T, cmap='viridis', annot=True, fmt='.1f', cbar=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(len(xedges) - 1) + 0.5)
    ax.set_yticks(np.arange(len(yedges) - 1) + 0.5)
    ax.set_xticklabels(np.arange(1, len(xedges)))
    ax.set_yticklabels(np.arange(1, len(yedges)))
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    # exit(42)

    experiment_start_cut_date = '2024-05-14T19:02:49+00:00'
    experiment_target_anonscore = 25
    target_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\mn1\\as25\\'
    all_cjs = {}
    all_stats = {}

    # round_logs = als.parse_client_coinjoin_logs(target_path)
    # exit(42)

    # prison_logs = analyse_prison_logs(target_path)
    # exit(42)

    problematic_sessions = ['mix1 0.1btc | 12 cjs | txid: 34']

    def filter_sessions(data: dict, remove_sessions: list):
        # Filter known problematic sessions
        for remove_session in remove_sessions:
            for session in list(data['anon_percentage_status'].keys()):
                if session.find(remove_session) != -1:
                    for stat_name in data.keys():
                        if session in data[stat_name].keys():
                            data[stat_name].pop(session)
        return data

    def analyze_mix(target_path, mix_name, experiment_target_anonscore, experiment_start_cut_date, problematic_sessions, all_cjs, all_stats):
        cjs, wallet_stats = analyze_as25(target_path, mix_name, experiment_target_anonscore, experiment_start_cut_date)
        wallet_stats = filter_sessions(wallet_stats, problematic_sessions)
        for to_remove in problematic_sessions:
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
        als.merge_dicts(cjs, all_cjs)
        als.merge_dicts(wallet_stats, all_stats)
        return cjs, all_stats


    analyze_mix(target_path, 'mix1', experiment_target_anonscore, experiment_start_cut_date, problematic_sessions, all_cjs, all_stats)
    analyze_mix(target_path, 'mix2', experiment_target_anonscore, experiment_start_cut_date, problematic_sessions, all_cjs, all_stats)
    analyze_mix(target_path, 'mix3', experiment_target_anonscore, experiment_start_cut_date, problematic_sessions, all_cjs, all_stats)
    assert len(all_stats['anon_percentage_status']) == 23, f'Unexpected number of coinjoin sessions {len(all_stats['anon_percentage_status'])}'

    plot_cj_anonscores(all_stats['anon_percentage_status'], f'All wallets, progress towards fully anonymized liquidity (as={experiment_target_anonscore}); total sessions={len(all_stats['anon_percentage_status'])}',
                       'privacy progress (%)')
    plot_cj_anonscores(all_stats['observed_remix_liquidity_ratio_cumul'], f'All wallets, cumullative remix liquidity ratio; total sessions={len(all_stats['observed_remix_liquidity_ratio_cumul'])}',
                       'cummulative remix ratio')
    plot_cj_anonscores(all_stats['skipped_cjtxs'],
                       f'All wallets, skipped cjtxs;total sessions={len(all_stats['skipped_cjtxs'])}',
                       'num cjtxs skipped')
    plot_cj_anonscores(all_stats['num_inputs'],
                       f'All wallets, number of inputs;total sessions={len(all_stats['num_inputs'])}',
                       'number of inputs')
    plot_cj_anonscores(all_stats['num_outputs'],
                       f'All wallets, number of outputs;total sessions={len(all_stats['num_outputs'])}',
                       'number of outputs')
    x, y = [], []
    for session in all_stats['num_inputs'].keys():
        x.extend(all_stats['num_inputs'][session])
        y.extend(all_stats['num_outputs'][session])
    plot_cj_heatmap(x, y, 'number of inputs', 'number of outputs','Frequency of inputs to outputs pairs')

    sessions_lengths = [len(all_cjs['sessions'][session]['coinjoins']) for session in all_cjs['sessions'].keys()]
    print(f'Sessions lengths: median={round(np.median(sessions_lengths), 2)}, average={round(np.average(sessions_lengths), 2)}, min={min(sessions_lengths)}, max={max(sessions_lengths)}')

    remix_ratios = [max(all_stats['observed_remix_liquidity_ratio_cumul'][session]) for session in all_stats['observed_remix_liquidity_ratio_cumul'].keys()]
    print(f'Remix ratios: median={round(np.median(remix_ratios), 2)}, average={round(np.average(remix_ratios), 2)}, min={round(min(remix_ratios), 2)}, max={round(max(remix_ratios), 2)}')

    num_inputs = list(chain.from_iterable(all_stats['num_inputs'][session] for session in all_stats['num_inputs']))
    print(f'Input stats: median={np.median(num_inputs)}, average={round(np.average(num_inputs), 2)}, min={min(num_inputs)}, max={max(num_inputs)}')

    num_outputs = list(chain.from_iterable(all_stats['num_outputs'][session] for session in all_stats['num_outputs']))
    print(f'Output stats: median={np.median(num_outputs)}, average={round(np.average(num_outputs), 2)}, min={min(num_outputs)}, max={max(num_outputs)}')

    progress_100 = len([all_stats['anon_percentage_status'][session][0] for session in all_stats['anon_percentage_status'] if all_stats['anon_percentage_status'][session][0] > 99])
    print(f'Anonscore target of {experiment_target_anonscore} hit already during first coinjoin for {progress_100} of {len(all_stats['anon_percentage_status'])} sessions {round(progress_100 / len(all_stats['anon_percentage_status']) * 100, 2)}%')

    exit(42)


    # base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\mn1\\tmp\\'
    # merged = merge_coins_files(base_path, 'mix2_coins.json', 'mix2_coins_20240528.json')
    # dmp.save_json_to_file_pretty(os.path.join(base_path, 'mix2_coins_merged.json'), merged)
    # merged = merge_coins_files(base_path, 'mix1_coins.json', 'mix1_coins_20240528.json')
    # dmp.save_json_to_file_pretty(os.path.join(base_path, 'mix1_coins_merged.json'), merged)
    # exit(42)

    # TODO: Fraction of coins already above AS=25, yet remixed again
    # TODO: num inputs / outputs
    # TODO: Prison time distribution


