import copy
import os

from matplotlib import pyplot as plt

import parse_dumplings as dmp
import cj_analysis as als


SATS_IN_BTC = 100000000


def plot_cj_anonscores(data: dict, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for cj_session in data.keys():
        line_style = ':'
        if cj_session.find('0.1btc') != -1:
            line_style = 'solid'
        if cj_session.find('0.2btc') != -1:
            line_style = '-.'
        ax.plot(range(1, len(data[cj_session]) + 1), data[cj_session], label=cj_session, linestyle=line_style)

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
    ax.set_ylabel('privacy progress (%)')
    plt.show()


def analyze_as25(target_base_path: str, mix_name: str, target_as: int, experiment_start_date: str):
    target_path = os.path.join(target_base_path, f'{mix_name}_history.json')
    history_all = dmp.load_json_from_file(target_path)['result']
    target_path = os.path.join(target_base_path, f'{mix_name}_coins.json')
    coins = dmp.load_json_from_file(target_path)['result']

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

    # Detect separate coinjoin sessions and split based on them.
    # Assumption: 1 non-coinjoin tx followed by one or more coinjoin session, finished again with non-coinjoin tx
    cjtxs = {'sessions': {}}

    # If last tx is coinjoin, add one artificial non-coinjoin one
    if history[-1]['islikelycoinjoin'] is True:
        artificial_end = copy.deepcopy(history[-1])
        artificial_end['islikelycoinjoin'] = False
        artificial_end['tx'] = '0000000000000000000000000000000000000000000000000000000000000000'
        artificial_end['label'] = 'artificial end merge'
        history.append(artificial_end)

    def get_session_label(mix_name: str, session_size_inputs: int, segment: list, session_funding_tx: dict) -> str:
        # Two options for session label
        cjsession_label_short_date = f'{mix_name} {round(session_size_inputs / SATS_IN_BTC, 1)}btc | {len(segment)} cjs | ' + \
                                     session_funding_tx['datetime'] + ' ' + session_funding_tx['tx'][0:8]
        cjsession_label_short_notxid = f'{mix_name} {round(session_size_inputs / SATS_IN_BTC, 1)}btc | {len(segment)} cjs | {session_funding_tx['tx'][0:2]} '
        cjsession_label_short = cjsession_label_short_date
        cjsession_label_short = cjsession_label_short_notxid
        return cjsession_label_short

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

            session_cjtxs[tx['tx']] = record
        else:
            # Non-coinjoin transaction detected (initial or final merge)
            if len(tx['outputs']) == 1 and tx['outputs'][list(tx['outputs'].keys())[0]]['amount'] > 0:
                session_size_inputs = tx['outputs'][list(tx['outputs'].keys())[0]]['amount']

            if len(session_cjtxs) > 0:
                session_label = get_session_label(mix_name, session_size_inputs, session_cjtxs, session_funding_tx)
                cjtxs['sessions'][session_label] = {'coinjoins': {session_funding_tx['tx']: record}}
                session_cjtxs = {}

            session_funding_tx = tx



    # # Append remaining stats which may not be terminated by merge transaction
    # if len(session_segment) > 0:
    #     cjsession_label_short = get_session_label(mix_name, session_size_inputs, session_segment, session_funding_tx)
    #     print(f' |--> \"{cjsession_label_short}\"', end='')
    # if len(output_weighted_anonscore_list) > 0:
    #     stats['all_cjs_weight_anonscore'][cjsession_label_short] = output_weighted_anonscore_list
    # if len(anon_percentage_status_list) > 0:
    #     stats['anon_percentage_status'][cjsession_label_short] = anon_percentage_status_list
    # if len(session_segment) > 0:
    #     mix_segments.append(session_segment)



    # Compute basic statistics
    mix_segments = []
    session_segment = []
    session_size_inputs = 0
    session_funding_tx = ''
    stats = {}
    stats['all_cjs_weight_anonscore'] = {}
    stats['anon_percentage_status'] = {}
    output_weighted_anonscore_list = []
    anon_percentage_status_list = []
    num_inputs_list = []
    num_outputs_list = []
    session_coins = {}
    for cjtx in history:
        if cjtx['islikelycoinjoin'] is True:
            print(f'#', end='')
            assert len(cjtx['outputs']) != 0, f'No coins assigned to {cjtx['tx']}'

            for index in cjtx['inputs']:  # Remove coins from session_coins spend by this cj
                session_coins.pop(cjtx['inputs'][index]['address'], None)
            for index in cjtx['outputs']:  # Add coins to session_coins created by this cj
                session_coins[cjtx['outputs'][index]['address']] = cjtx['outputs'][index]

            # Get statistics about number of inputs and outputs
            num_inputs_list.append(len(cjtx['inputs']))
            num_outputs_list.append(len(cjtx['outputs']))

            for index in cjtx['outputs']:
                if cjtx['outputs'][index]['anonymityScore'] < target_as:
                    print("\033[31m" + f' {round(cjtx['outputs'][index]['anonymityScore'], 1)}' + "\033[0m", end='')
                    # if cjtx['outputs'][index]['anonymityScore'] == 1:
                    #     print(f' {cjtx['outputs'][index]['address']}', end='')
                else:
                    print("\033[32m" + f' {round(cjtx['outputs'][index]['anonymityScore'], 1)}' + "\033[0m", end='')

                # Compute cummulative weighted anonscore
                output_weighted_anonscore += (cjtx['outputs'][index]['amount'] / session_size_inputs) * cjtx['outputs'][index]['anonymityScore']


            output_weighted_anonscore_list.append(output_weighted_anonscore)
            #print(f' {round(output_weighted_anonscore, 1)}', end='')
            session_segment.append(cjtx['tx'])
            anon_percentage_status = 0
            for address in session_coins.keys():
                if session_coins[address]['anonymityScore'] > 1:
                    effective_as = min(session_coins[address]['anonymityScore'], target_as)
                    anon_percentage_status += (effective_as / target_as) * (session_coins[address]['amount'] / session_size_inputs)

            WARN_TOO_HIGH_PRIVACY_PROGRESS = False
            if WARN_TOO_HIGH_PRIVACY_PROGRESS and anon_percentage_status > 1:
                print(f'Too large anon_percentage_status: {cjtx['tx']}')
            print(f' {round(anon_percentage_status * 100, 1)}%', end='')
            anon_percentage_status_list.append(anon_percentage_status * 100)
        else:
            # Non-coinjoin transaction detected (initial or final merge)
            session_end_merge_tx = f'{len(session_segment)} cjs | ' + cjtx['label'] + ' ' + cjtx['datetime'] + ' ' + cjtx['tx']
            print("\033[34m" + f' * ' + session_end_merge_tx + "\033[0m", end='')

            # If anything available, prepare session label
            if len(anon_percentage_status_list) > 0:
                cjsession_label_short = get_session_label(mix_name, session_size_inputs, session_segment, session_funding_tx)
                print(f' |--> \"{cjsession_label_short}\"', end='')
            print('')

            # Store computed data
            if len(anon_percentage_status_list) > 0:
                assert cjsession_label_short not in stats['anon_percentage_status'], f'Duplicate session label {cjsession_label_short}'
                stats['anon_percentage_status'][cjsession_label_short] = anon_percentage_status_list
            anon_percentage_status_list = []
            if len(output_weighted_anonscore_list) > 0:
                stats['all_cjs_weight_anonscore'][cjsession_label_short] = output_weighted_anonscore_list
            output_weighted_anonscore_list = []
            output_weighted_anonscore = 0
            if len(session_segment) > 0:
                mix_segments.append(session_segment)
            session_segment = []

            # Determine initial size of input before mixing (single utxo) - will be used as input for next mix session
            if len(cjtx['outputs']) == 1 and cjtx['outputs'][list(cjtx['outputs'].keys())[0]]['amount'] > 0:
                session_size_inputs = cjtx['outputs'][list(cjtx['outputs'].keys())[0]]['amount']
                print(f'{session_size_inputs} ', end='')
            # Store initial non-coinjoin tx which stared this session
            session_funding_tx = cjtx
            # Empty session coins pool
            session_coins = {}

    # Append remaining stats which may not be terminated by merge transaction
    if len(output_weighted_anonscore_list) > 0:
        cjsession_label_short = get_session_label(mix_name, session_size_inputs, session_segment, session_funding_tx)
        print(f' |--> \"{cjsession_label_short}\"', end='')
    if len(output_weighted_anonscore_list) > 0:
        stats['all_cjs_weight_anonscore'][cjsession_label_short] = output_weighted_anonscore_list
    if len(anon_percentage_status_list) > 0:
        stats['anon_percentage_status'][cjsession_label_short] = anon_percentage_status_list
    if len(session_segment) > 0:
        mix_segments.append(session_segment)

    print(f'\n{mix_name}: Total experiments: {len(mix_segments)}, total txs={len(history)}, total coins: {len(coins)}')
    print()
    print(mix_segments)

    return stats


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


if __name__ == "__main__":

    # # Load all prison coin files, merge and compute statistics
    # hex_outpoint = "82A23500AD90C8C42F00F2DA0A4C265C0D0A91543C5D3A037F44436F14B8D9039A000000"
    # txid, index = parse_outpoint(hex_outpoint)
    # print(f"TXID: {txid}")
    # print(f"Index: {index}")
#    exit(42)

    # base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\mn1\\tmp\\'
    # merged = merge_coins_files(base_path, 'mix2_coins.json', 'mix2_coins_20240528.json')
    # dmp.save_json_to_file_pretty(os.path.join(base_path, 'mix2_coins_merged.json'), merged)
    # merged = merge_coins_files(base_path, 'mix1_coins.json', 'mix1_coins_20240528.json')
    # dmp.save_json_to_file_pretty(os.path.join(base_path, 'mix1_coins_merged.json'), merged)
    # exit(42)

    experiment_start_cut_date = '2024-05-14T19:02:49+00:00'
    experiment_target_anonscore = 25
    target_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\mn1\\as25\\'
    all_stats = {}

    #problematic_sessions = ['0.1btc | 12 cjs | 2024-05-26T07:47:11+00:00']
    problematic_sessions = ['mix1 0.1btc | 12 cjs | 34']

    def filter_sessions(data: dict, remove_sessions: list):
        # Filter known problematic sessions
        for remove_session in remove_sessions:
            for session in list(data['anon_percentage_status'].keys()):
                if session.find(remove_session) != -1:
                    data['anon_percentage_status'].pop(session)
        return data

    def analyze_mix(target_path, mix_name, experiment_target_anonscore, experiment_start_cut_date, problematic_sessions, all_stats):
        wallet_stats = analyze_as25(target_path, mix_name, experiment_target_anonscore, experiment_start_cut_date)
        wallet_stats = filter_sessions(wallet_stats, problematic_sessions)
        plot_cj_anonscores(wallet_stats['anon_percentage_status'],
                           f'Wallet mix1, progress towards fully anonymized liquidity (anonscore threshold); total sessions={len(wallet_stats['anon_percentage_status'])}')
        als.merge_dicts(wallet_stats, all_stats)
        return all_stats


    all_stats = analyze_mix(target_path, 'mix1', experiment_target_anonscore, experiment_start_cut_date, problematic_sessions, all_stats)
    all_stats = analyze_mix(target_path, 'mix2', experiment_target_anonscore, experiment_start_cut_date, problematic_sessions, all_stats)
    all_stats = analyze_mix(target_path, 'mix3', experiment_target_anonscore, experiment_start_cut_date, problematic_sessions, all_stats)
    assert len(all_stats['anon_percentage_status']) == 23, 'Unexpected number of coinjoin sessions'

    plot_cj_anonscores(all_stats['anon_percentage_status'], f'All wallets, progress towards fully anonymized liquidity (anonscore threshold); total sessions={len(all_stats['anon_percentage_status'])}')

    exit(42)
