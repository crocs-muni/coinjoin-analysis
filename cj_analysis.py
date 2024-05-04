import logging
import numpy as np
from datetime import datetime
from enum import Enum

SATS_IN_BTC = 100000000


class PRECOMP_STRPTIME():
    precomp_strptime = {}

    def strptime(self, datestr: str, datestr_format: str) -> datetime:
        if datestr not in self.precomp_strptime:
            self.precomp_strptime[datestr] = datetime.strptime(datestr, datestr_format)
        return self.precomp_strptime[datestr]


precomp_datetime = PRECOMP_STRPTIME()


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
        for message in self.summary_messages:
            logging.info(message)


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


def get_ratio_string(numerator, denominator) -> str:
    if denominator != 0:
        return f'{numerator}/{denominator} ({round(numerator/float(denominator) * 100, 1)}%)'
    else:
        return f'{numerator}/{0} (0%)'


def get_inputs_type_list(coinjoins, sorted_cj_time, event_type, in_or_out: str, burn_time_from, burn_time_to, analyze_values):
    if analyze_values:
        return [sum([coinjoins[cjtx['txid']][in_or_out][index]['value'] for index in coinjoins[cjtx['txid']][in_or_out].keys()
                     if coinjoins[cjtx['txid']][in_or_out][index]['mix_event_type'] == event_type.name and
                     coinjoins[cjtx['txid']][in_or_out][index].get('burn_time_cjtxs', -1) in range(burn_time_from, burn_time_to + 1)])
            for cjtx in sorted_cj_time]
    else:
        return [sum([1 for index in coinjoins[cjtx['txid']][in_or_out].keys()
                     if coinjoins[cjtx['txid']][in_or_out][index]['mix_event_type'] == event_type.name and
                     coinjoins[cjtx['txid']][in_or_out][index].get('burn_time_cjtxs', -1) in range(burn_time_from, burn_time_to + 1)])
            for cjtx in sorted_cj_time]


def plot_inputs_type_ratio(mix_id: str, data: dict, initial_cj_index: int, ax, analyze_values: bool, normalize_values: bool):
    """
    Ratio between various types of inputs (fresh, remixed, remixed_friends)
    :param mix_id:
    :param data:
    :param ax:
    :param analyze_values if true, then size of inputs is analyzed, otherwise only numbers
    :return:
    """
    coinjoins = data['coinjoins']
    cj_time = [{'txid': cjtxid, 'broadcast_time': precomp_datetime.strptime(coinjoins[cjtxid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")} for cjtxid in coinjoins.keys()]
    sorted_cj_time = sorted(cj_time, key=lambda x: x['broadcast_time'])
    #sorted_cj_time = sorted_cj_time[0:500]

    no_remix = {'inputs': [], 'outputs': []}
    for cjtx in sorted_cj_time:
        if sum([1 for index in coinjoins[cjtx['txid']]['inputs'].keys()
                if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name]) == 0:
            logging.warning(f'No input remix detected for {cjtx}')
            no_remix['inputs'].append(cjtx['txid'])
        if sum([1 for index in coinjoins[cjtx['txid']]['outputs'].keys()
             if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name]) == 0:
            logging.warning(f'No output remix detected for {cjtx}')
            no_remix['outputs'].append(cjtx['txid'])

    logging.warning(f'Txs with no input&output remix: {set(no_remix['inputs']).intersection(set(no_remix['outputs']))}')

    input_types_nums = {}
    for event_type in MIX_EVENT_TYPE:
        if analyze_values:
            # Sum of values of inputs is taken
            input_types_nums[event_type.name] = [sum([coinjoins[cjtx['txid']]['inputs'][index]['value'] for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                            if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == event_type.name])
                                            for cjtx in sorted_cj_time]
        else:
            # Only number of inputs is taken
            input_types_nums[event_type.name] = [sum([1 for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                        if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == event_type.name])
                                   for cjtx in sorted_cj_time]

    event_type = MIX_EVENT_TYPE.MIX_REMIX
    BURN_TIME_RANGES = [('1-2', 1, 2), ('3-5', 3, 5), ('6-19', 6, 19), ('20+', 20, 1999), ('2000+', 2000, 1000000)]
    for range_val in BURN_TIME_RANGES:
        input_types_nums[f'{event_type.name}_{range_val[0]}'] = get_inputs_type_list(coinjoins, sorted_cj_time, event_type, 'inputs', range_val[1], range_val[2], analyze_values)

    short_exp_name = mix_id

    # Normalize all values into range 0-1 (only MIX_ENTER, MIX_REMIX and MIX_REMIX_FRIENDS are considered for base total)
    input_types_nums_normalized = {}
    total_values = (np.array(input_types_nums[MIX_EVENT_TYPE.MIX_ENTER.name]) + np.array(input_types_nums[MIX_EVENT_TYPE.MIX_REMIX.name]) +
                    np.array(input_types_nums[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name]) + np.array(input_types_nums[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name]))
    # Normalize all values including 'MIX_REMIX_1-2' etc.
    for item in input_types_nums.keys():
        input_types_nums_normalized[item] = np.array(input_types_nums[item]) / total_values

    print(f'MIX_ENTER median ratio: {round(np.median(input_types_nums_normalized[MIX_EVENT_TYPE.MIX_ENTER.name]) * 100, 2)}%')
    print(f'MIX_REMIX median ratio: {round(np.median(input_types_nums_normalized[MIX_EVENT_TYPE.MIX_REMIX.name]) * 100, 2)}%')
    for range_val in BURN_TIME_RANGES:
        remix_name = f'{event_type.name}_{range_val[0]}'
        print(f'{remix_name} median ratio: {round(np.median(input_types_nums_normalized[remix_name]) * 100, 2)}%')
    print(f'MIX_REMIX_FRIENDS median ratio: {round(np.median(input_types_nums_normalized[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name]) * 100, 2)}%')
    print(f'MIX_REMIX_FRIENDS_WW1 median ratio: {round(np.median(input_types_nums_normalized[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name]) * 100, 2)}%')

    # Convert non-normalized values from sats to btc
    for item in input_types_nums.keys():
        input_types_nums[item] = np.array(input_types_nums[item]) / SATS_IN_BTC

    # Set normalized or non-normalized version to use
    input_types = input_types_nums_normalized if normalize_values else input_types_nums

    bar_width = 0.3
    categories = range(0, len(sorted_cj_time))

    # New version with separated remixes
    bars = []
    bars.append((input_types[MIX_EVENT_TYPE.MIX_ENTER.name], 'MIX_ENTER', 'blue', 0.9))
    #bars.append((input_types_nums[MIX_EVENT_TYPE.MIX_REMIX.name], 'MIX_REMIX', 'orange', 0.5))
    bars.append((input_types['MIX_REMIX_1-2'], 'MIX_REMIX_1-2', 'gold', 0.9))
    bars.append((input_types['MIX_REMIX_3-5'], 'MIX_REMIX_3-5', 'orange', 0.5))
    bars.append((input_types['MIX_REMIX_6-19'], 'MIX_REMIX_6-19', 'moccasin', 0.5))
    bars.append((input_types['MIX_REMIX_20+'], 'MIX_REMIX_20+', 'lightcoral', 0.7))
    bars.append((input_types['MIX_REMIX_2000+'], 'MIX_REMIX_2000+', 'sienna', 1))
    bars.append((input_types[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name], 'MIX_REMIX_FRIENDS', 'green', 0.5))
    bars.append((input_types[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name], 'MIX_REMIX_FRIENDS_WW1', 'green', 0.9))

    # Draw all inserted bars atop each other
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

    ax.set_title(f'Type of inputs for given cjtx ({'values' if analyze_values else 'number'})\n{short_exp_name}')
    ax.set_xlabel('Coinjoin in time')
    if analyze_values and normalize_values:
        ax.set_ylabel('Fraction of inputs sizes')
    if analyze_values and not normalize_values:
        ax.set_ylabel('Inputs sizes (btc)')
    if not analyze_values and normalize_values:
        ax.set_ylabel('Fraction of input numbers')
    if not analyze_values and not normalize_values:
        ax.set_ylabel('Input numbers')


def plot_mix_liquidity(mix_id: str, data: dict, initial_liquidity, time_liqiudity: dict, initial_cj_index: int, ax):
    coinjoins = data['coinjoins']
    cj_time = [{'txid': cjtxid, 'broadcast_time': precomp_datetime.strptime(coinjoins[cjtxid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")} for cjtxid in coinjoins.keys()]
    sorted_cj_time = sorted(cj_time, key=lambda x: x['broadcast_time'])

    mix_enter = [sum([coinjoins[cjtx['txid']]['inputs'][index]['value'] for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_ENTER.name])
                           for cjtx in sorted_cj_time]
    mix_remixfriend = [sum([coinjoins[cjtx['txid']]['inputs'][index]['value'] for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name])
                           for cjtx in sorted_cj_time]
    mix_remixfriend_ww1 = [sum([coinjoins[cjtx['txid']]['inputs'][index]['value'] for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX_FRIENDS_WW1.name])
                           for cjtx in sorted_cj_time]
    mix_leave = [sum([coinjoins[cjtx['txid']]['outputs'][index]['value'] for index in coinjoins[cjtx['txid']]['outputs'].keys()
                                    if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_LEAVE.name])
                               for cjtx in sorted_cj_time]
    mix_stay = [sum([coinjoins[cjtx['txid']]['outputs'][index]['value'] for index in coinjoins[cjtx['txid']]['outputs'].keys()
                                    if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_STAY.name])
                               for cjtx in sorted_cj_time]


    cjtx_cummulative_liquidity = []
    curr_liquidity = initial_liquidity[0]  # Take last cummulative liquidity (MIX_ENTERxxx - MIX_LEAVE) from previous interval
    assert len(mix_enter) == len(mix_leave) == len(mix_remixfriend) == len(mix_remixfriend_ww1) == len(mix_stay), logging.error(f'Mismatch in length of input/out sum arrays: {len(mix_enter)} vs. {len(mix_leave)}')
    # Change in liquidity as observed by each coinjoin (increase directly when mix_enter, decrease directly even when mix_leave happens later)
    for index in range(0, len(mix_enter)):
        curr_liquidity = curr_liquidity + mix_enter[index] + mix_remixfriend[index] + mix_remixfriend_ww1[index] - mix_leave[index]
        cjtx_cummulative_liquidity.append(curr_liquidity)

    # Cummulative liquidity never remixed or leaving mix (MIX_STAY coins)
    stay_liquidity = []
    curr_stay_liquidity = initial_liquidity[1]  # Take last cummulative liquidity (MIX_STAY) from previous interval
    for index in range(0, len(mix_stay)):
        curr_stay_liquidity = curr_stay_liquidity + mix_stay[index]
        stay_liquidity.append(curr_stay_liquidity)

    # Plot in btc
    liquidity_btc = [item / SATS_IN_BTC for item in cjtx_cummulative_liquidity]
    stay_liquidity_btc = [item / SATS_IN_BTC for item in stay_liquidity]
    #x_ticks = range(initial_cj_index, initial_cj_index + len(liquidity_btc))
    ax.plot(liquidity_btc, color='royalblue', alpha=0.6)
    #ax.plot(stay_liquidity_btc, color='royalblue', alpha=0.6, linestyle='--')
    ax.set_ylabel('btc in mix', color='royalblue')
    ax.tick_params(axis='y', colors='royalblue')

    return cjtx_cummulative_liquidity, stay_liquidity


def plot_mining_fee_rates(mix_id: str, data: dict, mining_fees: dict, ax):
    coinjoins = data['coinjoins']
    cj_time = [{'txid': cjtxid,
                'broadcast_time': precomp_datetime.strptime(coinjoins[cjtxid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")} for
               cjtxid in coinjoins.keys()]
    sorted_cj_time = sorted(cj_time, key=lambda x: x['broadcast_time'])

    # For each coinjoin find the closest fee rate record and plot it
    fee_rates = []
    fee_start_index = 0
    for cj in sorted_cj_time:
        timestamp = cj['broadcast_time'].timestamp()
        while timestamp > mining_fees[fee_start_index]['timestamp']:
            fee_start_index = fee_start_index + 1
        closest_fee = mining_fees[fee_start_index - 1]['avgFee_90']
        fee_rates.append(closest_fee)

    if ax:
        ax.plot(fee_rates, color='gray', alpha=0.4, linewidth=1, linestyle='--')
        ax.tick_params(axis='y', colors='gray', labelsize=6)
        ax.set_ylabel('Mining fee rate sats/vB (90th percentil)', color='gray', fontsize='6')

    return fee_rates


def plot_num_wallets(mix_id: str, data: dict, ax):
    coinjoins = data['coinjoins']
    cj_time = [{'txid': cjtxid,
                'broadcast_time': precomp_datetime.strptime(coinjoins[cjtxid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")} for
               cjtxid in coinjoins.keys()]
    sorted_cj_time = sorted(cj_time, key=lambda x: x['broadcast_time'])

    # For each coinjoin find the closest fee rate record and plot it
    AVG_NUM_INPUTS = 1.765  # value taken from simulations for all distributions
    num_wallets = [len(coinjoins[cj['txid']]['inputs']) / AVG_NUM_INPUTS for cj in sorted_cj_time]

    # Alternative plot:

    if ax:
        AVG_WINDOWS = 10
        num_wallets_avg = compute_averages(num_wallets, AVG_WINDOWS)
        ax.plot(num_wallets_avg, color='green', alpha=0.2, linewidth=1, linestyle='-')
        ax.tick_params(axis='y', colors='green', labelsize=6)
        ax.set_ylabel('Estimated number of active wallets', color='green', fontsize='6')

    return num_wallets


def analyze_input_out_liquidity(coinjoins, postmix_spend, premix_spend, mix_protocol: MIX_PROTOCOL, ww1_coinjoins={}, ww1_postmix_spend={}):
    """
    Requires performance speedup, will not finish (after 8 hours) for Whirlpool with very large number of coins
    :param coinjoins:
    :param postmix_spend:
    :param premix_spend:
    :param mix_protocol:
    :param ww1_coinjoins:
    :param ww1_postmix_spend:
    :return:
    """
    logging.debug('analyze_input_out_liquidity() started')
    liquidity_events = []
    total_inputs = 0
    total_mix_entering = 0
    total_mix_friends = 0
    total_outputs = 0
    total_mix_leaving = 0
    total_mix_staying = []
    total_utxos = 0
    broadcast_times = {cjtx: precomp_datetime.strptime(coinjoins[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for cjtx in coinjoins.keys()}
    broadcast_times.update({tx: precomp_datetime.strptime(postmix_spend[tx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for tx in postmix_spend.keys()})
    # Sort coinjoins based on time
    cj_time = [{'txid': cjtxid, 'broadcast_time': precomp_datetime.strptime(coinjoins[cjtxid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")} for cjtxid in coinjoins.keys()]
    sorted_cj_times = sorted(cj_time, key=lambda x: x['broadcast_time'])
    #coinjoins_list = [cj['txid'] for cj in sorted_cj_times]
    coinjoins_index = {}  # Precomputed mapping of txid to index for fast buntime computation
    for i in range(0, len(sorted_cj_times)):
        coinjoins_index[sorted_cj_times[i]['txid']] = i

    for cjtx in coinjoins:
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
                    coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs'] = coinjoins_index[cjtx] - coinjoins_index[spending_tx]
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
                        logging.warning(f'Could not find spend_by_tx {spend_by_tx} in postmix_spend txs')
                    else:
                        coinjoins[cjtx]['outputs'][output]['burn_time'] = round((broadcast_times[spend_by_tx] - broadcast_times[cjtx]).total_seconds(), 0)
                    total_mix_leaving += 1
                    coinjoins[cjtx]['outputs'][output]['mix_event_type'] = MIX_EVENT_TYPE.MIX_LEAVE.name
                else:
                    # Mix spend: The output is spent by next coinjoin tx => stays in mix
                    coinjoins[cjtx]['outputs'][output]['mix_event_type'] = MIX_EVENT_TYPE.MIX_REMIX.name
                    coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs'] = coinjoins_index[spend_by_tx] - coinjoins_index[cjtx]
                    coinjoins[cjtx]['outputs'][output]['burn_time'] = round((broadcast_times[spend_by_tx] - broadcast_times[cjtx]).total_seconds(), 0)

    SM.print(f'  {get_ratio_string(total_mix_entering, total_inputs)} Inputs entering mix / total inputs used by mix transactions')
    SM.print(f'  {get_ratio_string(total_mix_friends, total_inputs)} Friends inputs re-entering mix / total inputs used by mix transactions')
    SM.print(f'  {get_ratio_string(total_mix_leaving, total_outputs)} Outputs leaving mix / total outputs by mix transactions')
    SM.print(f'  {get_ratio_string(len(total_mix_staying), total_outputs)} Outputs staying in mix / total outputs by mix transactions')
    SM.print(f'  {sum(total_mix_staying) / SATS_IN_BTC} btc, total value staying in mix')

    logging.debug('analyze_input_out_liquidity() finished')


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
