import logging
import numpy as np
from datetime import datetime
from enum import Enum

SATS_IN_BTC = 100000000


class MIX_EVENT_TYPE(Enum):
    MIX_ENTER = 'MIX_ENTER'  # New liquidity coming to mix
    MIX_LEAVE = 'MIX_LEAVE'  # Liquidity leaving mix (postmix spend)
    MIX_REMIX = 'MIX_REMIX'  # Remixed value within mix
    MIX_REMIX_FRIENDS = 'MIX_REMIX_FRIENDS'  # Remixed value within mix, but not directly, but one hop friends (WW2)
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


def extract_txid_from_inout_string(inout_string):
    if isinstance(inout_string, str):
        if inout_string.startswith('vin') or inout_string.startswith('vout'):
            return inout_string[inout_string.find('_') + 1: inout_string.rfind('_')], inout_string[inout_string.rfind('_')+1:]
        else:
            assert False, f'Invalid inout string {inout_string}'
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
    cj_time = [{'txid': cjtxid, 'broadcast_time': datetime.strptime(coinjoins[cjtxid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")} for cjtxid in coinjoins.keys()]
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
    bars.append((input_types['MIX_REMIX_2000+'], 'MIX_REMIX_2000+', 'peru', 0.7))
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
        ax.set_ylabel('Fraction of input values')
    if analyze_values and not normalize_values:
        ax.set_ylabel('Size of inputs')
    if not analyze_values and normalize_values:
        ax.set_ylabel('Fraction of number of inputs')
    if analyze_values and not normalize_values:
        ax.set_ylabel('Number of inputs')


def plot_mix_liquidity(mix_id: str, data: dict, initial_liquidity: int, initial_cj_index: int, ax):
    coinjoins = data['coinjoins']
    cj_time = [{'txid': cjtxid, 'broadcast_time': datetime.strptime(coinjoins[cjtxid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")} for cjtxid in coinjoins.keys()]
    sorted_cj_time = sorted(cj_time, key=lambda x: x['broadcast_time'])

    output_types_nums = {}
    mix_enter = [sum([coinjoins[cjtx['txid']]['inputs'][index]['value'] for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_ENTER.name])
                           for cjtx in sorted_cj_time]
    mix_remixfriend = [sum([coinjoins[cjtx['txid']]['inputs'][index]['value'] for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name])
                           for cjtx in sorted_cj_time]
    mix_leave = [sum([coinjoins[cjtx['txid']]['outputs'][index]['value'] for index in coinjoins[cjtx['txid']]['outputs'].keys()
                                    if coinjoins[cjtx['txid']]['outputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_LEAVE.name])
                               for cjtx in sorted_cj_time]

    liquidity = []
    curr_liquidity = initial_liquidity  # Take liquidity from previous interval
    assert len(mix_enter) == len(mix_leave) == len(mix_remixfriend), logging.error(f'Mismatch in length of input/out sum arrays: {len(mix_enter)} vs. {len(mix_leave)}')
    for index in range(0, len(mix_enter)):
        curr_liquidity = curr_liquidity + mix_enter[index] + mix_remixfriend[index] - mix_leave[index]
        liquidity.append(curr_liquidity)

    # Plot in btc
    liquidity_btc = [item / SATS_IN_BTC for item in liquidity]
    #x_ticks = range(initial_cj_index, initial_cj_index + len(liquidity_btc))
    ax.plot(liquidity_btc, color='royalblue')
    ax.set_ylabel('btc in mix', color='royalblue')
    ax.tick_params(axis='y', colors='royalblue')

    return liquidity[-1]


def analyze_input_out_liquidity(coinjoins, postmix_spend, premix_spend, mix_protocol: MIX_PROTOCOL):
    logging.debug('analyze_input_out_liquidity() started')
    liquidity_events = []
    total_inputs = 0
    total_mix_entering = 0
    total_mix_friends = 0
    total_outputs = 0
    total_mix_leaving = 0
    total_mix_staying = []
    total_utxos = 0
    broadcast_times = {cjtx: datetime.strptime(coinjoins[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for cjtx in coinjoins.keys()}
    broadcast_times.update({tx: datetime.strptime(postmix_spend[tx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for tx in postmix_spend.keys()})
    # Sort coinjoins based on time
    cj_time = [{'txid': cjtxid, 'broadcast_time': datetime.strptime(coinjoins[cjtxid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")} for cjtxid in coinjoins.keys()]
    sorted_cj_times = sorted(cj_time, key=lambda x: x['broadcast_time'])
    coinjoins_list = [cj['txid'] for cj in sorted_cj_times]

    for cjtx in coinjoins:
        for input in coinjoins[cjtx]['inputs']:
            total_inputs += 1
            if 'spending_tx' in coinjoins[cjtx]['inputs'][input].keys():
                spending_tx, index = extract_txid_from_inout_string(coinjoins[cjtx]['inputs'][input]['spending_tx'])
                if spending_tx not in coinjoins.keys():
                    # Direct previous transaction is from outside the mix => potentially new input liquidity
                    if mix_protocol == MIX_PROTOCOL.WASABI2:
                        # Either: 1. New fresh liquidity entered or 2. One hop friend mixing (if WW2)
                        if spending_tx in postmix_spend.keys():
                            # Friends do not pay tx
                            coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name
                            total_mix_friends += 1

                            # coinjoins[cjtx]['inputs'][input]['burn_time'] = round(
                            #     (broadcast_times[cjtx] - broadcast_times[spending_tx]).total_seconds(), 0)
                            # coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs'] = coinjoins_list.index(
                            #     cjtx) - coinjoins_list.index(spending_tx)
                        else:
                            total_mix_entering += 1
                            coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_ENTER.name
                    else:
                        # All other protocols than WW2 do not have 'friends do not pay'
                        total_mix_entering += 1
                        coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_ENTER.name
                else:  # Direct mix to mix transaction
                    coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_REMIX.name
                    coinjoins[cjtx]['inputs'][input]['burn_time'] = round((broadcast_times[cjtx] - broadcast_times[spending_tx]).total_seconds(), 0)
                    coinjoins[cjtx]['inputs'][input]['burn_time_cjtxs'] = coinjoins_list.index(cjtx) - coinjoins_list.index(spending_tx)
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
                    total_mix_leaving += 1
                    coinjoins[cjtx]['outputs'][output]['mix_event_type'] = MIX_EVENT_TYPE.MIX_LEAVE.name
                else:
                    # Mix spend: The output is spent by next coinjoin tx => stays in mix
                    coinjoins[cjtx]['outputs'][output]['mix_event_type'] = MIX_EVENT_TYPE.MIX_REMIX.name
                    coinjoins[cjtx]['outputs'][output]['burn_time_cjtxs'] = coinjoins_list.index(spend_by_tx) - coinjoins_list.index(cjtx)
                    coinjoins[cjtx]['outputs'][output]['burn_time'] = round((broadcast_times[spend_by_tx] - broadcast_times[cjtx]).total_seconds(), 0)

    SM.print(f'  {get_ratio_string(total_mix_entering, total_inputs)} Inputs entering mix / total inputs used by mix transactions')
    SM.print(f'  {get_ratio_string(total_mix_friends, total_inputs)} Friends inputs re-entering mix / total inputs used by mix transactions')
    SM.print(f'  {get_ratio_string(total_mix_leaving, total_outputs)} Outputs leaving mix / total outputs by mix transactions')
    SM.print(f'  {get_ratio_string(len(total_mix_staying), total_outputs)} Outputs staying in mix / total outputs by mix transactions')
    SM.print(f'  {sum(total_mix_staying) / SATS_IN_BTC} btc, total value staying in mix')

    logging.debug('analyze_input_out_liquidity() finished')