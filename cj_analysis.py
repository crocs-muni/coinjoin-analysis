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


def plot_inputs_type_ratio(mix_id: str, data: dict, initial_cj_index: int, ax, analyze_values: bool):
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

    for cjtx in sorted_cj_time:
        if sum([1 for index in coinjoins[cjtx['txid']]['inputs'].keys()
                if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == MIX_EVENT_TYPE.MIX_REMIX.name]) == 0:
            logging.warning(f'No remix detected for {cjtx}')

    input_types_nums = {}
    for type in MIX_EVENT_TYPE:
        if analyze_values:
            # Sum of values of inputs is taken
            input_types_nums[type.name] = [sum([coinjoins[cjtx['txid']]['inputs'][index]['value'] for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                        if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == type.name])
                                   for cjtx in sorted_cj_time]
        else:
            # Only number of inputs is taken
            input_types_nums[type.name] = [sum([1 for index in coinjoins[cjtx['txid']]['inputs'].keys()
                                        if coinjoins[cjtx['txid']]['inputs'][index]['mix_event_type'] == type.name])
                                   for cjtx in sorted_cj_time]

    short_exp_name = mix_id

    input_types_nums_normalized = {}
    total_values = (np.array(input_types_nums[MIX_EVENT_TYPE.MIX_ENTER.name]) + np.array(input_types_nums[MIX_EVENT_TYPE.MIX_REMIX.name])
                    + np.array(input_types_nums[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name]))
    input_types_nums_normalized[MIX_EVENT_TYPE.MIX_ENTER.name] = np.array(input_types_nums[MIX_EVENT_TYPE.MIX_ENTER.name]) / total_values
    input_types_nums_normalized[MIX_EVENT_TYPE.MIX_REMIX.name] = np.array(input_types_nums[MIX_EVENT_TYPE.MIX_REMIX.name]) / total_values
    input_types_nums_normalized[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name] = np.array(input_types_nums[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name]) / total_values

    input_types_nums = input_types_nums_normalized
    print(f'MIX_ENTER median ratio: {round(np.median(input_types_nums_normalized[MIX_EVENT_TYPE.MIX_ENTER.name]) * 100, 2)}%')
    print(f'MIX_REMIX median ratio: {round(np.median(input_types_nums_normalized[MIX_EVENT_TYPE.MIX_REMIX.name]) * 100, 2)}%')
    print(f'MIX_REMIX_FRIENDS median ratio: {round(np.median(input_types_nums_normalized[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name]) * 100, 2)}%')

    bar_width = 0.3
    categories = range(0, len(sorted_cj_time))
    first = (input_types_nums[MIX_EVENT_TYPE.MIX_ENTER.name], 'MIX_ENTER', 'blue', 0.9)
    second = (input_types_nums[MIX_EVENT_TYPE.MIX_REMIX.name], 'MIX_REMIX', 'orange', 0.5)
    third = (input_types_nums[MIX_EVENT_TYPE.MIX_REMIX_FRIENDS.name], 'MIX_REMIX_FRIENDS', 'green', 0.9)

    ax.bar(categories, first[0], bar_width, label=f'{first[1]} {short_exp_name}', alpha=first[3], color=first[2], linewidth=0)
    ax.bar(categories, second[0], bar_width, label=f'{second[1]} {short_exp_name}', alpha=second[3], color=second[2],
            bottom=np.array(first[0]), linewidth=0)
    ax.bar(categories, third[0], bar_width, label=f'{third[1]} {short_exp_name}', alpha=third[3], color=third[2],
            bottom=np.array(first[0]) + np.array(second[0]), linewidth=0)

    #ax.set_xticklabels(range(initial_cj_index, initial_cj_index + len(sorted_cj_time)))
    # current_ticks = ax.get_xticks()[0]
    # current_ticks_positions, current_tick_labels = ax.get_xticks(minor=False)
    # ticks_to_change = [tick for tick in current_ticks]
    # new_tick_labels = {tick: str(tick) for tick in current_ticks}  # Dictionary with new tick labels
    # ax.set_xticks(list(new_tick_labels.keys()), list(new_tick_labels.values()))
    #ax.set_xticks(current_ticks, [tick + initial_cj_index for tick in current_ticks])
    ax.set_title(f'Type of inputs for given cjtx ({'values' if analyze_values else 'number'})\n{short_exp_name}')
    ax.set_xlabel('Coinjoin in time')
    ax.set_ylabel('Fraction of inputs')


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