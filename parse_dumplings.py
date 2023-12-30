import copy
import os
from datetime import datetime
from enum import Enum
from collections import defaultdict
from jsonpickle import json

SATS_IN_BTC = 100000000
VerboseTransactionInfoLineSeparator = ':::'
VerboseInOutInfoInLineSeparator = '}'


class MIX_EVENT_TYPE(Enum):
    MIX_ENTER = 'MIX_ENTER'  # New liquidity coming to mix
    MIX_LEAVE = 'MIX_LEAVE'  # Liquidity leaving mix (postmix spend)
    MIX_REMIX = 'MIX_REMIX'  # Remixed value within mix
    MIX_STAY = 'MIX_STAY'    # Mix output not yet spend (may be remixed or leave mix later)


def get_ratio_string(numerator, denominator) -> str:
    if denominator != 0:
        return f'{numerator}/{denominator} ({round(numerator/float(denominator) * 100, 1)}%)'
    else:
        return f'{numerator}/{0} (0%)'



def get_input_name_string(txid, index):
    return f'vin_{txid}_{index}'


def get_output_name_string(txid, index):
    return f'vout_{txid}_{index}'


def extract_txid_from_inout_string(inout_string):
    if inout_string.startswith('vin') or inout_string.startswith('vout'):
        return inout_string[inout_string.find('_') + 1: inout_string.rfind('_')], inout_string[inout_string.rfind('_')+1:]
    else:
        assert False, f'Invalid inout string {inout_string}'


def load_coinjoin_stats_from_file(target_file):
    cj_stats = {}
    #print(f'Processing file {target_file}')
    with open(target_file, "r") as file:
        for line in file.readlines():
            parts = line.split(VerboseTransactionInfoLineSeparator)
            record = {}
            tx_id = None if parts[0] is None else parts[0]
            record['txid'] = tx_id
            block_hash = None if parts[1] is None else parts[1]
            record['block_hash'] = block_hash
            block_index = None if parts[2] is None else int(parts[2])
            record['block_index'] = block_index
            block_time = None if parts[3] is None else datetime.fromtimestamp(int(parts[3]))
            record['broadcast_time'] = block_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Be careful, broadcast time and blocktime can be significantly different

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
                this_input['value'] = float(segments[2]) / SATS_IN_BTC  # BUGBUG:keep in sats and correct analyzis code instead
                this_input['wallet_name'] = 'real_unknown'
                this_input['address'] = segments[3]
                this_input['script_type'] = segments[4]

                record['inputs'][f'{index}'] = this_input
                index += 1

            outputs = [output.strip('{') for output in parts[5].split(VerboseInOutInfoInLineSeparator)] if parts[5] else None
            record['outputs'] = {}
            index = 0
            for output in outputs:
                segments = output.split('+')
                this_output = {}
                this_output['value'] = float(segments[0]) / SATS_IN_BTC  # BUGBUG:keep in sats and correct analyzis code instead
                this_output['wallet_name'] = 'real_unknown'
                this_output['address'] = segments[1]  # BUGBUG: this is not address but likely script itself - needs for decoding
                this_output['script_type'] = segments[2]

                record['outputs'][f'{index}'] = this_output
                index += 1

            cj_stats[tx_id] = record

    # backward reference to spending transaction output is already set ('spending_tx'), now set also forward link ('spend_by_tx')
    for txid in cj_stats.keys():
        for index in cj_stats[txid]['inputs'].keys():
            input = cj_stats[txid]['inputs'][index]

            if 'spending_tx' in input.keys():
                tx, vout = extract_txid_from_inout_string(input['spending_tx'])
                # Try to find transaction and set its record
                if tx in cj_stats.keys() and vout in cj_stats[tx]['outputs'].keys():
                    cj_stats[tx]['outputs'][vout]['spend_by_tx'] = get_input_name_string(txid, index)

    return cj_stats


def load_coinjoin_stats(base_path):
    coinjoin_stats = {}
    files = []
    if os.path.exists(base_path):
        files = os.listdir(base_path)
    else:
        print('Path {} does not exists'.format(base_path))

    for file in files:
        target_file = os.path.join(base_path, file)
        coinjoin_stats[target_file]['coinjoins'] = load_coinjoin_stats_from_file(target_file)

    return coinjoin_stats


def analyze_input_out_liquidity(coinjoins, postmix_spend):
    liquidity_events = []
    total_inputs = 0
    total_mix_entering = 0
    total_outputs = 0
    total_mix_leaving = 0
    total_mix_staying = 0
    total_utxos = 0
    for cjtx in coinjoins:
        for input in coinjoins[cjtx]['inputs']:
            total_inputs += 1
            spending_tx, index = extract_txid_from_inout_string(coinjoins[cjtx]['inputs'][input]['spending_tx'])
            if spending_tx not in coinjoins.keys():
                # Previous transaction is from outside the mix => new fresh liquidity entered
                total_mix_entering += 1
                coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_ENTER.name
            else:
                coinjoins[cjtx]['inputs'][input]['mix_event_type'] = MIX_EVENT_TYPE.MIX_REMIX.name

        for output in coinjoins[cjtx]['outputs']:
            total_outputs += 1
            if 'spend_by_tx' not in coinjoins[cjtx]['outputs'][output].keys():
                # This output is not spend by any tx => still utxo (stays within mixing pool)
                total_utxos += 1
                total_mix_staying += 1
                coinjoins[cjtx]['outputs'][output]['mix_event_type'] = MIX_EVENT_TYPE.MIX_STAY.name
            else:
                # This output is spend, figure out if by other mixing transaction or postmix spend
                spend_by_tx, index = extract_txid_from_inout_string(coinjoins[cjtx]['outputs'][output]['spend_by_tx'])
                if spend_by_tx not in coinjoins.keys():
                    # Postmix spend: the spending transaction is outside mix => liquidity out
                    assert spend_by_tx in postmix_spend.keys(), "could not find spend_by_tx"
                    total_mix_leaving += 1
                    coinjoins[cjtx]['outputs'][output]['mix_event_type'] = MIX_EVENT_TYPE.MIX_LEAVE.name
                else:
                    # Mix spend: The output is spent by next coinjoin tx => stays in mix
                    coinjoins[cjtx]['outputs'][output]['mix_event_type'] = MIX_EVENT_TYPE.MIX_REMIX.name

    print(f'  {get_ratio_string(total_mix_entering, total_inputs)} Inputs entering mix / total inputs used by mix transactions')
    print(f'  {get_ratio_string(total_mix_leaving, total_outputs)} Outputs leaving mix / total outputs by mix transactions')
    print(f'  {get_ratio_string(total_mix_staying, total_outputs)} Outputs staying in mix / total outputs by mix transactions')


def extract_wallets_info(data):
    wallets_info = {}
    txs_data = data['coinjoins']
    for cjtxid in txs_data.keys():
        for index in txs_data[cjtxid]['inputs'].keys():
            target_addr = txs_data[cjtxid]['inputs'][index]['address']
            wallet_name = txs_data[cjtxid]['inputs'][index]['wallet_name']
            if wallet_name not in wallets_info.keys():
                wallets_info[wallet_name] = {}
            wallets_info[wallet_name][target_addr] = {'address': target_addr}
        for index in txs_data[cjtxid]['outputs'].keys():
            target_addr = txs_data[cjtxid]['outputs'][index]['address']
            wallet_name = txs_data[cjtxid]['outputs'][index]['wallet_name']
            if wallet_name not in wallets_info.keys():
                wallets_info[wallet_name] = {}
            wallets_info[wallet_name][target_addr] = {'address': target_addr}
    return wallets_info


def extract_rounds_info(data):
    rounds_info = {}
    txs_data = data['coinjoins']
    for cjtxid in txs_data.keys():
        # Create basic round info from coinjoin data
        rounds_info[cjtxid] = {"cj_tx_id": cjtxid, "round_start_timestamp": txs_data[cjtxid]['broadcast_time'],
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
    for txid in data['postmix'].keys():
        for index in data['postmix'][txid]['inputs'].keys():
            input = data['postmix'][txid]['inputs'][index]
            if 'spending_tx' in input.keys():
                tx, vout = extract_txid_from_inout_string(input['spending_tx'])
                # Try to find transaction in mix (coinjoins) and set its record
                if tx in data['coinjoins'].keys() and vout in data['coinjoins'][tx]['outputs'].keys():
                    data['coinjoins'][tx]['outputs'][vout]['spend_by_tx'] = get_input_name_string(txid, index)

    return data


def load_coinjoins(target_path: str, mix_filename: str, postmix_filename: str, premix_filename: str =None) -> dict:
    # All mixes are having mixing coinjoins and postmix spends
    data = {'rounds': {}, 'filename': os.path.join(target_path, mix_filename),
            'coinjoins': load_coinjoin_stats_from_file(os.path.join(target_path, mix_filename)),
            'postmix': load_coinjoin_stats_from_file(os.path.join(target_path, postmix_filename))}

    # Only Samourai Whirlpool is having premix tx (TX0)
    if premix_filename is not None:
        data['premix'] = load_coinjoin_stats_from_file(os.path.join(target_path, premix_filename))

    # Set spending transactions also between mix and postmix
    data = compute_mix_postmix_link(data)

    data['wallets_info'] = extract_wallets_info(data)
    data['rounds'] = extract_rounds_info(data)

    return data


def propagate_cluster_name_for_all_inputs(cluster_name, postmix_txs, txid, mix_txs):
    # Set same cluster id for all inputs
    for input in postmix_txs[txid]['inputs']:
        postmix_txs[txid]['inputs'][input]['cluster_id'] = cluster_name
        if 'spending_tx' in postmix_txs[txid]['inputs'][input]:
            tx, vout = extract_txid_from_inout_string(postmix_txs[txid]['inputs'][input]['spending_tx'])
            # Try to find transaction and set its record (postmix txs, coinjoin txs)
            if tx in postmix_txs.keys() and vout in postmix_txs[tx]['outputs'].keys():
                postmix_txs[tx]['outputs'][vout]['cluster_id'] = cluster_name
            if tx in mix_txs.keys() and vout in mix_txs[tx]['outputs'].keys():
                mix_txs[tx]['outputs'][vout]['cluster_id'] = cluster_name


def analyze_postmix_spends(tx_dict: dict) -> dict:
    """
    Simple chain analysis heuristics:
    1. N:1 Merges (many inputs, single output)
    2. 1:1 Resend (one input, one output)
    :param tx_dict: input dict with transactions
    :return: updated dict with transactions
    """
    postmix_txs = tx_dict['postmix']
    mix_txs = tx_dict['coinjoins']

    print('### Simple chain analysis')
    # N:1 Merge (many inputs, one output), including # 1:1 Resend (one input, one output)
    new_cluster_index = 0   # Unique cluster index (used if not already set)
    cluster_name = 'unassigned'  # Index to use
    offset = new_cluster_index  # starting offset of cluster index used to compute number of assigned indexes
    for txid in postmix_txs.keys():
        if len(postmix_txs[txid]['outputs']) == 1:
            # Find or use existing cluster index
            if 'cluster_id' in postmix_txs[txid]['outputs']['0']:
                cluster_name = postmix_txs[txid]['outputs']['0']['cluster_id']
            else:
                # New cluster index
                new_cluster_index += 1
                cluster_name = f'c_{new_cluster_index}'

            # Set output cluster id
            postmix_txs[txid]['outputs']['0']['cluster_id'] = cluster_name
            # Set same cluster id for all merged inputs
            propagate_cluster_name_for_all_inputs(cluster_name, postmix_txs, txid, mix_txs)

    # Compute total number of inputs used in postmix spending
    total_inputs = sum([len(postmix_txs[txid]['inputs']) for txid in postmix_txs.keys()])
    print(f' {get_ratio_string(new_cluster_index - offset, total_inputs)} N:1 postmix merges detected')

    return tx_dict


def analyze_coinjoin_blocks(data):
    same_block_coinjoins = defaultdict(list)
    for txid in data['coinjoins'].keys():
        same_block_coinjoins[data['coinjoins'][txid]['block_hash']].append(txid)
    filtered_dict = {key: value for key, value in same_block_coinjoins.items() if len(value) > 1}
    print(f' {get_ratio_string(len(filtered_dict), len(data['coinjoins']))} coinjoins in same block')
    #print(f'{filtered_dict}')


def process_coinjoins(target_path, mix_filename, postmix_filename, premix_filename=None):
    data = load_coinjoins(target_path, mix_filename, postmix_filename, premix_filename)

    print('*******************************************')
    print(f'{mix_filename} coinjoins: {len(data['coinjoins'])}')

    analyze_input_out_liquidity(data['coinjoins'], data['postmix'])

    analyze_postmix_spends(data)

    analyze_coinjoin_blocks(data)

    return data


def filter_liquidity_events(data):
    events = {}
    for txid in data['coinjoins']:
        events[txid] = copy.deepcopy(data['coinjoins'][txid])
        events[txid].pop('block_hash')
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
                    if item not in ['value', 'wallet_name', 'mix_event_type']:
                        events[txid]['inputs'][input].pop(item)
        # Process outputs
        events[txid]['num_outputs'] = len(events[txid]['outputs'])
        for output in list(events[txid]['outputs'].keys()):
            if ('mix_event_type' not in events[txid]['outputs'][output]
                    or events[txid]['outputs'][output]['mix_event_type'] not in [MIX_EVENT_TYPE.MIX_ENTER.name, MIX_EVENT_TYPE.MIX_LEAVE.name]):
                # Remove whole given output
                events[txid]['outputs'].pop(output)
            else:
                # Remove all unnecessary data
                for item in events[txid]['outputs'][output].copy():
                    if item not in ['value', 'wallet_name', 'mix_event_type']:
                        events[txid]['outputs'][output].pop(item)
    return events


def process_and_save_coinjoins(mix_id: str, target_path: os.path, mix_filename: str, postmix_filename: str, premix_filename: str=None):
    # Process and save full conjoin information
    data = process_coinjoins(target_path, mix_filename, postmix_filename, premix_filename)
    if SAVE_BASE_FILES:
        with open(os.path.join(target_path, f'{mix_id}_txs.json'), "w") as file:
            file.write(json.dumps(dict(sorted(data.items())), indent=4))

    # Filter only liquidity-relevant events to maintain smaller file
    events = filter_liquidity_events(data)
    with open(os.path.join(target_path, f'{mix_id}_events.json'), "w") as file:
        file.write(json.dumps(dict(sorted(events.items())), indent=4))


if __name__ == "__main__":
    FULL_TX_SET = True
    SAVE_BASE_FILES = False
    target_base_path = 'c:\\!blockchains\\CoinJoin\\Dumplings_Stats_20231113\\'
    target_path = os.path.join(target_base_path, 'Scanner')

    if FULL_TX_SET:
        # All transactions
        process_and_save_coinjoins('whirlpool', target_path, 'SamouraiCoinJoins.txt', 'SamouraiPostMixTxs.txt', 'SamouraiTx0s.txt')
        process_and_save_coinjoins('wasabi', target_path, 'WasabiCoinJoins.txt', 'WasabiPostMixTxs.txt')
        process_and_save_coinjoins('wasabi2', target_path, 'Wasabi2CoinJoins.txt', 'Wasabi2PostMixTxs.txt')
    else:
        # Smaller set for debugging
        process_and_save_coinjoins('wasabi_test', target_path, 'wasabi_mix_test.txt', 'wasabi_postmix_test.txt')
        process_and_save_coinjoins('wasabi2_test', target_path, 'wasabi2_mix_test.txt', 'wasabi2_postmix_test.txt')
        process_and_save_coinjoins('whirlpool_test', target_path, 'sam_mix_test.txt', 'sam_postmix_test.txt')

