import os
from datetime import datetime
from jsonpickle import json

SATS_IN_BTC = 100000000
VerboseTransactionInfoLineSeparator = ':::'
VerboseInOutInfoInLineSeparator = '}'


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
    total_inputs = 0
    total_mix_entering = 0
    total_outputs = 0
    total_mix_leaving = 0
    total_utxos = 0
    for cjtx in coinjoins:
        for input in coinjoins[cjtx]['inputs']:
            total_inputs += 1
            spending_tx, index = extract_txid_from_inout_string(coinjoins[cjtx]['inputs'][input]['spending_tx'])
            if spending_tx not in coinjoins.keys():
                # Previous transaction is from outside the mix => new fresh liquidity entered
                total_mix_entering += 1

        for output in coinjoins[cjtx]['outputs']:
            total_outputs += 1
            if 'spend_by_tx' not in coinjoins[cjtx]['outputs'][output].keys():
                # This output is not spend by any tx => still utxo (stays within mixing pool)
                total_utxos += 1
                total_mix_leaving += 1
            else:
                # This output is spend, figue out if by other mixing transaction or postmix spend
                spend_by_tx, index = extract_txid_from_inout_string(coinjoins[cjtx]['outputs'][output]['spend_by_tx'])
                if spend_by_tx not in coinjoins.keys():
                    # Postmix spend: the spending transaction is outside mix => liquidity out
                    assert spend_by_tx in postmix_spend.keys(), "could not find spend_by_tx"
                    total_mix_leaving += 1
                # else: # Mix spend: The output is spent by next coinjoin tx => stay in mix

    print(f'Inputs entering mix / total inputs used by mix transactions = {total_mix_entering}/{total_inputs} ({round(total_mix_entering/float(total_inputs) * 100, 1)}%)')
    print(f'Outputs leaving mix / total outputs created by mix transactions =  {total_mix_leaving}/{total_outputs} ({round(total_mix_leaving/float(total_outputs) * 100, 1)}%)')


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


def load_coinjoins(target_path: str, mix_filename: str, postmix_filename: str, premix_filename: str =None) -> dict:
    # All mixes are having mixing coinjoins and postmix spends
    data = {'rounds': {}, 'filename': os.path.join(target_path, mix_filename),
            'coinjoins': load_coinjoin_stats_from_file(os.path.join(target_path, mix_filename)),
            'postmix': load_coinjoin_stats_from_file(os.path.join(target_path, postmix_filename))}

    # Only Samourai Whirlpool is having premix tx (TX0)
    if premix_filename is not None:
        data['premix'] = load_coinjoin_stats_from_file(os.path.join(target_path, premix_filename))

    data['wallets_info'] = extract_wallets_info(data)
    data['rounds'] = extract_rounds_info(data)

    return data


def process_coinjoins(target_path, mix_filename, postmix_filename, premix_filename=None):
    data = load_coinjoins(target_path, mix_filename, postmix_filename, premix_filename)

    print('*******************************************')
    print(f'{mix_filename} coinjoins: {len(data['coinjoins'])}')
    analyze_input_out_liquidity(data['coinjoins'], data['postmix'])

    return data


if __name__ == "__main__":
    FULL_TX_SET = False
    target_base_path = 'c:\\!blockchains\\CoinJoin\\Dumplings_Stats_20231113\\'
    target_path = os.path.join(target_base_path, 'Scanner')

    if FULL_TX_SET:
        # All transactions
        process_coinjoins(target_path, 'SamouraiCoinJoins.txt', 'SamouraiPostMixTxs.txt', 'SamouraiTx0s.txt')
        process_coinjoins(target_path, 'WasabiCoinJoins.txt', 'WasabiPostMixTxs.txt')
        process_coinjoins(target_path, 'Wasabi2CoinJoins.txt', 'Wasabi2PostMixTxs.txt')
    else:
        # Smaller set for debugging
        process_coinjoins(target_path, 'wasabi_mix_test.txt', 'wasabi_postmix_test.txt')
        data = process_coinjoins(target_path, 'wasabi2_mix_test.txt', 'wasabi2_postmix_test.txt')
        with open(os.path.join(target_path, 'wasabi2.json'), "w") as file:
            file.write(json.dumps(dict(sorted(data.items())), indent=4))
        process_coinjoins(target_path, 'sam_mix_test.txt', 'sam_postmix_test.txt')

