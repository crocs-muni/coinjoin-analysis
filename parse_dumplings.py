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

    with open(target_file, "r") as file:
        for line in file.readlines():
            parts = line.split(VerboseTransactionInfoLineSeparator)
            record = {}
            id = None if parts[0] is None else parts[0]
            record['txid'] = id
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
                this_input['wallet_name'] = 'unknown'
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
                this_output['wallet_name'] = 'unknown'
                this_output['address'] = segments[1]  # BUGBUG: this is not address but likely script itself - needs for decoding
                this_output['script_type'] = segments[2]

                record['outputs'][f'{index}'] = this_output
                index += 1

            cj_stats[id] = record

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


def analyze_coinjoins(coinjoins):
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
                # Previous transaction is outside mix => new fresh liquidity
                total_mix_entering += 1

        for output in coinjoins[cjtx]['outputs']:
            total_outputs += 1
            if 'spend_by_tx' not in coinjoins[cjtx]['outputs'][output].keys():
                # This output is not spend by any => utxo
                total_utxos += 1
            else:
                spend_by_tx, index = extract_txid_from_inout_string(coinjoins[cjtx]['outputs'][output]['spend_by_tx'])
                if spend_by_tx not in coinjoins.keys():
                    # The spending transaction is outside mix => liquidity out
                    total_mix_leaving += 1
                #else: # The output is spent by next coinjoin tx => stay in mix

    print(f'Total inputs entering = {total_mix_entering}')
    print(f'Total outputs leaving = {total_mix_leaving}')
    print(f'Total inputs/outputs = {total_inputs}/{total_outputs}')
    print(f'Total utxos = {total_utxos}')


if __name__ == "__main__":
    target_base_path = 'c:\\!blockchains\\CoinJoin\\Dumplings_Stats_20231113\\'
    target_path = os.path.join(target_base_path, 'Scanner', 'SamouraiCoinJoins.txt')
    #target_path = os.path.join(target_base_path, 'Scanner', 'sam_test.txt')
    whirlpool = {}
    whirlpool['wallets_info'] = {}
    whirlpool['rounds'] = {}
    whirlpool['coinjoins'] = load_coinjoin_stats_from_file(target_path)
    num_coinjoins = len(whirlpool['coinjoins'])
    print(f'Whirlpool coinjoins: {num_coinjoins}')

    #parse_cj_logs.compute_link_between_inputs_and_outputs(whirlpool, [cjtxid for cjtxid in whirlpool.keys()])

    # with open(target_path + '.json', "w") as file:
    #     file.write(json.dumps(dict(sorted(whirlpool.items())), indent=4))

    # Analyze loaded data
    analyze_coinjoins(whirlpool['coinjoins'])

    # target_path = os.path.join(target_base_path, 'Scanner', 'Wasabi2CoinJoins.txt')
    # wasabi2 = load_coinjoin_stats_from_file(target_path)
    # print(f'Wasabi2 coinjoins: {len(wasabi2)}')
    # with open(target_path + '.json', "w") as file:
    #     file.write(json.dumps(dict(sorted(wasabi2.items())), indent=4))


    #coinjoin_txs = load_coinjoin_stats(os.path.join(target_base_path, 'Scanner'))
