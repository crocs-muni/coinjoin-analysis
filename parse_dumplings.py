
import os
from datetime import datetime

SATS_IN_BTC = 100000000
VerboseTransactionInfoLineSeparator = ':::'
VerboseInOutInfoInLineSeparator = '}'


def get_input_name_string(txid, index):
    return f'vin_{txid}_{index}'


def get_output_name_string(txid, index):
    return f'vout_{txid}_{index}'


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
                this_input['wtf'] = segments[3]
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


if __name__ == "__main__":
    target_base_path = 'c:\\!blockchains\\CoinJoin\\Dumplings_Stats_20231113\\'
    test = load_coinjoin_stats_from_file(os.path.join(target_base_path, 'Scanner', 'Wasabi2CoinJoins.txt'))
    print(len(test))
    #coinjoin_txs = load_coinjoin_stats(os.path.join(target_base_path, 'Scanner'))
