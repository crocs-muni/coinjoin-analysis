import re
import subprocess
import json
import sys
import wcli
from itertools import zip_longest
import graphviz
from graphviz import Digraph
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime, timedelta


BTC_CLI_PATH = 'C:\\bitcoin-25.0\\bin\\bitcoin-cli'
WASABIWALLET_DATA_DIR = 'c:\\Users\\xsvenda\\AppData\\Roaming\\WalletWasabi'
TX_AD_CUT_LEN = 10  # length of displayed address or txid
WALLET_COLORS = {}
UNKNOWN_WALLET_STRING = 'UNKNOWN'
PRINT_COLLATED_COORD_CLIENT_LOGS = False
INSERT_WALLET_NODES = False


# colors used for different wallet clusters. Avoid following colors : 'red' (used for cjtx)
COLORS = ['darkorange', 'green', 'lightblue', 'gray', 'aquamarine', 'darkorchid1', 'cornsilk3', 'chocolate',
   'deeppink1', 'cadetblue', 'darkgreen', 'black']


def read_lines_for_round(filename, round_id):
    """
    Returns all lines from given log file which references target round id
    :param filename: name of file with logs
    :param round_id: target round id
    :return: list of log lines with round_id
    """
    lines_with_round = []
    target_string = 'Round ({})'.format(round_id)
    try:
        with open(filename, 'r') as file:
            for line in file:
                if target_string in line:
                    lines_with_round.append(line)
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return lines_with_round


def find_round_ids(filename, regex_pattern, group_names):
    """
    Extracts all round_ids which from provided file which match regexec pattern and its specified part given by group_name.
    Function is more generic as any group_name from regex_pattern can be specified, not only round_id
    :param filename: name of file with logs
    :param regex_pattern: regex pattern which is matched to every line
    :param group_name: name of item specified in regex pattern, which is extracted
    :return: list of dictionaries for all specified group_names
    """
    hits = {}

    try:
        with open(filename, 'r') as file:
            for line in file:
                for match in re.finditer(regex_pattern, line):
                    hit_group = {}
                    for group_name in group_names:  # extract all provided group names
                        if group_name in match.groupdict():
                            hit_group[group_name] = match.group(group_name).strip()
                    # insert into dictionary with key equal to value of first hit group
                    key_name = match.group(group_names[0]).strip()
                    hits[key_name] = hit_group

    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return hits


def find_round_cjtx_mapping(filename, regex_pattern, round_id, cjtx):
    """
    Extracts mapping between round id and its coinjoin tx id.
    :param filename: name of file with logs
    :param regex_pattern: regex pattern to match log line where mapping is found
    :param round_id: name in regex for round id item
    :param cjtx: name in regex for coinjointx id item
    :return: dictionary of mapping between round_id and coinjoin tx id
    """
    mapping = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                for match in re.finditer(regex_pattern, line):
                    if round_id in match.groupdict() and cjtx in match.groupdict():
                        mapping[match.group(round_id).strip()] = match.group(cjtx).strip()
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return mapping


def print_round_logs(filename, round_id):
    """
    Print log lines for given round id with visual separator afterwards
    :param filename: file with logs
    :param round_id: target round id
    """
    round_logs = read_lines_for_round(filename, round_id)
    [print(line.rstrip()) for line in round_logs]
    print('**************************************')
    print('**************************************')
    print('**************************************')


def run_command(command, verbose):
    """
    Execute shell command and return results
    :param command: command line to be executed
    :param verbose: if True, print intermediate results
    :return: command results with stdout, stderr and returncode (see subprocess CompletedProcess for documentation)
    """
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        if verbose:
            if result.returncode == 0:
                print("Command executed successfully.")
                print("Output:")
                print(result.stdout)
            else:
                print("Command failed.")
                print("Error:")
                print(result.stderr)
    except Exception as e:
        print("An error occurred:", e)

    return result


def get_input_address(txid, txid_in_out):
    """
    Returns address which was used in transaction given by 'txid' as 'txid_in_out' output index
    :param txid:
    :param txid_in_out:
    :return:
    """
    result = run_command(
        '{} -regtest getrawtransaction {} true'.format(BTC_CLI_PATH, txid), False)
    tx_info = result.stdout

    try:
        parsed_data = json.loads(tx_info)
        outputs = parsed_data['vout']
        for output in outputs:
            if output['n'] == txid_in_out:
                return output['scriptPubKey']['address'], parsed_data

    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)

    return None


def extract_tx_info(txid):
    """
    Extract input and output addresses
    :param txid:
    :return:
    """
    result = run_command(
        '{} -regtest getrawtransaction {} true'.format(BTC_CLI_PATH, txid), False)
    tx_info = result.stdout

    tx_record = {}
    input_addresses = {}
    input_txids = {}
    output_addresses = {}
    try:
        parsed_data = json.loads(tx_info)
        tx_record = {}

        tx_record['txid'] = txid
        #tx_record['raw_tx_json'] = parsed_data
        tx_record['inputs'] = {}
        tx_record['outputs'] = {}

        inputs = parsed_data['vin']
        index = 0
        for input in inputs:
            in_address, in_full_info = get_input_address(input['txid'], input['vout'])  # we need to read and parse previous transaction to obtain address
            tx_record['inputs'][index] = {}
            #tx_record['inputs'][index]['full_info'] = in_full_info
            tx_record['inputs'][index]['address'] = in_address
            tx_record['inputs'][index]['txid'] = input['txid']
            input_addresses[index] = in_address  # store address to index of the input
            index = index + 1

        outputs = parsed_data['vout']
        for output in outputs:
            output_addresses[output['n']] = output['scriptPubKey']['address']

            tx_record['outputs'][output['n']] = {}
            #tx_record['outputs'][output['n']]['full_info'] = output
            tx_record['outputs'][output['n']]['address'] = output['scriptPubKey']['address']
            tx_record['outputs'][output['n']]['value'] = output['value']

    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)

    return tx_record


def graphviz_insert_wallet(wallet_name, graphdot):
    if INSERT_WALLET_NODES:
        graphdot.attr('node', shape='diamond')
        graphdot.attr('node', fillcolor='green')
        graphdot.attr('node', color=WALLET_COLORS[wallet_name])
        graphdot.attr('node', style='filled')
        graphdot.attr('node', fontsize='20')
        graphdot.node(wallet_name)


def graphviz_insert_address(addr, fill_color, graphdot):
    if TX_AD_CUT_LEN > 0:
        addr = addr[:TX_AD_CUT_LEN] + '...'

    graphdot.attr('node', shape='ellipse')
    graphdot.attr('node', fillcolor=fill_color)
    graphdot.attr('node', color='gray')
    graphdot.attr('node', style='filled')
    graphdot.attr('node', fontsize='20')
    graphdot.attr('node', id=addr)
    graphdot.attr('node', label='{}'.format(addr))
    graphdot.node(addr)


def graphviz_insert_cjtxid(coinjoin_tx, graphdot):
    cjtxid = coinjoin_tx['txid']
    if TX_AD_CUT_LEN > 0:
        cjtxid = cjtxid[:TX_AD_CUT_LEN] + '...'

    graphdot.attr('node', shape='box')
    graphdot.attr('node', fillcolor='red')
    graphdot.attr('node', color='black')
    graphdot.attr('node', style='filled')
    graphdot.attr('node', fontsize='20')
    graphdot.attr('node', id=cjtxid)
    graphdot.attr('node', label='cjtxid:\n{}\n{}\n{}'.format(cjtxid, coinjoin_tx['round_start_time'], coinjoin_tx['broadcast_time']))
    graphdot.node(cjtxid)


def graphviz_insert_wallet_address_mapping(wallet_name, addr, graphdot):
    if INSERT_WALLET_NODES:
        if TX_AD_CUT_LEN > 0:
            addr = addr[:TX_AD_CUT_LEN] + '...'
        graphdot.edge(wallet_name, addr, color=WALLET_COLORS[wallet_name], style='dotted', dir='none')


def graphviz_insert_address_cjtx_mapping(addr, coinjoin_txid, edge_color, graphdot):
    if TX_AD_CUT_LEN > 0:
        addr = addr[:TX_AD_CUT_LEN] + '...'
        coinjoin_txid = coinjoin_txid[:TX_AD_CUT_LEN] + '...'
    graphdot.edge(addr, coinjoin_txid, color=edge_color, style='dashed')


def graphviz_insert_cjtx_address_mapping(coinjoin_txid, addr, value_size, edge_color, graphdot):
    if TX_AD_CUT_LEN > 0:
        addr = addr[:TX_AD_CUT_LEN] + '...'
        coinjoin_txid = coinjoin_txid[:TX_AD_CUT_LEN] + '...'
    if value_size < 1:
        width = '1'
    elif value_size < 5:
        width = '3'
    elif value_size < 10:
        width = '5'
    elif value_size < 15:
        width = '7'
    elif value_size < 20:
        width = '9'
    else:
        width = '11'

    graphdot.edge(coinjoin_txid, addr, color=edge_color, style='solid', label="{}₿".format(value_size), penwidth=width)


def print_tx_info(cjtx, address_wallet_mapping, graphdot):
    """
    Prints mapping between addresses in given coinjoin transaction.
    :param input_addresses: input addresses to cjtx
    :param output_addresses: output addresses from cjtx
    :param address_wallet_mapping: mapping between address and controlling wallet
    :param coinjoin_txid: transaction id of coinjoin transaction
    :param graphdot: graphviz engine
    :return:
    """

    used_wallets = {}
    input_addresses = [cjtx['inputs'][index]['address'] for index in cjtx['inputs'].keys()]
    output_addresses = [cjtx['outputs'][index]['address'] for index in cjtx['outputs'].keys()]
    for addr in input_addresses + output_addresses:
        if addr in address_wallet_mapping.keys():
            used_wallets[address_wallet_mapping[addr]] = 1
        else:
            print('Missing wallet mapping for {}'.format(addr))

    cjtxid = cjtx['txid']
    used_wallets = sorted(used_wallets.keys())
    # print all inputs mapped to their outputs
    for wallet_name in used_wallets:
        print('Wallet `{}`'.format(wallet_name))

        for index in cjtx['inputs'].keys():
            addr = cjtx['inputs'][index]['address']
            if address_wallet_mapping[addr] == wallet_name:
                print('  ({}):{}'.format(wallet_name, addr))
                graphviz_insert_address(addr, WALLET_COLORS[wallet_name], graphdot)
                graphviz_insert_wallet_address_mapping(wallet_name, addr, graphdot)  # wallet to address
                graphviz_insert_address_cjtx_mapping(addr, cjtxid, WALLET_COLORS[wallet_name], graphdot)  # address to coinjoin txid

        #with graphdot.subgraph() as group:
        for index in cjtx['outputs'].keys():
            addr = cjtx['outputs'][index]['address']
            if address_wallet_mapping[addr] == wallet_name:
                print('  -> ({}):{}'.format(wallet_name, addr))
                graphviz_insert_address(addr, WALLET_COLORS[wallet_name], graphdot)
                graphviz_insert_wallet_address_mapping(wallet_name, addr, graphdot)  # wallet to address
                graphviz_insert_cjtx_address_mapping(cjtxid, addr, cjtx['outputs'][index]['value'], WALLET_COLORS[wallet_name], graphdot)  # coinjoin to addr


def analyze_coinjoin_stats(cjtx_stats, address_wallet_mapping, wallets_info):
    """
    Analyze coinjoin transactions statistics
    :param cjtx_stats:
    :param address_wallet_mapping:
    :return:
    """

    #
    # Number of distinct wallets in coinjoins
    #
    cjtx_wallet_frequencies = {}
    for cjtx in cjtx_stats.keys():
        cjtx_wallet_frequencies[cjtx] = {}
        used_wallets = []
        for input_index in cjtx_stats[cjtx]['inputs'].keys():  # iterate over input addresses
            addr = cjtx_stats[cjtx]['inputs'][input_index]['address']
            if addr in address_wallet_mapping.keys():
                used_wallets.append(address_wallet_mapping[addr])
            else:
                print('Missing wallet mapping for {}'.format(addr))

        for wallet in sorted(wallets_info.keys()):
            cjtx_wallet_frequencies[cjtx][wallet] = used_wallets.count(wallet)

    # Distribution of number of inputs from different wallets
    for cjtx in cjtx_wallet_frequencies.keys():
        cjtx_stats[cjtx]['num_wallets_involved'] = sum(1 for value in cjtx_wallet_frequencies[cjtx].values() if value != 0)

    wallets_in_stats = [cjtx_stats[value]['num_wallets_involved'] for value in cjtx_stats.keys()]
    wallets_in_frequency = Counter(wallets_in_stats)
    wallets_in_frequency_all = {}
    for wallet_num in range(0, max(wallets_in_stats) + 1):
        if wallet_num in wallets_in_frequency.keys():
            wallets_in_frequency_all[wallet_num] = wallets_in_frequency[wallet_num]
        else:
            wallets_in_frequency_all[wallet_num] = 0
    wallets_in_frequency_all_ordered = [wallets_in_frequency_all[key] for key in sorted(wallets_in_frequency_all.keys())]
    plt.bar(range(0, len(wallets_in_frequency_all_ordered)), wallets_in_frequency_all_ordered)
    plt.xticks(range(0, len(wallets_in_frequency_all_ordered)), range(0, len(wallets_in_frequency_all_ordered)))
    plt.xlabel('Number of distinct wallets')
    plt.ylabel('Number of coinjoins')
    plt.title('Number of separate wallets participating in coinjoin tx')
    plt.savefig('num_wallets_in_round.png')
    plt.close()

    #
    # How many times given wallet participated in coinjoin?
    #
    wallets_used = []
    for wallet_name in wallets_info.keys():
        wallet_times_used = 0
        for cjtx in cjtx_stats.keys():  # go over all coinjoin transactions
            wallet_times_used_in_cjtx = 0
            for index in cjtx_stats[cjtx]['inputs']:
                if cjtx_stats[cjtx]['inputs'][index]['wallet_name'] == wallet_name:
                    wallet_times_used_in_cjtx = wallet_times_used_in_cjtx + 1
            wallet_times_used = wallet_times_used + wallet_times_used_in_cjtx
        wallets_used.append(wallet_times_used)

    plt.bar(wallets_info.keys(), wallets_used)
    plt.xlabel('Wallet name')
    plt.xticks(rotation=45)
    plt.ylabel('Number of participations')
    plt.title('Number of times given wallet was participating in coinjoin')
    plt.savefig('num_times_wallet_used.png')
    plt.close()

    #
    # Number of distinct wallets in coinjoins in time
    #
    SLOT_WIDTH_SECONDS = 3600
    experiment_start_time = datetime.strptime(cjtx_stats[list(cjtx_stats.keys())[0]]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")
    slot_start_time = experiment_start_time
    slot_last_time = datetime.strptime(cjtx_stats[list(cjtx_stats.keys())[-1]]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")
    diff_seconds = (slot_last_time - slot_start_time).total_seconds()
    num_slots = int(diff_seconds // SLOT_WIDTH_SECONDS)
    cjtx_in_hours = {hour: [] for hour in range(0, num_slots + 1)}
    for cjtx in cjtx_stats.keys():  # go over all coinjoin transactions
        timestamp = datetime.strptime(cjtx_stats[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")
        cjtx_hour = int((timestamp - slot_start_time).total_seconds() // SLOT_WIDTH_SECONDS)
        cjtx_in_hours[cjtx_hour].append(cjtx)

    plt.plot([len(cjtx_in_hours[cjtx_hour]) for cjtx_hour in cjtx_in_hours.keys()])
    x_ticks = []
    for slot in cjtx_in_hours.keys():
        x_ticks.append((experiment_start_time + slot * timedelta(seconds=SLOT_WIDTH_SECONDS)).strftime("%Y-%m-%d %H:%M:%S"))

    plt.xticks(range(0, len(x_ticks)), x_ticks)
    plt.xticks(rotation=45, fontsize=6)
    plt.ylim(0)
    plt.subplots_adjust(bottom=0.2)
    plt.ylabel('Number of coinjoin transactions')
    plt.title('Number of coinjoin transactions in given time period')
    plt.savefig('num_coinjoins_in_time.png')
    plt.close()


def parse_coinjoin_logs(wallets_info, graphdot):
    #WASABIWALLET_DATA_DIR = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\20230901_5minRound_3parallel_max10inputs\\'
    #coord_input_file = '{}\\Backend\\Logs_4paralell_1hour.txt'.format(WASABIWALLET_DATA_DIR)
    #client_input_file = '{}\\Client\\Logs_4paralell_1hour.txt'.format(WASABIWALLET_DATA_DIR)

    #WASABIWALLET_DATA_DIR = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\20230822_90secRound_4parallel_1hour\\'
    #coord_input_file = '{}\\Backend\\Logs_4paralell_1hour_debug.txt'.format(WASABIWALLET_DATA_DIR)
    #client_input_file = '{}\\Client\\Logs_4paralell_1hour_debug.txt'.format(WASABIWALLET_DATA_DIR)

    #WASABIWALLET_DATA_DIR = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol1\\20230830_5minRound_3parallel_overnight'
    #WASABIWALLET_DATA_DIR = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol1\\20230901_5minRound_3parallel_max10inputs'
    #coord_input_file = '{}\\Backend\\Logs.txt'.format(WASABIWALLET_DATA_DIR)
    #client_input_file = '{}\\Client\\Logs.txt'.format(WASABIWALLET_DATA_DIR)
    # coord_input_file = '{}\\Backend\\Logs_debug.txt'.format(WASABIWALLET_DATA_DIR)
    # client_input_file = '{}\\Client\\Logs_debug.txt'.format(WASABIWALLET_DATA_DIR)
    # coord_input_file = '{}\\Backend\\Logs_20230829_5minRound_3parallel_overnight.txt'.format(WASABIWALLET_DATA_DIR)
    # client_input_file = '{}\\Client\\Logs_20230829_5minRound_3parallel_overnight.txt'.format(WASABIWALLET_DATA_DIR)

    # coord_input_file = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\20230830_5minRound_3parallel_overnight\\Logs_backend_20230830_5minRound_3parallel_overnight.txt'
    # client_input_file = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\20230830_5minRound_3parallel_overnight\\Logs_client_20230830_5minRound_3parallel_overnight.txt'
    # coord_input_file = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\20230830_5minRound_3parallel_overnight\\Logs_backend_20230829_5minRound_3parallel_overnight.txt'
    # client_input_file = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\20230830_5minRound_3parallel_overnight\\Logs_client_20230829_5minRound_3parallel_overnight.txt'

    #WASABIWALLET_DATA_DIR = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol2\\debug\\WalletWasabi'



    coord_input_file = '{}\\Backend\\Logs.txt'.format(WASABIWALLET_DATA_DIR)
    client_input_file = '{}\\Client\\Logs.txt'.format(WASABIWALLET_DATA_DIR)


    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Created round with params: MaxSuggestedAmount:'([0-9\.]+)' BTC?"
    start_round_ids = find_round_ids(coord_input_file, regex_pattern, ['round_id', 'timestamp'])
    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Successfully broadcast the coinjoin: (?P<cj_tx_id>[0-9a-f]*)\.?"
    success_coinjoin_round_ids = find_round_ids(coord_input_file, regex_pattern, ['round_id', 'timestamp'])
    round_cjtx_mapping = find_round_cjtx_mapping(coord_input_file, regex_pattern, 'round_id', 'cj_tx_id')

    # find all ids which have complete log from round creation (Created round with params)
    # to cj tx broadcast (Successfully broadcast the coinjoin)
    full_round_ids = [key_value for key_value in success_coinjoin_round_ids.keys() if key_value in start_round_ids.keys()]

    cjtx_stats = {}
    for round_id in full_round_ids:
        # Find coinjoin transaction id and create record if not already
        cjtxid = round_cjtx_mapping[round_id]
        if cjtxid not in cjtx_stats.keys():
            cjtx_stats[cjtxid] = {}

        # extract input and output addreses
        tx_record = extract_tx_info(round_cjtx_mapping[round_id])
        tx_record['round_id'] = round_id
        tx_record['round_start_time'] = start_round_ids[round_id]['timestamp']
        tx_record['broadcast_time'] = success_coinjoin_round_ids[round_id]['timestamp']
        cjtx_stats[cjtxid] = tx_record

    # print only logs with full rounds
    #[print_round_logs(coord_input_file, id) for id in full_round_ids]
    print('\n\nTotal complete rounds found: {}'.format(len(full_round_ids)))

    # 2023-08-22 11:06:35.181 [21] DEBUG	CoinJoinClient.CreateRegisterAndConfirmCoinsAsync (469)	Round (5f3425c1f2e0cc81c9a74a213abf1ea3f128247d6be78ecd259158a5e1f9b66c): Inputs(4) registration started - it will end in: 00:01:22.
    #regex_pattern = r"(.*) \[.+(?P<method>CoinJoinClient\..*) \([0-9]+\).*Round \((?P<round_id>.*)\): Inputs\((?P<num_inputs>[0-9]+)\) registration started - it will end in: ([0-9:]+)\."
    #client_start_round_ids = find_round_ids(coord_input_file, regex_pattern, 'round_id')
    # 2023-08-22 11:06:51.466 [10] INFO	AliceClient.RegisterInputAsync (105)	Round (5f3425c1f2e0cc81c9a74a213abf1ea3f128247d6be78ecd259158a5e1f9b66c), Alice (94687969-bf26-1dfd-af98-2365e708b893): Registered 80b9c8615226e03d2474d8ad481c2db7505cb2715b10d83ee9c95106aaa3dcfd-0.
    #regex_pattern = r"(.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Successfully broadcast the coinjoin: (?P<cj_tx_id>[0-9a-f]*)\.?"

    #cjtx_stats = {}
    address_wallet_mapping = {}
    for round_id in full_round_ids:
        cjtxid = round_cjtx_mapping[round_id]
        #input_addresses = cjtx_stats[cjtxid]['inputs']
        #output_addresses = cjtx_stats[cjtxid]['outputs']
        # Build mapping of addresses to wallets names ('unknown' if not mapped)
        address_wallet_mapping.update(
            {cjtx_stats[cjtxid]['inputs'][addr_index]['address']: UNKNOWN_WALLET_STRING for addr_index in cjtx_stats[cjtxid]['inputs'].keys()})
        address_wallet_mapping.update(
            {cjtx_stats[cjtxid]['outputs'][addr_index]['address']: UNKNOWN_WALLET_STRING for addr_index in cjtx_stats[cjtxid]['outputs'].keys()})

        for index in cjtx_stats[cjtxid]['inputs'].keys():
            addr = cjtx_stats[cjtxid]['inputs'][index]['address']
            for wallet_name in wallets_info.keys():
                for waddr in wallets_info[wallet_name]:
                    if addr == waddr['address']:
                        address_wallet_mapping[addr] = wallet_name
                        cjtx_stats[cjtxid]['inputs'][index]['wallet_name'] = wallet_name

        for index in cjtx_stats[cjtxid]['outputs'].keys():
            addr = cjtx_stats[cjtxid]['outputs'][index]['address']
            for wallet_name in wallets_info.keys():
                for waddr in wallets_info[wallet_name]:
                    if addr == waddr['address']:
                        address_wallet_mapping[addr] = wallet_name
                        cjtx_stats[cjtxid]['outputs'][index]['wallet_name'] = wallet_name

        #
        # Print collated round logs from coordinator and client for a given round
        #
        if PRINT_COLLATED_COORD_CLIENT_LOGS:
            coord_round_logs = read_lines_for_round(coord_input_file, round_id)
            client_round_logs = read_lines_for_round(client_input_file, round_id)

            sorted_combined_list = sorted(coord_round_logs + client_round_logs)
            for line in sorted_combined_list:
                line = line.replace(" INFO", " INFO ")
                if line in client_round_logs:
                    print("  " + line.rstrip())
                else:
                    print(line.rstrip())

        #
        # Print and visualize inputs and outputs for given coinjoin transaction
        #
        print('**************************************')
        print('Address input-output mapping for cjtx: {}'.format(round_cjtx_mapping[round_id]))
        graphviz_insert_cjtxid(cjtx_stats[cjtxid], graphdot)
        print_tx_info(cjtx_stats[cjtxid], address_wallet_mapping, graphdot)
        print('**************************************')
        print('**************************************')

    analyze_coinjoin_stats(cjtx_stats, address_wallet_mapping, wallets_info)

    return cjtx_stats


def load_wallets_info():
    """
    Loads information about wallets and their addresses using Wasabi RPC
    :return: dictionary for all loaded wallets with retrieved info
    """
    WALLET_NAME_TEMPLATE = 'Wallet'
    MAX_WALLETS = 10
    wcli.WASABIWALLET_DATA_DIR = WASABIWALLET_DATA_DIR
    wcli.VERBOSE = False
    wallets_info = {}
    wallet_names = ['{}{}'.format(WALLET_NAME_TEMPLATE, index) for index in range(1, MAX_WALLETS + 1)]
    for wallet_name in wallet_names:
        if wcli.wcli(['selectwallet', wallet_name, 'pswd']) is not None:
            print('Wallet `{}` found.'.format(wallet_name))
            wcli.wcli(['getwalletinfo'])
            wallet_addresses = wcli.wcli(['listkeys'])
            if wallet_addresses is not None:
                wallets_info[wallet_name] = wallet_addresses
        else:
            print('Wallet `{}` not found.'.format(wallet_name))
    return wallets_info


if __name__ == "__main__":
    cjtx_stats = {}

    WALLET_INFO_VIA_RPC = True
    if WALLET_INFO_VIA_RPC:
        print("Loading current wallets info from WasabiWallet RPC")
        # Load wallets info via WasabiWallet RPC
        wallets_info = load_wallets_info()
        # Save wallets info into json
        print("Saving current wallets into wallets_info.json")
        with open("wallets_info.json", "w") as file:
            file.write(json.dumps(dict(sorted(wallets_info.items())), indent=4))
    else:
        print("Loading current wallets info from wallets_info.json")
        print("WARNING: wallets info may be outdated, if wallets were active after information retrieval via RPC "
              "(watch for {} wallet name string in results).".format(UNKNOWN_WALLET_STRING))
        with open("wallets_info.json", "r") as file:
            wallets_info = json.load(file)

    print("Wallets info loaded.")
    cjtx_stats['wallets_info'] = wallets_info

    # Prepare Graph
    dot2 = Digraph(comment='CoinJoin={}'.format("XX"))
    graph_label = ''
    graph_label += 'Coinjoin visualization\n.'
    dot2.attr(rankdir='LR', size='8,5')
    dot2.attr(size='120,80')

    color_ctr = 0
    for wallet_name in wallets_info.keys():
        WALLET_COLORS[wallet_name] = COLORS[color_ctr]
        graphviz_insert_wallet(wallet_name, dot2)
        color_ctr = color_ctr + 1
    # add special color for 'unknown' wallet
    WALLET_COLORS[UNKNOWN_WALLET_STRING] = 'red'
    graphviz_insert_wallet(UNKNOWN_WALLET_STRING, dot2)

    # Parse and visualize conjoin
    cjtx_stats['coinjoins'] = parse_coinjoin_logs(wallets_info, dot2)
    #Save coinjoin transactions info into json
    with open("coinjoin_tx_info.json", "w") as file:
        file.write(json.dumps(dict(sorted(cjtx_stats.items())), indent=4))

    # render resulting graphviz
    dot2.render('coinjoin_{}'.format(""), view=True)



# problems

# NOT_ENOUGH_PARTICIPANTS 2023-09-02 09:50:18.482 [50] INFO	Arena.StepInputRegistrationPhaseAsync (159)	Round (ad2a5479a5e335436bbad21c3ccfce91ec155475c7014e86592f886b3edd0ed4): Not enough inputs (0) in InputRegistration phase. The minimum is (5). MaxSuggestedAmount was '43000.00000000' BTC.
# WRONG_PHASE 2023-09-02 10:09:06.983 [59] WARNING	IdempotencyRequestCache.GetCachedResponseAsync (79)	WalletWasabi.WabiSabi.Backend.Models.WrongPhaseException: Round (1244dd436283015f2b6ac8d5b258421a6092d651c9316f1a2f9579257bef932e): Wrong phase (ConnectionConfirmation).
# ??? 2023-09-02 10:17:45.038 [48] INFO	LateResponseLoggerFilter.OnException (18)	Request 'ConfirmConnection' missing the phase 'InputRegistration,ConnectionConfirmation' ('00:00:00' timeout) by '738764.08:16:45.0188191'. Round id '85bcc20df3cecd986072e5041e0260c635b1d404dc942da0affb127c28159904'.
# NOT_ENOUGH_FUNDS 2023-09-02 10:18:48.814 [47] WARNING	IdempotencyRequestCache.GetCachedResponseAsync (79)	WalletWasabi.WabiSabi.Backend.Models.WabiSabiProtocolException: Not enough funds
# SIGNING_PHASE_TIMOUT 2023-09-02 10:31:13.421 [41] WARNING	Arena.StepTransactionSigningPhaseAsync (341)	Round (bfc40253b8e3d918d901fdd0326a7ade327e6139b3dbf19c263889c5bb51f2aa): Signing phase failed with timed out after 60 seconds.
# ALICE_REMOVED 2023-09-02 10:31:13.433 [41] INFO	Arena.FailTransactionSigningPhaseAsync (393)	Round (bfc40253b8e3d918d901fdd0326a7ade327e6139b3dbf19c263889c5bb51f2aa): Removed 1 alices, because they didn't sign. Remainig: 6
# INPUT BANNED 2023-09-02 21:01:59.661 [44] WARNING	IdempotencyRequestCache.GetCachedResponseAsync (79)	WalletWasabi.WabiSabi.Backend.Models.WabiSabiProtocolException: Input banned
# ALL_ALICES_REMOVED 2023-09-02 21:05:38.366 [38] INFO	Arena.FailTransactionSigningPhaseAsync (393)	Round (05e9b2acc8200109b5a2ceb4290193be27d2c1576ce41c9f83f3ac82a25bf66d): Removed 10 alices, because they didn't sign. Remainig: 0
# FILLED_SOME_ADDITIONAL_INPUTS 2023-09-02 21:57:33.400 [66] WARNING	Arena.TryAddBlameScriptAsync (584)	Round (91a14faef01faaad7aa05ab20e06ee29b11ffcd9a25c5db290c5c63ecdc93a90): Filled up the outputs to build a reasonable transaction because some alice failed to provide its output. Added amount: '12.39780793'.

# RequestTimeStatista.Display (60)	Response times for the last 60 minutes: