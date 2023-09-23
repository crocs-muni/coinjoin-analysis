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
import os.path
from enum import Enum

BTC_CLI_PATH = 'C:\\bitcoin-25.0\\bin\\bitcoin-cli'
WASABIWALLET_DATA_DIR = 'c:\\Users\\xsvenda\\AppData\\Roaming'
TX_AD_CUT_LEN = 10  # length of displayed address or txid
WALLET_COLORS = {}
UNKNOWN_WALLET_STRING = 'UNKNOWN'
COORDINATOR_WALLET_STRING = 'Coordinator'
PRINT_COLLATED_COORD_CLIENT_LOGS = False
INSERT_WALLET_NODES = False
ASSUME_COORDINATOR_WALLET = True
VERBOSE = False



class CJ_LOG_TYPES(Enum):
    ROUND_STARTED = 'ROUND_STARTED'
    BLAME_ROUND_STARTED = 'BLAME_ROUND_STARTED'
    COINJOIN_BROADCASTED = 'COINJOIN_BROADCASTED'
    INPUT_BANNED = 'INPUT_BANNED'
    NOT_ENOUGH_FUNDS = 'NOT_ENOUGH_FUNDS'
    NOT_ENOUGH_PARTICIPANTS = 'NOT_ENOUGH_PARTICIPANTS'
    WRONG_PHASE = 'WRONG_PHASE'
    MISSING_PHASE_BY_TIME = 'MISSING_PHASE_BY_TIME'
    SIGNING_PHASE_TIMEOUT = 'SIGNING_PHASE_TIMEOUT'
    ALICE_REMOVED = 'ALICE_REMOVED'
    FILLED_SOME_ADDITIONAL_INPUTS = 'FILLED_SOME_ADDITIONAL_INPUTS'


# colors used for different wallet clusters. Avoid following colors : 'red' (used for cjtx)
COLORS = ['darkorange', 'green', 'lightblue', 'gray', 'aquamarine', 'darkorchid1', 'cornsilk3', 'chocolate',
          'deeppink1', 'cadetblue', 'darkgreen', 'black']
LINE_STYLES = ['-', '--', '-.', ':']


def prepare_display_address(addr):
    if TX_AD_CUT_LEN > 0:
        addr = addr[:TX_AD_CUT_LEN] + '...' + '({})'.format(str(int(len(addr) / 2)))
    return addr


def prepare_display_cjtxid(cjtxid):
    if TX_AD_CUT_LEN > 0:
        cjtxid = cjtxid[:TX_AD_CUT_LEN] + '...' + '({})'.format(str(int(len(cjtxid) / 2)))
    return cjtxid


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
        # tx_record['raw_tx_json'] = parsed_data
        tx_record['inputs'] = {}
        tx_record['outputs'] = {}

        inputs = parsed_data['vin']
        index = 0
        for input in inputs:
            in_address, in_full_info = get_input_address(input['txid'], input[
                'vout'])  # we need to read and parse previous transaction to obtain address
            tx_record['inputs'][index] = {}
            # tx_record['inputs'][index]['full_info'] = in_full_info
            tx_record['inputs'][index]['address'] = in_address
            tx_record['inputs'][index]['txid'] = input['txid']
            tx_record['inputs'][index]['value'] = input['value']
            input_addresses[index] = in_address  # store address to index of the input
            index = index + 1

        outputs = parsed_data['vout']
        for output in outputs:
            output_addresses[output['n']] = output['scriptPubKey']['address']

            tx_record['outputs'][output['n']] = {}
            # tx_record['outputs'][output['n']]['full_info'] = output
            tx_record['outputs'][output['n']]['address'] = output['scriptPubKey']['address']
            tx_record['outputs'][output['n']]['value'] = output['value']

    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None

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
    addr = prepare_display_address(addr)

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
    cjtxid = prepare_display_cjtxid(cjtxid)

    graphdot.attr('node', shape='box')
    graphdot.attr('node', fillcolor='red')
    graphdot.attr('node', color='black')
    graphdot.attr('node', style='filled')
    graphdot.attr('node', fontsize='20')
    graphdot.attr('node', id=cjtxid)
    if 'is_blame_round' in coinjoin_tx.keys() and coinjoin_tx['is_blame_round']:
        graphdot.attr('node', label='cjtxid:\n{}\n{}\n{}\nBLAME ROUND'.format(cjtxid, coinjoin_tx['round_start_time'],
                                                                              coinjoin_tx['broadcast_time']))
    else:
        graphdot.attr('node', label='cjtxid:\n{}\n{}\n{}'.format(cjtxid, coinjoin_tx['round_start_time'],
                                                                 coinjoin_tx['broadcast_time']))
    graphdot.node(cjtxid)


def graphviz_insert_wallet_address_mapping(wallet_name, addr, graphdot):
    if INSERT_WALLET_NODES:
        addr = prepare_display_address(addr)
        graphdot.edge(wallet_name, addr, color=WALLET_COLORS[wallet_name], style='dotted', dir='none')


def prepare_node_attribs(coinjoin_txid, addr, value_size):
    addr = prepare_display_address(addr)
    coinjoin_txid = prepare_display_cjtxid(coinjoin_txid)
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
    return coinjoin_txid, addr, width


def graphviz_insert_address_cjtx_mapping(addr, coinjoin_txid, value_size, edge_color, graphdot):
    coinjoin_txid, addr, width = prepare_node_attribs(coinjoin_txid, addr, value_size)
    graphdot.edge(addr, coinjoin_txid, color=edge_color, label="{}₿".format(value_size) if value_size > 0 else '',
                  style='dashed')


def graphviz_insert_cjtx_address_mapping(coinjoin_txid, addr, value_size, edge_color, graphdot):
    coinjoin_txid, addr, width = prepare_node_attribs(coinjoin_txid, addr, value_size)
    graphdot.edge(coinjoin_txid, addr, color=edge_color, style='solid', label="{}₿".format(value_size), penwidth=width)


def print_tx_info(cjtx, address_wallet_mapping, graphdot):
    """
    Prints mapping between addresses in given coinjoin transaction.
    :param cjtx: coinjoin transaction
    :param address_wallet_mapping: mapping between address and controlling wallet
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

    used_wallets = sorted(used_wallets.keys())
    # print all inputs mapped to their outputs
    for wallet_name in used_wallets:
        print('Wallet `{}`'.format(wallet_name))
        for index in cjtx['inputs'].keys():
            addr = cjtx['inputs'][index]['address']
            if address_wallet_mapping[addr] == wallet_name:
                print('  ({}):{}'.format(wallet_name, addr))

        for index in cjtx['outputs'].keys():
            addr = cjtx['outputs'][index]['address']
            if address_wallet_mapping[addr] == wallet_name:
                print('  -> ({}):{}'.format(wallet_name, addr))


def graphviz_tx_info(cjtx, address_wallet_mapping, graphdot):
    """
    Insert nodes and edges of provided transaction into graphviz engine
    :param cjtx: coinjoin transaction
    :param address_wallet_mapping: mapping between address and controlling wallet
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
    # Insert tx inputs and outputs into graphviz engine
    for wallet_name in used_wallets:
        for index in cjtx['inputs'].keys():
            addr = cjtx['inputs'][index]['address']
            if address_wallet_mapping[addr] == wallet_name:
                graphviz_insert_address(addr, WALLET_COLORS[wallet_name], graphdot)
                graphviz_insert_wallet_address_mapping(wallet_name, addr, graphdot)  # wallet to address
                value = cjtx['inputs'][index]['value'] if 'value' in cjtx['inputs'][index] else 0
                graphviz_insert_address_cjtx_mapping(addr, cjtxid, value, WALLET_COLORS[wallet_name],
                                                     graphdot)  # address to coinjoin txid

        for index in cjtx['outputs'].keys():
            addr = cjtx['outputs'][index]['address']
            if address_wallet_mapping[addr] == wallet_name:
                graphviz_insert_address(addr, WALLET_COLORS[wallet_name], graphdot)
                graphviz_insert_wallet_address_mapping(wallet_name, addr, graphdot)  # wallet to address
                graphviz_insert_cjtx_address_mapping(cjtxid, addr, cjtx['outputs'][index]['value'],
                                                     WALLET_COLORS[wallet_name], graphdot)  # coinjoin to addr


def random_line_style():
    return random.choice(LINE_STYLES)


def insert_percentages_annotations(bar_data, fig):
    total = sum(bar_data)
    percentages = [(value / total) * 100 for value in bar_data]
    for i, percentage in enumerate(percentages):
        fig.text(i, bar_data[i], f'{percentage:.0f}%', ha='center')


def analyze_coinjoin_stats(cjtx_stats, base_path):
    """
    Analyze coinjoin transactions statistics
    :param cjtx_stats:
    :param address_wallet_mapping:
    :return:
    """
    print("Starting analyze_coinjoin_stats() analysis")

    coinjoins = cjtx_stats['coinjoins']
    address_wallet_mapping = cjtx_stats['address_wallet_mapping']
    wallets_info = cjtx_stats['wallets_info']
    rounds = cjtx_stats['rounds']

    # Create a larger figure
    fig = plt.figure(figsize=(12, 8))

    # Create four subplots with their own axes
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    #
    # Number of coinjoins per given time interval (e.g., hour)
    #
    SLOT_WIDTH_SECONDS = 3600
    broadcast_times = [datetime.strptime(coinjoins[item]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f") for item in
                       coinjoins.keys()]
    broadcast_times.extend(
        [datetime.strptime(rounds[item]['round_start_timestamp'], "%Y-%m-%d %H:%M:%S.%f") for item in rounds.keys() if
         'round_start_timestamp' in rounds[item]])
    experiment_start_time = min(broadcast_times)
    slot_start_time = experiment_start_time
    slot_last_time = max(broadcast_times)
    diff_seconds = (slot_last_time - slot_start_time).total_seconds()
    num_slots = int(diff_seconds // SLOT_WIDTH_SECONDS)
    cjtx_in_hours = {hour: [] for hour in range(0, num_slots + 1)}
    cjtx_blame_in_hours = {hour: [] for hour in range(0, num_slots + 1)}
    rounds_started_in_hours = {hour: [] for hour in range(0, num_slots + 1)}
    for cjtx in coinjoins.keys():  # go over all coinjoin transactions
        timestamp = datetime.strptime(coinjoins[cjtx]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")
        cjtx_hour = int((timestamp - slot_start_time).total_seconds() // SLOT_WIDTH_SECONDS)
        cjtx_in_hours[cjtx_hour].append(cjtx)
        if coinjoins[cjtx].get('is_blame_round', False):
            cjtx_blame_in_hours[cjtx_hour].append(cjtx)
    for round_id in rounds.keys():  # go over all rounds (not only the successful ones)
        if 'round_start_timestamp' in rounds[round_id]:
            timestamp = datetime.strptime(rounds[round_id]['round_start_timestamp'], "%Y-%m-%d %H:%M:%S.%f")
            round_hour = int((timestamp - slot_start_time).total_seconds() // SLOT_WIDTH_SECONDS)
            rounds_started_in_hours[round_hour].append(round_id)
        else:
            print('Missing start timestamp for {}'.format(round_id))
    if cjtx_in_hours[len(cjtx_in_hours.keys()) - 1] == []:  # remove last slot if no coinjoins are available there
        # while cjtx_in_hours[len(cjtx_in_hours.keys()) - 1] == [] and \
        #     cjtx_blame_in_hours[len(cjtx_blame_in_hours.keys()) - 1] == []:
        #     #rounds_started_in_hours[len(rounds_started_in_hours.keys()) - 1] == []:  # remove all dates from back with no coinjoin
        del cjtx_in_hours[len(cjtx_in_hours.keys()) - 1]
        del cjtx_blame_in_hours[len(cjtx_blame_in_hours.keys()) - 1]
        del rounds_started_in_hours[len(rounds_started_in_hours.keys()) - 1]
    ax1.plot([len(rounds_started_in_hours[hour]) for hour in rounds_started_in_hours.keys()], label='Rounds started',
             color='blue')
    ax1.plot([len(cjtx_in_hours[cjtx_hour]) for cjtx_hour in cjtx_in_hours.keys()], label='All coinjoins finished',
             color='green')
    ax1.plot([len(cjtx_blame_in_hours[cjtx_hour]) for cjtx_hour in cjtx_blame_in_hours.keys()],
             label='Blame coinjoins finished', color='orange')
    ax1.legend()
    x_ticks = []
    for slot in cjtx_in_hours.keys():
        x_ticks.append(
            (experiment_start_time + slot * timedelta(seconds=SLOT_WIDTH_SECONDS)).strftime("%Y-%m-%d %H:%M:%S"))
    ax1.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    ax1.set_ylim(0)
    ax1.set_ylabel('Number of coinjoin transactions')
    ax1.set_title('Number of coinjoin transactions in given time period')

    #
    # Number of coinjoins per given time interval (e.g., hour)
    #
    SLOT_WIDTH_SECONDS = 3600
    broadcast_times = []
    for round_id in rounds.keys():
        if 'logs' in rounds[round_id]:
            for entry in rounds[round_id]['logs']:
                broadcast_times.append(datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S.%f"))
    experiment_start_time = min(broadcast_times)
    slot_start_time = experiment_start_time
    slot_last_time = max(broadcast_times)
    diff_seconds = (slot_last_time - slot_start_time).total_seconds()
    num_slots = int(diff_seconds // SLOT_WIDTH_SECONDS)
    logs_in_hours = {}
    for type in CJ_LOG_TYPES:
        logs_in_hours[type.name] = {hour: [] for hour in range(0, num_slots + 1)}

    for round_id in rounds.keys():  # go over all logs, insert into vector based on the log type
        if 'logs' in rounds[round_id]:
            for entry in rounds[round_id]['logs']:
                timestamp = datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
                log_hour = int((timestamp - slot_start_time).total_seconds() // SLOT_WIDTH_SECONDS)
                logs_in_hours[entry['type']][log_hour].append(log_hour)
    index = 0
    for log_type in logs_in_hours.keys():
        num_logs_of_type = sum(logs_in_hours[log_type][log_hour])
        if num_logs_of_type > 0:
            linestyle = LINE_STYLES[index % len(LINE_STYLES)]
            if log_type not in (CJ_LOG_TYPES.ROUND_STARTED.name, CJ_LOG_TYPES.BLAME_ROUND_STARTED.name,
                                CJ_LOG_TYPES.COINJOIN_BROADCASTED.name):
                ax2.plot([len(logs_in_hours[log_type][log_hour]) for log_hour in logs_in_hours[log_type].keys()],
                         label='{} ({})'.format(log_type, num_logs_of_type), linestyle=linestyle)
        index = index + 1
    ax2.legend(fontsize=6)
    x_ticks = []
    for slot in cjtx_in_hours.keys():
        x_ticks.append(
            (experiment_start_time + slot * timedelta(seconds=SLOT_WIDTH_SECONDS)).strftime("%Y-%m-%d %H:%M:%S"))
    ax2.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    ax2.set_ylim(0)
    ax2.set_ylabel('Number of logs')
    ax2.set_title('Number of logs in a given time period')

    #
    # Number of distinct wallets in coinjoins (including coordinator)
    #
    cjtx_wallet_frequencies = {}
    for cjtx in coinjoins.keys():
        cjtx_wallet_frequencies[cjtx] = {}
        used_wallets = []
        for input_index in coinjoins[cjtx]['inputs'].keys():  # iterate over input addresses
            addr = coinjoins[cjtx]['inputs'][input_index]['address']
            if addr in address_wallet_mapping.keys():
                used_wallets.append(address_wallet_mapping[addr])
            else:
                print('Missing wallet mapping for {}'.format(addr))

        for wallet in sorted(wallets_info.keys()):
            cjtx_wallet_frequencies[cjtx][wallet] = used_wallets.count(wallet)

    # Distribution of number of inputs from different wallets
    for cjtx in cjtx_wallet_frequencies.keys():
        coinjoins[cjtx]['num_wallets_involved'] = sum(
            1 for value in cjtx_wallet_frequencies[cjtx].values() if value != 0)

    wallets_in_stats = [coinjoins[value]['num_wallets_involved'] for value in coinjoins.keys()]
    wallets_in_frequency = Counter(wallets_in_stats)
    wallets_in_frequency_all = {}
    for wallet_num in range(0, max(wallets_in_stats) + 1):
        if wallet_num in wallets_in_frequency.keys():
            wallets_in_frequency_all[wallet_num] = wallets_in_frequency[wallet_num]
        else:
            wallets_in_frequency_all[wallet_num] = 0
    wallets_in_frequency_all_ordered = [wallets_in_frequency_all[key] for key in
                                        sorted(wallets_in_frequency_all.keys())]
    ax3.bar(range(0, len(wallets_in_frequency_all_ordered)), wallets_in_frequency_all_ordered)
    insert_percentages_annotations(wallets_in_frequency_all_ordered, ax3)  # Annotate the bars with percentages

    ax3.set_xticks(range(0, len(wallets_in_frequency_all_ordered)))
    ax3.set_xticklabels(range(0, len(wallets_in_frequency_all_ordered)))
    ax3.set_xlabel('Number of distinct wallets')
    ax3.set_ylabel('Number of coinjoins')
    ax3.set_title('Number of coinjoins with specific number of distinct wallets')

    #
    # How many times given wallet participated in coinjoin?
    #
    wallets_used = []
    for wallet_name in wallets_info.keys():
        wallet_times_used = 0
        for cjtx in coinjoins.keys():  # go over all coinjoin transactions
            wallet_times_used_in_cjtx = 0
            for index in coinjoins[cjtx]['inputs']:
                if 'wallet_name' in coinjoins[cjtx]['inputs'][index]:
                    if coinjoins[cjtx]['inputs'][index]['wallet_name'] == wallet_name:
                        wallet_times_used_in_cjtx = wallet_times_used_in_cjtx + 1
                else:
                    print('Missing wallet name for cjtx {}, input: {}'.format(cjtx, index))
            wallet_times_used = wallet_times_used + wallet_times_used_in_cjtx
        wallets_used.append(wallet_times_used)

    ax4.bar(wallets_info.keys(), wallets_used)
    insert_percentages_annotations(wallets_used, ax4)
    ax4.set_xlabel('Wallet name')
    x_ticks = list(wallets_info.keys())
    ax4.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
    ax4.set_ylabel('Number of participations')
    ax4.set_title('Number of inputs given wallet provided to coinjoin txs')

    experiment_name = os.path.basename(base_path)
    plt.suptitle('{}'.format(experiment_name), fontsize=16)  # Adjust the fontsize and y position as needed
    plt.subplots_adjust(bottom=0.1, wspace=0.5, hspace=0.5)
    save_file = os.path.join(base_path, "coinjoin_stats.png")
    plt.savefig(save_file, dpi=300)
    plt.close()
    print('Basic coinjoins statistics saved into {}'.format(save_file))


def build_address_wallet_mapping(cjtx_stats):
    address_wallet_mapping = {}
    for cjtxid in cjtx_stats['coinjoins'].keys():
        # Build mapping of addresses to wallets names ('unknown' if not mapped)
        address_wallet_mapping.update(
            {cjtx_stats['coinjoins'][cjtxid]['inputs'][addr_index]['address']: UNKNOWN_WALLET_STRING for addr_index in
             cjtx_stats['coinjoins'][cjtxid]['inputs'].keys()})
        address_wallet_mapping.update(
            {cjtx_stats['coinjoins'][cjtxid]['outputs'][addr_index]['address']: UNKNOWN_WALLET_STRING for addr_index in
             cjtx_stats['coinjoins'][cjtxid]['outputs'].keys()})

        for index in cjtx_stats['coinjoins'][cjtxid]['inputs'].keys():
            addr = cjtx_stats['coinjoins'][cjtxid]['inputs'][index]['address']
            for wallet_name in cjtx_stats['wallets_info'].keys():
                for waddr in cjtx_stats['wallets_info'][wallet_name]:
                    if addr == waddr['address']:
                        address_wallet_mapping[addr] = wallet_name
                        cjtx_stats['coinjoins'][cjtxid]['inputs'][index]['wallet_name'] = wallet_name

        for index in cjtx_stats['coinjoins'][cjtxid]['outputs'].keys():
            addr = cjtx_stats['coinjoins'][cjtxid]['outputs'][index]['address']
            for wallet_name in cjtx_stats['wallets_info'].keys():
                for waddr in cjtx_stats['wallets_info'][wallet_name]:
                    if addr == waddr['address']:
                        address_wallet_mapping[addr] = wallet_name
                        cjtx_stats['coinjoins'][cjtxid]['outputs'][index]['wallet_name'] = wallet_name

    return address_wallet_mapping


def parse_coinjoin_logs(base_directory):
    coord_input_file = '{}\\WalletWasabi\\Backend\\Logs.txt'.format(base_directory)

    print('Parsing coinjoin-relevant data from coordinator logs {}...'.format(coord_input_file), end='')
    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Created round with params: MaxSuggestedAmount:'([0-9\.]+)' BTC?"
    start_round_ids = find_round_ids(coord_input_file, regex_pattern, ['round_id', 'timestamp'])
    # 2023-09-05 08:56:50.892 [38] INFO	Arena.CreateBlameRoundAsync (417)	Blame Round (c05a3b73cebffc79956e1e3abf3d9020b3e02e05f01eb7fb0d01dbcd26d64be7): Blame round created from round '05a9dfe6244d2f4004d2927798ecd42d557bbaefe61de58f67a0265f5710a2da'.
    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Blame Round \((?P<round_id>.*)\): Blame round created from round '(?P<orig_round_id>.*)'?"
    start_blame_rounds_id = find_round_ids(coord_input_file, regex_pattern, ['round_id', 'timestamp'])
    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Successfully broadcast the coinjoin: (?P<cj_tx_id>[0-9a-f]*)\.?"
    success_coinjoin_round_ids = find_round_ids(coord_input_file, regex_pattern, ['round_id', 'timestamp', 'cj_tx_id'])
    # round_cjtx_mapping = find_round_cjtx_mapping(coord_input_file, regex_pattern, 'round_id', 'cj_tx_id')
    round_cjtx_mapping = {round_id: success_coinjoin_round_ids[round_id]['cj_tx_id'] for round_id in
                          success_coinjoin_round_ids.keys()}
    print('done')

    # find all ids which have complete log from round creation (Created round with params)
    # to cj tx broadcast (Successfully broadcast the coinjoin)
    full_round_ids = [key_value for key_value in success_coinjoin_round_ids.keys() if
                      key_value in start_round_ids.keys()]
    missing_start_round_ids = [key_value for key_value in success_coinjoin_round_ids.keys() if
                               key_value not in start_round_ids.keys()]
    full_blame_round_ids = [key_value for key_value in success_coinjoin_round_ids.keys() if
                            key_value in start_blame_rounds_id.keys()]
    # Merge standard and blame rounds
    full_round_ids = full_round_ids + full_blame_round_ids
    start_round_ids.update(start_blame_rounds_id)

    print('Total fully finished coinjoins found: {}'.format(len(full_round_ids)))
    print('Total blame coinjoins found: {}'.format(len(full_blame_round_ids)))
    print('Parsing separate coinjoin transactions ', end='')
    cjtx_stats = {}
    for round_id in full_round_ids:
        # extract input and output addresses
        tx_record = extract_tx_info(round_cjtx_mapping[round_id])
        if tx_record is not None:
            # Find coinjoin transaction id and create record if not already
            cjtxid = round_cjtx_mapping[round_id]
            if cjtxid not in cjtx_stats.keys():
                cjtx_stats[cjtxid] = {}

            tx_record['round_id'] = round_id
            tx_record['round_start_time'] = start_round_ids[round_id]['timestamp']
            tx_record['broadcast_time'] = success_coinjoin_round_ids[round_id]['timestamp']
            tx_record['is_blame_round'] = True if round_id in start_blame_rounds_id.keys() else False
            cjtx_stats[cjtxid] = tx_record
        else:
            print('ERROR: decoding transaction for tx={} (round id={})'.format(round_cjtx_mapping[round_id], round_id))
        print('.', end='')

    # print only logs with full rounds
    # [print_round_logs(coord_input_file, id) for id in full_round_ids]
    print('\n\nTotal complete rounds found: {}'.format(len(full_round_ids)))

    # 2023-08-22 11:06:35.181 [21] DEBUG	CoinJoinClient.CreateRegisterAndConfirmCoinsAsync (469)	Round (5f3425c1f2e0cc81c9a74a213abf1ea3f128247d6be78ecd259158a5e1f9b66c): Inputs(4) registration started - it will end in: 00:01:22.
    # regex_pattern = r"(.*) \[.+(?P<method>CoinJoinClient\..*) \([0-9]+\).*Round \((?P<round_id>.*)\): Inputs\((?P<num_inputs>[0-9]+)\) registration started - it will end in: ([0-9:]+)\."
    # client_start_round_ids = find_round_ids(coord_input_file, regex_pattern, 'round_id')
    # 2023-08-22 11:06:51.466 [10] INFO	AliceClient.RegisterInputAsync (105)	Round (5f3425c1f2e0cc81c9a74a213abf1ea3f128247d6be78ecd259158a5e1f9b66c), Alice (94687969-bf26-1dfd-af98-2365e708b893): Registered 80b9c8615226e03d2474d8ad481c2db7505cb2715b10d83ee9c95106aaa3dcfd-0.
    # regex_pattern = r"(.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Successfully broadcast the coinjoin: (?P<cj_tx_id>[0-9a-f]*)\.?"

    print('Total fully finished coinjoins processed: {}'.format(len(cjtx_stats.keys())))

    return cjtx_stats


def insert_type(items, type_info):
    for round_id, value in items.items():
        value.update({'type': type_info.name})


def insert_by_round_id(rounds_logs, events):
    for round_id, value in events.items():
        if round_id not in rounds_logs:
            rounds_logs[round_id] = {}
        if 'logs' not in rounds_logs[round_id]:
            rounds_logs[round_id]['logs'] = []
        rounds_logs[round_id]['logs'].append(value)


def parse_coinjoin_errors(cjtx_stats, base_directory):
    coord_input_file = '{}\\WalletWasabi\\Backend\\Logs.txt'.format(base_directory)

    print('Parsing coinjoin-relevant error data from coordinator logs {}...'.format(coord_input_file), end='')

    rounds_logs = {}
    #
    # Round-dependent information
    #
    # Round id to coinjoin txid mapping
    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Successfully broadcast the coinjoin: (?P<cj_tx_id>[0-9a-f]*)\.?"
    success_coinjoin_round_ids = find_round_ids(coord_input_file, regex_pattern, ['round_id', 'timestamp', 'cj_tx_id'])
    insert_type(success_coinjoin_round_ids, CJ_LOG_TYPES.COINJOIN_BROADCASTED)
    insert_by_round_id(rounds_logs, success_coinjoin_round_ids)
    # If round resulted in successful coinjoin tx, add explicit entry
    for round_id in rounds_logs.keys():
        if round_id in success_coinjoin_round_ids.keys():
            rounds_logs[round_id] = {}
            rounds_logs[round_id]['logs'] = []
            rounds_logs[round_id]['cj_tx_id'] = success_coinjoin_round_ids[round_id]['cj_tx_id']

    # Start of a round
    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Created round with params: MaxSuggestedAmount:'([0-9\.]+)' BTC?"
    start_round_ids = find_round_ids(coord_input_file, regex_pattern, ['round_id', 'timestamp'])
    insert_type(start_round_ids, CJ_LOG_TYPES.ROUND_STARTED)
    insert_by_round_id(rounds_logs, start_round_ids)
    for round_id in rounds_logs.keys():
        if round_id in start_round_ids.keys():
            rounds_logs[round_id]['round_start_timestamp'] = start_round_ids[round_id]['timestamp']

    # Start of a blame round
    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Blame Round \((?P<round_id>.*)\): Blame round created from round '(?P<orig_round_id>.*)'?"
    start_blame_rounds_id = find_round_ids(coord_input_file, regex_pattern, ['round_id', 'timestamp'])
    insert_type(start_blame_rounds_id, CJ_LOG_TYPES.BLAME_ROUND_STARTED)
    insert_by_round_id(rounds_logs, start_blame_rounds_id)
    for round_id in rounds_logs.keys():
        if round_id in start_blame_rounds_id.keys():
            rounds_logs[round_id]['round_start_timestamp'] = start_blame_rounds_id[round_id]['timestamp']

    # MISSING_PHASE_BY_TIME 2023-09-02 10:17:45.038 [48] INFO	LateResponseLoggerFilter.OnException (18)	Request 'ConfirmConnection' missing the phase 'InputRegistration,ConnectionConfirmation' ('00:00:00' timeout) by '738764.08:16:45.0188191'. Round id '85bcc20df3cecd986072e5041e0260c635b1d404dc942da0affb127c28159904'.
    regex_pattern = r"(?P<timestamp>.*) \[.+LateResponseLoggerFilter.OnException.*Request '(?P<request_name>.*)' missing the phase '(?P<phase_missed>.*)' \('(?P<timeout_value>.*)' timeout\) by '(?P<timeout_missed>.*)'. Round id '(?P<round_id>.*)'.?"
    missing_phase_by_time = find_round_ids(coord_input_file, regex_pattern,
                                           ['round_id', 'timestamp', 'request_name', 'phase_missed', 'timeout_value',
                                            'timeout_missed'])
    insert_type(missing_phase_by_time, CJ_LOG_TYPES.MISSING_PHASE_BY_TIME)
    insert_by_round_id(rounds_logs, missing_phase_by_time)

    # FILLED_SOME_ADDITIONAL_INPUTS 2023-09-02 21:57:33.400 [66] WARNING	Arena.TryAddBlameScriptAsync (584)	Round (91a14faef01faaad7aa05ab20e06ee29b11ffcd9a25c5db290c5c63ecdc93a90): Filled up the outputs to build a reasonable transaction because some alice failed to provide its output. Added amount: '12.39780793'.
    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\.TryAddBlameScriptAsync.*) \(.*Round \((?P<round_id>.*)\): Filled up the outputs to build a reasonable transaction because some alice failed to provide its output. Added amount: '(?P<amount_added>[0-9\.]+).?'.?"
    filled_additional_inputs = find_round_ids(coord_input_file, regex_pattern,
                                              ['round_id', 'timestamp', 'amount_added'])
    insert_type(filled_additional_inputs, CJ_LOG_TYPES.FILLED_SOME_ADDITIONAL_INPUTS)
    insert_by_round_id(rounds_logs, filled_additional_inputs)

    # ALICE_REMOVED 2023-09-02 10:31:13.433 [41] INFO	Arena.FailTransactionSigningPhaseAsync (393)	Round (bfc40253b8e3d918d901fdd0326a7ade327e6139b3dbf19c263889c5bb51f2aa): Removed 1 alices, because they didn't sign. Remainig: 6
    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Removed (?P<num_alices_removed>[0-9]+) alices, because they didn't sign. Remaini[n]*g: (?P<num_alices_remaining>[0-9]+).?"
    alices_removed = find_round_ids(coord_input_file, regex_pattern,
                                    ['round_id', 'timestamp', 'num_alices_removed', 'num_alices_remaining'])
    insert_type(alices_removed, CJ_LOG_TYPES.ALICE_REMOVED)
    insert_by_round_id(rounds_logs, alices_removed)
    all_alices_removed = {key: value for key, value in alices_removed.items() if
                          alices_removed[key]['num_alices_remaining'] == '0'}

    # SIGNING_PHASE_TIMOUT 2023-09-02 10:31:13.421 [41] WARNING	Arena.StepTransactionSigningPhaseAsync (341)	Round (bfc40253b8e3d918d901fdd0326a7ade327e6139b3dbf19c263889c5bb51f2aa): Signing phase failed with timed out after 60 seconds.
    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Signing phase failed with timed out after (?P<timeout_length>[0-9]+) seconds.?"
    signing_phase_timeout = find_round_ids(coord_input_file, regex_pattern, ['round_id', 'timestamp', 'timeout_length'])
    insert_type(signing_phase_timeout, CJ_LOG_TYPES.SIGNING_PHASE_TIMEOUT)
    insert_by_round_id(rounds_logs, signing_phase_timeout)

    # WRONG_PHASE 2023-09-02 10:09:06.983 [59] WARNING	IdempotencyRequestCache.GetCachedResponseAsync (79)	WalletWasabi.WabiSabi.Backend.Models.WrongPhaseException: Round (1244dd436283015f2b6ac8d5b258421a6092d651c9316f1a2f9579257bef932e): Wrong phase (ConnectionConfirmation).
    regex_pattern = r"(?P<timestamp>.*) .* WARNING.*WalletWasabi\.WabiSabi\.Backend\.Models\.WrongPhaseException: Round \((?P<round_id>.*)\): Wrong phase \((?P<phase_info>[a-zA-Z0-9]+)\).?"
    wrong_phase = find_round_ids(coord_input_file, regex_pattern,
                                 ['round_id', 'timestamp', 'num_participants', 'min_participants_required'])
    insert_type(wrong_phase, CJ_LOG_TYPES.WRONG_PHASE)
    insert_by_round_id(rounds_logs, wrong_phase)

    # NOT_ENOUGH_PARTICIPANTS 2023-09-02 09:50:18.482 [50] INFO	Arena.StepInputRegistrationPhaseAsync (159)	Round (ad2a5479a5e335436bbad21c3ccfce91ec155475c7014e86592f886b3edd0ed4): Not enough inputs (0) in InputRegistration phase. The minimum is (5). MaxSuggestedAmount was '43000.00000000' BTC.
    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Not enough inputs \((?P<num_participants>[0-9]+)\) in InputRegistration phase\. The minimum is \((?P<min_participants_required>[0-9]+)\)\. MaxSuggestedAmount was '([0-9\.]+)' BTC?"
    not_enough_participants = find_round_ids(coord_input_file, regex_pattern,
                                             ['round_id', 'timestamp', 'num_participants', 'min_participants_required'])
    insert_type(not_enough_participants, CJ_LOG_TYPES.NOT_ENOUGH_PARTICIPANTS)
    insert_by_round_id(rounds_logs, not_enough_participants)

    #
    # Round-independent information
    #
    if 'no_round' not in rounds_logs.keys():
        rounds_logs['no_round'] = []

    # INPUT BANNED 2023-09-02 21:01:59.661 [44] WARNING	IdempotencyRequestCache.GetCachedResponseAsync (79)	WalletWasabi.WabiSabi.Backend.Models.WabiSabiProtocolException: Input banned
    regex_pattern = r"(?P<timestamp>.*) .* WARNING.*WabiSabiProtocolException: Input banned.?"
    input_banned = find_round_ids(coord_input_file, regex_pattern, ['timestamp'])
    insert_type(input_banned, CJ_LOG_TYPES.INPUT_BANNED)
    rounds_logs['no_round'].append(input_banned)

    # NOT_ENOUGH_FUNDS 2023-09-02 10:18:48.814 [47] WARNING	IdempotencyRequestCache.GetCachedResponseAsync (79)	WalletWasabi.WabiSabi.Backend.Models.WabiSabiProtocolException: Not enough funds
    regex_pattern = r"(?P<timestamp>.*) .* WARNING.*WabiSabiProtocolException: Not enough funds.?"
    not_enough_funds = find_round_ids(coord_input_file, regex_pattern, ['timestamp'])
    insert_type(not_enough_funds, CJ_LOG_TYPES.NOT_ENOUGH_FUNDS)
    rounds_logs['no_round'].append(not_enough_funds)

    if 'rounds' not in cjtx_stats.keys():
        cjtx_stats['rounds'] = {}
    cjtx_stats['rounds'].update(rounds_logs)

    print()
    return cjtx_stats


def visualize_cjtx_graph(cjtx_stats, cjtxid, address_wallet_mapping, graphdot):
    """
    Print and visualize inputs and outputs for given coinjoin transaction
        :param cjtx_stats:
    :param cjtx_stats: list of all transactions
    :param cjtxid: transaction id of coinjoin transaction
    :param address_wallet_mapping: mapping between addresses and their wallets
    :param graphdot: graphviz engine
    :return:
    """
    if VERBOSE:
        print('**************************************')
        print('Address input-output mapping {}'.format(cjtxid))
        print_tx_info(cjtx_stats[cjtxid], address_wallet_mapping, graphdot)
        print('**************************************')

    # Insert into graphviz engine
    graphviz_insert_cjtxid(cjtx_stats[cjtxid], graphdot)
    graphviz_tx_info(cjtx_stats[cjtxid], address_wallet_mapping, graphdot)


def load_wallets_info():
    """
    Loads information about wallets and their addresses using Wasabi RPC
    :return: dictionary for all loaded wallets with retrieved info
    """
    WALLET_NAME_TEMPLATE = 'Wallet'
    WALLET_NAME_TEMPLATE = 'SimplePassiveWallet'
    MAX_WALLETS = 16
    wcli.WASABIWALLET_DATA_DIR = os.path.join(WASABIWALLET_DATA_DIR, "WalletWasabi")
    wcli.VERBOSE = False
    wallets_info = {}
    wallet_names = ['{}{}'.format(WALLET_NAME_TEMPLATE, index) for index in range(1, MAX_WALLETS + 1)]
    wallet_names.append('DistributorWallet')
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


def visualize_coinjoins(cjtx_stats, base_path):
    address_wallet_mapping = cjtx_stats['address_wallet_mapping']

    # Prepare Graph
    dot2 = Digraph(comment='CoinJoin={}'.format("XX"))
    graph_label = ''
    graph_label += 'Coinjoin visualization\n.'
    dot2.attr(rankdir='LR', size='8,5')
    dot2.attr(size='120,80')

    color_ctr = 0

    for wallet_name in cjtx_stats['wallets_info'].keys():
        WALLET_COLORS[wallet_name] = COLORS[color_ctr]
        graphviz_insert_wallet(wallet_name, dot2)
        color_ctr = color_ctr + 1
    # add special color for 'unknown' wallet
    WALLET_COLORS[UNKNOWN_WALLET_STRING] = 'red'
    WALLET_COLORS[COORDINATOR_WALLET_STRING] = 'grey30'
    graphviz_insert_wallet(UNKNOWN_WALLET_STRING, dot2)
    graphviz_insert_wallet(COORDINATOR_WALLET_STRING, dot2)

    for cjtxid in cjtx_stats['coinjoins'].keys():
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

        # Visualize into large connected graph
        visualize_cjtx_graph(cjtx_stats['coinjoins'], cjtxid, address_wallet_mapping, dot2)

    # render resulting graphviz
    print(
        'Going to render coinjoin relations graph (may take several minutes if larger number of coinjoins are visualized) ... ',
        end='')
    save_file = os.path.join(base_path, "coinjoin_graph")
    dot2.render(save_file, view=True)
    print('done')


def obtain_wallets_info(base_path, load_wallet_info_via_rpc):
    wallets_file = os.path.join(base_path, "wallets_info.json")
    if load_wallet_info_via_rpc:
        print("Loading current wallets info from WasabiWallet RPC")
        # Load wallets info via WasabiWallet RPC
        wallets_info = load_wallets_info()
        # Save wallets info into json
        print("Saving current wallets into {}".format(wallets_file))
        with open(wallets_file, "w") as file:
            file.write(json.dumps(dict(sorted(wallets_info.items())), indent=4))
    else:
        print("Loading current wallets info from {}".format(wallets_file))
        print("WARNING: wallets info may be outdated, if wallets were active after information retrieval via RPC "
              "(watch for {} wallet name string in results).".format(UNKNOWN_WALLET_STRING))
        with open(wallets_file, "r") as file:
            wallets_info = json.load(file)

    print("Wallets info loaded.")
    return wallets_info


def fix_coordinator_wallet_addresses(cjtx_stats):
    """
    Check all wallets with not yet assigned wallet name and assume that all which are 32B long are coming from coordinator.
    :param cjtx_stats:
    :return:
    """
    if COORDINATOR_WALLET_STRING not in cjtx_stats['wallets_info']:
        cjtx_stats['wallets_info'][COORDINATOR_WALLET_STRING] = []
    for cjtxid in cjtx_stats['coinjoins'].keys():
        for index in cjtx_stats['coinjoins'][cjtxid]['inputs'].keys():
            target_addr = cjtx_stats['coinjoins'][cjtxid]['inputs'][index]['address']
            if 'wallet_name' not in cjtx_stats['coinjoins'][cjtxid]['inputs'][index] and len(target_addr) == 64:
                cjtx_stats['coinjoins'][cjtxid]['inputs'][index]['wallet_name'] = COORDINATOR_WALLET_STRING
                cjtx_stats['wallets_info'][COORDINATOR_WALLET_STRING].append({'address': target_addr})
        for index in cjtx_stats['coinjoins'][cjtxid]['outputs'].keys():
            target_addr = cjtx_stats['coinjoins'][cjtxid]['outputs'][index]['address']
            if 'wallet_name' not in cjtx_stats['coinjoins'][cjtxid]['outputs'][index] and len(target_addr) == 64:
                cjtx_stats['coinjoins'][cjtxid]['outputs'][index]['wallet_name'] = COORDINATOR_WALLET_STRING
                cjtx_stats['wallets_info'][COORDINATOR_WALLET_STRING].append({'address': target_addr})

    # Rebuild full wallet mapping
    cjtx_stats['address_wallet_mapping'] = build_address_wallet_mapping(cjtx_stats)
    return cjtx_stats


def process_experiment(base_path):
    # WASABIWALLET_DATA_DIR = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol2\\20230903_5minRound_3parallel_max20inputs_fromFresh30btx\\'
    # WASABIWALLET_DATA_DIR = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol2\\20230903_5minRound_1parallel_max10inputs_fromFresh30btx\\'
    # WASABIWALLET_DATA_DIR = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol2\\20230903_5minRound_3parallel_max20inputs_fromFresh30btx'
    # WASABIWALLET_DATA_DIR = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol2\\20230902_5minRound_3parallel_max10inputs_fromFresh30btx_samererun'
    # WASABIWALLET_DATA_DIR = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol2\\20230902_5minRound_3parallel_max10inputs_fromFresh30btx'
    # WASABIWALLET_DATA_DIR = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol2\\20230905_3minRound_3parallel_max10inputs_fromFresh30btx'

    WASABIWALLET_DATA_DIR = base_path

    save_file = os.path.join(WASABIWALLET_DATA_DIR, "coinjoin_tx_info.json")
    if LOAD_TXINFO_FROM_FILE:
        # Load parsed coinjoin transactions again
        with open(save_file, "r") as file:
            cjtx_stats = json.load(file)
    else:
        # Load wallets info
        cjtx_stats = {}
        cjtx_stats['wallets_info'] = obtain_wallets_info(WASABIWALLET_DATA_DIR, LOAD_WALLETS_INFO_VIA_RPC)

        # Parse conjoins from logs
        cjtx_stats['coinjoins'] = parse_coinjoin_logs(WASABIWALLET_DATA_DIR)

        # Build mapping between address and controlling wallet
        cjtx_stats['address_wallet_mapping'] = build_address_wallet_mapping(cjtx_stats)

        # Assume coordinator for all 32B addresses
        if ASSUME_COORDINATOR_WALLET:
            cjtx_stats = fix_coordinator_wallet_addresses(cjtx_stats)

        # Analyze error states
        if PARSE_ERRORS:
            parse_coinjoin_errors(cjtx_stats, WASABIWALLET_DATA_DIR)

        # Save parsed coinjoin transactions info into json
        with open(save_file, "w") as file:
            file.write(json.dumps(dict(sorted(cjtx_stats.items())), indent=4))

    # Assume coordinator for all 32B addresses
    if ASSUME_COORDINATOR_WALLET:
        cjtx_stats = fix_coordinator_wallet_addresses(cjtx_stats)
        print('Potential coordinator addresses recomputed, saving...', end='')
        with open(save_file, "w") as file:
            file.write(json.dumps(dict(sorted(cjtx_stats.items())), indent=4))

    # Analyze various coinjoins statistics
    analyze_coinjoin_stats(cjtx_stats, WASABIWALLET_DATA_DIR)

    # Visualize coinjoins
    to_visualize = dict(list(cjtx_stats['coinjoins'].items())[:10])
    cjtx_stats['coinjoins'] = to_visualize
    if GENERATE_COINJOIN_GRAPH:
        visualize_coinjoins(cjtx_stats, WASABIWALLET_DATA_DIR)

    print('All done.')


def process_multiple_experiments(base_path):
    files = []
    if os.path.exists(base_path):
        files = os.listdir(base_path)
    else:
        print('Path {} does not exists'.format(base_path))

    for file in files:
        target_base_path = os.path.join(base_path, file)
        if os.path.isdir(target_base_path):
            test_path = os.path.join(target_base_path, 'WalletWasabi')
            if os.path.exists(test_path):
                print('****************************')
                print('Analyzing experiment {}'.format(target_base_path))
                process_experiment(target_base_path)


if __name__ == "__main__":
    FRESH_COINJOIN = False
    if FRESH_COINJOIN:
        # Extract all info from running wallet and fullnode
        LOAD_TXINFO_FROM_FILE = False  # If True, existence of coinjoin_tx_info.json is assumed and all data are read from it. Bitcoin Core/Wallet RPC is not required
        LOAD_WALLETS_INFO_VIA_RPC = True  # If False, existance of wallets_info.json with wallet information is assumed and loaded from
        GENERATE_COINJOIN_GRAPH = False
        PARSE_ERRORS = True
    else:
        # Just recompute analysis
        LOAD_TXINFO_FROM_FILE = True
        LOAD_WALLETS_INFO_VIA_RPC = False
        GENERATE_COINJOIN_GRAPH = False
        PARSE_ERRORS = False

    GENERATE_COINJOIN_GRAPH = True

    target_base_path = 'c:\\Users\\xsvenda\\AppData\\Roaming\\'
    target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol3\\20230922_2000Round_3parallel_max10inputs_10wallets_1btc'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol3\\20230920_1000Round_3parallel_max10inputs_10wallets_1btc'
    process_experiment(target_base_path)

    # BASE_PATH = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\debug\\'
    # process_multiple_experiments(BASE_PATH)

# Count number of unique / same inputs
