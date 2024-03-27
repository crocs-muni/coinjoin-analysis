import copy
import csv
import math
import re
import subprocess
import json
import sys
import wcli
from itertools import zip_longest
from graphviz import Digraph
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime, timedelta
import os.path
from enum import Enum
sys.path.append('boltzmann/boltzmann/')
import ludwig
from cProfile import Profile
from pstats import SortKey, Stats
from decimal import Decimal
import jsonpickle
from multiprocessing.pool import ThreadPool, Pool
from tqdm import tqdm
import anonymity_score
import bitcoinlib.transactions
from bitcoinlib.transactions import Transaction

BTC_CLI_PATH = 'C:\\bitcoin-25.0\\bin\\bitcoin-cli'
WASABIWALLET_DATA_DIR = 'c:\\Users\\xsvenda\\AppData\\Roaming'
TX_AD_CUT_LEN = 16  # length of displayed address or txid
WALLET_COLORS = {}
UNKNOWN_WALLET_STRING = 'UNKNOWN'
COORDINATOR_WALLET_STRING = 'Coordinator'
PRINT_COLLATED_COORD_CLIENT_LOGS = False
INSERT_WALLET_NODES = False
ASSUME_COORDINATOR_WALLET = False
VERBOSE = False
NUM_THREADS = 1
SATS_IN_BTC = 100000000
PRE_2_0_4_VERSION = False
MAX_NUM_DISPLAY_UTXO = 1000
GLOBAL_IMG_SUFFIX = '.3'

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
    UTXO_IN_PRISON = 'UTXO_IN_PRISON'


class CJ_ALICE_TYPES(Enum):
    ALICE_REGISTERED = 'ALICE_REGISTERED'
    ALICE_CONNECTION_CONFIRMED = 'ALICE_CONNECTION_CONFIRMED'
    ALICE_READY_TO_SIGN = 'ALICE_READY_TO_SIGN'
    ALICE_POSTED_SIGNATURE = 'ALICE_POSTED_SIGNATURE'


# colors used for different wallet clusters. Avoid following colors : 'red' (used for cjtx)
COLORS = ['darkorange', 'green', 'lightblue', 'gray', 'aquamarine', 'darkorchid1', 'cornsilk3', 'chocolate',
          'deeppink1', 'cadetblue', 'darkgreen', 'burlywood4', 'cyan', 'darkgray', 'darkslateblue', 'dodgerblue4',
          'greenyellow', 'indigo', 'lightslateblue', 'plum3', 'tan1', 'black']
LINE_STYLES = ['-', '--', '-.', ':']


def float_equals(a, b, tolerance=1e-9):
    return abs(a - b) < tolerance


# Define a custom JSON encoder that handles Decimal objects
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)


class DecimalDecoder(json.JSONDecoder):
    def decode(self, s):
        obj = super(DecimalDecoder, self).decode(s)
        return self._decode_decimal(obj)

    def _decode_decimal(self, obj):
        if isinstance(obj, str):
            try:
                return Decimal(obj)
            except ValueError:
                pass
        elif isinstance(obj, list):
            return [self._decode_decimal(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._decode_decimal(value) for key, value in obj.items()}
        return obj


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
                    if key_name not in hits.keys():
                        hits[key_name] = []
                    hits[key_name].append(hit_group)

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


def get_input_address(txid, txid_in_out, raw_txs: dict = {}):
    """
    Returns address which was used in transaction given by 'txid' as 'txid_in_out' output index
    :param txid: transaction id to read input address from
    :param txid_in_out: index in vout to read input address from
    :param raw_txs: pre-computed database of transactions
    :return:
    """

    if len(raw_txs) > 0 and txid in raw_txs.keys():
        tx_info = raw_txs[txid]
    else:
        result = run_command(
            '{} -regtest getrawtransaction {} true'.format(BTC_CLI_PATH, txid), False)
        tx_info = json.loads(result.stdout)

    try:
        parsed_data = tx_info
        outputs = parsed_data['vout']
        for output in outputs:
            if output['n'] == txid_in_out:
                return output['scriptPubKey']['address'], parsed_data

    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)

    return None, None


def extract_tx_info(txid: str, raw_txs: dict):
    """
    Extract input and output addresses
    :param txid: transaction to parse
    :param raw_txs: dictionary with pre-loaded transactions
    :return: parsed transaction record
    """

    USE_PRELOADED = True if len(raw_txs) > 0 and txid in raw_txs.keys() else False

    if USE_PRELOADED:
        # Use pre-loaded transactions if available
        tx_info = raw_txs[txid]
    else:
        # retrieve from fullnode via RPC
        result = run_command(
            '{} -regtest getrawtransaction {} true'.format(BTC_CLI_PATH, txid), False)
        if result.returncode != 0:
            print(f'Cannot retrieve tx info for {txid} with {result.stderr} error')
            return None
        tx_info = json.loads(result.stdout)

    tx_record = {}
    input_addresses = {}
    input_txids = {}
    output_addresses = {}
    try:
        parsed_data = tx_info
        tx_record = {}

        tx_record['txid'] = txid
        # tx_record['raw_tx_json'] = parsed_data
        tx_record['inputs'] = {}
        tx_record['outputs'] = {}

        inputs = parsed_data['vin']
        index = 0
        for input in inputs:
            # we need to read and parse previous transaction to obtain address and other information
            in_address, in_full_info = get_input_address(input['txid'], input['vout'], raw_txs)

            tx_record['inputs'][index] = {}
            # tx_record['inputs'][index]['full_info'] = in_full_info
            tx_record['inputs'][index]['address'] = in_address
            tx_record['inputs'][index]['txid'] = input['txid']
            tx_record['inputs'][index]['value'] = in_full_info['vout'][input['vout']]['value']
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
        graphdot.attr('node', penwidth='1')
        graphdot.attr('node', style='filled')
        graphdot.attr('node', fontsize='20')
        graphdot.attr('node', id=wallet_name)
        graphdot.attr('node', label='{}'.format(wallet_name))
        graphdot.node() if GENERATE_COINJOIN_GRAPH_BLIND else graphdot.node(wallet_name)


def graphviz_insert_address(addr, fill_color, graphdot):
    addr = prepare_display_address(addr)

    graphdot.attr('node', shape='ellipse')
    graphdot.attr('node', fillcolor=fill_color)
    graphdot.attr('node', color='gray')
    graphdot.attr('node', penwidth='1')
    graphdot.attr('node', style='filled')
    graphdot.attr('node', fontsize='20')
    graphdot.attr('node', id=addr)
    if not GENERATE_COINJOIN_GRAPH_BLIND:
        graphdot.attr('node', label='{}'.format(addr))
    graphdot.node(addr)


def graphviz_insert_cjtxid(coinjoin_tx, graphdot):
    cjtxid = coinjoin_tx['txid']
    cjtxid = prepare_display_cjtxid(cjtxid)

    graphdot.attr('node', shape='box')
    graphdot.attr('node', fillcolor='white')
    graphdot.attr('node', color='green')
    graphdot.attr('node', penwidth='3')
    graphdot.attr('node', style='filled')
    graphdot.attr('node', fontsize='20')
    graphdot.attr('node', id=cjtxid)
    tx_entropy = coinjoin_tx['analysis']['processed']['tx_entropy'] if 'analysis' in coinjoin_tx.keys() else -1
    tx_entropy_str = "{:.1f}".format(tx_entropy) if tx_entropy >= 0 else '?'
    num_dlinks = coinjoin_tx['analysis']['processed']['num_deterministic_links'] if 'analysis' in coinjoin_tx.keys() else '?'
    if tx_entropy == 0:  # Highlight coinjoins with no entropy addition
        graphdot.attr('node', fillcolor='red')
        graphdot.attr('node', color='black')
        graphdot.attr('node', penwidth='10')
    else:
        graphdot.attr('node', fillcolor='white')
        graphdot.attr('node', color='green')
        graphdot.attr('node', penwidth=str(tx_entropy * 5))

    if 'is_blame_round' in coinjoin_tx.keys() and coinjoin_tx['is_blame_round']:
        graphdot.attr('node', label='coinjoin tx:\n{}\n{}\n{}\ntx_entropy={}\nnum_dlinks={}\nBLAME ROUND'.format(cjtxid, coinjoin_tx['round_start_time'],
                                                                              coinjoin_tx['broadcast_time'], tx_entropy_str, num_dlinks))
    else:
        graphdot.attr('node', label='coinjoin tx:\n{}\n{}\n{}\ntx_entropy={}\nnum_dlinks={}'.format(cjtxid, coinjoin_tx['round_start_time'],
                                                                 coinjoin_tx['broadcast_time'], tx_entropy_str, num_dlinks))
    if GENERATE_COINJOIN_GRAPH_BLIND:
        graphdot.attr('node', label='')
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


def graphviz_insert_address_cjtx_mapping(addr, coinjoin_txid, value_size, edge_color, vin_index, graphdot):
    coinjoin_txid, addr, width = prepare_node_attribs(coinjoin_txid, addr, value_size)
    label = "{}{:.8f}₿".format('[{}] '.format(vin_index), value_size if value_size > 0 else '')
    if GENERATE_COINJOIN_GRAPH_BLIND:
        label = ''
    graphdot.edge(addr, coinjoin_txid, color=edge_color, label=label, style='solid', penwidth=width) \


def graphviz_insert_cjtx_address_mapping(coinjoin_txid, addr, value_size, entropy, edge_color, vout_index, graphdot):
    coinjoin_txid, addr, width = prepare_node_attribs(coinjoin_txid, addr, value_size)
    label = "{}{:.8f}₿".format('[{}] '.format(vout_index), value_size if value_size > 0 else '')
    label += " e={}".format(round(entropy, 1)) if entropy > 0 else ''
    if GENERATE_COINJOIN_GRAPH_BLIND:
        label = ''
    graphdot.edge(coinjoin_txid, addr, color=edge_color, style='solid', label=label, penwidth=width)


def graphviz_insert_address_address_mapping(add1, addr2, edge_color, graphdot):
    add1 = prepare_display_address(add1)
    addr2 = prepare_display_address(addr2)
    graphdot.edge(add1, addr2, color=edge_color, style='dotted', penwidth='10')


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
    for index in cjtx['inputs'].keys():
        addr = cjtx['inputs'][index]['address']
        value = cjtx['inputs'][index]['value'] if 'value' in cjtx['inputs'][index] else 0
        wallet_name = address_wallet_mapping[addr]
        graphviz_insert_address(addr, WALLET_COLORS[wallet_name], graphdot)
        graphviz_insert_wallet_address_mapping(wallet_name, addr, graphdot)  # wallet to address
        graphviz_insert_address_cjtx_mapping(addr, cjtxid, value, WALLET_COLORS[wallet_name],
                                             index, graphdot)  # address to coinjoin txid

    for index in cjtx['outputs'].keys():
        addr = cjtx['outputs'][index]['address']
        value = cjtx['outputs'][index]['value'] if 'value' in cjtx['outputs'][index] else 0
        wallet_name = address_wallet_mapping[addr]
        graphviz_insert_address(addr, WALLET_COLORS[wallet_name], graphdot)
        graphviz_insert_wallet_address_mapping(wallet_name, addr, graphdot)  # wallet to address
        entropy = cjtx['outputs'][index]['anon_score'] if 'anon_score' in cjtx['outputs'][index] else 0
        graphviz_insert_cjtx_address_mapping(cjtxid, addr, value, entropy,
                                             WALLET_COLORS[wallet_name], index, graphdot)  # coinjoin to addr
    # insert detected deterministic links
    if 'analysis' in cjtx.keys():
        for link in cjtx['analysis']['processed']['deterministic_links']:
            addr1 = link[0][0]
            addr2 = link[1][0]
            edge_color = WALLET_COLORS[address_wallet_mapping[addr1]]
            edge_color = 'black'
            if address_wallet_mapping[addr1] != address_wallet_mapping[addr2]:
                print('ERROR: {} Deterministic link mismatch {} to {}'.format(cjtxid, address_wallet_mapping[addr1], address_wallet_mapping[addr2]))

            if len(cjtx['analysis']['processed']['deterministic_links']) < 100:  # do not draw cases with too many deterministic links
                graphviz_insert_address_address_mapping(addr1, addr2, edge_color, graphdot)


def random_line_style():
    return random.choice(LINE_STYLES)


def insert_percentages_annotations(bar_data, fig):
    total = sum(bar_data)
    percentages = [(value / total) * 100 for value in bar_data]
    for i, percentage in enumerate(percentages):
        fig.text(i, bar_data[i], f'{percentage:.1f}%', ha='center')


# Function to calculate sliding window average
def sliding_window_average(data, window_size):
    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


def calculate_entropy(values):
    # Count the occurrences of each unique value
    value_counts = Counter(values)
    # Calculate the total number of values
    total_count = len(values)

    # Calculate the entropy
    entropy = 0.0
    for count in value_counts.values():
        probability = count / total_count
        entropy -= probability * math.log2(probability)

    # Initialize the minimum entropy to a large value
    min_entropy = float('inf')
    # Calculate the entropy for each unique value
    for value in set(values):
        probability = value_counts[value] / total_count
        entropy = -probability * math.log2(probability)
        # Update the minimum entropy if needed
        if entropy < min_entropy:
            min_entropy = entropy

    return entropy, min_entropy


class CoinJoinPlots:
    plot = None  # Base MathplotLib
    fig = None

    axes = []  # List of all created axes (accessible via index)

    # Named axes for specific analysis (is set to specific one from axes[])
    ax_coinjoins = None
    ax_logs = None
    ax_num_inoutputs = None
    ax_available_utxos = None
    ax_utxo_denom_anonscore = None
    ax_wallets_distrib = None

    ax_utxo_provided = None

    ax_anonscore_distrib_wallets = None
    ax_anonscore_from_outputs = None

    ax_utxo_entropy_from_outputs = None
    ax_utxo_entropy_from_outputs_inoutsize = None
    ax_initial_final_utxos = None

    # ax_mining_fees = None

    ax_boltzmann_entropy_inoutsize = None
    ax_boltzmann_entropy = None
    ax_boltzmann_entropy_roundtime = None
    ax_boltzmann_txcombinations = None
    ax_boltzmann_entropy_roundtime_wallets = None

    def __init__(self, plot):
        self.plot = plot
        self.clear()

    def new_figure(self, subplots_rows=4, subplots_columns=5, set_default_layout=True):
        # Create four subplots with their own axes
        SCALE_FACTOR = 6
        self.fig = self.plot.figure(figsize=(subplots_columns*SCALE_FACTOR*2, subplots_rows*SCALE_FACTOR))
        for index in range(1, subplots_rows * subplots_columns + 1):
            self.axes.append(self.fig.add_subplot(subplots_rows, subplots_columns, index))

        # Set default layout suitable for analysis of single experiment
        if set_default_layout:
            self.ax_coinjoins = self.axes[0]
            self.ax_logs = self.axes[1]
            self.ax_num_inoutputs = self.axes[2]
            self.ax_available_utxos = self.axes[3]
            self.ax_wallets_distrib = self.axes[4]

            self.ax_utxo_provided = self.axes[5]

            self.ax_anonscore_distrib_wallets = self.axes[6]
            self.ax_anonscore_from_outputs = self.axes[7]

            self.ax_utxo_entropy_from_outputs = self.axes[8]
            self.ax_utxo_entropy_from_outputs_inoutsize = self.axes[9]
            self.ax_utxo_denom_anonscore = self.axes[10]
            self.ax_initial_final_utxos = self.axes[11]

            # self.ax_mining_fees = ax11

            self.ax_boltzmann_entropy_inoutsize = self.axes[12]
            self.ax_boltzmann_entropy = self.axes[13]
            self.ax_boltzmann_entropy_roundtime = self.axes[14]
            self.ax_boltzmann_txcombinations = self.axes[15]
            self.ax_boltzmann_entropy_roundtime_wallets = self.axes[16]

    def savefig(self, save_file, experiment_name: str):
        self.plot.suptitle('{}'.format(experiment_name), fontsize=16)  # Adjust the fontsize and y position as needed
        self.plot.subplots_adjust(bottom=0.1, wspace=0.1, hspace=0.5)
        self.plot.savefig(save_file, dpi=300)
        return save_file

    def clear(self):
        self.plot.clf()
        self.plot.close()
        self.fig = None

        self.axes = []  # List of all created axes (accessible via index)

        # Named axes for specific analysis (is set to specific one from axes[])
        self.ax_coinjoins = None
        self.ax_logs = None
        self.ax_utxo_denom_anonscore = None
        self.ax_num_inoutputs = None
        self.ax_available_utxos = None
        self.ax_wallets_distrib = None

        self.ax_utxo_provided = None

        self.ax_anonscore_distrib_wallets = None
        self.ax_anonscore_from_outputs = None

        self.ax_utxo_entropy_from_outputs = None
        self.ax_utxo_entropy_from_outputs_inoutsize = None
        self.ax_initial_final_utxos = None

        # self.ax_mining_fees = None

        self.ax_boltzmann_entropy_inoutsize = None
        self.ax_boltzmann_entropy = None
        self.ax_boltzmann_entropy_roundtime = None
        self.ax_boltzmann_txcombinations = None
        self.ax_boltzmann_entropy_roundtime_wallets = None


def analyze_coinjoin_stats(cjtx_stats, base_path, cjplt: CoinJoinPlots, short_exp_name: str=''):
    """
    Analyze coinjoin transactions statistics and plot it to provided CoinJoinPlots structure.
    If given subgraph attribute is None, then only analysis is performed without plotting.
    :param cjtx_stats:
    :param address_wallet_mapping:
    :param cjplt: plots for coinjoin graphs
    @param short_exp_name if provided, short experiment name is put into legend
    :return:
    """

    experiment_name = os.path.basename(base_path)
    print(f"Starting analyze_coinjoin_stats() analysis for {base_path}")
    coinjoins = cjtx_stats['coinjoins']
    address_wallet_mapping = cjtx_stats['address_wallet_mapping']
    wallets_info = cjtx_stats['wallets_info']
    wallets_coins = cjtx_stats['wallets_coins']
    rounds = cjtx_stats['rounds']
    cj_time = []
    for cjtxid in coinjoins.keys():
        cj_time.append({'txid':cjtxid, 'broadcast_time': datetime.strptime(coinjoins[cjtxid]['broadcast_time'], "%Y-%m-%d %H:%M:%S.%f")})
    sorted_cj_time = sorted(cj_time,  key=lambda x: x['broadcast_time'])

    # Add placeholder for analytics computed
    if 'analysis' not in cjtx_stats.keys():
        cjtx_stats['analysis'] = {}
    cjtx_stats['analysis']['path'] = base_path

    # Compute same output size statistics
    for cjtx in coinjoins.keys():
        if 'analysis2' not in coinjoins[cjtx]:
            coinjoins[cjtx]['analysis2'] = {}
        out_values = [coinjoins[cjtx]['outputs'][index]['value'] for index in coinjoins[cjtx]['outputs'].keys()]
        utxo_value_counts = Counter(out_values)
        coinjoins[cjtx]['analysis2']['outputs_frequency'] = utxo_value_counts
        coinjoins[cjtx]['analysis2']['outputs_different_values'] = len(utxo_value_counts)
        # If all outputs are equal, then sameness ratio is 1. If all different, then 1/num_outputs
        coinjoins[cjtx]['analysis2']['outputs_sameness_ratio'] = utxo_value_counts[max(utxo_value_counts, key=lambda k: utxo_value_counts[k])] / len(coinjoins[cjtx]['outputs'])
        coinjoins[cjtx]['analysis2']['outputs_entropy'], coinjoins[cjtx]['analysis2']['outputs_min_entropy'] = calculate_entropy(out_values)

        # Store number of other same output values for every output
        coinjoins[cjtx]['analysis2']['outputs'] = coinjoins[cjtx]['outputs']
        for index in coinjoins[cjtx]['outputs'].keys():
            coinjoins[cjtx]['analysis2']['outputs'][index]['num_others_with_same_values'] = utxo_value_counts[coinjoins[cjtx]['analysis2']['outputs'][index]['value']]

    num_boltzmann_analyzed = sum([1 for cjtxid in coinjoins.keys() if 'analysis' in coinjoins[cjtxid] and coinjoins[cjtxid]['analysis']['processed']['successfully_analyzed'] is True])

    # Compute mining fee contribution in given coinjoin by seperate wallets
    for cjtx in coinjoins.keys():
        sum_inputs = sum(coinjoins[cjtx]['inputs'][index]['value'] for index in coinjoins[cjtx]['inputs'].keys())
        sum_outputs = sum(coinjoins[cjtx]['outputs'][index]['value'] for index in coinjoins[cjtx]['outputs'].keys())
        mining_fee = sum_inputs - sum_outputs
        coinjoins[cjtx]['analysis2']['mining_fee'] = mining_fee

        for wallet_name in cjtx_stats['wallets_info'].keys():
            sum_inputs = sum(
                coinjoins[cjtx]['inputs'][index]['value'] for index in coinjoins[cjtx]['inputs'].keys() if wallet_name == coinjoins[cjtx]['inputs'][index]['wallet_name'])
            sum_outputs = sum(
                coinjoins[cjtx]['outputs'][index]['value'] for index in coinjoins[cjtx]['outputs'].keys()
                if 'wallet_name' in coinjoins[cjtx]['outputs'][index].keys() and wallet_name == coinjoins[cjtx]['outputs'][index]['wallet_name'])
            mining_and_coinjoin_fee_payed = sum_inputs - sum_outputs

            if sum_inputs > 0:  # save only if some value was found
                if 'mining_and_coinjoin_fee_payed' not in coinjoins[cjtx]['analysis2'].keys():
                    coinjoins[cjtx]['analysis2']['mining_and_coinjoin_fee_payed'] = {}
                coinjoins[cjtx]['analysis2']['mining_and_coinjoin_fee_payed'][wallet_name] = int(mining_and_coinjoin_fee_payed)

    # Detect inputs which achieved base anonscore, yet were used again in mixing
    BASE_ANONSCORE_LIMIT = 5  # TODO: Extract exact value from configuration files
    num_overmixed_utxos = 0
    total_utxos = 0
    for wallet_name in wallets_info.keys():
        for cjtx in coinjoins:
            for index in coinjoins[cjtx]['outputs'].keys():
                output = coinjoins[cjtx]['outputs'][index]
                if 'wallet_name' in output.keys() and wallet_name == output['wallet_name'] and 'anon_score' in output.keys():
                    if output['anon_score'] >= BASE_ANONSCORE_LIMIT and 'spend_by_txid' in output:
                        #print(f'Overly mixed output: {get_output_name_string(cjtx, index)}, {output['anon_score']}')
                        num_overmixed_utxos += 1
                    total_utxos += 1
    if total_utxos > 0:
        print(f'Total overmixed outputs: {num_overmixed_utxos} / {total_utxos} ({round(num_overmixed_utxos/total_utxos, 1)*100}%)')
    else:
        print('Total overmixed outputs: nothing found.')


    # Compute number of new / mixed / outgoing utxos for each coinjoin
    # unmixed_utxos_in_wallets ... # utxo not yet mixed even once (not output of any previous coinjoin)
    # mixed_utxos_in_wallets ... # utxo mixed at least once, but still mixed later on (input of some later coinjoin)
    # finish_utxos_in_wallets ... # utxos not mixed in any subsequent coinjoin

    # For each coinjoin, reconstruct set of coins available in wallets and eligible for mixing
    # Use [round_start_time, broadcast_time] for more precise interval when coin is evaluated
    for cjtx in [cjtx['txid'] for cjtx in sorted_cj_time]:
        unmixed_utxos = []
        mixed_utxos = []
        finished_utxos = []
        #cjtx_creation_time = coinjoins[cjtx]['round_start_time']
        cjtx_creation_time = coinjoins[cjtx]['broadcast_time']

        if 'unmixed_utxos_in_wallets' not in coinjoins[cjtx]['analysis2'].keys() and \
            'unmixed_utxos_in_wallets' not in coinjoins[cjtx]['analysis2'].keys() and \
            'mixed_utxos_in_wallets' not in coinjoins[cjtx]['analysis2'].keys():

            for wallet_name in wallets_coins.keys():
                for coin in wallets_coins[wallet_name]:
                    # get coin destroy time - if not used by any tx, then set to year 9999
                    coin_destroy_time = '9999-01-25 09:44:48.000' if 'destroy_time' not in coin.keys() else coin['destroy_time']
                    if coin['create_time'] < cjtx_creation_time:
                        if cjtx_creation_time <= coin_destroy_time:
                            # Coin was available for mixing at coinjoin round registration time
                            # Now identify if it was unmixed, already mixed but not enough (will be another coinjoin) or not mixed again
                            if coin['txid'] not in coinjoins.keys():  # Coin's creating tx was not coinjoin => unmixed
                                unmixed_utxos.append(coin)
                            elif 'spentBy' in coin.keys() and coin['spentBy'] in coinjoins.keys() and coin['spentBy'] != cjtx:
                                # Coin was already mixed (coming from coinjoin), but will be mixed again by future coinjoin
                                mixed_utxos.append(coin)
                            else:
                                # Coin is either unspent till end of all coinjoins or was spent by non-coinjoin tx (later)
                                # If anonscore is set (>=0), then check if it is below default mixing limit
                                if 0 <= coin['anonymityScore'] < BASE_ANONSCORE_LIMIT:
                                    mixed_utxos.append(coin)
                                else:
                                    finished_utxos.append(coin)
                        else:
                            # Coin was created, but also destroyed before this coinjoin time, do nothing
                            a = 5
                    else:
                        # Coin was created only later in time after this coinjoin, do nothing
                        a = 5

            coinjoins[cjtx]['analysis2']['unmixed_utxos_in_wallets'] = unmixed_utxos
            coinjoins[cjtx]['analysis2']['mixed_utxos_in_wallets'] = mixed_utxos
            coinjoins[cjtx]['analysis2']['finished_utxos_in_wallets'] = finished_utxos

    print('Starting graph plotting')
    #
    # Number of coinjoins per given time interval (e.g., hour or 10 minutes)
    #
    SLOT_WIDTH_SECONDS = 600
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

    if cjplt.ax_coinjoins:
        cjplt.ax_coinjoins.plot([len(rounds_started_in_hours[hour]) for hour in rounds_started_in_hours.keys()], label=f'Rounds started {short_exp_name}',
                 color='blue')
        cjplt.ax_coinjoins.plot([len(cjtx_in_hours[cjtx_hour]) for cjtx_hour in cjtx_in_hours.keys()], label=f'All coinjoins finished {short_exp_name}',
                 color='green')
        cjplt.ax_coinjoins.plot([len(cjtx_blame_in_hours[cjtx_hour]) for cjtx_hour in cjtx_blame_in_hours.keys()],
                 label=f'Blame coinjoins finished {short_exp_name}', color='orange')

    logs_in_hours = {}
    logs_in_hours[CJ_LOG_TYPES.UTXO_IN_PRISON.name] = {hour: [] for hour in range(0, num_slots + 1)}
    for round_id in rounds.keys():  # go over all logs, insert into vector based on the log type
        if 'logs' in rounds[round_id]:
            for entry in rounds[round_id]['logs']:
                if entry['type'] == CJ_LOG_TYPES.UTXO_IN_PRISON.name:
                    timestamp = datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
                    log_hour = int((timestamp - slot_start_time).total_seconds() // SLOT_WIDTH_SECONDS)
                    if log_hour in logs_in_hours[entry['type']].keys():
                        logs_in_hours[entry['type']][log_hour].append(log_hour)
                    else:
                        print('WARNING: missing log entry time slot {} for {}/{}, ignoring it'.format(log_hour, entry['type'], timestamp))
    if cjplt.ax_coinjoins:
        cjplt.ax_coinjoins.plot([len(logs_in_hours[CJ_LOG_TYPES.UTXO_IN_PRISON.name][log_hour]) for log_hour in logs_in_hours[CJ_LOG_TYPES.UTXO_IN_PRISON.name].keys()],
                         label=f'(UTXOs in prison) {short_exp_name}', color='lightgray', linestyle='--')

    x_ticks = []
    for slot in cjtx_in_hours.keys():
        x_ticks.append(
            (experiment_start_time + slot * timedelta(seconds=SLOT_WIDTH_SECONDS)).strftime("%Y-%m-%d %H:%M:%S"))
    if cjplt.ax_coinjoins:
        cjplt.ax_coinjoins.legend()
        cjplt.ax_coinjoins.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
        cjplt.ax_coinjoins.set_ylim(0)
        cjplt.ax_coinjoins.set_ylabel('Number of coinjoin transactions')
        cjplt.ax_coinjoins.set_title(f'Number of coinjoin transactions in given time period\n{experiment_name}')

    # SAVE_ANALYTICS
    cjtx_stats['analysis']['rounds_started_all'] = [x_ticks, [len(rounds_started_in_hours[hour]) for hour in rounds_started_in_hours.keys()]]
    cjtx_stats['analysis']['coinjoins_finished_all'] = [x_ticks, [len(cjtx_in_hours[hour]) for hour in cjtx_in_hours.keys()]]
    cjtx_stats['analysis']['coinjoins_finished_blame'] = [x_ticks, [len(cjtx_blame_in_hours[hour]) for hour in cjtx_blame_in_hours.keys()]]
    cjtx_stats['analysis']['utxos_in_prison'] = [x_ticks, [len(logs_in_hours[CJ_LOG_TYPES.UTXO_IN_PRISON.name][hour]) for hour in logs_in_hours[CJ_LOG_TYPES.UTXO_IN_PRISON.name].keys()]]


    #
    # Number of coinjoin logs per given time interval (e.g., hour)
    #
    broadcast_times = []
    for round_id in rounds.keys():
        if 'logs' in rounds[round_id]:
            for entry in rounds[round_id]['logs']:
                broadcast_times.append(datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S.%f"))
        else:
            if 'coinjoins' in rounds[round_id]:
                for entry in rounds[round_id]['logs']:
                    broadcast_times.append(datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S.%f"))

    experiment_start_time = min(broadcast_times) if len(broadcast_times) > 0 else datetime(2009, 1, 1)
    slot_start_time = experiment_start_time
    slot_last_time = max(broadcast_times) if len(broadcast_times) > 0 else datetime(2009, 1, 1)
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
                if log_hour in logs_in_hours[entry['type']].keys():
                    logs_in_hours[entry['type']][log_hour].append(log_hour)
                else:
                    print('WARNING: missing log entry time slot {} for {}/{}, ignoring it'.format(log_hour, entry['type'],
                                                                                                timestamp))

    index = 0
    for log_type in logs_in_hours.keys():
        num_logs_of_type = sum([len(logs_in_hours[log_type][log_hour]) for log_hour in logs_in_hours[log_type].keys()])
        if num_logs_of_type > 0:
            linestyle = LINE_STYLES[index % len(LINE_STYLES)]
            if log_type not in (CJ_LOG_TYPES.ROUND_STARTED.name, CJ_LOG_TYPES.BLAME_ROUND_STARTED.name,
                                CJ_LOG_TYPES.COINJOIN_BROADCASTED.name):
                if cjplt.ax_logs:
                    cjplt.ax_logs.plot([len(logs_in_hours[log_type][log_hour]) for log_hour in logs_in_hours[log_type].keys()],
                         label=f'{log_type} ({num_logs_of_type}) {short_exp_name}', linestyle=linestyle)
        index = index + 1
    x_ticks = []
    for slot in cjtx_in_hours.keys():
        x_ticks.append(
            (experiment_start_time + slot * timedelta(seconds=SLOT_WIDTH_SECONDS)).strftime("%Y-%m-%d %H:%M:%S"))
    if cjplt.ax_logs:
        cjplt.ax_logs.legend(fontsize=6)
        cjplt.ax_logs.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
        cjplt.ax_logs.set_ylim(0)
        cjplt.ax_logs.set_ylabel('Number of logs')
        cjplt.ax_logs.set_title(f'Number of logs in a given time period\n{experiment_name}')

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
    if cjplt.ax_wallets_distrib:
        cjplt.ax_wallets_distrib.bar(range(0, len(wallets_in_frequency_all_ordered)), wallets_in_frequency_all_ordered)
        insert_percentages_annotations(wallets_in_frequency_all_ordered, cjplt.ax_wallets_distrib)  # Annotate the bars with percentages

        cjplt.ax_wallets_distrib.set_xticks(range(0, len(wallets_in_frequency_all_ordered)))
        cjplt.ax_wallets_distrib.set_xticklabels(range(0, len(wallets_in_frequency_all_ordered)))
        cjplt.ax_wallets_distrib.tick_params(axis='x', labelsize=6)
        cjplt.ax_wallets_distrib.set_xlabel('Number of distinct wallets')
        cjplt.ax_wallets_distrib.set_ylabel('Number of coinjoins')
        cjplt.ax_wallets_distrib.set_xlim(0.5)
        cjplt.ax_wallets_distrib.set_title(f'Number of coinjoins with specific number of distinct wallets  (all transactions)\n{experiment_name}')

    # SAVE_ANALYTICS
    cjtx_stats['analysis']['coinjoin_number_different_wallets'] = [list(range(0, len(wallets_in_frequency_all_ordered))), wallets_in_frequency_all_ordered]


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
                        wallet_times_used_in_cjtx += 1
                else:
                    print('Missing wallet name for cjtx {}, input: {}'.format(cjtx, index))
            wallet_times_used = wallet_times_used + wallet_times_used_in_cjtx
        wallets_used.append((wallet_name, wallet_times_used))

    unused_wallets = [wallet_times_used[0] for wallet_times_used in wallets_used if wallet_times_used[1] == 0]
    wallets_used_times = [wallet_times_used[1] for wallet_times_used in wallets_used]
    print(f'Number of wallets with no inputs mixed: {len(unused_wallets)} out of {len(wallets_used)} total')
    print(f'  {unused_wallets}')

    x_ticks = list(wallets_info.keys())

    # SAVE_ANALYTICS
    cjtx_stats['analysis']['wallets_no_input_mixed'] = [[], unused_wallets]
    cjtx_stats['analysis']['wallets_times_used_as_input'] = [x_ticks, wallets_used_times]

    if cjplt.ax_utxo_provided:
        cjplt.ax_utxo_provided.bar(wallets_info.keys(), wallets_used_times)
        insert_percentages_annotations(wallets_used_times, cjplt.ax_utxo_provided)
        cjplt.ax_utxo_provided.set_xlabel('Wallet name')
        cjplt.ax_utxo_provided.set_xticks(range(0, len(x_ticks)), x_ticks, rotation=45, fontsize=6)
        cjplt.ax_utxo_provided.set_ylabel('Number of participations')
        cjplt.ax_utxo_provided.set_title(f'Number of inputs given wallet provided to coinjoin txs (all transactions)\n{experiment_name}')
        if len(x_ticks) >= 100:
            cjplt.ax_utxo_provided.tick_params(axis='x', labelsize=4)
        if len(x_ticks) >= 200:
            cjplt.ax_utxo_provided.tick_params(axis='x', labelsize=2)

    #
    # Number of inputs/outputs/wallets in coinjoins
    #
    outputs_data = [len(coinjoins[cjtx['txid']]['outputs']) for cjtx in sorted_cj_time]

    x = range(1, len(outputs_data))
    outputs_data_avg10 = sliding_window_average(outputs_data, 10)
    outputs_data_avg100 = sliding_window_average(outputs_data, 100)
    inputs_data = [len(coinjoins[cjtx['txid']]['inputs']) for cjtx in sorted_cj_time]
    if cjplt.ax_num_inoutputs:
        cjplt.ax_num_inoutputs.plot(outputs_data, label=f'#outputs {short_exp_name}', linestyle='-.', color='red', )
        if len(outputs_data_avg100) > 1:
            cjplt.ax_num_inoutputs.plot(x[100 - 2:], outputs_data_avg100, label='#outs avg(100)', linewidth=3, color='red', linestyle='-.')
        cjplt.ax_num_inoutputs.plot(inputs_data, label=f'#inputs {short_exp_name}', color='blue')

        # # Compute ratio between inputs and outputs
        # inout_ratio = [float(outputs_data[index]) / float(inputs_data[index]) for index in range(0, len(inputs_data))]
        # cjplt.ax_num_inoutputs.plot(inout_ratio, label='ratio out/in', color='magenta')

    # Number of different wallets involved time
    num_wallets_in_time_data = [coinjoins[cjtx['txid']]['num_wallets_involved'] for cjtx in sorted_cj_time]
    num_wallets_in_time_data_avg100 = sliding_window_average(num_wallets_in_time_data, 100)
    x = range(1, len(num_wallets_in_time_data))
    if cjplt.ax_num_inoutputs:
        cjplt.ax_num_inoutputs.plot(num_wallets_in_time_data, label=f'#wallets {short_exp_name}', color='orange', linestyle='--')
        if len(num_wallets_in_time_data_avg100) > 1:
            cjplt.ax_num_inoutputs.plot(x[100-2:], num_wallets_in_time_data_avg100, label=f'#wallets avg(100) {short_exp_name}', linewidth=3, color='orange', linestyle='--')

        cjplt.ax_num_inoutputs.set_xlabel('Coinjoin in time')
        cjplt.ax_num_inoutputs.set_ylabel('Number of inputs/outputs/wallets')
        #cjplt.ax_num_inoutputs.set_yscale('log')
        cjplt.ax_num_inoutputs.legend()
        cjplt.ax_num_inoutputs.set_title(f'Number of inputs/outputs/wallets of cj transactions (all transactions)\n{experiment_name}')

    # SAVE_ANALYTICS
    cjtx_stats['analysis']['coinjoin_number_inputs_in_time'] = [list(range(0, len(inputs_data))), inputs_data]
    cjtx_stats['analysis']['coinjoin_number_outputs_in_time'] = [list(range(0, len(inputs_data))), outputs_data]
    cjtx_stats['analysis']['coinjoin_number_wallets_involved_in_time'] = [list(range(0, len(inputs_data))), num_wallets_in_time_data]

    #
    #  Number of txos of unmined/mixed/finished during start of given coinjoin
    #
    num_unmixed_utxos_per_cj = [len(coinjoins[cjtx['txid']]['analysis2']['unmixed_utxos_in_wallets']) for cjtx in sorted_cj_time]
    num_mixed_data_per_cj = [len(coinjoins[cjtx['txid']]['analysis2']['mixed_utxos_in_wallets']) for cjtx in sorted_cj_time]
    num_finished_data_per_cj = [len(coinjoins[cjtx['txid']]['analysis2']['finished_utxos_in_wallets']) for cjtx in sorted_cj_time]
    if cjplt.ax_available_utxos:
        bar_width = 0.1
        categories = range(0, len(outputs_data))
        cjplt.ax_available_utxos.bar(categories, num_unmixed_utxos_per_cj, bar_width,
                                     label=f'unmixed {short_exp_name}', alpha=0.3, color='blue')
        cjplt.ax_available_utxos.bar(categories, num_mixed_data_per_cj, bar_width, bottom=num_unmixed_utxos_per_cj,
                                   label=f'mixed {short_exp_name}', color='orange', alpha=0.3)
        cjplt.ax_available_utxos.bar(categories, num_finished_data_per_cj, bar_width,
                                   bottom=np.array(num_unmixed_utxos_per_cj) + np.array(num_mixed_data_per_cj),
                                   label=f'finished {short_exp_name}', color='green', alpha=0.3)
        cjplt.ax_available_utxos.plot(categories, num_unmixed_utxos_per_cj, label=f'unmixed {short_exp_name}',
                                      linewidth=3, color='blue', linestyle='--', alpha=0.3)
        cjplt.ax_available_utxos.plot(categories, num_mixed_data_per_cj, label=f'mixed {short_exp_name}',
                                     linewidth=3, color='orange', linestyle='-.', alpha=0.3)
        cjplt.ax_available_utxos.plot(categories, num_finished_data_per_cj, label=f'finished {short_exp_name}',
                                     linewidth=3, color='green', linestyle=':', alpha=0.3)

        cjplt.ax_available_utxos.set_xlabel('Coinjoin in time')
        cjplt.ax_available_utxos.set_ylabel('Number of txos')
        cjplt.ax_available_utxos.legend(loc='lower left')
        cjplt.ax_available_utxos.set_title(f'Number of txos available in wallets when cjtx is starting registration (all transactions)\n{experiment_name}')

    #
    # Number of different wallets involved time
    #
    # Now redundant, moved to ax_num_inoutputs
    # num_wallets_in_time_data = [coinjoins[cjtx['txid']]['num_wallets_involved'] for cjtx in sorted_cj_time]
    # num_wallets_in_time_data_avg100 = sliding_window_average(num_wallets_in_time_data, 100)
    # x = range(1, len(num_wallets_in_time_data))
    # ax_num_participating_wallets.plot(num_wallets_in_time_data, label='Number of participating wallets')
    # ax_num_participating_wallets.plot(x[100-2:], num_wallets_in_time_data_avg100, label='Number of participating wallets avg(100)', linewidth=3, color='red')
    # ax_num_participating_wallets.set_xlabel('Coinjoin in time')
    # ax_num_participating_wallets.set_ylabel('Number of wallets')
    # ax_num_participating_wallets.legend()
    # ax_num_participating_wallets.set_title(f'Number of wallets participating in coinjoin transaction (all transactions)\n{experiment_name}')

    #
    # Wallet efficiency in time
    #
    # efficiency_data = [coinjoins[cjtx['txid']]['analysis']['processed']['efficiency'] for cjtx in sorted_cj_time if 'analysis' in coinjoins[cjtx['txid']].keys()
    #                    and coinjoins[cjtx['txid']]['analysis']['processed']['successfully_analyzed'] is True]
    # x = range(1, len(efficiency_data))
    # efficiency_data_avg10 = sliding_window_average(efficiency_data, 10)
    # efficiency_data_avg100 = sliding_window_average(efficiency_data, 100)
    # ax5.plot(efficiency_data, label='Wallet efficiency (%)')
    # #ax5.plot(x[10-2:], efficiency_data_avg10, label='Wallet efficiency avg(10) (%)')
    # ax5.plot(x[100-2:], efficiency_data_avg100, label='Wallet efficiency avg(100) (%)', linewidth=3, color='green')
    # ax5.set_xlabel('Coinjoin in time')
    # ax5.set_ylabel('Wallet efficiency (%)')
    # ax5.legend()
    # ax5.set_title(f'Wallet efficiency with respect to perfect coinjoin (only successfully analyzed transactions!)\n{experiment_name}')

    #
    # Transaction boltzmann entropy and deterministic links
    #
    tx_entropy_data = [coinjoins[cjtx['txid']]['analysis']['processed']['tx_entropy'] for cjtx in sorted_cj_time
                       if 'analysis' in coinjoins[cjtx['txid']].keys() and coinjoins[cjtx['txid']]['analysis']['processed']['successfully_analyzed'] is True]
    x = range(1, len(tx_entropy_data))
    tx_entropy_data_avg100 = sliding_window_average(tx_entropy_data, 100)
    if cjplt.ax_boltzmann_entropy:
        cjplt.ax_boltzmann_entropy.plot(tx_entropy_data, label=f'Tx entropy (bits) {short_exp_name}')
        cjplt.ax_boltzmann_entropy.plot(x[100-2:], tx_entropy_data_avg100, label=f'Tx entropy avg(100) (bits) {short_exp_name}', linewidth=3, color='red')
        cjplt.ax_boltzmann_entropy.plot([coinjoins[cjtx['txid']]['analysis']['processed']['num_deterministic_links'] for cjtx in sorted_cj_time
                  if 'analysis' in coinjoins[cjtx['txid']].keys() and coinjoins[cjtx['txid']]['analysis']['processed']['successfully_analyzed'] is True], label=f'Number of deterministic links (#) {short_exp_name}')
        cjplt.ax_boltzmann_entropy.set_xlabel('Coinjoin in time')
        cjplt.ax_boltzmann_entropy.legend()
        cjplt.ax_boltzmann_entropy.set_title('Transaction entropy and deterministic links (Boltzmann, only successfully analyzed transactions! {}/{})\n{}'.format(num_boltzmann_analyzed, len(coinjoins.keys()), experiment_name))

    #
    # tx entropy versus coinjoin round start-broadcast time
    #
    x_cat, y_cat, x_cat_blame, y_cat_blame = [], [], [], []
    for cjtx in sorted_cj_time:
        if ('analysis' in coinjoins[cjtx['txid']].keys()
                and coinjoins[cjtx['txid']]['analysis']['processed']['successfully_analyzed'] is True):
            time_diff = (datetime.strptime(coinjoins[cjtx['txid']]['broadcast_time'],"%Y-%m-%d %H:%M:%S.%f")
                         - datetime.strptime(coinjoins[cjtx['txid']]['round_start_time'],"%Y-%m-%d %H:%M:%S.%f")).total_seconds()
            tx_entropy = coinjoins[cjtx['txid']]['analysis']['processed']['tx_entropy']
            if coinjoins[cjtx['txid']]['is_blame_round'] is False:
                x_cat.append(time_diff)
                y_cat.append(tx_entropy)
            else:
                x_cat_blame.append(time_diff)
                y_cat_blame.append(tx_entropy)

    if cjplt.ax_boltzmann_entropy_roundtime:
        cjplt.ax_boltzmann_entropy_roundtime.scatter(x_cat, y_cat, label=f'Normal round {short_exp_name}', color='green')
        cjplt.ax_boltzmann_entropy_roundtime.scatter(x_cat_blame, y_cat_blame, label=f'Blame round {short_exp_name}', color='red')
        cjplt.ax_boltzmann_entropy_roundtime.set_xlabel('Length of round (seconds)')
        cjplt.ax_boltzmann_entropy_roundtime.set_ylabel('Transaction entropy (bits)')
        cjplt.ax_boltzmann_entropy_roundtime.legend()
        cjplt.ax_boltzmann_entropy_roundtime.set_title('Dependency of tx entropy on round duration (Boltzmann, only successfully analyzed transactions! {}/{}\n{})'.format(num_boltzmann_analyzed, len(coinjoins.keys()), experiment_name))

    #
    # tx boltzmann entropy versus coinjoin round start-broadcast time colored by wallet
    #
    wallet_offset = 0
    for wallet_name in wallets_info.keys():
        x_cat, y_cat = [], []
        for cjtx in sorted_cj_time:
            if ('analysis' in coinjoins[cjtx['txid']].keys()
                    and coinjoins[cjtx['txid']]['analysis']['processed']['successfully_analyzed'] is True):
                time_diff = (datetime.strptime(coinjoins[cjtx['txid']]['broadcast_time'],"%Y-%m-%d %H:%M:%S.%f")
                             - datetime.strptime(coinjoins[cjtx['txid']]['round_start_time'],"%Y-%m-%d %H:%M:%S.%f")).total_seconds()
                tx_entropy = coinjoins[cjtx['txid']]['analysis']['processed']['tx_entropy']
                wallets_used_intx = [coinjoins[cjtx['txid']]['inputs'][index]['wallet_name'] for index in coinjoins[cjtx['txid']]['inputs'].keys()]
                if wallet_name in wallets_used_intx:
                    x_cat.append(time_diff + wallet_offset)
                    y_cat.append(tx_entropy)

        if cjplt.ax_boltzmann_entropy_roundtime_wallets:
            cjplt.ax_boltzmann_entropy_roundtime_wallets.scatter(x_cat, y_cat, label=f'{wallet_name} {short_exp_name}', alpha=0.5, s=10)

    if cjplt.ax_boltzmann_entropy_roundtime_wallets:
        cjplt.ax_boltzmann_entropy_roundtime_wallets.set_xlabel('Length of round (seconds)')
        cjplt.ax_boltzmann_entropy_roundtime_wallets.set_ylabel('Transaction entropy (bits)')
        cjplt.ax_boltzmann_entropy_roundtime_wallets.legend()
        cjplt.ax_boltzmann_entropy_roundtime_wallets.set_title('Dependency of tx entropy on round duration (Boltzmann, only successfully analyzed transactions! {}/{}\n{})'.format(num_boltzmann_analyzed, len(coinjoins.keys()), experiment_name))

    #
    # tx entropy versus input/output value size
    #
    x_cat_in, y_cat_in, x_cat_out, y_cat_out = [], [], [], []
    for cjtx in sorted_cj_time:
        if ('analysis' in coinjoins[cjtx['txid']].keys()
                and coinjoins[cjtx['txid']]['analysis']['processed']['successfully_analyzed'] is True):
            cjtx_record_processed = coinjoins[cjtx['txid']]['analysis']['processed']
            tx_entropy = cjtx_record_processed['tx_entropy']
            for input in cjtx_record_processed['inputs']:
                x_cat_in.append(input[1])
                y_cat_in.append(tx_entropy)
            for output in cjtx_record_processed['outputs']:
                x_cat_out.append(output[1])
                y_cat_out.append(tx_entropy)

    if cjplt.ax_boltzmann_entropy_inoutsize:
        cjplt.ax_boltzmann_entropy_inoutsize.scatter(x_cat_in, y_cat_in, label=f'Tx inputs {short_exp_name}', color='green', s=1)
        cjplt.ax_boltzmann_entropy_inoutsize.scatter(x_cat_out, y_cat_out, label=f'Tx outputs {short_exp_name}', color='red', s=1)
        if len(x_cat_in) > 0 or len(x_cat_out) > 0:  # logscale only if some data were inserted
            cjplt.ax_boltzmann_entropy_inoutsize.set_xscale('log')
        cjplt.ax_boltzmann_entropy_inoutsize.set_xlabel('Value (sats) (log scale)')
        cjplt.ax_boltzmann_entropy_inoutsize.set_ylabel('Transaction entropy (bits)')
        #cjplt.ax_boltzmann_entropy_inoutsize.ticklabel_format(style='plain', axis='x')
        cjplt.ax_boltzmann_entropy_inoutsize.ticklabel_format(style='plain', axis='y')
        #cjplt.ax_boltzmann_entropy_inoutsize.set_xlim(0, 250000)
        cjplt.ax_boltzmann_entropy_inoutsize.legend()
        cjplt.ax_boltzmann_entropy_inoutsize.set_title('Dependency of tx entropy on size of inputs / outputs (Boltzmann, only successfully analyzed transactions! {}/{}\n{})'.format(num_boltzmann_analyzed, len(coinjoins.keys()), experiment_name))

    #
    # anonscore versus coijoin in time (how is anonscore of utxos evolving with more coinjoins in time executed?)
    #
    for wallet_name in wallets_info.keys():
        x_cat_out, y_cat_out = [], []
        cj_index = 0
        for cjtx in sorted_cj_time:
            for index in coinjoins[cjtx['txid']]['outputs'].keys():
                output = coinjoins[cjtx['txid']]['outputs'][index]
                if 'wallet_name' in output.keys() and wallet_name == output['wallet_name'] and 'anon_score' in output.keys():
                    x_cat_out.append(cj_index)
                    y_cat_out.append(coinjoins[cjtx['txid']]['outputs'][index]['anon_score'])

            cj_index = cj_index + 1

        if len(x_cat_out) > 0:
            if cjplt.ax_anonscore_from_outputs:
                cjplt.ax_anonscore_from_outputs.scatter(x_cat_out, y_cat_out, label=f'{wallet_name} {short_exp_name}', s=1)

    if cjplt.ax_anonscore_from_outputs:
        cjplt.ax_anonscore_from_outputs.set_xlabel('Coinjoin in time')
        cjplt.ax_anonscore_from_outputs.set_ylabel('Wasabi anonscore')
        cjplt.ax_anonscore_from_outputs.ticklabel_format(style='plain', axis='y')
        if len(wallets_info.keys()) <= 15:
            cjplt.ax_anonscore_from_outputs.legend(ncol=5, fontsize='small')
        cjplt.ax_anonscore_from_outputs.set_title(f'Wasabi Anonscore of outputs in time (all transactions)\n{experiment_name}')

    # SAVE_ANALYTICS
    # Nothing

    #
    # anonscore distribution for wallets
    #
    hist_wallets = {}
    bar_wallets = {}
    wallet_index = 0
    max_score = 1
    x_y_utxo_anonscore = []
    for wallet_name in wallets_info.keys():
        x_cat_out, y_cat_out = [], []
        cj_index = 0
        for cjtx in sorted_cj_time:
            for index in coinjoins[cjtx['txid']]['outputs'].keys():
                output = coinjoins[cjtx['txid']]['outputs'][index]
                if ('wallet_name' in output.keys() and wallet_name == output['wallet_name'] and
                        'anon_score' in output.keys() and output['anon_score'] > 1):
                    x_cat_out.append(cj_index)
                    y_cat_out.append(output['anon_score'])
                    if 'spend_by_txid' not in coinjoins[cjtx['txid']]['outputs'][index].keys():
                        x_y_utxo_anonscore.append((output['value'], output['anon_score']))
                    if max_score < output['anon_score']:  # keep the maximum
                        max_score = output['anon_score']
            cj_index = cj_index + 1

        if len(x_cat_out) > 0:
            data = np.array(y_cat_out)
            hist, bins = np.histogram(data, bins=20, range=(1, max_score))  # You can adjust the number of bins
            x = (bins[:-1] + bins[1:]) / 2
            bar_positions = (bins[:-1] + bins[1:]) / 2
            linestyle = LINE_STYLES[wallet_index % len(LINE_STYLES)]
            hist_wallets[wallet_name] = hist
            bar_wallets[wallet_name] = bar_positions

            #ax_anonscore_distrib_wallets.bar(x + wallet_index * bar_width * 0.9, y, width=bar_width, align='center', alpha=0.5, label='{} outputs'.format(wallet_name))
            #ax_anonscore_distrib_wallets.plot(x_smooth, y_smooth, label='{} outputs'.format(wallet_name))
            ticks = [round(value, 1) for value in x]
            if cjplt.ax_anonscore_distrib_wallets:
                cjplt.ax_anonscore_distrib_wallets.set_xticks(ticks)
                cjplt.ax_anonscore_distrib_wallets.plot(bar_positions, hist, label=f'{wallet_name} {short_exp_name}', linestyle=linestyle)

        wallet_index = wallet_index + 1

    # Insert average anonscore over all wallets (average from histograms and average from histogram bar positions is computed)
    hist_sum = [sum(elements)/len(hist_wallets.keys()) for elements in zip(*hist_wallets.values())]
    bar_positions_sum = [sum(elements)/len(bar_wallets.keys()) for elements in zip(*bar_wallets.values())]
    if cjplt.ax_anonscore_distrib_wallets:
        cjplt.ax_anonscore_distrib_wallets.plot(bar_positions_sum, np.array(hist_sum), label=f'AVERAGE {short_exp_name}', linestyle='solid',
                                          color='red', linewidth=3)
        cjplt.ax_anonscore_distrib_wallets.set_xlabel('Anonscore')
        cjplt.ax_anonscore_distrib_wallets.set_ylabel('Number of UTXOs')
        cjplt.ax_anonscore_distrib_wallets.ticklabel_format(style='plain', axis='y')
        if len(wallets_info.keys()) <= 15:
            cjplt.ax_anonscore_distrib_wallets.legend(ncol=5, fontsize='small')
        cjplt.ax_anonscore_distrib_wallets.set_title(f'Frequency of Wasabi anonscore of wallet outputs (all transactions)\n{experiment_name}')

    # SAVE_ANALYTICS:
    anonscore_histogram = []
    for index in range(0, len(hist_sum)):
        anonscore_histogram.append((hist_sum[index], bar_positions_sum[index]))
    cjtx_stats['analysis']['wallets_anonscore_histogram_avg'] = [bar_positions_sum, hist_sum]

    #
    # tx entropy versus input/output value size
    #
    x_cat_in, y_cat_in, x_cat_out, y_cat_out, x_cat_out_every_utxo, y_cat_out_every_utxo = [], [], [], [], [], []
    x_y_spend_txo = []
    x_y_utxo, x_y_txo = [], []
    for cjtx in sorted_cj_time:
        tx_entropy = coinjoins[cjtx['txid']]['analysis2']['outputs_entropy']
        for index in coinjoins[cjtx['txid']]['inputs'].keys():
            x_cat_in.append(coinjoins[cjtx['txid']]['inputs'][index]['value'])
            y_cat_in.append(tx_entropy)
        for index in coinjoins[cjtx['txid']]['outputs'].keys():
            x_cat_out.append(coinjoins[cjtx['txid']]['outputs'][index]['value'])
            y_cat_out.append(tx_entropy)
        # Anonymity set for every output
        for index in coinjoins[cjtx['txid']]['analysis2']['outputs'].keys():
            coin_value = coinjoins[cjtx['txid']]['analysis2']['outputs'][index]['value']
            ratio_entropy = (coinjoins[cjtx['txid']]['analysis2']['outputs'][index]['num_others_with_same_values']
                             / len(coinjoins[cjtx['txid']]['analysis2']['outputs']))
            if 'spend_by_txid' not in coinjoins[cjtx['txid']]['outputs'][index].keys():
                # This output is not spend by any other => utxo
                x_cat_out_every_utxo.append(coin_value)
                y_cat_out_every_utxo.append(ratio_entropy)
                x_y_utxo.append((coin_value, ratio_entropy))
            else:
                # This output is already spend
                x_y_spend_txo.append((coin_value, ratio_entropy))
                x_y_txo.append((coin_value, ratio_entropy))

    # Add lines corresponding to standard denominations
    std_denoms = [
        5000, 6561, 8192, 10000, 13122, 16384, 19683, 20000, 32768, 39366, 50000, 59049, 65536, 100000, 118098,
        131072, 177147, 200000, 262144, 354294, 500000, 524288, 531441, 1000000, 1048576, 1062882, 1594323, 2000000,
        2097152, 3188646, 4194304, 4782969, 5000000, 8388608, 9565938, 10000000, 14348907, 16777216, 20000000,
        28697814, 33554432, 43046721, 50000000, 67108864, 86093442, 100000000, 129140163, 134217728, 200000000,
        258280326, 268435456, 387420489, 500000000, 536870912, 774840978, 1000000000, 1073741824, 1162261467,
        2000000000, 2147483648, 2324522934, 3486784401, 4294967296, 5000000000, 6973568802, 8589934592, 10000000000,
        10460353203, 17179869184, 20000000000, 20920706406, 31381059609, 34359738368, 50000000000, 62762119218,
        68719476736, 94143178827, 100000000000, 137438953472
    ]

    def display_denomination_dependency_graph(x_y_spend_txo, x_y_utxo, ax_graph, y_label: str, graph_title: str,
                                              main_color: str):
        assert len(x_y_utxo) > 0
        # Add all spent outputs created in time
        if ax_graph:
            ax_graph.scatter([coin[0] for coin in x_y_spend_txo], [coin[1] for coin in x_y_spend_txo],
                            label=f'Tx outputs (spend) {short_exp_name}', color='red', s=1)

        # Overlay with currently valid UTXOs, make proportional size of utxos based on number of occurrences
        utxo_value_counts = Counter(x_y_utxo)
        assert len(utxo_value_counts) > 0
        # For nice legend entry, pick the most common utxo (wrt occurrences) and plot single one to have it shown in legend
        coin = utxo_value_counts.most_common()[int(len(utxo_value_counts)/2)]
        if ax_graph:
            ax_graph.scatter([coin[0][0]], [coin[0][1]], label=f'Tx outputs (unspend,\nsized by occurence) {short_exp_name}', color=main_color, s=30*coin[1],
                                                       alpha=0.1)
        # Now display all utxos (possibly limited up to MAX_NUM_DISPLAY_UTXO utxos if required)
        if len(utxo_value_counts) > MAX_NUM_DISPLAY_UTXO:
            print(f'Limiting number of UTXOs displayed from {len(utxo_value_counts)} to {MAX_NUM_DISPLAY_UTXO} (MAX_NUM_DISPLAY_UTXO)')
        if ax_graph:
            for coin in list(utxo_value_counts.keys())[0:min(MAX_NUM_DISPLAY_UTXO, len(utxo_value_counts))]:
                ax_graph.scatter([coin[0]], [coin[1]], color=main_color, s=30*utxo_value_counts[coin], alpha=0.1)

        # Compute and display median entropy for standard denominations
        avg_denom_entropy_or_anonscore = {}
        for denom in std_denoms:
            denom_utxos = [coin[1] for coin in x_y_utxo if coin[0] == denom]
            if len(denom_utxos) > 0:
                avg_denom_entropy_or_anonscore[denom] = np.median(denom_utxos)
        if ax_graph:
            ax_graph.plot(list(avg_denom_entropy_or_anonscore.keys()), list(avg_denom_entropy_or_anonscore.values()),
                          color=main_color, linewidth=3, label=f'median {short_exp_name}')

        max_value_used = 0
        if len(x_y_spend_txo) > 0:
            max_value_used = max([coin[0] for coin in x_y_spend_txo])
        if len(x_cat_out_every_utxo) > 0:
            if max(x_cat_out_every_utxo) > max_value_used:
                max_value_used = max(x_cat_out_every_utxo)

        # Plot vertical lines to highlight standard denominations
        if ax_graph:
            for value in std_denoms:
                if value <= max_value_used:
                    ax_graph.axvline(x=value, color='gray', linestyle='--', linewidth=0.1)
            if len(x_cat_in) > 0 or len(x_cat_out) > 0:  # logscale only if some data were inserted
                ax_graph.set_xscale('log')

        if ax_graph:
            ax_graph.set_xlabel('Value (sats) (log scale)')
            ax_graph.set_ylabel(y_label)
            ax_graph.ticklabel_format(style='plain', axis='y')
            ax_graph.legend()
            ax_graph.set_title(graph_title)

        return avg_denom_entropy_or_anonscore

    avg_denom_entropy = display_denomination_dependency_graph(x_y_spend_txo, x_y_utxo, cjplt.ax_utxo_entropy_from_outputs_inoutsize,
                                          'Transaction entropy', f'Dependency of UTXO entropy on size of inputs / outputs (all transactions)\n{experiment_name}',
                                                              'green')
    avg_denom_anonscore = {}
    if len(x_y_utxo_anonscore) > 0:
        avg_denom_anonscore = display_denomination_dependency_graph(x_y_spend_txo, x_y_utxo_anonscore, cjplt.ax_utxo_denom_anonscore,
                                              'Anonscore', f'Dependency of UTXO anonscore on outputs (all transactions)\n{experiment_name}',
                                                                    'darkblue')
    # SAVE_ANALYTICS
    cjtx_stats['analysis']['coinjoin_txos_entropy_on_size'] = [[coin[0] for coin in x_y_spend_txo], [coin[1] for coin in x_y_spend_txo]]
    cjtx_stats['analysis']['coinjoin_utxos_entropy_on_size_median'] = [list(avg_denom_entropy.keys()), list(avg_denom_entropy.values())]
    cjtx_stats['analysis']['coinjoin_utxos_anonscore_on_size_median'] = [list(avg_denom_anonscore.keys()), list(avg_denom_anonscore.values())]


    #
    # Initial and final output sizes for every wallet
    #
    x_txo_initial, y_txo_initial, x_txo_final, y_txo_final = [], [], [], []
    x_y_txo_initial, x_y_txo_final = [], []

    for cjtx in sorted_cj_time:
        # Every wallet's utxo as it was before mixing ('spending_tx' does not exist)
        for index in coinjoins[cjtx['txid']]['inputs'].keys():
            coin_value = coinjoins[cjtx['txid']]['inputs'][index]['value']
            wallet_name = coinjoins[cjtx['txid']]['inputs'][index]['wallet_name']
            if 'spending_tx' not in coinjoins[cjtx['txid']]['inputs'][index].keys():
                # This output is not spend by any other => output from the mix in the wallet
                x_txo_initial.append(wallet_name)
                y_txo_initial.append(coin_value)
                x_y_txo_initial.append((wallet_name, coin_value))

        # Every for every wallet as it was after mixing ('spend_by' does not exist)
        for index in coinjoins[cjtx['txid']]['analysis2']['outputs'].keys():
            coin_value = coinjoins[cjtx['txid']]['analysis2']['outputs'][index]['value']
            wallet_name = coinjoins[cjtx['txid']]['analysis2']['outputs'][index]['wallet_name']
            if 'spend_by_txid' not in coinjoins[cjtx['txid']]['outputs'][index].keys():
                # This output is not spend by any other => output from the mix in the wallet
                x_txo_final.append(wallet_name)
                y_txo_final.append(coin_value)
                x_y_txo_final.append((wallet_name, coin_value))

    # Add all outputs created in time
    SORT_WALLETS = False
    UTXO_POINT_SIZE = 1
    if SORT_WALLETS:
        combined_data = list(zip(x_txo_initial, y_txo_initial))
        sorted_data = sorted(combined_data, key=lambda x: x[0])
        x_txo_initial, y_txo_initial = zip(*sorted_data)
    if cjplt.ax_initial_final_utxos:
        cjplt.ax_initial_final_utxos.scatter(x_txo_initial, y_txo_initial, label=f'Initial mix inputs {short_exp_name}', color='red', s=UTXO_POINT_SIZE, alpha=0.2)
    if SORT_WALLETS:
        combined_data = list(zip(x_txo_final, y_txo_final))
        sorted_data = sorted(combined_data, key=lambda x: x[0])
        x_txo_final, y_txo_final = zip(*sorted_data)
    stripped_wallet_names = [value.split('-')[-1] for value in list(set(x_txo_initial))]
    if cjplt.ax_initial_final_utxos:
        cjplt.ax_initial_final_utxos.scatter(x_txo_final, y_txo_final, label=f'Final mix outputs {short_exp_name}', color='green', s=UTXO_POINT_SIZE, alpha=0.2)
        cjplt.ax_initial_final_utxos.set_xticks(range(0, len(stripped_wallet_names)), stripped_wallet_names, rotation=45, fontsize=6)

    # Make proportional size of utxos based on number of occurences
    # value_counts = Counter(x_y_txo_initial)
    # coin = value_counts.most_common()[int(len(value_counts)/2)]
    # ax_initial_final_utxos.scatter([coin[0][0]], [coin[0][1]], label='Initial mix inputs,\nsized by occurence)', color='red', s=30*coin[1],
    #                                                alpha=0.1)
    # value_counts = Counter(x_y_txo_final)
    # coin = value_counts.most_common()[int(len(value_counts)/2)]
    # ax_initial_final_utxos.scatter([coin[0][0]], [coin[0][1]], label='Final mix outputs,\nsized by occurence)', color='green', s=30*coin[1],
    #                                                alpha=0.1)

    # Plot lines corresponding to standard denominations
    if cjplt.ax_initial_final_utxos:
        for value in std_denoms:
            if value <= max(y_txo_initial):
                cjplt.ax_initial_final_utxos.axhline(y=value, color='gray', linestyle='--', linewidth=0.1)

        if len(y_txo_initial) > 0 or len(y_txo_final) > 0:  # logscale only if some data were inserted
            cjplt.ax_initial_final_utxos.set_yscale('log')
        cjplt.ax_initial_final_utxos.set_ylabel('Value (sats) (log scale)')
        cjplt.ax_initial_final_utxos.set_xlabel('Wallet name')
        cjplt.ax_initial_final_utxos.legend()
        cjplt.ax_initial_final_utxos.set_title(f'Presence of wallet utxos at beginning and end of mixing\n{experiment_name}')


    #
    # Wallet efficiency in time
    #
    num_combinations_data = [coinjoins[cjtx['txid']]['analysis']['processed']['num_combinations_detected'] for cjtx in sorted_cj_time if 'analysis' in coinjoins[cjtx['txid']].keys()
                       and coinjoins[cjtx['txid']]['analysis']['processed']['successfully_analyzed'] is True]
    x = range(1, len(num_combinations_data))
    num_combinations_data_avg100 = sliding_window_average(num_combinations_data, 100)
    if cjplt.ax_boltzmann_txcombinations:
        cjplt.ax_boltzmann_txcombinations.plot(num_combinations_data, label=f'Number of tx combinations detected {short_exp_name}')
        cjplt.ax_boltzmann_txcombinations.plot(x[100-2:], num_combinations_data_avg100, label=f'Number of tx combinations detected avg(100) {short_exp_name}', linewidth=3, color='green')
        cjplt.ax_boltzmann_txcombinations.set_xlabel('Coinjoin in time')
        cjplt.ax_boltzmann_txcombinations.set_ylabel('Number of combinations (log scale)')
        cjplt.ax_boltzmann_txcombinations.ticklabel_format(style='plain', axis='y')
        if len(num_combinations_data) > 0 or len(x_cat_out) > 0:  # logscale only if some data were inserted
            cjplt.ax_boltzmann_txcombinations.set_yscale('log')
        cjplt.ax_boltzmann_txcombinations.legend()
        cjplt.ax_boltzmann_txcombinations.set_title(f'Number of tx combinations found in time (Boltzmann, only successfully analyzed transactions!)\n{experiment_name}')

    #
    # Num non-unique output values in time
    #
    entropy_in_time_data = [coinjoins[cjtx['txid']]['analysis2']['outputs_entropy'] for cjtx in sorted_cj_time]
    min_entropy_in_time_data = [coinjoins[cjtx['txid']]['analysis2']['outputs_min_entropy'] for cjtx in sorted_cj_time]
    #non_unique_in_time_data = [coinjoins[cjtx['txid']]['analysis2']['outputs_sameness_ratio'] * 100 for cjtx in sorted_cj_time]
    min_entropy_in_time_data_avg100 = sliding_window_average(min_entropy_in_time_data, 100)
    entropy_in_time_data_avg100 = sliding_window_average(entropy_in_time_data, 100)
    x = range(1, len(min_entropy_in_time_data))
    if cjplt.ax_utxo_entropy_from_outputs:
        cjplt.ax_utxo_entropy_from_outputs.plot(entropy_in_time_data, label=f'Outputs entropy {short_exp_name}', color='lightblue')
        cjplt.ax_utxo_entropy_from_outputs.plot(min_entropy_in_time_data, label=f'Outputs min entropy {short_exp_name}', color='lightgreen')
        cjplt.ax_utxo_entropy_from_outputs.plot(x[100-2:], entropy_in_time_data_avg100, label=f'Outputs entropy avg(100) {short_exp_name}', linewidth=3, color='darkblue')
        cjplt.ax_utxo_entropy_from_outputs.plot(x[100-2:], min_entropy_in_time_data_avg100, label=f'Outputs min entropy avg(100) {short_exp_name}', linewidth=3, color='darkgreen')
        cjplt.ax_utxo_entropy_from_outputs.set_xlabel('Coinjoin in time')
        cjplt.ax_utxo_entropy_from_outputs.set_ylabel('Entropy')
        cjplt.ax_utxo_entropy_from_outputs.set_ylim(0)
        cjplt.ax_utxo_entropy_from_outputs.legend()
        cjplt.ax_utxo_entropy_from_outputs.set_title(f'Entropy of outputs of coinjoin transactions (all transactions)\n{experiment_name}')

    # SAVE_ANALYTICS
    cjtx_stats['analysis']['coinjoin_utxos_entropy_in_time'] = [list(range(0, len(entropy_in_time_data))), entropy_in_time_data]
    cjtx_stats['analysis']['coinjoin_utxos_minentropy_in_time'] = [list(range(0, len(min_entropy_in_time_data))), min_entropy_in_time_data]

    #!!! Add how many coinjoins given wallet participated in. Weighted number by number of input txos.


# 'os^vD'
SCENARIO_MARKER_MAPPINGS = {'paretosum-static-.*-30utxo': '*', 'paretosum-static-.*-5utxo': '^',
                            'uniformsum-static-.*-30utxo': 'D', 'uniformsum-static-.*-5utxo': 's'}


def get_marker_style(experiment_name: str, mapping_set: dict) -> str:
    for pattern, marker in mapping_set.items():
        match = re.search(pattern, experiment_name)
        if match:
            return marker

    return 'o'  # return standard circle if nothing matched


def visualize_scatter_num_wallets(fig, multi_cjtx_stats: dict, dataset_name: str, dataset_name_index: int, funct_eval, normalize_x: bool, normalize_y: bool, x_label: str, y_label: str, graph_title: str):
    for experiment in multi_cjtx_stats:
        # Establish number of wallets overall, remove wallets which did not participated
        num_wallets = len(multi_cjtx_stats[experiment]['wallets_info'].keys()) - len(multi_cjtx_stats[experiment]['analysis']['wallets_no_input_mixed'][1])
        inputs = multi_cjtx_stats[experiment]['analysis'][dataset_name][dataset_name_index]
        # Normalize y axis values
        if normalize_y:  # Normalize by size of values
            sum_values = sum(inputs)
            inputs = [value / sum_values for value in inputs]
        result = funct_eval(inputs)
        if normalize_x:  # Normalize by number of values
            result /= len(inputs)
        marker_style = get_marker_style(experiment, SCENARIO_MARKER_MAPPINGS)
        fig.scatter([num_wallets], [result], label=f'{os.path.basename(experiment)}', alpha=0.6, s=40, marker=marker_style)
    fig.set_xlabel(x_label)
    fig.set_ylabel(y_label)
    fig.legend(ncol=5, fontsize='4')
    fig.set_title(graph_title)


def visualize_scatter_num_wallets_weighted(fig, multi_cjtx_stats: dict, dataset_name: str, funct_eval, normalize: bool, x_label: str, y_label: str, graph_title: str):
    for experiment in multi_cjtx_stats:
        # Establish number of wallets overall, remove wallets which did not participated
        num_wallets = len(multi_cjtx_stats[experiment]['wallets_info'].keys()) - len(multi_cjtx_stats[experiment]['analysis']['wallets_no_input_mixed'][1])

        # Obtain desired dataset to be processed
        inputs = multi_cjtx_stats[experiment]['analysis'][dataset_name]

        # If required, normalize the dataset (all lists separately)
        if normalize is True:
            normalized_lists = []
            for data_list in inputs:
                normalized_list = [x / sum(data_list) for x in data_list]
                normalized_lists.append(normalized_list)
            inputs = normalized_lists

        # Weight datasets
        weighted_input_data = [inputs[0][index] * inputs[1][index] for index in range(0, len(inputs[0]))]

        # Apply evaluation function
        result = funct_eval(weighted_input_data)

        # Plot result with dedicated marker
        marker_style = get_marker_style(experiment, SCENARIO_MARKER_MAPPINGS)
        fig.scatter([num_wallets], [result], label=f'{os.path.basename(experiment)}', alpha=0.6, s=40, marker=marker_style)
    fig.set_xlabel(x_label)
    fig.set_ylabel(y_label)
    fig.legend(ncol=5, fontsize='4')
    fig.set_title(graph_title)


def interpolate_values(original_list, new_length):
    ratio = new_length / len(original_list)
    interpolated_list = []

    # Case 1: list expansion : ratio > 1
    if ratio >= 1:
        remaining_previous_ratio = 0
        for orig_index in range(len(original_list)):
            value_to_distribute = original_list[orig_index]

            num_cells_to_fill = math.ceil(ratio + remaining_previous_ratio)
            remaining_previous_ratio = (ratio + remaining_previous_ratio) - num_cells_to_fill
            for i in range(num_cells_to_fill):
                if len(interpolated_list) < new_length:  # condition to prevent +1 overflow due to rounding errors
                    interpolated_list.append(float(value_to_distribute) / num_cells_to_fill)
                else:
                    # add to the last item (do not expand further)
                    interpolated_list[-1] += float(value_to_distribute) / num_cells_to_fill

    # Case 2: list compression : ratio < 1
    if ratio < 1:
        remaining_previous_cells = 0
        orig_index = 0
        while orig_index < len(original_list):
            num_cells_to_sum = math.floor(1 / ratio + remaining_previous_cells)
            remaining_previous_cells = 1 / ratio - num_cells_to_sum
            sum_value = 0
            for i in range(num_cells_to_sum):
                if orig_index + i < len(original_list):
                    sum_value += original_list[orig_index + i]
            orig_index += num_cells_to_sum
            if len(interpolated_list) < new_length:
                # Add new item
                interpolated_list.append(sum_value)
            else:
                # add to the last item (do not expand further)
                interpolated_list[-1] += sum_value

    assert len(interpolated_list) == new_length, f'Wrong resulting length of interpolated list, {new_length} vs. {len(interpolated_list)}'
    assert float_equals(sum(original_list), sum(interpolated_list)), f'Sum of values in interpolated list not preserved, {sum(original_list)} vs. {sum(interpolated_list)}'
    return interpolated_list


def weighted_wallets_participating(fig, multi_cjtx_stats: dict, dataset_name: str, funct_eval, normalize: bool, x_label: str, y_label: str, graph_title: str):
    # 1. Set normalized number of bins NUM_BINS (e.g., 100)
    # 2. Interpolate existing input list (e.g., number of cjtxs with specific number of wallets) to NUM_BINS for comparisond
    # 3. Compute weighted sum of bins (more the higher, the highest result is when all coinjoins had all wallets participating)

    # 1. Standardized number of bins
    NUM_BINS = 100

    for experiment in multi_cjtx_stats:
        # Establish number of wallets overall, remove wallets which did not participated
        num_wallets = len(multi_cjtx_stats[experiment]['wallets_info'].keys()) - len(multi_cjtx_stats[experiment]['analysis']['wallets_no_input_mixed'][1])

        # Obtain desired dataset to be processed
        inputs_orig = multi_cjtx_stats[experiment]['analysis'][dataset_name]
        inputs = inputs_orig

        # 2. Recompute to given number of bins (two cases 1. num_wallets > num_bins and 2. num_wallets <= num_bins)
        inputs[1] = interpolate_values(inputs[1], NUM_BINS)
        inputs[0] = [index for index in range(0, NUM_BINS)]

        # If required, normalize the dataset (all lists separately)
        if normalize is True:
            inputs[1] = [x / sum(inputs[1]) for x in inputs[1]]

        # Weight datasets
        weighted_input_data = [inputs[0][index] * inputs[1][index] for index in range(0, len(inputs[0]))]

        # Apply evaluation function
        result = funct_eval(weighted_input_data)

        # Normalize
        if normalize is True:
            result = result / NUM_BINS

        # Plot result with dedicated marker
        marker_style = get_marker_style(experiment, SCENARIO_MARKER_MAPPINGS)
        fig.scatter([num_wallets], [result], label=f'{os.path.basename(experiment)}', alpha=0.5, s=20, marker=marker_style)
    fig.set_xlabel(x_label)
    fig.set_ylabel(y_label)
    fig.legend(ncol=5, fontsize='4')
    fig.set_title(graph_title)


def analyze_aggregated_coinjoin_stats(multi_cjtx_stats, base_path):
    """
    Analyze and visualize aggregated coinjoin transactions statistics
    :param multi_cjtx_stats: statistics from multiple separate coinjoin experiments
    :param base_path:  base path for analysis
    :return:
    """

    print("Starting analyze_aggregated_coinjoin_stats() analysis")

    # Create four subplots with their own axes
    fig = plt.figure(figsize=(48, 24))
    ax1 = fig.add_subplot(3, 3, 1)
    ax2 = fig.add_subplot(3, 3, 2)
    ax3 = fig.add_subplot(3, 3, 3)
    ax4 = fig.add_subplot(3, 3, 4)
    ax5 = fig.add_subplot(3, 3, 5)
    ax6 = fig.add_subplot(3, 3, 6)
    ax7 = fig.add_subplot(3, 3, 7)
    ax8 = fig.add_subplot(3, 3, 8)
    ax9 = fig.add_subplot(3, 3, 9)

    # Basic principle:
    #   1. Compute one (or more) data points per single experiment
    #   2. Visualize all data points as scatterplot


    #
    # Weighted results computations FUNCT[item[0] * item[1] for item in data]
    #
    # Average anonscore for different base parameter (number of wallets participating)
    visualize_scatter_num_wallets_weighted(ax1, multi_cjtx_stats, 'wallets_anonscore_histogram_avg', np.median,
                                           False, 'Number of wallets in mix', 'Median anonscore',
                                  f'Dependency of {'wallets_anonscore_histogram_avg'} on number of wallets')

    # DONE: Histogram of number of wallets participating in coinjoins (normalized)
    weighted_wallets_participating(ax4, multi_cjtx_stats, 'coinjoin_number_different_wallets', sum,
                                           True, 'Number of wallets in mix', 'Fraction of different wallets in coinjoins',
                                           f'Dependency of {'coinjoin_number_different_wallets'} on number of wallets')

    #
    # Absolute number computations FUNCT[data[data_index]]
    # Relative number computations FUNCT[data[data_index]] / len(data[data_index])
    #

    # DONE: Number of coinjoins finished for different base parameter (number of wallets participating)
    visualize_scatter_num_wallets(ax2, multi_cjtx_stats, 'coinjoins_finished_all',
                                  1, sum, False, False, 'Number of wallets in mix', 'Number of coinjoins',
                                  f'Dependency of {'coinjoins_finished_all'} on number of wallets')

    # DONE: Number of utxos in prison
    visualize_scatter_num_wallets(ax3, multi_cjtx_stats, 'utxos_in_prison',
                                  1, sum, False, False, 'Number of wallets in mix', 'Number of utxos in prison',
                                  f'Dependency of {'utxos_in_prison'} on number of wallets')

    # Median frequency a wallet was used in mix
    visualize_scatter_num_wallets(ax5, multi_cjtx_stats, 'wallets_times_used_as_input',
                                  1, np.median, True, True, 'Number of wallets in mix', 'Fraction of participation',
                                  f'Dependency of {'wallets_times_used_as_input'} on number of wallets')

    # Average number of inputs into cjtxs
    visualize_scatter_num_wallets(ax6, multi_cjtx_stats, 'coinjoin_number_inputs_in_time',
                                  1, sum, True, False, 'Number of wallets in mix', 'Number of inputs of cjtx',
                                  f'Dependency of {'coinjoin_number_inputs_in_time'} (average number of inputs) on number of wallets')

    # Average number of outputs into cjtxs
    visualize_scatter_num_wallets(ax7, multi_cjtx_stats, 'coinjoin_number_outputs_in_time',
                                  1, sum, True, False, 'Number of wallets in mix', 'Number of outputs of cjtx',
                                  f'Dependency of {'coinjoin_number_outputs_in_time'} (average number of outputs) on number of wallets')

    # Average entropy of coinjoins
    visualize_scatter_num_wallets(ax8, multi_cjtx_stats, 'coinjoin_utxos_entropy_in_time',
                                  1, sum, True, False, 'Number of wallets in mix', 'Entropy',
                                  f'Dependency of {'coinjoin_utxos_entropy_in_time'} (average entropy of coinjoin round) on number of wallets')

    # coinjoin_txos_entropy_in_time

    # Save results into file
    experiment_name = os.path.basename(base_path)
    plt.suptitle('{}'.format(experiment_name), fontsize=16)  # Adjust the fontsize and y position as needed
    plt.subplots_adjust(bottom=0.1, wspace=0.1, hspace=0.5)
    save_file = os.path.join(base_path, f"aggregated_coinjoin_stats{GLOBAL_IMG_SUFFIX}.png")
    plt.savefig(save_file, dpi=300)
    plt.close()
    print('Aggregated coinjoins statistics saved into {}'.format(save_file))


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
                if addr in cjtx_stats['wallets_info'][wallet_name].keys():
                    address_wallet_mapping[addr] = wallet_name
                    cjtx_stats['coinjoins'][cjtxid]['inputs'][index]['wallet_name'] = wallet_name

        for index in cjtx_stats['coinjoins'][cjtxid]['outputs'].keys():
            addr = cjtx_stats['coinjoins'][cjtxid]['outputs'][index]['address']
            for wallet_name in cjtx_stats['wallets_info'].keys():
                if addr in cjtx_stats['wallets_info'][wallet_name].keys():
                    address_wallet_mapping[addr] = wallet_name
                    cjtx_stats['coinjoins'][cjtxid]['outputs'][index]['wallet_name'] = wallet_name

    return address_wallet_mapping


def parse_client_coinjoin_logs(cjtx_stats, base_directory):
    # Client logs parsing
    # 2023-10-23 16:23:30.303 [40] INFO	AliceClient.RegisterInputAsync (121)	Round (5455291d82b748469b5eb2e63d3859370c1f3823d4b8ca5cea7322f93b98af05), Alice (6eb340fe-d153-ac10-0246-252e8a866fbc): Registered 95cdc75886465b7e0a95b7f7e41a92c0ff92a8d2d075d426b92f0ca1b8424d2c-4.
    # 2023-10-23 16:23:38.053 [41] INFO	AliceClient.CreateRegisterAndConfirmInputAsync (77)	Round (5455291d82b748469b5eb2e63d3859370c1f3823d4b8ca5cea7322f93b98af05), Alice (6eb340fe-d153-ac10-0246-252e8a866fbc): Connection was confirmed.
    # 2023-10-23 16:24:05.939 [27] INFO	AliceClient.ReadyToSignAsync (223)	Round (5455291d82b748469b5eb2e63d3859370c1f3823d4b8ca5cea7322f93b98af05), Alice (6eb340fe-d153-ac10-0246-252e8a866fbc): Ready to sign.
    # 2023-10-23 16:24:46.110 [41] INFO	AliceClient.SignTransactionAsync (217)	Round (5455291d82b748469b5eb2e63d3859370c1f3823d4b8ca5cea7322f93b98af05), Alice (6eb340fe-d153-ac10-0246-252e8a866fbc): Posted a signature.
    client_input_file = os.path.join(base_directory, 'Logs.txt')

    print('Parsing coinjoin-relevant data from client logs {}...'.format(client_input_file), end='')

    # class CJ_ALICE_TYPES(Enum):
    #     ALICE_REGISTERED = 'ALICE_REGISTERED'
    #     ALICE_CONNECTION_CONFIRMED = 'ALICE_CONNECTION_CONFIRMED'
    #     ALICE_READY_TO_SIGN = 'ALICE_READY_TO_SIGN'
    #     ALICE_POSTED_SIGNATURE = 'ALICE_POSTED_SIGNATURE'


    alice_events_log = {}
    regex_pattern = r"(?P<timestamp>.*) \[.+(AliceClient.RegisterInputAsync.*) \(.*Round \((?P<round_id>.*)\), Alice \((?P<alice_id>.*)\): Registered (?P<tx_id>.*)-(?P<tx_out_index>[0-9]+)\.?"
    alice_events = find_round_ids(client_input_file, regex_pattern, ['round_id', 'timestamp', 'alice_id', 'tx_id', 'tx_out_index'])
    for round_id in alice_events.keys():
        for alice_event in alice_events[round_id]:
            alice_id = alice_event['alice_id']
            if alice_id not in alice_events_log.keys():
                alice_events_log[alice_id] = {}

            alice_events_log[alice_id][CJ_ALICE_TYPES.ALICE_REGISTERED.name] = alice_event

    regex_pattern = r"(?P<timestamp>.*) \[.+(AliceClient.CreateRegisterAndConfirmInputAsync.*) \(.*Round \((?P<round_id>.*)\), Alice \((?P<alice_id>.*)\): Connection was confirmed\.?"
    alice_events = find_round_ids(client_input_file, regex_pattern, ['round_id', 'timestamp', 'alice_id'])
    for round_id in alice_events.keys():
        for alice_event in alice_events[round_id]:
            alice_id = alice_event['alice_id']
            alice_events_log[alice_id][CJ_ALICE_TYPES.ALICE_CONNECTION_CONFIRMED.name] = alice_event

    regex_pattern = r"(?P<timestamp>.*) \[.+(AliceClient.ReadyToSignAsync.*) \(.*Round \((?P<round_id>.*)\), Alice \((?P<alice_id>.*)\): Ready to sign\.?"
    alice_events = find_round_ids(client_input_file, regex_pattern, ['round_id', 'timestamp', 'alice_id'])
    for round_id in alice_events.keys():
        for alice_event in alice_events[round_id]:
            alice_id = alice_event['alice_id']
            alice_events_log[alice_id][CJ_ALICE_TYPES.ALICE_READY_TO_SIGN.name] = alice_event

    regex_pattern = r"(?P<timestamp>.*) \[.+(AliceClient.SignTransactionAsync.*) \(.*Round \((?P<round_id>.*)\), Alice \((?P<alice_id>.*)\): Posted a signature\.?"
    alice_events = find_round_ids(client_input_file, regex_pattern, ['round_id', 'timestamp', 'alice_id'])
    for round_id in alice_events.keys():
        for alice_event in alice_events[round_id]:
            alice_id = alice_event['alice_id']
            alice_events_log[alice_id][CJ_ALICE_TYPES.ALICE_POSTED_SIGNATURE.name] = alice_event

    # Find and pair alice event logs to the right input
    #for cjtx_id in cjtx_stats['coinjoins'].keys():

    print('finished')


def parse_backend_coinjoin_logs(coord_input_file, raw_tx_db: dict = {}):
    print('Parsing coinjoin-relevant data from coordinator logs {}...'.format(coord_input_file), end='')
    if PRE_2_0_4_VERSION:
        regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Created round with params: MaxSuggestedAmount:'([0-9\.]+)' BTC?"
    else:
        regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Created round with parameters: MaxSuggestedAmount:'([0-9\.]+)' BTC?"
    start_round_ids = find_round_ids(coord_input_file, regex_pattern, ['round_id', 'timestamp'])
    # 2023-09-05 08:56:50.892 [38] INFO	Arena.CreateBlameRoundAsync (417)	Blame Round (c05a3b73cebffc79956e1e3abf3d9020b3e02e05f01eb7fb0d01dbcd26d64be7): Blame round created from round '05a9dfe6244d2f4004d2927798ecd42d557bbaefe61de58f67a0265f5710a2da'.
    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Blame Round \((?P<round_id>.*)\): Blame round created from round '(?P<orig_round_id>.*)'?"
    start_blame_rounds_id = find_round_ids(coord_input_file, regex_pattern, ['round_id', 'timestamp'])
    # 2023-10-07 11:04:56.723 [43] INFO	Arena.StepTransactionSigningPhaseAsync (374)	Round (dee277ed8fd5d1af24bf09126818b3cec362f52f9fc4323474c2ec5075454d1a): Successfully broadcast the coinjoin: 345386611e7a4543524a3c7fa27f14d511fbb70b1b8786d777b19fb265e95558.
    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Successfully broadcast the coinjoin: (?P<cj_tx_id>[0-9a-f]*)\.?"
    success_coinjoin_round_ids = find_round_ids(coord_input_file, regex_pattern, ['round_id', 'timestamp', 'cj_tx_id'])
    # round_cjtx_mapping = find_round_cjtx_mapping(coord_input_file, regex_pattern, 'round_id', 'cj_tx_id')
    round_cjtx_mapping = {round_id: success_coinjoin_round_ids[round_id][0]['cj_tx_id'] for round_id in     # take only the first record found
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
        tx_record = extract_tx_info(round_cjtx_mapping[round_id], raw_tx_db)
        if tx_record is not None:
            # Find coinjoin transaction id and create record if not already
            cjtxid = round_cjtx_mapping[round_id]
            if cjtxid not in cjtx_stats.keys():
                cjtx_stats[cjtxid] = {}

            tx_record['round_id'] = round_id
            tx_record['round_start_time'] = start_round_ids[round_id][0]['timestamp']
            tx_record['broadcast_time'] = success_coinjoin_round_ids[round_id][0]['timestamp']
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
        for index in value:
            index.update({'type': type_info.name})


def insert_by_round_id(rounds_logs, events):
    for round_id, value in events.items():
        if round_id not in rounds_logs:
            rounds_logs[round_id] = {}
        if 'logs' not in rounds_logs[round_id]:
            rounds_logs[round_id]['logs'] = []
        rounds_logs[round_id]['logs'].extend(value)


def parse_coinjoin_errors(cjtx_stats, coord_input_file):
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
            rounds_logs[round_id]['cj_tx_id'] = success_coinjoin_round_ids[round_id][0]['cj_tx_id']

    # Start of a round
    if PRE_2_0_4_VERSION:
        regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Created round with params: MaxSuggestedAmount:'([0-9\.]+)' BTC?"
    else:
        regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Created round with parameters: MaxSuggestedAmount:'([0-9\.]+)' BTC?"
    start_round_ids = find_round_ids(coord_input_file, regex_pattern, ['round_id', 'timestamp'])
    insert_type(start_round_ids, CJ_LOG_TYPES.ROUND_STARTED)
    insert_by_round_id(rounds_logs, start_round_ids)
    for round_id in rounds_logs.keys():
        if round_id in start_round_ids.keys():
            rounds_logs[round_id]['round_start_timestamp'] = start_round_ids[round_id][0]['timestamp']

    # Start of a blame round
    regex_pattern = r"(?P<timestamp>.*) \[.+(Arena\..*) \(.*Blame Round \((?P<round_id>.*)\): Blame round created from round '(?P<orig_round_id>.*)'?"
    start_blame_rounds_id = find_round_ids(coord_input_file, regex_pattern, ['round_id', 'timestamp'])
    insert_type(start_blame_rounds_id, CJ_LOG_TYPES.BLAME_ROUND_STARTED)
    insert_by_round_id(rounds_logs, start_blame_rounds_id)
    for round_id in rounds_logs.keys():
        if round_id in start_blame_rounds_id.keys():
            rounds_logs[round_id]['round_start_timestamp'] = start_blame_rounds_id[round_id][0]['timestamp']

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
                          alices_removed[key][0]['num_alices_remaining'] == '0'}

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
    """load_wallets_info
    Loads information about wallets and their addresses using Wasabi RPC
    :return: dictionary for all loaded wallets with retrieved info
    """
    MAX_WALLETS = 30
    wcli.WASABIWALLET_DATA_DIR = os.path.join(WASABIWALLET_DATA_DIR, "WalletWasabi")
    wcli.VERBOSE = False
    wallets_info = {}
    wallet_names = ['{}{}'.format('SimplePassiveWallet', index) for index in range(1, MAX_WALLETS + 1)]
    #wallet_names.extend(['{}{}'.format('Wallet', index) for index in range(1, MAX_WALLETS + 1)])
    for wallet_name in wallet_names:
        if wcli.LEGACY_API:
            if wcli.wcli(['selectwallet', wallet_name, 'pswd']) is not None:
                print('Wallet `{}` found.'.format(wallet_name))
                wcli.wcli(['getwalletinfo'])
                wallet_addresses = wcli.wcli(['listkeys'])
                wallet_coins_all = wcli.wcli(['listcoins'])
                wallet_coins_unspent = wcli.wcli(['listunspentcoins'])
                if wallet_addresses is not None:
                    wallets_info[wallet_name] = {addr_data['address']: addr_data for addr_data in wallet_addresses}
                    #wallets_info[wallet_name] = wallet_addresses
            else:
                print('Wallet `{}` not found.'.format(wallet_name))
        else:
            #if wcli.wcli(['-wallet={}'.format(wallet_name), 'getwalletinfo']) is not None:
            wcli.wcli(['-wallet={}'.format(wallet_name), 'getwalletinfo'])
            wallet_addresses = wcli.wcli(['-wallet={}'.format(wallet_name), 'listkeys'])
            wallet_coins_all = wcli.wcli(['-wallet={}'.format(wallet_name), 'listcoins'])
            wallet_coins_unspent = wcli.wcli(['-wallet={}'.format(wallet_name), 'listunspentcoins'])
            if wallet_addresses is not None:
                wallets_info[wallet_name] = {addr_data['address']: addr_data for addr_data in wallet_addresses}
                #wallets_info[wallet_name] = wallet_addresses
            else:
                print('Wallet `{}` not found.'.format(wallet_name))
    return wallets_info, wallet_coins_all, wallet_coins_unspent


def visualize_coinjoins(cjtx_stats, base_path='', output_name='coinjoin_graph', view_pdf=True):
    address_wallet_mapping = cjtx_stats['address_wallet_mapping']

    # Prepare Graph
    dot2 = Digraph(comment='CoinJoin={}'.format("XX"))
    graph_label = ''
    graph_label += 'Coinjoin visualization\n.'
    dot2.attr(rankdir='LR', size='8,5')
    dot2.attr(size='120,80')

    # Generate wallet color mapping
    color_ctr = 0
    for wallet_name in cjtx_stats['wallets_info'].keys():
        if color_ctr < len(COLORS):
            WALLET_COLORS[wallet_name] = COLORS[color_ctr]
        else:
            WALLET_COLORS[wallet_name] = 'grey{}'.format((color_ctr * 5) % 100)

        graphviz_insert_wallet(wallet_name, dot2)
        color_ctr = color_ctr + 1
    # add special color for 'unknown' wallet
    WALLET_COLORS[UNKNOWN_WALLET_STRING] = 'red'
    WALLET_COLORS[COORDINATOR_WALLET_STRING] = 'grey30'

    graphviz_insert_wallet(UNKNOWN_WALLET_STRING, dot2)
    graphviz_insert_wallet(COORDINATOR_WALLET_STRING, dot2)

    sorted_cjs = sorted(cjtx_stats['coinjoins'], key=lambda txid: cjtx_stats['coinjoins'][txid]['broadcast_time'])

    # for cjtxid in cjtx_stats['coinjoins'].keys():
    prev_cjtxid = None
    for cjtxid in sorted_cjs:
        # Visualize into large connected graph
        visualize_cjtx_graph(cjtx_stats['coinjoins'], cjtxid, address_wallet_mapping, dot2)
        if GENERATE_COINJOIN_GRAPH_LINEAR and prev_cjtxid:
            dot2.edge(prepare_display_cjtxid(prev_cjtxid), prepare_display_cjtxid(cjtxid), constraint='true')
        prev_cjtxid = cjtxid

    # render resulting graphviz
    save_file = os.path.join(base_path, output_name)
    dot2.render(save_file, view=view_pdf)


def get_input_name_string(input):
    return f'vin_{input[0]}_{input[1]['vin']}'


def get_output_name_string(output):
    return get_output_name_string(output[1]['txid'], output[1]['vout'])


def get_output_name_string(txid, output_index):
    return f'vout_{txid}_{output_index}'


def extract_txid_from_inout_string(inout_string):
    if inout_string.startswith('vin') or inout_string.startswith('vout'):
        return inout_string[inout_string.find('_') + 1: inout_string.rfind('_')]
    else:
        assert False, f'Invalid inout string {inout_string}'


def graphviz_insert_aggregated_connections(cjtx_stats, target_cjtxid, graphdot):
    """
    Insert connection colored by wallet and sized by sum of values from a same wallet to target_cjtxid inputs.
    :param target_cjtxid:
    :param graphdot:
    :return:
    """

    # For every wallet, find all inputs coming from any other previous conjoin transactions used in target_cjtxid
    # Insert edge (for every wallet) with thickness equivalent to sum of input values

    # Wallets used in target_cjtxid
    wallets_used = set([cjtx_stats[target_cjtxid]['inputs'][index]['wallet_name']
                        for index in cjtx_stats[target_cjtxid]['inputs'].keys()])

    cjtx = cjtx_stats[target_cjtxid]  # Shortcut for target_cjtxid transaction

    for other_cjtxid in cjtx_stats.keys():
        if other_cjtxid == target_cjtxid:  # Do not process itself
            continue

        # For every wallet, find sum of inputs between other_cjtxid and target_cjtxid
        for wallet_name in wallets_used:
            wallet_inputs_sum = 0
            for index in cjtx['inputs'].keys():
                if wallet_name == cjtx['inputs'][index]['wallet_name']:
                    value = cjtx['inputs'][index]['value'] if 'value' in cjtx['inputs'][index] else 1
                    if 'spending_tx' in cjtx['inputs'][index].keys():
                        txid, index = cjtx['inputs'][index]['spending_tx']
                        if txid == other_cjtxid:
                            wallet_inputs_sum += value
                    else:
                        print(f'WARNING: Missing input specification for {target_cjtxid}[inputs]{index}')

            # Insert single edge between target_cjtxid and other_cjtxid for shared inputs
            if wallet_inputs_sum > 0:  # Plot connection only if at least some satoshies are transfered
                width = math.log(int(wallet_inputs_sum), 2)  # The sizes might be very different, use log
                graphdot.edge(prepare_display_cjtxid(other_cjtxid), prepare_display_cjtxid(target_cjtxid),
                              color=WALLET_COLORS[wallet_name], style='solid', dir='forward', penwidth=f'{width}')


def visualize_coinjoins_aggregated(cjtx_stats, base_path='', output_name='coinjoin_graph', view_pdf=True):
    """
    Creates simplified graph of coinjoins where nodes are cj transactions and edges are inputs/outputs used.
     Colored and sized according to wallet.
    :param cjtx_stats: structure with all coinjoins
    :param base_path: string with description of path with measurements used as label in graph
    :param output_name: name of output file to save pdf
    :param view_pdf: If True, the pdf is shown after its rendering
    :return:
    """
    address_wallet_mapping = cjtx_stats['address_wallet_mapping']

    # Prepare Graph
    dot2 = Digraph(comment='CoinJoin={}'.format("XX"))
    graph_label = ''
    graph_label += 'Coinjoin visualization\n.'
    dot2.attr(rankdir='LR', size='8,5')
    dot2.attr(size='120,80')

    # Generate wallet color mapping
    color_ctr = 0
    for wallet_name in cjtx_stats['wallets_info'].keys():
        if color_ctr < len(COLORS):
            WALLET_COLORS[wallet_name] = COLORS[color_ctr]
        else:
            WALLET_COLORS[wallet_name] = 'grey{}'.format((color_ctr * 5) % 100)
        color_ctr = color_ctr + 1

    # add special color for 'unknown' wallet
    WALLET_COLORS[UNKNOWN_WALLET_STRING] = 'red'
    WALLET_COLORS[COORDINATOR_WALLET_STRING] = 'grey30'

    sorted_cjs = sorted(cjtx_stats['coinjoins'], key=lambda txid: cjtx_stats['coinjoins'][txid]['broadcast_time'])

    # Insert coinjoin transactions as nodes (+ attempt to order it in single "line")
    # Note: This does not seem to work perfectly - some ordering is enforced, but rendering engine
    # may ignore to some extent
    prev_cjtxid = None
    for cjtxid in sorted_cjs:
        # Visualize into large connected graph
        dot2.attr('node', rank='same')  # Force same rand for all inserted nodes
        graphviz_insert_cjtxid(cjtx_stats['coinjoins'][cjtxid], dot2)
        # Insert strong linearizing connection between transactions sorted by time of occurence
        if prev_cjtxid:
            dot2.edge(prepare_display_cjtxid(prev_cjtxid), prepare_display_cjtxid(cjtxid),
                      constraint='true', penwidth='10')
        prev_cjtxid = cjtxid

    # Insert edge according to wallet and size of its inputs
    for cjtxid in sorted_cjs:
        graphviz_insert_aggregated_connections(cjtx_stats['coinjoins'], cjtxid, dot2)

    # render resulting graphviz
    save_file = os.path.join(base_path, output_name)
    dot2.render(save_file, view=view_pdf)


def obtain_wallets_info(base_path, load_wallet_info_via_rpc, load_wallet_from_docker_files, all_tx_db):
    wallets_file = os.path.join(base_path, "wallets_info.json")
    wallets_coins_file = os.path.join(base_path, "wallets_coins.json")

    if load_wallet_info_via_rpc:
        print("Loading current wallets info from WasabiWallet RPC")
        # Load wallets info via WasabiWallet RPC
        wallets_info, wallets_coins_all, wallet_coins_unspent = load_wallets_info()
        # Save wallets info into json
        if not READ_ONLY_COINJOIN_TX_INFO:
            print("Saving current wallets into {}".format(wallets_file))
            with open(wallets_file, "w") as file:
                file.write(json.dumps(dict(sorted(wallets_info.items())), indent=4))
    elif load_wallet_from_docker_files:
        print("Loading current wallets info from pre-retrieved coin files wasabi-client-x/[coins.json][keys.json] ")

        # List folders corresponding to wallets
        base_path_wasabi = os.path.join(base_path, 'data')
        files = os.listdir(base_path_wasabi) if os.path.exists(base_path_wasabi) else print(f'Path {base_path_wasabi} does not exist')
        wallets_info = {}   # Information about wallet keys ('listkeys' RPC)
        wallets_coins_all = {}     # Information about wallet coins ('listcoins' RPC)
        # Load information from all wallet directories
        anonymity_by_address = {}
        for file in files:
            target_base_path = os.path.join(base_path_wasabi, file)
            # processs wallets one by one
            if os.path.isdir(target_base_path) and file.startswith('wasabi-client-'):
                print(f'Processing {target_base_path}')
                wallet_name = 'wallet-' + file[len('wasabi-client-'):]

                # Wallet coins (as obtained by 'listcoins' RPC)
                with open(os.path.join(target_base_path, 'coins.json'), "r") as file:
                    wallet_coins = json.load(file)
                    if isinstance(wallet_coins, str) and wallet_coins.lower() == 'timeout':
                        print(f'Loading wallet keys failed with {wallet_coins} for {target_base_path}')
                    else:
                        parsed_coins = anonymity_score.parse_wallet_coins(wallet_name, wallet_coins)
                        for coin in parsed_coins:
                            anonymity_by_address[coin.address] = coin

                    wallets_coins_all[wallet_name] = wallet_coins

                # Wallet addresses (as obtained by 'listkeys' RPC) - now extracted from 'keys.json' file
                with open(os.path.join(target_base_path, 'keys.json'), "r") as file:
                    wallet_keys = json.load(file)
                    if isinstance(wallet_keys, str) and wallet_keys.lower() == 'timeout':
                        print(f'Loading wallet keys failed with {wallet_keys} for {target_base_path}')
                    else:
                        wallets_info[wallet_name] = {addr_data['address']: addr_data for addr_data in wallet_keys}

                # Code below partially reconstruct wallet_addresses from parsed_coins,
                # but now we can read directly 'keys.json' so no longer in use
                # wallet_addresses = []
                # for coin in parsed_coins:
                #     json_representation = {"fullKeyPath": "", "internal": True, "keyState": 2, "label": "",
                #                            "scriptPubKey": "", "pubkey": "", "pubKeyHash": "", "address": coin.address}
                #     wallet_addresses.append(json_representation)
                # wallets_info[wallet_name] = wallet_addresses

        # Sometimes, all_tx_db is not complete for all logged coinjoins
        # Create artificial record for some future time
        highest_known_mine_time = max([all_tx_db[txid]['mine_time'] for txid in all_tx_db.keys()])
        datetime_obj = datetime.strptime(highest_known_mine_time, "%Y-%m-%d %H:%M:%S.%f")
        datetime_obj = datetime_obj + timedelta(minutes=10)
        highest_known_mine_time_next = datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Update coins with information about their lifetime information
        for wallet_name in wallets_coins_all.keys():
            for coin in wallets_coins_all[wallet_name]:
                if coin['txid'] in all_tx_db.keys():
                    coin['block_hash'] = all_tx_db[coin['txid']]['hash']
                    coin['create_time'] = all_tx_db[coin['txid']]['mine_time']
                else:
                    # Synthetic block info in case of missing block
                    coin['block_hash'] = 'synthetic_block'
                    coin['create_time'] = highest_known_mine_time_next

                if 'spentBy' in coin.keys():
                    if coin['spentBy'] in all_tx_db.keys():
                        coin['destroy_time'] = all_tx_db[coin['spentBy']]['mine_time']
                    else:
                        # Synthetic destroy time in case of missing block
                        coin['destroy_time'] = highest_known_mine_time_next


        # Serialize parsed coins for all wallets into 'serialized_annonymity.json' file
        # (as expected by subsequent analysis)
        list_of_coins = list(anonymity_by_address.values())
        encoded = jsonpickle.encode(list_of_coins)
        serialization_path = os.path.join(base_path, 'serialized_annonymity.json')
        with open(serialization_path, "w") as f:
            f.write(encoded)

        # Save wallets info into json
        wallets_file = os.path.join(base_path, 'wallets_info.json')
        print("Saving current wallets into {}".format(wallets_file))
        with open(wallets_file, "w") as file:
            file.write(json.dumps(dict(sorted(wallets_info.items())), indent=4))
        # Save wallets coins into json
        wallets_coins_file = os.path.join(base_path, 'wallets_coins.json')
        print("Saving current wallets coins {}".format(wallets_coins_file))
        with open(wallets_coins_file, "w") as file:
            file.write(json.dumps(dict(sorted(wallets_coins_all.items())), indent=4))
    else:
        print("Loading current wallets info from {}".format(wallets_file))
        print("WARNING: wallets info may be outdated, if wallets were active after information retrieval via RPC "
              "(watch for {} wallet name string in results).".format(UNKNOWN_WALLET_STRING))
        with open(wallets_file, "r") as file:
            wallets_info = json.load(file)
        with open(wallets_coins_file, "r") as file:
            wallets_coins_all = json.load(file)

    print("Wallets info loaded.")

    return wallets_info, wallets_coins_all


def fix_coordinator_wallet_addresses(cjtx_stats):
    """
    Check all wallets with not yet assigned wallet name and assume that all which are 32B long are coming from coordinator.
    :param cjtx_stats:
    :return:
    """
    COORDINATOR_WALLET_ADDR_LEN = 44
    #COORDINATOR_WALLET_ADDR_LEN = 64
    if COORDINATOR_WALLET_STRING not in cjtx_stats['wallets_info']:
        cjtx_stats['wallets_info'][COORDINATOR_WALLET_STRING] = {}
    for cjtxid in cjtx_stats['coinjoins'].keys():
        for index in cjtx_stats['coinjoins'][cjtxid]['inputs'].keys():
            target_addr = cjtx_stats['coinjoins'][cjtxid]['inputs'][index]['address']
            if 'wallet_name' not in cjtx_stats['coinjoins'][cjtxid]['inputs'][index] and len(target_addr) == COORDINATOR_WALLET_ADDR_LEN:
                cjtx_stats['coinjoins'][cjtxid]['inputs'][index]['wallet_name'] = COORDINATOR_WALLET_STRING
                cjtx_stats['wallets_info'][COORDINATOR_WALLET_STRING][target_addr] = {'address': target_addr}
        for index in cjtx_stats['coinjoins'][cjtxid]['outputs'].keys():
            target_addr = cjtx_stats['coinjoins'][cjtxid]['outputs'][index]['address']
            if 'wallet_name' not in cjtx_stats['coinjoins'][cjtxid]['outputs'][index] and len(target_addr) == COORDINATOR_WALLET_ADDR_LEN:
                cjtx_stats['coinjoins'][cjtxid]['outputs'][index]['wallet_name'] = COORDINATOR_WALLET_STRING
                cjtx_stats['wallets_info'][COORDINATOR_WALLET_STRING][target_addr] = {'address': target_addr}

    # Rebuild full wallet mapping
    cjtx_stats['address_wallet_mapping'] = build_address_wallet_mapping(cjtx_stats)
    return cjtx_stats


def load_prison_data(cjtx_stats, base_path):
    prison_file = os.path.join(base_path, "WalletWasabi", "Backend", "WabiSabi", "Prison.txt")
    if not os.path.exists(prison_file):
        prison_file = os.path.join(base_path, "data", "wasabi-backend", "backend", "WabiSabi", "Prison.txt")
    items_in_prison = 0

    #detect prison version (<2.0.4 is different than >=2.0.4)

    if os.path.exists(prison_file):
        with open(prison_file, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                prison_log = {}
                prison_log['type'] = CJ_LOG_TYPES.UTXO_IN_PRISON.name
                prison_log['timestamp'] = datetime.fromtimestamp(int(row[0])).strftime("%Y-%m-%d %H:%M:%S.%f")
                prison_log['utxo'] = row[1]
                prison_log['reason'] = row[2]
                prison_log['round_id'] = ''
                prison_log['value'] = ''
                prison_log['adv_reason'] = ''

                if prison_log['reason'] == 'RoundDisruption':
                    prison_log['value'] = row[3]
                    prison_log['round_id'] = row[4]
                    prison_log['adv_reason'] = row[5]
                elif prison_log['reason'] == 'Cheating':
                    prison_log['round_id'] = row[3]
                else:
                    print('Unknown prison reason {}'.format(prison_log['reason']))

                cjtx_stats['rounds'][prison_log['round_id']]['logs'].append(prison_log)
                items_in_prison += 1

        print('Total {} records found in prison'.format(items_in_prison))
    else:
        print('WARNING: No prison file found at {}'.format(prison_file))
    return cjtx_stats


def load_anonscore_data(cjtx_stats, base_path):
    anonscore_file = os.path.join(base_path, "serialized_annonymity.json")

    if os.path.exists(anonscore_file):
        anon_scores = anonymity_score.deserialize_to_list(anonscore_file)

        anonscore_1 = 0
        anonscore_gt1 = 0
        # Pair anon score to outputs
        for item in anon_scores:
            if isinstance(item, anonymity_score.CoinWithAnonymity):
                item = item.as_dict()
            txid = item['txid']
            #out_index = str(item['index'])
            out_index = item['index']
            anon_score = item['annon_score']
            if anon_score > 1:
                anonscore_gt1 = anonscore_gt1 + 1
            else:
                anonscore_1 = anonscore_1 + 1

            if txid in cjtx_stats['coinjoins'].keys():
                # Some older inputs were having out_index key as string ('2' instead of 2), check
                if out_index in cjtx_stats['coinjoins'][txid]['outputs'].keys():
                    cjtx_stats['coinjoins'][txid]['outputs'][out_index]['anon_score'] = anon_score
                else:
                    cjtx_stats['coinjoins'][txid]['outputs'][str(out_index)]['anon_score'] = anon_score
            else:
                if isinstance(txid, str):
                    print(f'WARNING: Processing anon scores - tx {txid} not found in coinjoin list')
                else:
                    print('Strange, isinstance(txid, str) is false')

        print('Total {} UTXOs with only base anonscore (=1) and {} with better than base anonscore found.'.format(anonscore_1, anonscore_gt1))
    else:
        print('Anonscore statistics not found in {}'.format(anonscore_file))
    return cjtx_stats


def list_files(folder_path, suffix):
    json_files = [os.path.join(root, file) for root, dirs, files in os.walk(folder_path) for file in files if file.endswith(suffix)]
    return json_files


def load_rawtx_database(base_tx_path):
    tx_db = {}
    files = list_files(base_tx_path, '.json')
    for tx_file in files:
        with open(tx_file, "r") as file:
            raw_tx = json.load(file)
            result = run_command(
                '{} -regtest decoderawtransaction \"{}\" '.format(BTC_CLI_PATH, raw_tx['txRawHex']), False)
            tx_info = result.stdout
            parsed_tx_info = json.loads(tx_info)
            tx_db[parsed_tx_info['txid']] = tx_info

            # # Create a Transaction object from the raw hex
            # tx = Transaction.parse(raw_tx['txRawHex'], False, 'regtest')
            # tx_db[tx.txid] = tx.as_json()
    return tx_db


def load_tx_database_from_btccore(base_tx_path):
    tx_db = {}
    files = list_files(base_tx_path, '.json')
    for tx_file in files:  # Each file corresponds to whole block - may be multiple transactions
        with (open(tx_file, "r") as file):
            block_txs = json.load(file)
            for tx_info in block_txs['tx']:
                datetime_obj = datetime.utcfromtimestamp(block_txs['time'])
                tx_info['mine_time'] = datetime_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                tx_db[tx_info['txid']] = tx_info


            # # Create a Transaction object from the raw hex
            # tx = Transaction.parse(raw_tx['txRawHex'], False, 'regtest')
            # tx_db[tx.txid] = tx.as_json()
    return tx_db


def remove_link_between_inputs_and_outputs(coinjoins):
    for txid in coinjoins.keys():
        for index, input in coinjoins[txid]['inputs'].items():
            if 'spending_tx' in coinjoins[txid]['inputs'][index]:
                coinjoins[txid]['inputs'][index].pop('spending_tx')
        for index, output in coinjoins[txid]['outputs'].items():
            if 'spend_by_txid' in coinjoins[txid]['outputs'][index]:
                coinjoins[txid]['outputs'][index].pop('spend_by_txid')



def compute_link_between_inputs_and_outputs(coinjoins, sorted_cjs_in_scope):
    """
    Compute backward and forward connection between all transactions in sorted_cjs_in_scope list. As a result,
    for every input, 'spending_tx' record is inserted pointing to transaction and index of its output spent.
    For every output, 'spend_by_txid' is inserted pointing to transaction and its index which spents this output.
    :param coinjoins: structure with coinjoins
    :param sorted_cjs_in_scope: list of cj transactions to be used for calculating connections. Can be subset of
    coinjoins parameter - in such case, not all inputs and outputs will have 'spending_tx' and spend_by_txid' filled.
    :return: Updated structure with coinjoins
    """
    all_outputs = {}
    # Obtain all outputs as (address, value) tuples
    for tx_index in range(0, len(sorted_cjs_in_scope)):
        txid = sorted_cjs_in_scope[tx_index]
        for index, output in coinjoins[txid]['outputs'].items():
            all_outputs[output['address']] = (txid, index, output)  # (txid, output['address'], output['value'])

    # Check if such combination is in inputs of any other transaction in the scope
    for tx_index in range(0, len(sorted_cjs_in_scope)):
        txid = sorted_cjs_in_scope[tx_index]
        for index, input in coinjoins[txid]['inputs'].items():
            if input['address'] in all_outputs.keys() and input['value'] == all_outputs[input['address']][2]['value']:
                # we found corresponding input, mark it as used (tuple (txid, index))
                # Set also corresponding output 'spend_by_txid'
                target_output = all_outputs[input['address']]
                coinjoins[target_output[0]]['outputs'][target_output[1]]['spend_by_txid'] = (txid, index)
                coinjoins[txid]['inputs'][index]['spending_tx'] = (target_output[0], target_output[1])

    return coinjoins


def optimize_wallets_info(cjtx_stats):
    """
    Check if 'wallets_info' structure is realized as list. If yes, then change to more optimal dictionary
    :param cjtx_stats:
    :return: True if any optimization happened, False otherwise
    """
    optimized = False
    for wallet_name in cjtx_stats['wallets_info'].keys():
        if type(cjtx_stats['wallets_info'][wallet_name]) is list:
            # change to dictionary
            optimized = True
            optimized_as_dict = {addr_data['address']: addr_data for addr_data in cjtx_stats['wallets_info'][wallet_name]}
            cjtx_stats['wallets_info'][wallet_name] = optimized_as_dict
    return optimized


def optimize_txvalue_info(cjtx_stats):
    """
    Check if tx value is realized as an integer. If not, change from float to integer
    :param cjtx_stats:
    :return: True if any optimization happened, False otherwise
    """
    optimized = False
    float_value_detected = False
    for txid in cjtx_stats['coinjoins'].keys():
        for input in cjtx_stats['coinjoins'][txid]['inputs'].keys():
            if type(cjtx_stats['coinjoins'][txid]['inputs'][input]['value']) is float:
                float_value_detected = True
                break
        else:
            continue
        break

    if float_value_detected:
        # Change all computed values to integer sats instead of float (imprecise)
        optimized = True
        for txid in cjtx_stats['coinjoins'].keys():
            for input in cjtx_stats['coinjoins'][txid]['inputs'].keys():
                assert type(cjtx_stats['coinjoins'][txid]['inputs'][input]['value']) is float, 'non-float value'
                cjtx_stats['coinjoins'][txid]['inputs'][input]['value'] \
                    = int(cjtx_stats['coinjoins'][txid]['inputs'][input]['value'] * SATS_IN_BTC)
            for output in cjtx_stats['coinjoins'][txid]['outputs'].keys():
                assert type(cjtx_stats['coinjoins'][txid]['outputs'][output]['value']) is float, 'non-float value'
                cjtx_stats['coinjoins'][txid]['outputs'][output]['value'] \
                    = int(cjtx_stats['coinjoins'][txid]['outputs'][output]['value'] * SATS_IN_BTC)

    return optimized


def process_experiment(args):
    base_path = args[0]
    save_figs = args[1]
    WASABIWALLET_DATA_DIR = base_path
    print(f'INPUT PATH: {base_path}')
    save_file = os.path.join(WASABIWALLET_DATA_DIR, "coinjoin_tx_info.json")
    if LOAD_TXINFO_FROM_FILE:
        # Load parsed coinjoin transactions again
        with open(save_file, "r") as file:
            cjtx_stats = json.load(file)

        # Build mapping between address and controlling wallet
        if 'address_wallet_mapping' not in cjtx_stats.keys():
            cjtx_stats['address_wallet_mapping'] = build_address_wallet_mapping(cjtx_stats)
            if not READ_ONLY_COINJOIN_TX_INFO:
                with open(save_file, "w") as file:
                    file.write(json.dumps(dict(sorted(cjtx_stats.items())), indent=4))
    else:
        # Load transaction info from serialized files
        if LOAD_TXINFO_FROM_DOCKER_FILES:
            # Load tx from logs stored by Bitcoin fullnode - all transactions available
            tx_path = os.path.join(base_path, 'data', 'btc-node')
            RAW_TXS_DB = load_tx_database_from_btccore(tx_path)

            # Load tx from logs stored by wasabi coordinator - big limitation - only the coinjoin transaction
            # are stored, not their inputs => unused now
            # tx_path = os.path.join(base_path, 'data', 'wasabi-backend', 'WabiSabi', 'CoinJoinTransactions')
            # RAW_TXS_DB = load_rawtx_database(tx_path)

        # Load wallets info
        cjtx_stats = {}
        cjtx_stats['wallets_info'], cjtx_stats['wallets_coins'] = (
            obtain_wallets_info(WASABIWALLET_DATA_DIR, LOAD_WALLETS_INFO_VIA_RPC, LOAD_TXINFO_FROM_DOCKER_FILES, RAW_TXS_DB))

        # Parse conjoins from logs
        coord_input_file = os.path.join(WASABIWALLET_DATA_DIR, 'WalletWasabi/Backend/Logs.txt')
        if not os.path.exists(coord_input_file):  # if not found, try dockerized path
            coord_input_file = os.path.join(WASABIWALLET_DATA_DIR, 'data', 'wasabi-backend', 'backend', 'Logs.txt')
        cjtx_stats['coinjoins'] = parse_backend_coinjoin_logs(coord_input_file, RAW_TXS_DB)

        # Build mapping between address and controlling wallet
        cjtx_stats['address_wallet_mapping'] = build_address_wallet_mapping(cjtx_stats)

        # Assume coordinator for all 32B addresses
        if ASSUME_COORDINATOR_WALLET:
            cjtx_stats = fix_coordinator_wallet_addresses(cjtx_stats)

        # Analyze error states
        if PARSE_ERRORS:
            coord_input_file = os.path.join(WASABIWALLET_DATA_DIR, 'WalletWasabi/Backend/Logs.txt')
            if not os.path.exists(coord_input_file):  # if not found, try dockerized path
                coord_input_file = os.path.join(WASABIWALLET_DATA_DIR, 'data', 'wasabi-backend', 'backend', 'Logs.txt')
            parse_coinjoin_errors(cjtx_stats, coord_input_file)

        # Save parsed coinjoin transactions info into json
        if not READ_ONLY_COINJOIN_TX_INFO:
            with open(save_file, "w") as file:
                file.write(json.dumps(dict(sorted(cjtx_stats.items())), indent=4))

    # Older version of files used slow wallet address storage and float for sats, optimize and save if changed
    updated_cjtx_stats = optimize_wallets_info(cjtx_stats)
    updated_cjtx_stats |= optimize_txvalue_info(cjtx_stats)
    if updated_cjtx_stats is True:
        # Save optimized coinjoin transactions info into json
        if not READ_ONLY_COINJOIN_TX_INFO:
            with open(save_file, "w") as file:
                file.write(json.dumps(dict(sorted(cjtx_stats.items())), indent=4))

    # Assume coordinator for all 32B addresses
    if ASSUME_COORDINATOR_WALLET:
        cjtx_stats = fix_coordinator_wallet_addresses(cjtx_stats)
        print('Potential coordinator addresses recomputed, saving...', end='')
        if not READ_ONLY_COINJOIN_TX_INFO:
            with open(save_file, "w") as file:
                file.write(json.dumps(dict(sorted(cjtx_stats.items())), indent=4))
        print('done')

    # Compute coinjoin stats
    if RETRIEVE_TRANSACTION_INFO:
        print('Fetching transactions info for analysis...')
        result = ludwig.retrieve_txs(list(cjtx_stats['coinjoins'].keys()))
        for cjtxid in result.keys():
            cjtx_stats['coinjoins'][cjtxid]['raw_tx'] = jsonpickle.encode(result[cjtxid])
        if len(result) > 0:  # save if any update was done
            print('Saving updated transaction info (raw_tx)...', end='')
            if not READ_ONLY_COINJOIN_TX_INFO:
                with open(save_file, "w") as file:
                    file.write(json.dumps(dict(sorted(cjtx_stats.items())), indent=4))
            print('done')

    if LOAD_COMPUTED_TRANSACTION_INFO:
        # if available, use already computed tx entropy analysis
        #   (expected files 'tx_analysis_cjtxid'.json in 'tx' folder)
        tx_info_path = os.path.join(WASABIWALLET_DATA_DIR, "tx")
        if os.path.isdir(tx_info_path):
            for cjtxid in cjtx_stats['coinjoins'].keys():
                tx_analysis_path = os.path.join(tx_info_path, "tx_analysis_{}.json".format(cjtxid))
                if os.path.isfile(tx_analysis_path):
                    # Create analysis section of not yet
                    if 'analysis' not in cjtx_stats['coinjoins'][cjtxid].keys():
                        cjtx_stats['coinjoins'][cjtxid]['analysis'] = {}
                    # Load existing tx analysis
                    with open(tx_analysis_path, "r") as file:
                        tx_analysis = json.load(file)
                    cjtx_stats['coinjoins'][cjtxid]['analysis'].update(tx_analysis[cjtxid])

            print('Saving updated transaction info (tx_analysis)...', end='')
            if not READ_ONLY_COINJOIN_TX_INFO:
                with open(save_file, "w") as file:
                    file.write(json.dumps(dict(sorted(cjtx_stats.items())), indent=4))
            print('done')

    # Compute coinjoin stats
    if COMPUTE_COINJOIN_STATS or FORCE_COMPUTE_COINJOIN_STATS:
        if PARALLELIZE_COMPUTE_COINJOIN_STATS:
            def analyze_tx(cjtx: dict):
                tx_result = ludwig.analyze_txs_from_prefetched_simple(cjtx[0])
                for txid in cjtx[0].keys():
                    with open('{}/tx_analysis_{}.json'.format(WASABIWALLET_DATA_DIR, txid), "w") as file:
                        file.write(json.dumps(tx_result, indent=4))
                return tx_result

            to_analyze = []
            for cjtxid in cjtx_stats['coinjoins'].keys():
                if FORCE_COMPUTE_COINJOIN_STATS or 'analysis' not in cjtx_stats['coinjoins'][cjtxid]:
                    tx = jsonpickle.decode(cjtx_stats['coinjoins'][cjtxid]['raw_tx'])
                    to_analyze.append([{cjtxid: tx}])

            with tqdm(total=len(to_analyze)) as progress:
                for result in ThreadPool(NUM_THREADS).imap(analyze_tx, to_analyze):
                    progress.update(1)
                    for txid in result.keys():
                        cjtx_stats['coinjoins'][txid]['analysis'] = result[txid]
        else:
            for cjtxid in cjtx_stats['coinjoins'].keys():
                if FORCE_COMPUTE_COINJOIN_STATS or 'analysis' not in cjtx_stats['coinjoins'][cjtxid]:
                    tx = jsonpickle.decode(cjtx_stats['coinjoins'][cjtxid]['raw_tx'])
                    if tx is not None:
                        result = ludwig.analyze_txs_from_prefetched_simple({cjtxid: tx})

                        # insert analysis to cj info
                        cjtx_stats['coinjoins'][cjtxid]['analysis'] = result[cjtxid]

                # check for potentially incorrect mapping (as we know ground truth)
                if 'analysis' in cjtx_stats['coinjoins'][cjtxid]:
                    for link in cjtx_stats['coinjoins'][cjtxid]['analysis']['processed']['deterministic_links']:
                        if cjtx_stats['address_wallet_mapping'][link[0][0]] != cjtx_stats['address_wallet_mapping'][link[1][0]]:
                            print('ERROR: {} Deterministic link mismatch {} to {}'.format(cjtxid, cjtx_stats['address_wallet_mapping'][link[0][0]],
                                                                                          cjtx_stats['address_wallet_mapping'][link[1][0]]))
        if not READ_ONLY_COINJOIN_TX_INFO:
            with open(save_file, "w") as file:
                file.write(json.dumps(dict(sorted(cjtx_stats.items())), indent=4))
    print('Entropy analysis: {} txs out of {} successfully analyzed'.format(sum([1 for cjtxid in cjtx_stats['coinjoins'].keys() if 'analysis' in cjtx_stats['coinjoins'][cjtxid] and cjtx_stats['coinjoins'][cjtxid]['analysis']['processed']['successfully_analyzed'] is True]), len(cjtx_stats['coinjoins'].keys())))

    client_input_path = os.path.join(WASABIWALLET_DATA_DIR, 'WalletWasabi', 'Client')
    #parse_client_coinjoin_logs(cjtx_stats, client_input_path)
    # TODO: load client logs from multiple directories 'wasabi-client-x'

    load_prison_data(cjtx_stats, WASABIWALLET_DATA_DIR)

    load_anonscore_data(cjtx_stats, WASABIWALLET_DATA_DIR)

    if not READ_ONLY_COINJOIN_TX_INFO:
        with open(save_file, "w") as file:
            file.write(json.dumps(dict(sorted(cjtx_stats.items())), indent=4))

    # Establish linkage between inputs and outputs for further analysis
    remove_link_between_inputs_and_outputs(cjtx_stats['coinjoins'])
    compute_link_between_inputs_and_outputs(cjtx_stats['coinjoins'],
                                            [cjtxid for cjtxid in cjtx_stats['coinjoins'].keys()])

    # Analyze various coinjoins statistics.
    # Plot only if required (time-consuming, and not thread safe)
    cjplots = CoinJoinPlots(plt)  # Always provide coinjoin plots, but empty by default (no graph generation)
    if save_figs:
        cjplots.new_figure()  # If required, generate and set graph Axes for generation
    #short_exp_name = os.path.basename(WASABIWALLET_DATA_DIR)
    short_exp_name = ''
    analyze_coinjoin_stats(cjtx_stats, WASABIWALLET_DATA_DIR, cjplots, short_exp_name)
    if save_figs:  # If required, render and save visual graph
        experiment_name = os.path.basename(base_path)
        fig_save_file = os.path.join(base_path, f"coinjoin_stats{GLOBAL_IMG_SUFFIX}.png")
        fig_save_file = cjplots.savefig(fig_save_file, experiment_name)
        cjplots.clear()
        print(f'Basic coinjoins statistics saved into {fig_save_file}')

    # Save updated information with analysis results
    if not READ_ONLY_COINJOIN_TX_INFO and SAVE_ANALYTICS_TO_FILE:
        with open(save_file, "w") as file:
            file.write(json.dumps(dict(sorted(cjtx_stats.items())), indent=4))

    # Visualize coinjoins in agrregated fashion
    if GENERATE_COINJOIN_GRAPH_BLIND:
        visualize_coinjoins_aggregated(cjtx_stats, WASABIWALLET_DATA_DIR, 'coinjoin_graph_aggregated', False)

    # Visualize coinjoins in verbose mode (possibly only subset of coinjoins visualized to prevent graphviz overload)
    to_visualize = dict(list(cjtx_stats['coinjoins'].items())[:])
    #to_visualize = dict(list(cjtx_stats['coinjoins'].items())[:64])
    #to_visualize = dict(list(cjtx_stats['coinjoins'].items())[100:121])
    cjtx_stats['coinjoins'] = to_visualize
    if GENERATE_COINJOIN_GRAPH:
        print('Going to render coinjoin relations graph (may take several minutes if larger number of coinjoins '
              'are visualized) ... ', end='')
        visualize_coinjoins(cjtx_stats, WASABIWALLET_DATA_DIR, 'coinjoin_graph', False)
        print(' done')

    print('All done.')

    return cjtx_stats


def get_experiments_base_paths(base_path: str):
    """
    Search on-level subfolders for provided base_path for folders with experiments.
    Returns it as a list of paths.
    :param base_path: base folder with experiments as subfolders
    :return: list of folders which cointain recognized experiments
    """
    files = []
    if os.path.exists(base_path):
        files = os.listdir(base_path)
    else:
        print('Path {} does not exists'.format(base_path))

    # Search all paths to be processed
    paths_to_process = []
    for file in files:
        target_exp_base_path = os.path.join(base_path, file)
        if os.path.isdir(target_exp_base_path):
            if (os.path.exists(os.path.join(target_exp_base_path, 'WalletWasabi'))
                    or os.path.exists(os.path.join(target_exp_base_path, 'data'))):
                paths_to_process.append(target_exp_base_path)

    return paths_to_process


def process_multiple_experiments(base_path: str, save_figs=False, num_threads=-1):
    print(f'Starting analysis of multiple folders in {base_path}')

    results = {}  # Aggregated results

    # Collect paths with experiments
    paths_to_process = get_experiments_base_paths(base_path)

    if num_threads == -1:  # If num_threads argument is not provided, create separate threat for every experiment
        num_threads = len(paths_to_process)

    # Execute process_experiment in parallel
    with tqdm(total=len(paths_to_process)) as progress:
        # Mathplotlib is not threadsafe, therefore generate new figure and save results only in single-thread analysis
        assert not save_figs or num_threads == 1, 'Mathplotlib is not threadsafe, cannot save figures while threading'
        params = []  # Prepare parameters for each thread
        for path in paths_to_process:
            params.append((path, save_figs))
        for result in ThreadPool(num_threads).imap(process_experiment, params):
            progress.update(1)
            print(result['analysis']['path'])
            results[result['analysis']['path']] = result

    return results


def gather_paths_with_experiments(base_path):
    files = []
    if os.path.exists(base_path):
        files = os.listdir(base_path)
    else:
        print('Path {} does not exists'.format(base_path))

    # Search all paths to be processed
    paths_to_process = []
    for file in files:
        target_base_path = os.path.join(base_path, file)
        if os.path.isdir(target_base_path):
            if (os.path.exists(os.path.join(target_base_path, 'WalletWasabi'))
                    or os.path.exists(os.path.join(target_base_path, 'data'))):
                paths_to_process.append(target_base_path)

    return paths_to_process


def analyze_multiple_experiments(results: dict, base_path: str):
    print(f'Starting analysis of multiple experiments {results.keys()}')

    analyze_aggregated_coinjoin_stats(results, base_path)

    return results


def visualize_aggregated_graphs(experiment_paths_sorted, graphs, base_path: str, collate_same_num_wallets: bool):
    for graph_type in graphs:
        counted_wallets = Counter([num for path, num in experiment_paths_sorted])
        num_wallets, max_num_wallets = counted_wallets.most_common(1)[0]
        cjplots = CoinJoinPlots(plt)
        if 3 <= max_num_wallets <= 5:
            # Trying to put same number of wallets on same row
            num_columns = max_num_wallets
            num_rows = len(counted_wallets.keys())
        else:
            # Not trying to put same number of wallets on same row
            num_columns = 3
            num_rows = math.ceil(len(experiment_paths_sorted) / float(num_columns))  # enough rows to capture all graphs
        cjplots.new_figure(num_rows, num_columns, False)  # Create default graph (4x4), do NOT assign default layout

        # Perform processing for each desired analysis separately (note: not optimal as we are repeating analyses
        # for all other unused as well. But plotting, not analysis, is the most time-consuming operation.)
        axis_index = -1
        last_num_wallets = -1
        for experiment_path, num_wallets in experiment_paths_sorted:
            print(f'INPUT PATH: {experiment_path}')
            data_file = os.path.join(experiment_path, "coinjoin_tx_info.json")
            # Load parsed coinjoin transactions again
            with open(data_file, "r") as file:
                cjtx_stats = json.load(file)

            # Set targeted named subplot to next unused axes
            # Case 1: Collating results for same number of wallets (collate_same_num_wallets == True), use every subplot
            # Case 2: Subplots separately (collate_same_num_wallets == False) - try to put results for same number of wallets into same row
            # Case 3: not trying to allign and not collating - every subplot is used
            if collate_same_num_wallets:  # Case 1
                if num_wallets != last_num_wallets:
                    # Go to next subplot only if change in number of wallets is detected
                    axis_index += 1
                    last_num_wallets = num_wallets
            else:
                # Case 2: Trying to have same num wallets in the row
                if max_num_wallets == num_columns and num_wallets != last_num_wallets:
                    # Skip all remaining subplots in the row
                    if axis_index < 0:
                        axis_index = 0
                    else:
                        current_row = math.floor(axis_index / num_columns)
                        axis_index = (current_row + 1) * num_columns
                    last_num_wallets = num_wallets
                else:
                    # Case 3: Got to new subplot for every experiment (not trying to allign and not collating)
                    axis_index += 1

            assert axis_index < len(cjplots.axes), f'Invalid axis index: {axis_index}, max is {len(cjplots.axes) - 1}'
            setattr(cjplots, graph_type, cjplots.axes[axis_index])

            # Perform analysis, which will fill only desired analysis into yet unused subgraph
            analyze_coinjoin_stats(cjtx_stats, experiment_path, cjplots)

        # Save figure will all subgraphs
        experiment_name = os.path.basename(base_path)
        fig_save_file = os.path.join(base_path, f"cj_singletype_{graph_type}_{'collated_' if collate_same_num_wallets else ''}stats{GLOBAL_IMG_SUFFIX}.png")
        fig_save_file = cjplots.savefig(fig_save_file, experiment_name)
        cjplots.clear()
        print(f'Aggregated coinjoins statistics saved into {fig_save_file}')


def generate_aggregated_visualization(paths_to_process: list):
    """
    Generate separate graph picture from every analyzed property separately
    (specific graph from all experiments in one picture)
    :param paths_to_process:
    :param base_path:
    :return:
    """

    # Load all experiment paths found on paths_to_process
    for base_path in paths_to_process:
        experiment_paths = get_experiments_base_paths(base_path)

        # Sort paths based on some property (e.g., number of wallets incrementally)
        reg = 'static-(?P<num_wallets>[0-9]+)-'
        experiment_paths_temp = []
        for experiment_path in experiment_paths:
            match = re.search(reg, experiment_path)
            assert match, f'Failed to match regular expression \'{reg}\' to experiment_path \'{experiment_path}\''
            if match:
                num_wallets = int(match.groupdict()['num_wallets'])
                experiment_paths_temp.append((experiment_path, num_wallets))
        experiment_paths_sorted = sorted(experiment_paths_temp, key=lambda x: x[1])
        assert len(experiment_paths_sorted) == len(experiment_paths)

        # Visualize data batched based on the number of wallets (only selected types of graphs which will not overload the plot)
        graphs = ['ax_num_inoutputs', 'ax_available_utxos', 'ax_utxo_entropy_from_outputs', 'ax_utxo_entropy_from_outputs_inoutsize', 'ax_utxo_denom_anonscore']
        visualize_aggregated_graphs(experiment_paths_sorted, graphs, base_path, True)

        # Visualize data of given type into single graph
        graphs = ['ax_num_inoutputs', 'ax_available_utxos', 'ax_wallets_distrib', 'ax_utxo_provided', 'ax_anonscore_distrib_wallets',
                  'ax_anonscore_from_outputs', 'ax_utxo_entropy_from_outputs', 'ax_utxo_denom_anonscore',
                  'ax_utxo_entropy_from_outputs_inoutsize', 'ax_initial_final_utxos']
        visualize_aggregated_graphs(experiment_paths_sorted, graphs, base_path, False)


class AnalysisType(Enum):
    COLLECT_COINJOIN_DATA_LOCAL = 1
    COMPUTE_COINJOIN_TXINFO_REMOTE = 2
    ANALYZE_COINJOIN_DATA_LOCAL = 3
    COLLECT_COINJOIN_DATA_LOCAL_DOCKER = 4


if __name__ == "__main__":
    PROFILE_PERFORMANCE = False

    # Analysis type
    #cfg = AnalysisType.COLLECT_COINJOIN_DATA_LOCAL
    #cfg = AnalysisType.COLLECT_COINJOIN_DATA_LOCAL_DOCKER
    cfg = AnalysisType.ANALYZE_COINJOIN_DATA_LOCAL
    #cfg = AnalysisType.COMPUTE_COINJOIN_TXINFO_REMOTE

    print('Analysis configuration: {}'.format(cfg.name))

    RAW_TXS_DB = {}

    if cfg == AnalysisType.COLLECT_COINJOIN_DATA_LOCAL:
        # Extract all info from running wallet and fullnode

        # If True, preprocessed files extracted from dockerized instances are assumed.
        # If False, local instance with Wasabi wallet is assumed.
        LOAD_TXINFO_FROM_DOCKER_FILES = False
        # If True, existence of coinjoin_tx_info.json is assumed and all data are read from it.
        # Bitcoin Core/Wallet RPC is not required
        LOAD_TXINFO_FROM_FILE = False
        # If True, info is obtained via RPC from WalletWasabi client, otherwise existence of wallets_info.json
        # with wallet information is assumed and loaded from
        LOAD_WALLETS_INFO_VIA_RPC = True
        # If True, coordinator logs are parsed for various error logs
        PARSE_ERRORS = True
        # If True, transaction info is retrieved from Bitcoin Core via RPC
        RETRIEVE_TRANSACTION_INFO = True
        # If True, subfolder 'tx' is searched for existence of already computed transaction analysis results
        # (tx_analysis_*.json). Used when tx analysis is performed externally and not all at once
        # (e.g., with high paralellism)
        LOAD_COMPUTED_TRANSACTION_INFO = False
        # If True, then coinjoin_tx_info.json is not written to with intermediate updates
        READ_ONLY_COINJOIN_TX_INFO = False
        # If True, coinjoin analysis (Boltzmann links etc.) is performed if not yet done for given tx
        COMPUTE_COINJOIN_STATS = False
        # If True, recomputation of coinjoin analysis is performed even if already computed before
        FORCE_COMPUTE_COINJOIN_STATS = False
        # If True, multiple analysis threads (NUM_THREADS) are started for analysis
        PARALLELIZE_COMPUTE_COINJOIN_STATS = False
        # If True, graphviz tx graph is executed
        GENERATE_COINJOIN_GRAPH = False
        # If True, final analytics are saved into base coinjoin_tx_info.json file
        SAVE_ANALYTICS_TO_FILE = True
        # Base start path with data for processing (WalletWasabi and other folders)
        target_base_path = 'c:\\Users\\xsvenda\\AppData\\Roaming\\'
    elif cfg == AnalysisType.COLLECT_COINJOIN_DATA_LOCAL_DOCKER:
        # Extract info from static files previously created and collected from dockerized instances
        LOAD_TXINFO_FROM_DOCKER_FILES = True
        LOAD_TXINFO_FROM_FILE = False
        LOAD_WALLETS_INFO_VIA_RPC = False
        PARSE_ERRORS = True
        RETRIEVE_TRANSACTION_INFO = False
        LOAD_COMPUTED_TRANSACTION_INFO = False
        READ_ONLY_COINJOIN_TX_INFO = False
        COMPUTE_COINJOIN_STATS = False
        FORCE_COMPUTE_COINJOIN_STATS = False
        PARALLELIZE_COMPUTE_COINJOIN_STATS = False
        GENERATE_COINJOIN_GRAPH = False
        SAVE_ANALYTICS_TO_FILE = True
        target_base_path = 'c:\\Users\\xsvenda\\AppData\\Roaming\\'
    elif cfg == AnalysisType.ANALYZE_COINJOIN_DATA_LOCAL:
        # Just recompute analysis

        LOAD_TXINFO_FROM_DOCKER_FILES = False
        LOAD_TXINFO_FROM_FILE = True
        LOAD_WALLETS_INFO_VIA_RPC = False
        PARSE_ERRORS = False

        RETRIEVE_TRANSACTION_INFO = False

        LOAD_COMPUTED_TRANSACTION_INFO = True
        READ_ONLY_COINJOIN_TX_INFO = False

        COMPUTE_COINJOIN_STATS = False
        FORCE_COMPUTE_COINJOIN_STATS = False
        PARALLELIZE_COMPUTE_COINJOIN_STATS = False

        GENERATE_COINJOIN_GRAPH = True
        SAVE_ANALYTICS_TO_FILE = True

        target_base_path = 'c:\\Users\\xsvenda\\AppData\\Roaming\\'
    elif cfg == AnalysisType.COMPUTE_COINJOIN_TXINFO_REMOTE:
        # Compute only time-consuming transaction entropy analysis from pre-retrieved transactions

        LOAD_TXINFO_FROM_DOCKER_FILES = False
        LOAD_TXINFO_FROM_FILE = True
        LOAD_WALLETS_INFO_VIA_RPC = False
        GENERATE_COINJOIN_GRAPH = False
        PARSE_ERRORS = False

        RETRIEVE_TRANSACTION_INFO = False

        LOAD_COMPUTED_TRANSACTION_INFO = True
        READ_ONLY_COINJOIN_TX_INFO = False

        COMPUTE_COINJOIN_STATS = True
        FORCE_COMPUTE_COINJOIN_STATS = True
        PARALLELIZE_COMPUTE_COINJOIN_STATS = True
        NUM_THREADS = 100

        GENERATE_COINJOIN_GRAPH = False

        target_base_path = '/home/xsvenda/coinjoin/'

    GENERATE_COINJOIN_GRAPH = False
    GENERATE_COINJOIN_GRAPH_BLIND = False
    GENERATE_COINJOIN_GRAPH_LINEAR = False
    #FORCE_COMPUTE_COINJOIN_STATS = True
    #COMPUTE_COINJOIN_STATS = True
    #PARALLELIZE_COINJOIN_STATS = False
    #LOAD_WALLETS_INFO_VIA_RPC = False
    #LOAD_TXINFO_FROM_FILE = False
    ASSUME_COORDINATOR_WALLET = False

    #READ_ONLY_COINJOIN_TX_INFO = True
    SAVE_ANALYTICS_TO_FILE = False

    #target_base_path = 'c:\\Users\\xsvenda\\AppData\\Roaming\\'
    #target_base_path = '/home/xsvenda/coinjoin/'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol4\\20230930_1000Round_1parallel_max7inputs_10wallets_1Msats\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol4\\20230930_148Round_1parallel_max10inputs_10wallets_1Msats\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol4\\20230930_460Round_1parallel_max10inputs_10wallets_1Msats\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol4\\20230929_2000Round_1parallel_max4inputs_10wallets_100ksats\\second_part\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol4\\20230929_2000Round_1parallel_max4inputs_10wallets_100ksats\\first_part\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol4\\20230928_2000Round_1parallel_max4inputs_10wallets_0.2btc\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol4\\20231001_1000Round_1parallel_max5inputs_5ans5wallets_10M_1Msats\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol4\\20231002_1000Round_1parallel_max20inputs_5and5wallets_10M_1Msats\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol4\\20231003_500Round_1parallel_max100inputs_30wallets_10Msats\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol4\\20231004_500Round_1parallel_max100inputs_30wallets_10Msats\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231007_2000Rounds_1parallel_max4inputs_10wallets\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231007_147Rounds_1parallel_max4inputs_10wallets_inJailForFailures\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231008_2000Rounds_1parallel_max5inputs_10wallets\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231013_500Rounds_1parallel_max20inputs_10wallets_5x10Msats\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231014_1000Rounds_1parallel_max10inputs_10wallets_5x10Msats\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231015_1000Rounds_1parallel_max15inputs_10wallets_5x10Msats\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231016_1000Rounds_1parallel_max25inputs_10wallets_5x10Msats\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231017_1000Rounds_1parallel_max30inputs_10wallets_5x10Msats\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231019_1000Rounds_1parallel_max10inputs_10wallets_5x10Msats_noPrison\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231020_1000Rounds_1parallel_max40inputs_10wallets_5x10Msats_noPrison\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231017_1000Rounds_1parallel_max30inputs_10wallets_5x10Msats\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231018_1000Rounds_1parallel_max30inputs_10wallets_5x10Msats_noPrison\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol6\\test\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol6\\two-party_2023-11-21_16-30\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231023_11Rounds_1parallel_max6inputs_10wallets_5x10Msats\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol8\\disbalanced-delayed-20_2023-11-23_14-47\\'

    # Two experiments with exactly same settings
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol8\\pareto-delayed-50_2023-11-23_22-28\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol8\\pareto-delayed-50_2023-11-24_15-19\\'

    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol8\\2023-11-24_17-09_normal-static-150\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol8\\2023-11-25_09-35_pareto-delayed-150\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol8\\2023-11-25_18-34_normal-delayed-150\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol8\\2023-12-01_14-50_pareto-static-200-5utxo\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol8\\2023-11-28_14-56_pareto-delayed-250\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol8\\2023-12-04_13-32_paretosum-static-30-30utxo\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol8\\2023-12-04_16-55_paretosum-static-100-30utxo\\'


    # 50 wallets, max 100 inputs cjtx
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol9\\max100inputs\\2023-12-04_15-14_paretosum-static-50-30utxo\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol9\\max100inputs\\2023-12-05_10-11_paretosum-static-50-30utxo\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol9\\max100inputs\\2023-12-05_11-40_paretosum-static-50-30utxo\\'

    # 50 wallets, max 500 inputs cjtx
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol9\\max500inputs\\2023-12-19_15-16_paretosum-dynamic-50-30utxo-special_5wallets1utxo_max500inputs\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol9\\max500inputs\\2024-01-03_20-31_paretosum-dynamic-50-30utxo-special\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol9\\max500inputs\\2023-12-19_15-19_paretosum-dynamic-50-30utxo-special\\'

    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\debug\\20231023_11Rounds_1parallel_max6inputs_10wallets_5x10Msats\\'

    # real coinjoins imported
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol9\\whirlpool\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol9\\wasabi\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol9\\wasabi2\\'
    #

    # 500 wallets simulation, 500 inputs, kubernetes
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol10\\2024-01-05_08-40_pareto-static-500\\'
    target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol10\\2024-01-05_14-04_paretosum-dynamic-50-30utxo-special\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol10\\2024-01-07_01-39_pareto-dynamic-500\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol10\\2024-01-07_13-21_pareto-dynamic-500\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol10\\2024-01-07_16-20_paretosum-dynamic-500-5utxo\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol10\\2024-01-07_19-20_paretosum-dynamic-500-5utxo\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol10\\2024-01-08_01-23_paretosum-dynamic-500-5utxo\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol10\\2024-01-08_15-52_paretosum-dynamic-500-5utxo\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol10\\2024-01-09_01-03_paretosum-dynamic-500-5utxo\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol10\\2024-01-09_19-02_paretosum-dynamic-200-5utxo\\')

    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol11\\2024-01-09_22-06_paretosum-dynamic-500-5utxo\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol11\\2024-01-10_01-00_paretosum-dynamic-500-5utxo\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol11\\2024-01-11_00-02_paretosum-dynamic-500-30utxo-special\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol11\\2024-01-10_17-46_paretosum-dynamic-500-30utxo-special\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol11\\2024-01-12_12-03_uniform-static-500-30utxo\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol11\\2024-01-12_20-47_uniform-static-500-30utxo\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol11\\2024-01-17_11-55_uniform-static-500-30utxo\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol11\\2024-01-17_15-38_paretosum-dynamic-500-30utxo-special\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol11\\2024-01-18_11-48_uniform-static-500-30utxo\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol8\\2023-12-04_13-32_paretosum-static-30-30utxo\\')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol12\\grid_uniform_test2\\2024-01-25_11-43_paretosum-static-10-30utxo')
    #target_base_path = ('c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol12\\grid_paretosum-static-30utxo\\2024-01-26_19-35_paretosum-static-500-30utxo')
    target_base_path = None
    #
    if target_base_path is not None:
        if PROFILE_PERFORMANCE:
            with Profile() as profile:
                process_experiment((target_base_path, True))
                Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats()
        else:
            if os.path.exists(target_base_path):
                cjplots = CoinJoinPlots(plt)
                cjplots.new_figure()
                process_experiment((target_base_path, True))
            else:
                print(f'ERROR: Path {target_base_path} does not exist')

    # Aggregated analysis of multiple folders with experiments
    # Assumption: given base path contains one or more subfolders (experiments for related settings) with each containing 'data' subfolder (one experiment)
    # super_base_path is path where aggregated results from all

    # super_base_path = ''
    super_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol12\\'
    target_base_paths = [os.path.join(super_base_path, 'grid_paretosum-static-30utxo'),
                         os.path.join(super_base_path, 'grid_paretosum-static-5utxo'),
                         os.path.join(super_base_path, 'grid_uniformsum-static-1utxo'),
                         os.path.join(super_base_path, 'grid_uniformsum-static-5utxo'),
                         os.path.join(super_base_path, 'grid_uniformsum-static-30utxo'),
                         os.path.join(super_base_path, 'grid_pareto-static-1utxo'),
                         os.path.join(super_base_path, 'grid_pareto-static-5utxo'),
                         os.path.join(super_base_path, 'grid_pareto-static-30utxo'),
                         os.path.join(super_base_path, 'grid_pareto-static-1utxo')]


    # super_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol12\\'
    # target_base_paths = [os.path.join(super_base_path, 'grid_pareto-static-30utxo')]

    # super_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol12\\'
    # target_base_paths = [os.path.join(super_base_path, 'grid_uniform_test2'),
    #                      os.path.join(super_base_path, 'grid_uniform_test2')]

    # super_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol12\\'
    # target_base_paths = [os.path.join(super_base_path, 'unproccesed')]

    # super_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol12\\'
    # target_base_paths = [os.path.join(super_base_path, 'grid_uniform_test3')]

    # super_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol12\\'
    # target_base_paths = [os.path.join(super_base_path, 'grid_uniform_test3')]

    # super_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol12\\'
    # target_base_paths = [os.path.join(super_base_path, 'grid_100w')]

    # super_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol12\\'
    # target_base_paths = [os.path.join(super_base_path, 'grid_2500-3000utxo')]

    super_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol12\\'
    target_base_paths = [os.path.join(super_base_path, 'Wasabi2')]



    ##Generate aggregated visualizations
    # generate_aggregated_visualization(target_base_paths)
    # exit(42)

    NUM_THREADS = 1  # if -1, then every experiment has own thread
    SAVE_BASE_FIGS = True if NUM_THREADS == 1 else False
    # SAVE_BASE_FIGS = False  # Force no saving of figs
    all_results = {}
    for base_path in target_base_paths:
        # Analyze one batch of experiments under 'base_path'
        results = process_multiple_experiments(base_path, SAVE_BASE_FIGS, NUM_THREADS)
        analyze_multiple_experiments(results, base_path)
        all_results.update(results)

    # Analyze all batches together
    analyze_multiple_experiments(all_results, super_base_path)

    # # Generate aggregated visualizations
    generate_aggregated_visualization(target_base_paths)
