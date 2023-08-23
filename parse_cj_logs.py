import re
import subprocess
import json
import sys
import wcli
from itertools import zip_longest

BTC_CLI_PATH = 'C:\\bitcoin-25.0\\bin\\bitcoin-cli'
WASABIWALLET_DATA_DIR = 'c:\\Users\\xsvenda\\AppData\\Roaming\\WalletWasabi'


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


def find_round_ids(filename, regex_pattern, group_name):
    """
    Extracts all round_ids which from provided file which match regexec pattern and its specified part given by group_name.
    Function is more generic as any group_name from regex_pattern can be specified, not only round_id
    :param filename: name of file with logs
    :param regex_pattern: regex pattern which is matched to every line
    :param group_name: name of item specified in regex pattern, which is extracted
    :return: list of unique values found for 'group_name' matches
    """
    round_ids = {}

    try:
        with open(filename, 'r') as file:
            for line in file:
                for match in re.finditer(regex_pattern, line):
                    if group_name in match.groupdict():
                        round_id = match.group(group_name).strip()
                        round_ids[round_id] = 1

    except FileNotFoundError:
        print(f"File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return round_ids.keys()


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
                return output['scriptPubKey']['address']

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

    input_addresses = {}
    output_addresses = {}
    try:
        parsed_data = json.loads(tx_info)

        inputs = parsed_data['vin']
        index = 0
        for input in inputs:
            txid_in = input['txid']
            txid_in_out = input['vout']
            in_address = get_input_address(txid_in, txid_in_out)  # we need to read and parse previous transaction to obtain address
            input_addresses[index] = in_address  # store address to index of the input
            index = index + 1

        outputs = parsed_data['vout']
        for output in outputs:
            output_addresses[output['n']] = output['scriptPubKey']['address']

    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)

    return input_addresses, output_addresses


def print_tx_info(input_addresses, output_addresses, wallets_info):
    """
    Prints mapping between addresses. Unfinished for now
    :param input_addresses:
    :param output_addresses:
    :param wallets_info: information about wallets including list of addresses
    :return:
    """
    # Build mapping of addresses to wallets names ('unknown' if not mapped)
    address_wallet_mapping = {input_addresses[addr_index]: 'unknown' for addr_index in input_addresses}
    for addr in list(input_addresses.values()) + list(output_addresses.values()):
        for wallet_name in wallets_info.keys():
            for waddr in wallets_info[wallet_name]:
                if addr == waddr['address']:
                    address_wallet_mapping[addr] = wallet_name

    used_wallets = sorted({address_wallet_mapping[addr]: 1 for addr in list(input_addresses.values()) + list(output_addresses.values())}.keys())
    # print all inputs mapped to their outputs
    for wallet_name in used_wallets:
        print('Wallet `{}`'.format(wallet_name))
        for addr in sorted(list(input_addresses.values())):
            if address_wallet_mapping[addr] == wallet_name:
                print('  ({}):{}'.format(wallet_name, addr))
        for addr in sorted(list(output_addresses.values())):
            if address_wallet_mapping[addr] == wallet_name:
                print('  -> ({}):{}'.format(wallet_name, addr))


def parse_coinjoin_logs(wallets_info):
    coord_input_file = '{}\\Backend\\Logs_4paralell_1hour.txt'.format(WASABIWALLET_DATA_DIR)
    client_input_file = '{}\\Client\\Logs_4paralell_1hour.txt'.format(WASABIWALLET_DATA_DIR)
    regex_pattern = r"(.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Created round with params: MaxSuggestedAmount:'([0-9\.]+)' BTC?"
    start_round_ids = find_round_ids(coord_input_file, regex_pattern, 'round_id')
    regex_pattern = r"(.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Successfully broadcast the coinjoin: (?P<cj_tx_id>[0-9a-f]*)\.?"
    success_coinjoin_round_ids = find_round_ids(coord_input_file, regex_pattern, 'round_id')
    round_cjtx_mapping = find_round_cjtx_mapping(coord_input_file, regex_pattern, 'round_id', 'cj_tx_id')

    # find all ids which have complete log from round creation (Created round with params)
    # to cj tx broadcast (Successfully broadcast the coinjoin)
    full_round_ids = [id for id in success_coinjoin_round_ids if id in start_round_ids]

    # print only logs with full rounds
    #[print_round_logs(coord_input_file, id) for id in full_round_ids]
    print('\n\nTotal complete rounds found: {}'.format(len(full_round_ids)))

    # 2023-08-22 11:06:35.181 [21] DEBUG	CoinJoinClient.CreateRegisterAndConfirmCoinsAsync (469)	Round (5f3425c1f2e0cc81c9a74a213abf1ea3f128247d6be78ecd259158a5e1f9b66c): Inputs(4) registration started - it will end in: 00:01:22.
    #regex_pattern = r"(.*) \[.+(?P<method>CoinJoinClient\..*) \([0-9]+\).*Round \((?P<round_id>.*)\): Inputs\((?P<num_inputs>[0-9]+)\) registration started - it will end in: ([0-9:]+)\."
    #client_start_round_ids = find_round_ids(coord_input_file, regex_pattern, 'round_id')
    # 2023-08-22 11:06:51.466 [10] INFO	AliceClient.RegisterInputAsync (105)	Round (5f3425c1f2e0cc81c9a74a213abf1ea3f128247d6be78ecd259158a5e1f9b66c), Alice (94687969-bf26-1dfd-af98-2365e708b893): Registered 80b9c8615226e03d2474d8ad481c2db7505cb2715b10d83ee9c95106aaa3dcfd-0.
    #regex_pattern = r"(.*) \[.+(Arena\..*) \(.*Round \((?P<round_id>.*)\): Successfully broadcast the coinjoin: (?P<cj_tx_id>[0-9a-f]*)\.?"

    for round_id in full_round_ids:
        coord_round_logs = read_lines_for_round(coord_input_file, round_id)
        client_round_logs = read_lines_for_round(client_input_file, round_id)

        sorted_combined_list = sorted(coord_round_logs + client_round_logs)
        for line in sorted_combined_list:
            if line in client_round_logs:
                line = line.replace(" INFO", " INFO ")
                print("  " + line.rstrip())
            else:
                line = line.replace(" INFO", " INFO ")
                print(line.rstrip())

        print('**************************************')

        print('Address input-output mapping for cjtx: {}'.format(round_cjtx_mapping[round_id]))
        input_addresses, output_addresses = extract_tx_info(round_cjtx_mapping[round_id])
        print_tx_info(input_addresses, output_addresses, wallets_info)

        print('**************************************')
        print('**************************************')


def load_wallets_info():
    """
    Loads information about wallets and their addresses using Wasabi RPC
    :return: dictionary for all loaded wallets with retrieved info
    """
    WALLET_NAME_TEMPLATE = 'Wallet'
    MAX_WALLETS = 6
    wcli.WASABIWALLET_DATA_DIR = WASABIWALLET_DATA_DIR
    wcli.VERBOSE = False
    wallets_info = {}
    wallet_names = ['{}{}'.format(WALLET_NAME_TEMPLATE, index) for index in range(0, MAX_WALLETS + 1)]
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
    wallets_info = load_wallets_info()
    parse_coinjoin_logs(wallets_info)
