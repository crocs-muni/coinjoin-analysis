import copy
import os
import json
from collections import deque
from collections import Counter
import parse_cj_logs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SATS_IN_BTC = 100000000
VERBOSE = False
LINE_STYLES = ['-', '--', '-.', ':']


def bfs_with_limit(coinjoins, root, k):
    # Initialize a queue for BFS and a dictionary to track visited nodes and their depth
    queue = deque()
    visited = {}

    # Initialize the root node with depth 0
    queue.append((root, 0))
    visited[root] = 0

    while queue:
        node, depth = queue.popleft()

        # Perform your desired operation on the node at depth k
        # if depth == k:
        #     print("Node at depth {}:".format(k), node)  # You can replace this with your desired operation

        # Check if we have reached the depth limit
        if depth >= k:
            continue

        # Explore neighbors or children of the current node
        for neighbor in get_neighbors(coinjoins, node):
            if neighbor not in visited or visited[neighbor] > depth + 1:
                # Add unvisited neighbors to the queue with increased depth
                queue.append((neighbor, depth + 1))
                visited[neighbor] = depth + 1

    return visited


def get_neighbors(coinjoins, node):
    next_cjs = []
    for key, output in coinjoins[node]['outputs'].items():
        # We need to find input in other coinjoins, which is having same txid, address and value
        # BUGBUG: better would be to get real txid+index, but we do not have it now
        for other_cjtx in coinjoins.keys():
            for key, input in coinjoins[other_cjtx]['inputs'].items():
                #
                if input['txid'] == node and input['address'] == output['address'] and input['value'] == output['value']:
                    next_cjs.append(other_cjtx)

    return set(next_cjs)


def set_denomination_group_type1(items, cj_counter):
    out_values = [items[index]['value'] for index in items.keys()]
    value_counts = Counter(out_values)
    value_counts_sorted = value_counts.most_common()
    char_mapping = {}
    for i, value in enumerate(value_counts_sorted):
        if value[1] > 1:
            char_mapping[value[0]] = chr(ord('A') + i)
        else:
            char_mapping[value[0]] = 'x'

    num_different_values = len(value_counts)
    for key, output in items.items():
        output['denomination_group'] = '{}_{}'.format(char_mapping[output['value']], cj_counter)


def set_denomination_group_type2(items):
    out_values = [items[index]['value'] for index in items.keys()]
    value_counts = Counter(out_values)
    value_counts_sorted = value_counts.most_common()
    char_mapping = {}
    for i, value in enumerate(value_counts_sorted):
        if value[1] > 1:
            char_mapping[value[0]] = chr(ord('A') + i)
        else:
            char_mapping[value[0]] = int(value[0] * SATS_IN_BTC)

    num_different_values = len(value_counts)
    for key, output in items.items():
        output['denomination_group'] = char_mapping[output['value']]


def serialize_llm_transaction(cjtx):
    sentence_denom_groups_str = ''
    sentence_values_str = ''

    input_values_denom_group = []
    input_values = []
    for key, input in cjtx['inputs'].items():
        sentence_values_str += '{} '.format(input['coin_counter'])

        sentence_denom_groups_str += '{} '.format(input['denomination_group'])
        sats = int(input['value'] * SATS_IN_BTC)
        sentence_values_str += '{} '.format(sats)

        input_values.append((input['coin_counter'], sats))
        input_values_denom_group.append(input['denomination_group'])

    sentence_denom_groups_str += ', '
    sentence_values_str += ', '

    output_values_denom_group = []
    output_values = []
    for key, output in cjtx['outputs'].items():
        if output['distance_to_next_use'] > 1:  # Do not print values with no use (-1) and immediate use (=1)
            sentence_denom_groups_str += '_{} '.format(output['distance_to_next_use'])
        sentence_values_str += '{} '.format(output['coin_counter'])

        sentence_denom_groups_str += '{} '.format(output['denomination_group'])
        sats = -int(output['value'] * SATS_IN_BTC)
        sentence_values_str += '{} '.format(sats)

        output_values.append((output['coin_counter'], sats))
        output_values_denom_group.append((output['distance_to_next_use'], output['denomination_group']))

    sentence_denom_groups_str += '. '
    sentence_values_str += '. '

    return sentence_denom_groups_str, (input_values_denom_group, output_values_denom_group), sentence_values_str, (input_values, output_values)


def number_inputs_outputs_unique(sorted_cjs_in_scope, coinjoins_dupl):
    # Numbering of inputs and outputs for all subsequent coinjoins in scope

    cj_counter = 1  # Incremental counter of transactions
    coin_counter = 1  # Incremental counter for different coins used between coinjoins (output and corresponding input is the same coin)
    for txid in sorted_cjs_in_scope:
        # Assign groups of outputs with same denomination and assign denomination_group character based on that
        # Unique values are having special group 'change' with character '♥'
        # set_denomination_group_type1(coinjoins_dupl[txid]['inputs'], cj_counter)
        # set_denomination_group_type1(coinjoins_dupl[txid]['outputs'], cj_counter)

        # Unique values will keep its sats value
        set_denomination_group_type2(coinjoins_dupl[txid]['inputs'])
        set_denomination_group_type2(coinjoins_dupl[txid]['outputs'])

        # Assign new counter only to such inputs, which are not connected to already numbered outputs
        for key, input in coinjoins_dupl[txid]['inputs'].items():
            if 'coin_counter' not in input.keys():
                # Check if this input is not output from some previous transaction in the scope we already numbered
                already_numbered = False
                for other_cjtx in coinjoins_dupl.keys():
                    if txid == other_cjtx:
                        continue
                    for key, output in coinjoins_dupl[other_cjtx]['outputs'].items():
                        if other_cjtx == input['txid'] and output['address'] == input['address'] and output['value'] == \
                                input['value']:
                            if 'coin_counter' not in input.keys():
                                input['coin_counter'] = output['coin_counter']
                            else:
                                print('ERROR: inconsistent coin_counter')
                            already_numbered = True
                            break
                    if already_numbered:
                        break

                if not already_numbered:
                    # We found new input which is not numbered
                    input['coin_counter'] = coin_counter
                    coin_counter += 1

        # outputs are always getting new counter
        for key, output in coinjoins_dupl[txid]['outputs'].items():
            if 'coin_counter' not in output.keys():
                output['coin_counter'] = coin_counter
                coin_counter += 1
            else:
                print('ERROR: inconsistent coin_counter')
        cj_counter += 1

    return coinjoins_dupl


def compute_output_distance_to_next_cj(coinjoins_dupl, sorted_cjs_in_scope):
    for tx_index in range(0, len(sorted_cjs_in_scope)):
        txid = sorted_cjs_in_scope[tx_index]
        # outputs are always getting new counter
        for key, output in coinjoins_dupl[txid]['outputs'].items():
            distance = -1
            address = output['address']
            for tx_index_after in range(tx_index + 1, len(sorted_cjs_in_scope)):
                for index, input in coinjoins_dupl[sorted_cjs_in_scope[tx_index_after]]['inputs'].items():
                    if input['address'] == address and input['txid'] == txid:
                        distance = tx_index_after - tx_index
                        break
                if distance != -1:
                    break

            if 'distance_to_next_use' not in output.keys():
                output['distance_to_next_use'] = distance

    return coinjoins_dupl


def get_outputs_leaving_mix(coinjoins_dupl, sorted_cjs_in_scope, root_tx_id):
    wallets_involved = set([coinjoins_dupl[root_tx_id]['inputs'][index]['wallet_name']
                            for index in coinjoins_dupl[root_tx_id]['inputs'].keys()])
    wallet_leave_txs = {}
    for wallet in wallets_involved:
        wallet_leave_txs[wallet] = []

    for cjtx_id in sorted_cjs_in_scope:
        for output_index in coinjoins_dupl[cjtx_id]['outputs'].keys():
            output = coinjoins_dupl[cjtx_id]['outputs'][output_index]
            output['vout'] = output_index
            if 'wallet_name' in output and output['wallet_name'] in wallets_involved:  # Analyze only wallets which are having some input in the root transaction
                # Nobody spent this utxo or its anon score already reached the mixing limit (artificial leave of mix for testing)
                if 'spend_by_txid' not in output.keys() or output['anon_score'] > FORCE_LIMIT_ANON_SCORE:
                    # This output is not spent in any of the subsequent mixing transactions
                    wallet_leave_txs[output['wallet_name']].append((cjtx_id, output))
                else:
                    # This output is spent as input in some of the subsequent mixing transaction, ignore
                    # do nothing
                    break

    return wallet_leave_txs


def get_inputs_entering_mix(coinjoins_dupl, sorted_cjs_in_scope, root_tx_id):
    wallets_involved = set([coinjoins_dupl[root_tx_id]['inputs'][index]['wallet_name']
                            for index in coinjoins_dupl[root_tx_id]['inputs'].keys()])
    wallet_enter_txs = {}
    for wallet in wallets_involved:
        wallet_enter_txs[wallet] = []

    for cjtx_id in sorted_cjs_in_scope:
        for input_index in coinjoins_dupl[cjtx_id]['inputs'].keys():
            input = coinjoins_dupl[cjtx_id]['inputs'][input_index]
            input['vin'] = input_index
            # Analyze only wallets which are having some input in the root transaction
            if 'wallet_name' in input and input['wallet_name'] in wallets_involved:
                # This input is not output of some previous transaction in scope (=enters mix)
                if 'spending_tx' not in input.keys():
                    # This output is not spent in any of the subsequent mixing transactions
                    wallet_enter_txs[input['wallet_name']].append((cjtx_id, input))
                else:
                    # This output is spent as input in some of the subsequent mixing transaction, ignore
                    # do nothing
                    break

    return wallet_enter_txs


def compute_inputs_leave_distribution_fifo(coinjoins_dupl, sorted_cjs_in_scope, root_tx_id):
    """
    Computes the distribution of utxos leaving the mix using first in first out principle.
    The input of given wallet do firts mix coinjoin is linked to all subsequent coinjoin outputs (of this wallet),
    until the original value of input is fulfilled. Shall result in smaller number of hops to leave coinjoin mix
    :param coinjoins_dupl:
    :param sorted_cjs_in_scope:
    :param root_tx_id:
    :return:
    """

    # Idea:
    #   1. Find all inputs, which are not resulting from outputs from any previous transaction (entering mix).
    #   2. Find all outputs, which are not used as input in any other transaction in the scope (leaving mix).
    #   3. Identify the controlling wallet for every input entering mix and analyze them separately
    #   4. Assign outputs leaving mix to input entering mix on FIFO basis
    #      - ordering done based on the cjtx block height
    #      - outputs are assigned to single input as long as sum of outputs values is not more than input value

    # Assign link between inputs and outputs
    parse_cj_logs.compute_link_between_inputs_and_outputs(coinjoins_dupl, sorted_cjs_in_scope)

    inputs_enter = get_inputs_entering_mix(coinjoins_dupl, sorted_cjs_in_scope, root_tx_id)
    outputs_leave = get_outputs_leaving_mix(coinjoins_dupl, sorted_cjs_in_scope, root_tx_id)

    inputs_outputs_mapping = {wallet_name: {} for wallet_name in inputs_enter.keys()}

    for wallet_name in inputs_enter.keys():
        for input in inputs_enter[wallet_name]:
            input[1]['to_fulfill'] = int(input[1]['value'] * SATS_IN_BTC)
    for wallet_name in outputs_leave.keys():
        for output in outputs_leave[wallet_name]:
            output[1]['remainig_sats'] = int(output[1]['value'] * SATS_IN_BTC)

    # Iterate over all outputs in FIFO fashion until given input is not fully consumed
    for wallet_name in inputs_enter.keys():
        remaining_sats_to_fulfill = 0  # number of sats to be attributed to some input
        # outputs are assumed to be ordered - starting with ones which are leaving mix first
        for output in outputs_leave[wallet_name]:
            # inputs are assumed to be ordered - try to fullfil first inputs first
            for input in inputs_enter[wallet_name]:
                # Check if we are not assigning outputs earlier than inputs is taking place
                distance = sorted_cjs_in_scope.index(output[0]) - sorted_cjs_in_scope.index(input[0])
                if distance >= 0:
                    if input[1]['to_fulfill'] > 0:
                        # we have some sats to fulfill for this input

                        input_name = parse_cj_logs.get_input_name_string(input)
                        if input_name not in inputs_outputs_mapping[wallet_name].keys():
                            inputs_outputs_mapping[wallet_name][input_name] = []
                        inputs_outputs_mapping[wallet_name][input_name].append(output)
                        output_size = output[1]['remainig_sats']
                        if output_size <= input[1]['to_fulfill']:
                            input[1]['to_fulfill'] -= output_size
                            output[1]['remainig_sats'] = 0
                            break  # go for next output which will fulfill this input
                        else:
                            output[1]['remainig_sats'] -= input[1]['to_fulfill']
                            input[1]['to_fulfill'] = 0
                            # this input is fullfilled, but some sats are remaining in output - use it for next input
                    else:
                        continue  # this input is fullfilled, process another one

    wallet_leave_distribution = {}
    wallets_involved = set([coinjoins_dupl[root_tx_id]['inputs'][index]['wallet_name'] for index
                            in coinjoins_dupl[root_tx_id]['inputs'].keys()])
    for wallet in wallets_involved:
        wallet_leave_distribution[wallet] = {}

    for wallet_name in inputs_outputs_mapping.keys():
        for item_input in inputs_outputs_mapping[wallet_name].keys():
            txid = parse_cj_logs.extract_txid_from_inout_string(item_input)
            output_txids = [txid[0] for txid in inputs_outputs_mapping[wallet_name][item_input]]
            value_counts = Counter(output_txids)
            for item_output in value_counts.keys():
                distance = sorted_cjs_in_scope.index(item_output) - sorted_cjs_in_scope.index(txid)
                if distance < 0:
                    print(f'Negative distance in coinjoin outputs between {txid} and {item_output}')
                num_at_distance = value_counts[item_output]
                if distance in inputs_outputs_mapping[wallet_name]:
                    wallet_leave_distribution[wallet_name][distance] += num_at_distance
                else:
                    wallet_leave_distribution[wallet_name][distance] = num_at_distance
    return wallet_leave_distribution, inputs_outputs_mapping


def compute_inputs_leave_distribution(coinjoins_dupl, sorted_cjs_in_scope, root_tx_id):
    """
    Compute position of outputs which are not mixed anymore for wallets participating in root_tx_id tx.
    :param sorted_cjs_in_scope:
    :param root_tx_id:
    :return:
    """

    # Assign link between inputs and outputs
    parse_cj_logs.compute_link_between_inputs_and_outputs(coinjoins_dupl, sorted_cjs_in_scope)

    # Idea: 1. Find all outputs, which are not used as input in any other transaction in the scope (leaving mix).
    #       2. Identify the controlling wallet and check if such wallet is present in the root transaction.
    #       3. If yes, create list with distribution of outputs leaving mix of every wallet in the root transaction.
    # Simplification: we assume that wallet is coming with only one input to the root transaction.

    wallets_involved = set([coinjoins_dupl[root_tx_id]['inputs'][index]['wallet_name'] for index
                            in coinjoins_dupl[root_tx_id]['inputs'].keys()])

    wallet_leave_distribution = {}
    wallet_leave_txs = {}
    for wallet in wallets_involved:
        wallet_leave_txs[wallet] = []
        wallet_leave_distribution[wallet] = {}

    for cjtx_id in sorted_cjs_in_scope:
        for output in coinjoins_dupl[cjtx_id]['outputs'].values():
            if 'wallet_name' in output and output['wallet_name'] in wallets_involved:  # Analyze only wallets which are having some input in the root transaction
                # Nobody spent this utxo or its anon score already reached the mixing limit (artificial leave of mix for testing)
                if 'spend_by_txid' not in output.keys() or output['anon_score'] > FORCE_LIMIT_ANON_SCORE:
                    # This output is not spent in any of the subsequent mixing transactions
                    wallet_leave_txs[output['wallet_name']].append(cjtx_id)
                else:
                    # This output is spent as input in some of the subsequent mixing transaction, ignore
                    # do nothing
                    break

    for wallet_name in wallet_leave_txs.keys():
        value_counts = Counter(wallet_leave_txs[wallet_name])
        for item in value_counts.keys():
            distance = sorted_cjs_in_scope.index(item) - sorted_cjs_in_scope.index(root_tx_id)
            num_at_distance = value_counts[item]
            if distance in wallet_leave_distribution[wallet_name]:
                wallet_leave_distribution[wallet_name][distance] += num_at_distance
            else:
                wallet_leave_distribution[wallet_name][distance] = num_at_distance

    return wallet_leave_distribution, wallet_leave_txs


def generate_llm_inputs(cjtx_stats):
    """
    Generate inputs for LLM from collected data
    :param cjtx_stats: loaded dict with simulated coinjoins
    :param WASABIWALLET_DATA_DIR: base directory
    :return:
    """

    print('** CONFIGURATION USED **')
    print('MIN_WALLETS_IN_COINJOIN = {}'.format(MIN_WALLETS_IN_COINJOIN))
    print('ANALYSIS_DEPTH_LIMIT = {}'.format(ANALYSIS_DEPTH_LIMIT))
    print('Total initial coinjoin transactions: {}'.format(len(cjtx_stats['coinjoins'].keys())))

    results = {}
    print('Generating LLM data...', end='')
    # Take each coinjoin as starting point
    tx_processed_counter = -1

    coinjoin_txs_included = 0
    coinjoin_txs_dropped = 0
    test_vectors_total = 0
    test_vectors_same_wallet = 0
    test_vectors_different_wallet = 0

    visualize_copy = copy.deepcopy(cjtx_stats)

    # Process all coinjoins, check if they have enough wallets in root coinjoin and generate variations
    for start_cjtxid in cjtx_stats['coinjoins'].keys():
        tx_processed_counter += 1
        if tx_processed_counter % 80 == 0:
            print('')

        # Check if enough wallets are present in root coinjoin tx (<=MIN_WALLETS_IN_COINJOIN) - if not, skip it for generation
        num_wallets_in_coinjoin = set([cjtx_stats['coinjoins'][start_cjtxid]['inputs'][index]['wallet_name'] for index in cjtx_stats['coinjoins'][start_cjtxid]['inputs'].keys()])
        if len(num_wallets_in_coinjoin) < MIN_WALLETS_IN_COINJOIN:
            print('x', end='')
            coinjoin_txs_dropped += 1
            continue

        print('.', end='')  # Print progress
        coinjoin_txs_included += 1

        # Obtain subtree with connected coinjoin txs with up to ANALYSIS_DEPTH_LIMIT from the root tx
        # get coinjoins in scope (all connected from initial cjtxid)
        cjs_in_scope = bfs_with_limit(cjtx_stats['coinjoins'], start_cjtxid, ANALYSIS_DEPTH_LIMIT)

        # Identify all different denomination values over the whole subtree
        out_values = []
        for txid in cjs_in_scope.keys():
            out_values = out_values + [cjtx_stats['coinjoins'][txid]['inputs'][index]['value'] for index in
                                       cjtx_stats['coinjoins'][txid]['inputs'].keys()]
            out_values = out_values + [cjtx_stats['coinjoins'][txid]['outputs'][index]['value'] for index in
                                   cjtx_stats['coinjoins'][txid]['outputs'].keys()]
        value_counts = Counter(out_values)
        value_counts_sorted = value_counts.most_common()
        if VERBOSE:
            print('Number of different denominations: {}'.format(len(value_counts_sorted)))
            print('Number of unique denominations: {}'.format(sum([1 for value in value_counts_sorted if value[1] == 1])))

        # Duplicate obtained coinjoin structure for further processing to preserve the original one untouched
        coinjoins_dupl = {}
        for key in cjs_in_scope:
            if key in cjtx_stats['coinjoins'].keys():
                coinjoins_dupl[key] = copy.deepcopy(cjtx_stats['coinjoins'][key])

        if GENERATE_SUBPARTS_GRAPHS:
            visualize_copy['coinjoins'] = coinjoins_dupl
            parse_cj_logs.visualize_coinjoins(visualize_copy, os.path.join(WASABIWALLET_DATA_DIR, 'llm', 'minWallets={}_depth={}'.format(MIN_WALLETS_IN_COINJOIN, ANALYSIS_DEPTH_LIMIT)), 'cj_root={}_depth={}'.format(start_cjtxid[:8], ANALYSIS_DEPTH_LIMIT), False)

        # Sort obtained coinjoins by their broadcast time
        sorted_cjs_in_scope = sorted(coinjoins_dupl, key=lambda txid: coinjoins_dupl[txid]['broadcast_time'])

        # Number inputs and outputs for all subsequent coinjoins in scope
        number_inputs_outputs_unique(sorted_cjs_in_scope, coinjoins_dupl)

        # Assign number of skipped coinjoins for each output
        # (how many coinjoins were created before the output was used as input to next one)
        compute_output_distance_to_next_cj(coinjoins_dupl, sorted_cjs_in_scope)

        # Generate different variations with swapped inputs and outputs
        assert sorted_cjs_in_scope[0] == start_cjtxid
        root_tx_id = sorted_cjs_in_scope[0]
        last_tx_id = sorted_cjs_in_scope[-1]

        # Compute distribution of outputs of root_tx_id for every wallet which are NOT mixed any longer
        #wallet_leave_distribution, wallet_leave_txs = compute_inputs_leave_distribution_fifo(coinjoins_dupl, sorted_cjs_in_scope, root_tx_id)

        # This compute_inputs_leave_distribution() assigns all *possible* outputs to given input
        wallet_leave_distribution, wallet_leave_txs = compute_inputs_leave_distribution(coinjoins_dupl, sorted_cjs_in_scope, root_tx_id)

        results[start_cjtxid] = []
        num_inputs_first_tx = len(coinjoins_dupl[root_tx_id]['inputs'])
        num_outputs_last_tx = len(coinjoins_dupl[last_tx_id]['outputs'])
        num_variants = num_inputs_first_tx * num_outputs_last_tx

        if not PERMUTATE_INPUTS:
            num_inputs_first_tx = 1  # do not permutate inputs
        if not PERMUTATE_OUTPUTS:
            num_outputs_last_tx = 1  # do not permutate outputs

        # Serialize all coinjoins into single string with variations of the first and last transaction
        for input_variant in range(0, num_inputs_first_tx):
            for output_variant in range(0, num_outputs_last_tx):
                coinjoin_sentence_denom_groups_str = ''
                coinjoin_sentence_denom_groups = []
                coinjoin_sentence_values_str = ''
                coinjoin_sentence_values = []

                first_tx = copy.deepcopy(coinjoins_dupl[root_tx_id])
                last_tx = copy.deepcopy(coinjoins_dupl[last_tx_id])

                first_input_swap = copy.deepcopy(first_tx['inputs'][str(input_variant)])
                first_tx['inputs'][str(input_variant)] = copy.deepcopy(first_tx['inputs']['0'])
                first_tx['inputs']['0'] = copy.deepcopy(first_input_swap)

                last_output_swap = copy.deepcopy(last_tx['outputs'][str(output_variant)])
                last_tx['outputs'][str(output_variant)] = copy.deepcopy(last_tx['outputs'][str(num_outputs_last_tx - 1)])
                last_tx['outputs'][str(num_outputs_last_tx - 1)] = copy.deepcopy(last_output_swap)

                # First transaction
                sentence_denom_groups_str, sentence_denom_groups, sentence_values_str, sentence_values = serialize_llm_transaction(first_tx)
                coinjoin_sentence_denom_groups_str += sentence_denom_groups_str
                coinjoin_sentence_denom_groups.append(sentence_denom_groups)
                coinjoin_sentence_values_str += sentence_values_str
                coinjoin_sentence_values.append(sentence_values)

                # Intermediate transactions
                for txid in sorted_cjs_in_scope[1:-2]:  # omit first and last transaction
                    sentence_denom_groups_str, sentence_denom_groups, sentence_values_str, sentence_values = serialize_llm_transaction(
                        coinjoins_dupl[txid])
                    coinjoin_sentence_denom_groups_str += sentence_denom_groups_str
                    coinjoin_sentence_denom_groups.append(sentence_denom_groups)
                    coinjoin_sentence_values_str += sentence_values_str
                    coinjoin_sentence_values.append(sentence_values)

                # Last transaction
                if last_tx['txid'] != first_tx['txid']:
                    sentence_denom_groups_str, sentence_denom_groups, sentence_values_str, sentence_values = serialize_llm_transaction(last_tx)
                    coinjoin_sentence_denom_groups_str += sentence_denom_groups_str
                    coinjoin_sentence_denom_groups.append(sentence_denom_groups)
                    coinjoin_sentence_values_str += sentence_values_str
                    coinjoin_sentence_values.append(sentence_values)

                # Get wallet name for the very first and very last item
                first_input_wallet = first_tx['inputs']['0']['wallet_name']
                last_item = last_tx['outputs'][str(num_outputs_last_tx - 1)]
                if 'wallet_name' in last_item.keys():
                    last_output_wallet = last_item['wallet_name']
                else:
                    last_output_wallet = 'other'

                if VERBOSE:
                    print(coinjoin_sentence_denom_groups_str)
                    print(coinjoin_sentence_values_str)
                    print('first_in_wallet={}, last_out_wallet={}'.format(first_input_wallet, last_output_wallet))

                test_item = {}
                test_item['coinjoin_sentence_denom_groups'] = coinjoin_sentence_denom_groups_str
                test_item['coinjoin_sentence_denom_groups_plain_values'] = coinjoin_sentence_denom_groups
                test_item['coinjoin_sentence_values'] = coinjoin_sentence_values_str
                test_item['coinjoin_sentence_plain_values'] = coinjoin_sentence_values
                test_item['first_input_wallet'] = first_input_wallet
                test_item['last_output_wallet'] = last_output_wallet
                test_item['first_input_wallet_mix_leave_distribution'] = wallet_leave_distribution[first_input_wallet]
                test_item['all_wallets_mix_leave_distribution'] = wallet_leave_distribution
                test_item['root_cj_tx'] = root_tx_id
                test_item['last_cj_tx'] = last_tx_id
                test_item['input_index_as_first'] = input_variant
                test_item['output_index_as_last'] = output_variant

                test_vectors_total += 1
                if first_input_wallet == last_output_wallet:
                    test_vectors_same_wallet += 1
                else:
                    test_vectors_different_wallet += 1

                results[start_cjtxid].append(test_item)
    print(' done')

    print('** CONFIGURATION USED **')
    print('MIN_WALLETS_IN_COINJOIN = {}'.format(MIN_WALLETS_IN_COINJOIN))
    print('ANALYSIS_DEPTH_LIMIT = {}'.format(ANALYSIS_DEPTH_LIMIT))
    print('Total initial coinjoin transactions: {}'.format(len(cjtx_stats['coinjoins'].keys())))

    print('coinjoin_txs_included={}'.format(coinjoin_txs_included))
    print('coinjoin_txs_dropped={}'.format(coinjoin_txs_dropped))
    print('test_vectors_total={}'.format(test_vectors_total))
    print('test_vectors_same_wallet={}'.format(test_vectors_same_wallet))
    print('test_vectors_different_wallet={}'.format(test_vectors_different_wallet))

    return results


def visualize_wallet_leave_distribution(cjtx_stats, events_vector, base_path, filter_wallets=[]):
    fig, ax = plt.subplots(figsize=(20, 16))
    index = 0
    #for fraction_to_analyze in np.arange(0.1, 0.11, 0.1):  # Fraction of wallets to be analyzed (0.1 => only first 10% are analyzed)
    for fraction_to_analyze in [1]:
        sorted_cjs_in_scope = sorted(cjtx_stats['coinjoins'].keys(), key=lambda txid: cjtx_stats['coinjoins'][txid]['broadcast_time'])
        # Visualize distribution of wallet leave distribution for every wallet and every root transaction
        aggregated_wallet_distribution_per_wallet = {}
        for cjtxid in sorted_cjs_in_scope[0:int(len(sorted_cjs_in_scope) * fraction_to_analyze)]:
            for wallet_name in events_vector[cjtxid][0]['all_wallets_mix_leave_distribution']:
                aggregated_wallet_distribution_per_wallet[wallet_name] = {i: 0 for i in range(0, len(events_vector))}
        aggregated_wallet_distribution = {i: 0 for i in range(0, len(events_vector))}
        for cjtxid in sorted_cjs_in_scope[0:int(len(sorted_cjs_in_scope) * fraction_to_analyze)]:
            # take only the first entry as others are just permutation of it
            for wallet_name in events_vector[cjtxid][0]['all_wallets_mix_leave_distribution']:
                for num_to_leave in events_vector[cjtxid][0]['all_wallets_mix_leave_distribution'][wallet_name].keys():
                    aggregated_wallet_distribution[num_to_leave] += events_vector[cjtxid][0]['all_wallets_mix_leave_distribution'][wallet_name][num_to_leave]
                    aggregated_wallet_distribution_per_wallet[wallet_name][num_to_leave] += events_vector[cjtxid][0]['all_wallets_mix_leave_distribution'][wallet_name][num_to_leave]

        print(aggregated_wallet_distribution)

        for wallet_name in aggregated_wallet_distribution_per_wallet.keys():
            if len(filter_wallets) > 0 and wallet_name not in filter_wallets:
                continue
            x_data = list(aggregated_wallet_distribution_per_wallet[wallet_name].keys())
            y_data = list(aggregated_wallet_distribution_per_wallet[wallet_name].values())
            ax.plot(x_data, y_data, label=f'{wallet_name}/{round(fraction_to_analyze, 1)}', alpha=1, linestyle=LINE_STYLES[index % len(LINE_STYLES)])
            index += 1

        # Display all wallets aggregated
        x_all_wallets = list(aggregated_wallet_distribution.keys())
        y_all_wallets = list(aggregated_wallet_distribution.values())
        #y_values = normalized_y_values = np.array(y_values) / np.max(y_values) * 100
        ax.plot(x_all_wallets, y_all_wallets, label=f'All wallets/{round(fraction_to_analyze, 1)}', alpha=0.2)
        index += 1

    ax.set_xlabel('Number of hops to leave mix')
    ax.set_ylabel('Number of outputs to leave mix')
    ax.set_yscale('log')
    ax.set_title(f'Distribution of hops to leave mixing (FORCE_LIMIT_ANON_SCORE={FORCE_LIMIT_ANON_SCORE}')
    #ax.legend(ncol=3)
    save_file = os.path.join(base_path, "coinjoin_leaving_distribution.png")
    plt.savefig(save_file, dpi=300)
#    fig.show()
    plt.close()


def compute_distance_single_wallet(cjtx_stats):
    sorted_cjs_in_scope = sorted(cjtx_stats['coinjoins'], key=lambda txid: cjtx_stats['coinjoins'][txid]['broadcast_time'])

    # Filter only coinjoins for specific wallets
    #filter_wallets = ['wallet-004', 'wallet-005', 'wallet-008', 'wallet-031', 'wallet-039']
    for target_wallet in cjtx_stats['wallets_info'].keys():
        # Filter only coinjoins which have at least one input from target_wallet
        filter_coinjoin = {}
        filter_coinjoin['coinjoins'] = {}
        for txid in cjtx_stats['coinjoins'].keys():
            if target_wallet in [cjtx_stats['coinjoins'][txid]['inputs'][vin]['wallet_name'] for vin in cjtx_stats['coinjoins'][txid]['inputs']]:
                filter_coinjoin['coinjoins'][txid] = cjtx_stats['coinjoins'][txid]
        print(f'Found {len(filter_coinjoin['coinjoins'])} cjtxs for {target_wallet}')

        # Compute distance from first and last coinjoin
        if len(filter_coinjoin['coinjoins']) > 0:
            distance = 0
            for txid in sorted_cjs_in_scope:
                distance += 1
                if txid == list(filter_coinjoin['coinjoins'].keys())[0]:
                    print(f'  first input at: {sorted_cjs_in_scope.index(txid)}')
                    distance = 0  # Start computing distance from this one
                if txid == list(filter_coinjoin['coinjoins'].keys())[-1]:  # Up to last one
                    print(f'  last output at: {sorted_cjs_in_scope.index(txid)}')
                    print(f'  distance from first to last coinjoin: {distance}')
                    break
    print('done')


if __name__ == "__main__":
    MIN_WALLETS_IN_COINJOIN = 1  # If root coinjoin transaction does not reach this limit, it is not included in training data (inner txs can be below)
    ANALYSIS_DEPTH_LIMIT = 100  # Depth of the assumed coinjoin transactions (number of next other connected coinjoins)
    GENERATE_SUBPARTS_GRAPHS = False

    # Settings primarily relevant for mix leave distribution setup
    PERMUTATE_INPUTS = False  # If True, all inputs from the root cjtx will be permutated to take the first place
    PERMUTATE_OUTPUTS = False  # If True, all outputs from the latest cjtx will be permutated to take the last place
    #FORCE_LIMIT_ANON_SCORE = 1.2  # If anon score is higher than this value, we assume that output is not used as input in any other transaction
    FORCE_LIMIT_ANON_SCORE = 100  # If anon score is higher than this value, we assume that output is not used as input in any other transaction

    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231007_2000Rounds_1parallel_max4inputs_10wallets\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol8\\disbalanced-delayed-20_2023-11-23_14-47\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\debug\\20231007_2000Rounds_1parallel_max4inputs_10wallets\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231008_2000Rounds_1parallel_max5inputs_10wallets\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231019_1000Rounds_1parallel_max10inputs_10wallets_5x10Msats_noPrison\\'

    # Two experiments with exactly same settings
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol8\\pareto-delayed-50_2023-11-23_22-28\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol8\\pareto-delayed-50_2023-11-24_15-19\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol8\\2023-12-04_13-32_paretosum-static-30-30utxo\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol9\\2023-12-05_11-40_paretosum-static-50-30utxo\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol9\\2023-12-05_10-11_paretosum-static-50-30utxo\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol9\\max100inputs\\2023-12-04_15-14_paretosum-static-50-30utxo\\'
    target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol9\\max500inputs\\2023-12-19_15-16_paretosum-dynamic-50-30utxo-special\\'
    #

    print('Path used = {}'.format(target_base_path))
    WASABIWALLET_DATA_DIR = target_base_path

    save_file = os.path.join(WASABIWALLET_DATA_DIR, "coinjoin_tx_info.json")
    # Load parsed coinjoin transactions again
    with open(save_file, "r") as file:
        cjtx_stats = json.load(file)

    # Compute distance for first input to all outputs for single wallets
    compute_distance_single_wallet(cjtx_stats)

    #
    # Generate LLM inputs
    #
    #visualize_wallet_leave_distribution(cjtx_stats, [])
    events_vector = generate_llm_inputs(cjtx_stats)

    visualize_wallet_leave_distribution(cjtx_stats, events_vector, target_base_path)

    exit(1)
    # Save LLM inputs as json
    events_file = os.path.join(WASABIWALLET_DATA_DIR, "cjtx_events_minwallets={}_depth={}.json".format(MIN_WALLETS_IN_COINJOIN, ANALYSIS_DEPTH_LIMIT))
    with open(events_file, "w") as file:
        file.write(json.dumps(dict(sorted(events_vector.items())), indent=4))

    #
    # Save LLM inputs as numpy binary format for training with separated labels
    #
    GENERATE_NUMPY_BINARY = True
    if GENERATE_NUMPY_BINARY:
        tx_data = []
        tx_data_same_wallet = []
        tx_data_denom_groups = []
        tx_data_denom_groups_answer = []
        test_inputs_counter = 0
        for cjtxid in events_vector.keys():
            for index in range(0, len(events_vector[cjtxid])):
                test_input = events_vector[cjtxid][index]['coinjoin_sentence_plain_values']
                tx_data.append(test_input)
                test_input_answer = True if events_vector[cjtxid][index]['first_input_wallet'] == events_vector[cjtxid][index]['last_output_wallet'] else False
                tx_data_same_wallet.append(test_input_answer)

                test_input = events_vector[cjtxid][index]['coinjoin_sentence_denom_groups_plain_values']
                tx_data_denom_groups.append(test_input)
                test_input_answer = events_vector[cjtxid][index]['first_input_wallet_mix_leave_distribution']
                tx_data_denom_groups_answer.append(test_input_answer)
                test_inputs_counter += 1

        # Save llm values as numpy binary format
        print('Total test inputs = {}'.format(test_inputs_counter))
        np.save(os.path.join(WASABIWALLET_DATA_DIR, 'first_last_wallet_X.npy'), tx_data)
        np.save(os.path.join(WASABIWALLET_DATA_DIR, 'first_last_wallet_Y.npy'), tx_data_same_wallet)

        np.save(os.path.join(WASABIWALLET_DATA_DIR, 'denom_group_distrib_X.npy'), tx_data_denom_groups)
        np.save(os.path.join(WASABIWALLET_DATA_DIR, 'denom_group_distrib_Y.npy'), tx_data_denom_groups_answer)

    print('LLM data generation finished')