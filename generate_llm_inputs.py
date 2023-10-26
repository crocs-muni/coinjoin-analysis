import copy
import os
import json
from collections import deque
from collections import Counter
import parse_cj_logs

SATS_IN_BTC = 100000000
VERBOSE = False


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


def set_denomination_group(items, cj_counter):
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


def serialize_llm_transaction(cjtx):
    sentence_denom_groups = ''
    sentence_values = ''
    for key, input in cjtx['inputs'].items():
        sentence_denom_groups += '{} '.format(input['coin_counter'])
        sentence_values += '{} '.format(input['coin_counter'])

        sentence_denom_groups += '{} '.format(input['denomination_group'])
        sentence_values += '{} '.format(int(input['value'] * SATS_IN_BTC))

    sentence_denom_groups += ', '
    sentence_values += ', '
    for key, output in cjtx['outputs'].items():
        sentence_denom_groups += '{} '.format(output['coin_counter'])
        sentence_values += '{} '.format(output['coin_counter'])

        sentence_denom_groups += '{} '.format(output['denomination_group'])
        sentence_values += '{} '.format(-int(output['value'] * SATS_IN_BTC))

    sentence_denom_groups += '. '
    sentence_values += '. '

    return sentence_denom_groups, sentence_values


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

    for start_cjtxid in cjtx_stats['coinjoins'].keys():
        tx_processed_counter += 1
        if tx_processed_counter % 80 == 0:
            print('')

        # Check if enough wallets are present in root coinjoin tx - if not, skip it for generation
        num_wallets_in_coinjoin = set([cjtx_stats['coinjoins'][start_cjtxid]['inputs'][index]['wallet_name'] for index in cjtx_stats['coinjoins'][start_cjtxid]['inputs'].keys()])
        if len(num_wallets_in_coinjoin) < MIN_WALLETS_IN_COINJOIN:
            print('x', end='')
            coinjoin_txs_dropped += 1
            continue

        print('.', end='')  # Print progress
        coinjoin_txs_included +=1

        # Obtain subtree with connected coinjoin txs with up to ANALYSIS_DEPTH_LIMIT from the root tx
        # get coinjoins in scope (all connected from initial cjtxid)
        cjs_in_scope = bfs_with_limit(cjtx_stats['coinjoins'], start_cjtxid, ANALYSIS_DEPTH_LIMIT)

        cj_counter = 1  # Incremental counter of transactions
        coin_counter = 1  # Incremental counter for different coins used between coinjoins (output and corresponding input is the same coin)

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

        # Duplicate obtained coinjoin structure for further processing
        coinjoins_dupl = {}
        for key in cjs_in_scope:
            if key in cjtx_stats['coinjoins'].keys():
                coinjoins_dupl[key] = copy.deepcopy(cjtx_stats['coinjoins'][key])

        if GENERATE_SUBPARTS_GRAPHS:
            visualize_copy['coinjoins'] = coinjoins_dupl
            parse_cj_logs.visualize_coinjoins(visualize_copy, os.path.join(WASABIWALLET_DATA_DIR, 'llm'), 'cj_root={}_depth={}'.format(start_cjtxid[:8], ANALYSIS_DEPTH_LIMIT), False)

        # Sort obtained coinjoins by their broadcast time
        sorted_cjs_in_scope = sorted(coinjoins_dupl, key=lambda txid: coinjoins_dupl[txid]['broadcast_time'])

        # Numbering of inputs and outputs for all subsequent coinjoins in scope
        for txid in sorted_cjs_in_scope:
            # Assign groups of outputs with same denomination and assign denomination_group character based on that
            # Unique values are having special group 'change' with character '♥'
            set_denomination_group(coinjoins_dupl[txid]['inputs'], cj_counter)
            set_denomination_group(coinjoins_dupl[txid]['outputs'], cj_counter)

            # Assign new counter only to such inputs, which are not connected to already numbered outputs
            for key, input in coinjoins_dupl[txid]['inputs'].items():
                if 'coin_counter' not in input.keys():
                    # Check if this input is not output from some previous transaction in the scope we already numbered
                    already_numbered = False
                    for other_cjtx in coinjoins_dupl.keys():
                        if txid == other_cjtx:
                            continue
                        for key, output in coinjoins_dupl[other_cjtx]['outputs'].items():
                            if other_cjtx == input['txid'] and output['address'] == input['address'] and output['value'] == input['value']:
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

        # Generate different variations with swapped inputs and outputs
        assert sorted_cjs_in_scope[0] == start_cjtxid
        root_tx_id = sorted_cjs_in_scope[0]
        last_tx_id = sorted_cjs_in_scope[-1]

        results[start_cjtxid] = []
        num_inputs_first_tx = len(coinjoins_dupl[root_tx_id]['inputs'])
        num_outputs_last_tx = len(coinjoins_dupl[last_tx_id]['outputs'])
        num_variants = num_inputs_first_tx * num_outputs_last_tx

        # Serialize all coinjoins into single string with variantiosn of the first and last transaction
        for input_variant in range(0, num_inputs_first_tx):
            for output_variant in range(0, num_outputs_last_tx):
                coinjoin_sentence_denom_groups = ''
                coinjoin_sentence_values = ''

                first_tx = copy.deepcopy(coinjoins_dupl[root_tx_id])
                last_tx = copy.deepcopy(coinjoins_dupl[last_tx_id])

                first_input_swap = copy.deepcopy(first_tx['inputs'][str(input_variant)])
                first_tx['inputs'][str(input_variant)] = copy.deepcopy(first_tx['inputs']['0'])
                first_tx['inputs']['0'] = copy.deepcopy(first_input_swap)

                last_output_swap = copy.deepcopy(last_tx['outputs'][str(output_variant)])
                last_tx['outputs'][str(output_variant)] = copy.deepcopy(last_tx['outputs'][str(num_outputs_last_tx - 1)])
                last_tx['outputs'][str(num_outputs_last_tx - 1)] = copy.deepcopy(last_output_swap)

                sentence_denom_groups, sentence_values = serialize_llm_transaction(first_tx)
                coinjoin_sentence_denom_groups += sentence_denom_groups
                coinjoin_sentence_values += sentence_values

                for txid in sorted_cjs_in_scope[1:-2]:  # omit first and last transaction
                    sentence_denom_groups, sentence_values = serialize_llm_transaction(coinjoins_dupl[txid])
                    coinjoin_sentence_denom_groups += sentence_denom_groups
                    coinjoin_sentence_values += sentence_values

                sentence_denom_groups, sentence_values = serialize_llm_transaction(last_tx)
                coinjoin_sentence_denom_groups += sentence_denom_groups
                coinjoin_sentence_values += sentence_values

                # Get wallet name for the very first and very last item
                first_input_wallet = first_tx['inputs']['0']['wallet_name']
                last_item = last_tx['outputs'][str(num_outputs_last_tx - 1)]
                if 'wallet_name' in last_item.keys():
                    last_output_wallet = last_item['wallet_name']
                else:
                    last_output_wallet = 'other'

                if VERBOSE:
                    print(coinjoin_sentence_denom_groups)
                    print(coinjoin_sentence_values)
                    print('first_in_wallet={}, last_out_wallet={}'.format(first_input_wallet, last_output_wallet))

                test_item = {}
                test_item['coinjoin_sentence_denom_groups'] = coinjoin_sentence_denom_groups
                test_item['coinjoin_sentence_values'] = coinjoin_sentence_values
                test_item['first_input_wallet'] = first_input_wallet
                test_item['last_output_wallet'] = last_output_wallet
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


if __name__ == "__main__":
    MIN_WALLETS_IN_COINJOIN = 3  # If root coinjoin transaction does not reach this limit, it is not included in training data (inner txs can be below)
    ANALYSIS_DEPTH_LIMIT = 1  # Depth of the assumed coinjoin transactions (number of next other connected coinjoins)
    GENERATE_SUBPARTS_GRAPHS = True

    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231007_2000Rounds_1parallel_max4inputs_10wallets\\'
    target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\debug\\20231007_2000Rounds_1parallel_max4inputs_10wallets\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231008_2000Rounds_1parallel_max5inputs_10wallets\\'
    #target_base_path = 'c:\\!blockchains\\CoinJoin\\WasabiWallet_experiments\\sol5\\20231019_1000Rounds_1parallel_max10inputs_10wallets_5x10Msats_noPrison\\'
#
    print('Path used = {}'.format(target_base_path))
    WASABIWALLET_DATA_DIR = target_base_path

    save_file = os.path.join(WASABIWALLET_DATA_DIR, "coinjoin_tx_info.json")
    # Load parsed coinjoin transactions again
    with open(save_file, "r") as file:
        cjtx_stats = json.load(file)

    # Generate LLM inputs
    events_vector = generate_llm_inputs(cjtx_stats)
    events_file = os.path.join(WASABIWALLET_DATA_DIR, "cjtx_events_minwallets={}_depth={}.json".format(MIN_WALLETS_IN_COINJOIN, ANALYSIS_DEPTH_LIMIT))
    with open(events_file, "w") as file:
        file.write(json.dumps(dict(sorted(events_vector.items())), indent=4))

    print('LLM data generation finished')