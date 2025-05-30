'''
Created on 20160917
@author: LaurentMT
'''
import os
import math
import getopt
import traceback
import sys

# Adds boltzmann directory into path
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from boltzmann.utils.tx_processor import process_tx
from boltzmann.utils.bitcoind_rpc_wrapper import BitcoindRPCWrapper
from boltzmann.utils.bci_wrapper import BlockchainInfoWrapper
from boltzmann.utils.blockstream_wrapper import BlockstreamWrapper
from boltzmann.utils.transaction import Transaction

def display_results(mat_lnk, nb_cmbn, inputs, outputs, fees, intrafees, efficiency, duration):
    '''
    Displays the results for a given transaction
    Parameters:
        mat_lnk   = linkability matrix
        nb_cmbn   = number of combinations detected
        inputs    = list of input txos (tuples (address, amount))
        outputs   = list of output txos (tuples (address, amount))
        fees      = fees associated to this transaction
        intrafees = max intrafees paid/received by participants (tuple (max intrafees received, max intrafees paid))
        efficiency= wallet efficiency for this transaction (expressed as a percentage)
    '''
    results = {}
    print('\nInputs = ' + str(inputs))
    print('\nOutputs = ' + str(outputs))
    print('\nFees = %i satoshis' % fees)
    results['successfully_analyzed'] = True if mat_lnk is not None else False
    results['inputs'] = inputs
    results['outputs'] = outputs
    results['fees'] = fees
    results['duration'] = duration
    results['linkability_matrix'] = mat_lnk.tolist() if mat_lnk is not None else None
    if (intrafees[0] > 0) and (intrafees[1] > 0):
        print('\nHypothesis: Max intrafees received by a participant = %i satoshis' % intrafees[0])
        print('Hypothesis: Max intrafees paid by a participant = %i satoshis' % intrafees[1])
        results['max_intrafees_received'] = intrafees[0]
        results['max_intrafees_paid'] = intrafees[1]

    print('\nNb combinations = %i' % nb_cmbn)
    results['num_combinations_detected'] = nb_cmbn
    results['tx_entropy'] = 0
    if nb_cmbn > 0:
        print('Tx entropy = %f bits' % math.log2(nb_cmbn))
        results['tx_entropy'] = math.log2(nb_cmbn)

    results['efficiency'] = 0
    results['efficiency_bits'] = 0
    if efficiency is not None and efficiency > 0:
        print('Wallet efficiency = %f%% (%f bits)' % (efficiency*100, math.log2(efficiency)))
        results['efficiency'] = efficiency*100
        results['efficiency_bits'] = math.log2(efficiency)

    results['entropy_density'] = 0
    if nb_cmbn > 0:
        print('Entropy density = %f%%' % (math.log2(nb_cmbn) / (len(inputs) + len(outputs))))
        results['entropy_density'] = (math.log2(nb_cmbn) / (len(inputs) + len(outputs)))

    results['deterministic_links'] = []
    results['num_deterministic_links'] = -1
    if mat_lnk is None:
        if nb_cmbn == 0:
            print('\nSkipped processing of this transaction (too many inputs and/or outputs)')
    else:
        if nb_cmbn != 0:
            print('\nLinkability Matrix (probabilities) :')
            print(mat_lnk / nb_cmbn)
        else:
            print('\nLinkability Matrix (#combinations with link) :')
            print(mat_lnk)

        dlCount = 0
        print('\nDeterministic links :')
        for i in range(0, len(outputs)):
            for j in range(0, len(inputs)):
                if (mat_lnk[i,j] == nb_cmbn) and mat_lnk[i,j] != 0 :
                    print('%s & %s are deterministically linked' % (inputs[j], outputs[i]))
                    results['deterministic_links'].append((inputs[j], outputs[i]))
                    dlCount += 1
        results['num_deterministic_links'] = dlCount

        # deterministic link ratio:
        nbLinks = len(outputs) * len(inputs)
        ratioDL = dlCount / nbLinks
#        nRatioDL = 1.0 - ratioDL
        print('\nDeterministic link ratio = %f%%' % (ratioDL * 100))
        results['deterministic_link_ratio'] = ratioDL * 100

    return results


def main(txids, rpc, testnet, blockstream, options=['PRECHECK', 'LINKABILITY', 'MERGE_INPUTS'], max_duration=600, max_txos=12, max_cj_intrafees_ratio=0):
    '''
    Main function
    Parameters:
        txids                   = list of transactions txids to be processed
        rpc                     = use bitcoind's RPC interface (or blockchain.info web API)
        testnet                 = use testnet (blockchain.info by default)
        blockstream             = use blockstream data provider
        options                 = options to be applied during processing
        max_duration            = max duration allocated to processing of a single tx (in seconds)
        max_txos                = max number of txos. Txs with more than max_txos inputs or outputs are not processed.
        max_cj_intrafees_ratio  = max intrafees paid by the taker of a coinjoined transaction.
                                  Expressed as a percentage of the coinjoined amount.
    '''
    blockchain_provider = None
    provider_descriptor = ''
    if rpc:
        blockchain_provider = BitcoindRPCWrapper()
        provider_descriptor = 'local RPC interface'
    else:
        if blockstream == True:
            blockchain_provider = BlockstreamWrapper()
            provider_descriptor = 'remote Blockstream API'
        else:
            blockchain_provider = BlockchainInfoWrapper()
            provider_descriptor = 'remote blockchain.info API'

    print("DEBUG: Using %s" % provider_descriptor)

    results = {}
    for txid in txids:
        print('\n\n--- %s -------------------------------------' % txid)
        # retrieves the tx from local RPC or external data provider
        try:
            tx = blockchain_provider.get_tx(txid, not testnet) # working option

            # tested option
            #rpc_style_tx = blockchain_provider._get_decoded_tx(txid)
            #tx = blockchain_provider.get_tx_from_result(txid, rpc_style_tx)

            print("DEBUG: Tx fetched: {0}".format(str(tx)))
        except Exception as err:
            print('Unable to retrieve information for %s from %s: %s %s' % (txid, provider_descriptor, err, traceback.format_exc()))
            continue

        # Computes the entropy of the tx and the linkability of txos
        (mat_lnk, nb_cmbn, inputs, outputs, fees, intrafees, efficiency, duration) = process_tx(tx, options, max_duration, max_txos, max_cj_intrafees_ratio)
        results[txid] = {}
        #results[txid]['raw'] = (mat_lnk, nb_cmbn, inputs, outputs, fees, intrafees, efficiency)

        # Displays the results
        results[txid]['processed'] = display_results(mat_lnk, nb_cmbn, inputs, outputs, fees, intrafees, efficiency, duration)

    return results


def usage():
    '''
    Usage message for this module
    '''
    sys.stdout.write('python ludwig.py [--rpc] [--testnet] [--blockstream] [--duration=600] [--maxnbtxos=12] [--cjmaxfeeratio=0] [--options=PRECHECK,LINKABILITY,MERGE_FEES,MERGE_INPUTS,MERGE_OUTPUTS] [--txids=8e56317360a548e8ef28ec475878ef70d1371bee3526c017ac22ad61ae5740b8,812bee538bd24d03af7876a77c989b2c236c063a5803c720769fc55222d36b47,...]');
    sys.stdout.write('\n\n[-t OR --txids] = List of txids to be processed.')
    sys.stdout.write('\n\n[-p OR --rpc] = Use bitcoind\'s RPC interface as source of blockchain data')
    sys.stdout.write('\n\n[-T OR --testnet] = Use testnet interface as source of blockchain data')
    sys.stdout.write('\n\n[-b OR --blockstream] = Use Blockstream interface as source of blockchain data')
    sys.stdout.write('\n\n[-d OR --duration] = Maximum number of seconds allocated to the processing of a single transaction. Default value is 600')
    sys.stdout.write('\n\n[-x OR --maxnbtxos] = Maximum number of inputs or ouputs. Transactions with more than maxnbtxos inputs or outputs are not processed. Default value is 12.')
    sys.stdout.write('\n\n[-r OR --cjmaxfeeratio] = Max intrafees paid by the taker of a coinjoined transaction. Expressed as a percentage of the coinjoined amount. Default value is 0.')

    sys.stdout.write('\n\n[-o OR --options] = Options to be applied during processing. Default value is PRECHECK, LINKABILITY, MERGE_INPUTS')
    sys.stdout.write('\n    Available options are :')
    sys.stdout.write('\n    PRECHECK = Checks if deterministic links exist without processing the entropy of the transaction. Similar to Coinjoin Sudoku by K.Atlas.')
    sys.stdout.write('\n    LINKABILITY = Computes the entropy of the transaction and the txos linkability matrix.')
    sys.stdout.write('\n    MERGE_INPUTS = Merges inputs "controlled" by a same address. Speeds up computations.')
    sys.stdout.write('\n    MERGE_OUTPUTS = Merges outputs "controlled" by a same address. Speeds up computations but this option is not recommended.')
    sys.stdout.write('\n    MERGE_FEES = Processes fees as an additional output paid by a single participant. May speed up computations.')
    sys.stdout.flush()


def analyze_txs(txids):
    # Initializes parameters
    #max_txos = 12
    #max_txos = 16
    max_txos = 30
    max_duration = 60
    max_cj_intrafees_ratio = 0 #0.005
    options = ['PRECHECK', 'LINKABILITY', 'MERGE_INPUTS']
    argv = sys.argv[1:]
    # Processes arguments
    try:
        opts, args = getopt.getopt(argv, 'hpt:p:T:b:d:o:r:x:', ['help', 'rpc', 'testnet', 'blockstream', 'txids=', 'duration=', 'options=', 'cjmaxfeeratio=', 'maxnbtxos='])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    rpc = True
    testnet = False
    blockstream = True
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()
        elif opt in ('-p', '--rpc'):
            rpc = True
        elif opt in ('-T', '--testnet'):
            testnet = True
        elif opt in ('-b', '--blockstream'):
            blockstream = True
        elif opt in ('-d', '--duration'):
            max_duration = int(arg)
        elif opt in ('-x', '--maxnbtxos'):
            max_txos = int(arg)
        elif opt in ('-r', '--cjmaxfeeratio'):
            max_cj_intrafees_ratio = float(arg)
        elif opt in ('-t', '--txids'):
            txids = [t.strip() for t in arg.split(',')]
        elif opt in ('-o', '--options'):
            options = [t.strip() for t in arg.split(',')]

    os.environ['BOLTZMANN_RPC_USERNAME'] = 'user'
    os.environ['BOLTZMANN_RPC_PASSWORD'] = 'password'
    os.environ['BOLTZMANN_RPC_HOST'] = '127.0.0.1'
    os.environ['BOLTZMANN_RPC_PORT'] = '18443'
    # Processes computations
    return main(txids=txids, rpc=rpc, testnet=testnet, blockstream=blockstream, options=options, max_duration=max_duration, max_txos=max_txos, max_cj_intrafees_ratio=max_cj_intrafees_ratio)


def retrieve_txs(txids):
    rpc = True
    testnet = False
    os.environ['BOLTZMANN_RPC_USERNAME'] = 'user'
    os.environ['BOLTZMANN_RPC_PASSWORD'] = 'password'
    os.environ['BOLTZMANN_RPC_HOST'] = '127.0.0.1'
    os.environ['BOLTZMANN_RPC_PORT'] = '18443'

    blockchain_provider = None
    provider_descriptor = ''
    if rpc:
        blockchain_provider = BitcoindRPCWrapper()
        provider_descriptor = 'local RPC interface'
    else:
        blockchain_provider = BlockstreamWrapper()
        provider_descriptor = 'remote Blockstream API'

    print("DEBUG: Using %s" % provider_descriptor)

    results = {}
    for txid in txids:
        print('\n\n--- %s -------------------------------------' % txid)
        # retrieves the tx from local RPC or external data provider
        try:
            tx = blockchain_provider.get_tx(txid, not testnet)  # working option
            # rpc_style_tx = blockchain_provider._get_decoded_tx(txid)
            # rpc_style_tx['block_height'] = blockchain_provider._get_block_height(txid, rpc_style_tx)
            # rpc_style_tx['time'] = None
            # rpc_style_tx['hash'] = rpc_style_tx['txid']
            #
            # rpc_style_tx['inputs'] = list()
            # for txin in rpc_style_tx['vin']:
            #     inpt = blockchain_provider._rpc_to_bci_input(txin)
            #     rpc_style_tx['inputs'].append(inpt)
            #
            # rpc_style_tx['out'] = list()
            # for txout in rpc_style_tx['vout']:
            #     outpt = blockchain_provider._rpc_to_bci_output(txout)
            #     rpc_style_tx['out'].append(outpt)

            print("DEBUG: Tx fetched: {0}".format(txid))
        except Exception as err:
            print('Unable to retrieve information for %s from %s: %s %s' % (txid, provider_descriptor, err, traceback.format_exc()))
            continue
        results[txid] = tx
        #results[txid] = rpc_style_tx
    return results


def analyze_txs_from_prefetched_simple(txs_info):
    max_txos = 30
    max_duration = 60
    max_cj_intrafees_ratio = 0 #0.005
    options = ['PRECHECK', 'LINKABILITY', 'MERGE_INPUTS']
    return analyze_txs_from_prefetched(txs_info, options, max_duration, max_txos, max_cj_intrafees_ratio)


def analyze_txs_from_prefetched(txs_info, options=['PRECHECK', 'LINKABILITY', 'MERGE_INPUTS'], max_duration=600, max_txos=12, max_cj_intrafees_ratio=0):
    results = {}
    for txid in txs_info.keys():
        print('\n\n--- %s -------------------------------------' % txid)
        try:
            # tx = blockchain_provider.get_tx(txid, not testnet)
            #rpc_style_tx = txs_info[txid]
            #tx = Transaction(rpc_style_tx)
            #tx = blockchain_provider.get_tx_from_result(txid, rpc_style_tx)
            tx = txs_info[txid]

            print("DEBUG: Tx loaded: {0}".format(str(tx)))
        except Exception as err:
            print('Unable to retrieve information for %s : %s %s' % (
            txid, err, traceback.format_exc()))
            continue

        # Computes the entropy of the tx and the linkability of txos
        (mat_lnk, nb_cmbn, inputs, outputs, fees, intrafees, efficiency, duration) = process_tx(tx, options, max_duration,
                                                                                      max_txos,
                                                                                      max_cj_intrafees_ratio)
        results[txid] = {}
        # results[txid]['raw'] = (mat_lnk, nb_cmbn, inputs, outputs, fees, intrafees, efficiency)

        # Displays the results
        results[txid]['processed'] = display_results(mat_lnk, nb_cmbn, inputs, outputs, fees, intrafees, efficiency, duration)

    return results


if __name__ == '__main__':
    txids = ['f5b1d707ba209ad6a7208effecea8e519081cbaf5d915926f07a5a53aa51ffab',
             'b507d6ac62f50b856ec486357ef2a243a772814223d4d4657561b74b826abe97']

    analyze_txs(txids)

