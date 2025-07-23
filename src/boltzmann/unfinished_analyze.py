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
