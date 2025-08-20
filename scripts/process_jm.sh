#!/usr/bin/env bash

# Prepare expected environment
BASE_PATH=$HOME
source $BASE_PATH/btc/coinjoin-analysis/scripts/activate_env.sh

# Restore false positives from WW1 and WW2 into inputs for JoinMarket
python3 -m cj_process.parse_dumplings --cjtype jm --env_vars "RESTORE_FALSE_POSITIVES_FOR_OTHERS=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log


# Extract and process Dumplings results
python3 -m cj_process.parse_dumplings --cjtype jm --action process_dumplings --env_vars "" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Copy already known false positives from false_cjtxs.json
for dir in joinmarket_all; do
    cp $BASE_PATH/btc/coinjoin-analysis/data/joinmarket/false_cjtxs.json $TMP_DIR/Scanner/$dir/
done

# Download historical fee rates
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/joinmarket_all/fee_rates.json
#for dir in joinmarket_all; do
#    cp $TMP_DIR/Scanner/joinmarket_all/fee_rates.json $TMP_DIR/Scanner/$dir/fee_rates.json
#done

# Run false positives detection
python3 -m cj_process.parse_dumplings --cjtype jm --action detect_false_positives --env_vars "" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Analyse liquidity 
python3 -m cj_process.parse_dumplings --cjtype jm --target-path $TMP_DIR/ --env_vars "ANALYSIS_LIQUIDITY=True" | tee parse_dumplings.py.log

