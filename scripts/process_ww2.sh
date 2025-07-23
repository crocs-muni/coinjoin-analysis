#!/bin/bash
BASE_PATH=$HOME

TMP_DIR="$BASE_PATH/btc/dumplings_temp2"


# Start processing in virtual environment
source $BASE_PATH/btc/coinjoin-analysis/venv/bin/activate 

# Go to analysis folder with scripts
cd $BASE_PATH/btc/coinjoin-analysis/src

# Copy processed metadata 
#cp $BASE_PATH/btc/coinjoin-analysis/data/wasabi2/wasabi2_wallet_predictions.json $TMP_DIR/Scanner/


# Extract and process Dumplings results
python3 -m cj_process.parse_dumplings --cjtype ww2 --action process_dumplings --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Copy already known false positives from false_cjtxs.json
for dir in wasabi2 wasabi2_others wasabi2_zksnacks; do
    cp $BASE_PATH/btc/coinjoin-analysis/data/wasabi2/false_cjtxs.json $TMP_DIR/Scanner/$dir/
done

# Download historical fee rates
curl -sSL "https://mempool.space/api/v1/mining/blocks/fee-rates/all" > $TMP_DIR/Scanner/wasabi2/fee_rates.json
for dir in wasabi2_others wasabi2_zksnacks; do
    cp $TMP_DIR/Scanner/wasabi2/fee_rates.json $TMP_DIR/Scanner/$dir/fee_rates.json
done

# Run false positives detection
python3 -m cj_process.parse_dumplings --cjtype ww2 --action detect_false_positives --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run coordinators detection
for dir in wasabi2 wasabi2_others wasabi2_zksnacks; do
    cp $BASE_PATH/btc/coinjoin-analysis/data/wasabi2/txid_coord.json $TMP_DIR/Scanner/$dir/
    cp $BASE_PATH/btc/coinjoin-analysis/data/wasabi2/txid_coord_t.json $TMP_DIR/Scanner/$dir/
done
python3 -m cj_process.parse_dumplings --cjtype ww2 --action detect_coordinators --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run split of post-zksnacks coordinators
python3 -m cj_process.parse_dumplings --cjtype ww2 --action split_coordinators --target-path $TMP_DIR/ | tee parse_dumplings.py.log
# Copy fee rates into newly created folders (selected ones)
for dir in kruw gingerwallet opencoordinator wasabicoordinator coinjoin_nl wasabist dragonordnance mega btip strange_2025; do
    cp $TMP_DIR/Scanner/wasabi2/fee_rates.json $TMP_DIR/Scanner/wasabi2_$dir/
    cp $TMP_DIR/Scanner/wasabi2/false_cjtxs.json $TMP_DIR/Scanner/wasabi2_$dir/
done

# Run detection of Bybit hack
python3 -m cj_process.parse_dumplings --cjtype ww2 --env_vars="ANALYSIS_BYBIT_HACK=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Analyse liquidity 
python3 -m cj_process.parse_dumplings --cjtype ww2 --target-path $TMP_DIR/ --env_vars "ANALYSIS_LIQUIDITY=True" | tee parse_dumplings.py.log

# Run generation of aggregated plots for all coordinators
python3 -m cj_process.parse_dumplings --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=False" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of plots for single intervals (only for selected coordinators)
python3 -m cj_process.parse_dumplings --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True;MIX_IDS=['wasabi2_zksnacks', 'wasabi2_kruw']" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of (time-consuming) multigraph plots (only for selected coordinators)
python3 -m cj_process.parse_dumplings --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=True;MIX_IDS=['wasabi2_zksnacks', 'wasabi2_kruw']" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of multigraph plots for all coordinators (very time consuming)
#python3 -m cj_process.parse_dumplings --cjtype ww2 --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Another visualization graphs (older)
python3 -m cj_process.parse_dumplings --cjtype ww2 --target-path $TMP_DIR/ --env_vars "VISUALIZE_ALL_COINJOINS_INTERVALS=True" | tee parse_dumplings.py.log



