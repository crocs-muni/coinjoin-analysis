#!/bin/bash
BASE_PATH=$HOME

TMP_DIR="$BASE_PATH/btc/dumplings_temp2"

# Start processing in virtual environment
source $BASE_PATH/btc/coinjoin-analysis/venv/bin/activate 

# Go to analysis folder with scripts
cd $BASE_PATH/btc/coinjoin-analysis/src

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

# Run generation of aggregated plots 
python3 -m cj_process.parse_dumplings --cjtype jm --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=False" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of multigraph plots
#python3 -m cj_process.parse_dumplings --cjtype jm  --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of plots only for specific intervals
#python3 -m cj_process.parse_dumplings --cjtype jm --action plot_coinjoins --target-path $TMP_DIR/ --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True;interval_start_date='2015-01-01 00:00:00.000'" | tee parse_dumplings.py.log

# Another visualization graphs (older)
#python3 -m cj_process.parse_dumplings --cjtype sw --target-path $TMP_DIR/ --env_vars "VISUALIZE_ALL_COINJOINS_INTERVALS=True;interval_start_date='2025-05-30 00:00:07.000';MIX_IDS=['whirlpool_ashigaru_25M', 'whirlpool_ashigaru_2_5M']" | tee parse_dumplings.py.log


#
# Run check for created files
#
#python3 -m cj_process.file_check $TMP_DIR/Scanner/  | tee parse_dumplings.py.log
