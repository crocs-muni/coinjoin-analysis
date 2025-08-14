#!/usr/bin/env bash

# Prepare expected environment
BASE_PATH=$HOME
source $BASE_PATH/btc/coinjoin-analysis/scripts/activate_env.sh

# Run generation of aggregated plots 
python3 -m cj_process.parse_dumplings --cjtype jm --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=False" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of multigraph plots
#python3 -m cj_process.parse_dumplings --cjtype jm  --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=True" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of plots only for specific intervals
python3 -m cj_process.parse_dumplings --cjtype jm --action plot_coinjoins --target-path $TMP_DIR/ --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True;interval_start_date='2015-01-01 00:00:00.000'" | tee parse_dumplings.py.log

# Another visualization graphs (older)
#python3 -m cj_process.parse_dumplings --cjtype sw --target-path $TMP_DIR/ --env_vars "VISUALIZE_ALL_COINJOINS_INTERVALS=True;interval_start_date='2025-05-30 00:00:07.000';MIX_IDS=['whirlpool_ashigaru_25M', 'whirlpool_ashigaru_2_5M']" | tee parse_dumplings.py.log

