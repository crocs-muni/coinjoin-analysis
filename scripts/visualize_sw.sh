#!/usr/bin/env bash

# Prepare expected environment
BASE_PATH=$HOME
source $BASE_PATH/btc/coinjoin-analysis/scripts/activate_env.sh

# Run generation of aggregated plots 
python3 -m cj_process.parse_dumplings --cjtype sw --action plot_coinjoins --env_vars "PLOT_REMIXES_MULTIGRAPH=False" --target-path $TMP_DIR/ | tee parse_dumplings.py.log

# Run generation of plots only for intervals
python3 -m cj_process.parse_dumplings --cjtype sw --action plot_coinjoins --target-path $TMP_DIR/ --env_vars "PLOT_REMIXES_SINGLE_INTERVAL=True" | tee parse_dumplings.py.log

