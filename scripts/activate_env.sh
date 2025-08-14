#!/usr/bin/env bash

BASE_PATH=$HOME
TMP_DIR="$BASE_PATH/btc/dumplings_temp2"

# Start processing in virtual environment
source $BASE_PATH/btc/coinjoin-analysis/venv/bin/activate 

# Go to analysis folder with scripts
cd $BASE_PATH/btc/coinjoin-analysis/src

